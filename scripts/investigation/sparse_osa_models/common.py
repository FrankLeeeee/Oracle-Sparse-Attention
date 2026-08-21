# SPDX-License-Identifier: Apache-2.0
"""Shared runner for the dense-vs-OSA comparison on the capped-window models.

Extends the Self-Forcing study (scripts/investigation/sparse_osa) to the three
other block-causal video models: Rolling Forcing 1.3B (rolling-window joint
denoising), LongLive-2.0 5B (8-frame chunks, 32-frame window) and
LingBot-World v2 14B causal (realtime I2V, 3-frame chunks). Rolling Forcing
and LongLive-2 run through one-shot `sglang generate`; LingBot only rolls out
chunks inside a realtime WebSocket session, so its runs start `sglang serve`
once per sparse config and drive one session per (prompt, duration).

A run's figure of merit is the *achieved* cumulative read density the backend
reports (dense fallbacks counted as 1.0). Unlike full-context Self-Forcing,
these models cap their attention window, so OSA's kept-whole frames (own
chunk + sink + recent) are a much larger fraction of the visible keys and the
density floor sits higher.
"""

import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa_models")
WS_CLIENT = REPO / "scripts/investigation/runtime_breakdown/lingbot_ws_client.py"
SEED = 42

# 16 fps Wan models: pixel frames = 4 * latent - 3; 24 fps LongLive-2.
MODELS = {
    "rolling_forcing": {
        "path": "frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers",
        "frames": {5: 81, 20: 321},
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
        "fps": 16,
    },
    "longlive2": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "frames": {5: 125, 20: 477},
        "resolutions": {"480p": (832, 480), "720p": (1280, 704)},
        "kind": "generate",
        "fps": 24,
    },
    "lingbot_world_v2": {
        "path": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "frames": {5: 81, 20: 321},
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "realtime",
        "fps": 16,
    },
}

# Same prompts as the Self-Forcing study: p0 is the main single-prompt
# experiment, p1-p5 the multi-prompt validation.
PROMPTS = {
    "p0_tokyo": (
        "A stylish woman walks down a Tokyo street filled with warm glowing "
        "neon and animated city signage. She wears a black leather jacket, a "
        "long red dress, and black boots, and carries a black purse. She wears "
        "sunglasses and red lipstick. She walks confidently and casually. The "
        "street is damp and reflective, creating a mirror effect of the "
        "colorful lights. Many pedestrians walk about."
    ),
    "p1_forest": (
        "A dynamic and chaotic scene in a dense forest during a heavy "
        "rainstorm, capturing a real girl frantically running through the "
        "foliage. Her wild hair flows behind her as she sprints, her arms "
        "flailing and her face contorted in fear and desperation. Behind her, "
        "various animals—rabbits, deer, and birds—are also running, creating "
        "a frenzied atmosphere. The girl's clothes are soaked, clinging to "
        "her body, and she is screaming and shouting as she tries to escape. "
        "The background is a blur of greenery and rain-drenched trees, with "
        "occasional glimpses of the darkening sky. A wide-angle shot from a "
        "low angle, emphasizing the urgency and chaos of the moment."
    ),
    "p2_plating": (
        "A dynamic over-the-shoulder perspective of a chef meticulously "
        "plating a dish in a bustling kitchen. The chef, a middle-aged man "
        "with a neatly trimmed beard and focused expression, deftly arranges "
        "ingredients on a pristine white plate. His hands move with "
        "precision, each gesture deliberate and practiced. The background "
        "shows a crowded kitchen with steaming pots, whirring blenders, and "
        "the clatter of utensils. Bright lights highlight the scene, casting "
        "shadows across the busy workspace. The camera angle captures the "
        "chef's detailed work from behind, emphasizing his skill and "
        "dedication."
    ),
    "p3_raccoon": (
        "A playful raccoon is seen playing an electronic guitar, strumming "
        "the strings with its front paws. The raccoon has distinctive black "
        "facial markings and a bushy tail. It sits comfortably on a small "
        "stool, its body slightly tilted as it focuses intently on the "
        "instrument. The setting is a cozy, dimly lit room with vintage "
        "posters on the walls, adding a retro vibe. The raccoon's expressive "
        "eyes convey a sense of joy and concentration. Medium close-up shot, "
        "focusing on the raccoon's face and hands interacting with the "
        "guitar."
    ),
    "p4_teacup": (
        "A close-up shot of a ceramic teacup slowly pouring water into a "
        "glass mug. The water flows smoothly from the spout of the teacup "
        "into the mug, creating gentle ripples as it fills up. Both cups have "
        "detailed textures, with the teacup having a matte finish and the "
        "glass mug showcasing clear transparency. The background is a blurred "
        "kitchen countertop, adding context without distracting from the "
        "central action. The pouring motion is fluid and natural, emphasizing "
        "the interaction between the two cups."
    ),
    "p5_tsunami": (
        "A dramatic and dynamic scene in the style of a disaster movie, "
        "depicting a powerful tsunami rushing through a narrow alley in "
        "Bulgaria. The water is turbulent and chaotic, with waves crashing "
        "violently against the walls and buildings on either side. The alley "
        "is lined with old, weathered houses, their facades partially "
        "submerged and splintered. The camera angle is low, capturing the "
        "full force of the tsunami as it surges forward, creating a sense of "
        "urgency and danger. People can be seen running frantically, adding "
        "to the chaotic atmosphere. The scene is filled with dramatic "
        "lighting and intense motion, emphasizing the destructive power of "
        "the tsunami."
    ),
}
MAIN_PROMPT = "p0_tokyo"


def osa_config(density: float) -> dict:
    """The steady-state OSA knob set used across the whole study."""
    return {
        "density": density,
        "sink_latent_frames": 1,
        "num_recent_frames": 1,
    }


SHEET_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DENSITY_LINE = re.compile(
    r"attention density so far: ([0-9.]+) over (\d+) calls \((\d+) dense\)"
)
STAGE_LINE = re.compile(r"\[(\w+Stage)\] finished in ([0-9.]+) seconds")
E2E_LINE = re.compile(r"Pixel data generated successfully in ([0-9.]+) seconds")

GENERATE_TIMEOUT_S = 7200
SERVER_READY_TIMEOUT_S = 1800
GPU_WAIT_TIMEOUT_S = 14400


def base_env(gpu: int, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        SGLANG_DIFFUSION_SYNC_STAGE_PROFILING="1",
    )
    env.update(extra or {})
    return env


def gpu_used_mib(gpu: int) -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(output.strip())


def wait_for_free_gpu(gpu: int, *, max_used_mib: int = 2048) -> None:
    """Wall-time runs on a shared box: wait out co-tenants, don't measure through them."""
    deadline = time.time() + GPU_WAIT_TIMEOUT_S
    warned = False
    while time.time() < deadline:
        if gpu_used_mib(gpu) <= max_used_mib:
            return
        if not warned:
            print(f"[gpu{gpu}] busy, waiting for it to free up", flush=True)
            warned = True
        time.sleep(60)
    raise RuntimeError(f"gpu {gpu} still busy after {GPU_WAIT_TIMEOUT_S}s")


def sparse_args(method: str | None, method_config: dict | None) -> list[str]:
    args = []
    if method is not None:
        args += ["--sparse-attention", method]
        if method_config:
            args += ["--sparse-attention-config", json.dumps(method_config)]
    return args


def parse_log(log: pathlib.Path) -> dict:
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    out: dict = {}
    densities = DENSITY_LINE.findall(text)
    if densities:
        density, calls, dense_calls = densities[-1]
        out.update(
            density=float(density),
            density_calls=int(calls),
            density_dense_calls=int(dense_calls),
        )
    for stage, seconds in STAGE_LINE.findall(text):
        key = "denoise_s" if stage.endswith("DenoisingStage") else None
        if stage.endswith("DecodingStage"):
            key = "vae_decode_s"
        if key:
            out[key] = out.get(key, 0.0) + float(seconds)
    e2e = E2E_LINE.findall(text)
    if e2e:
        out["e2e_s"] = float(e2e[-1])
    return out


def run_generate(
    *,
    model: str,
    out_dir: pathlib.Path,
    gpu: int,
    port_base: int,
    duration: int,
    res: str = "720p",
    prompt_key: str = MAIN_PROMPT,
    method: str | None = None,
    method_config: dict | None = None,
    wait_gpu: bool = True,
) -> dict:
    """One `sglang generate` run; returns parsed timings and density."""
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "run.log"
    args = [
        "sglang",
        "generate",
        "--model-path",
        spec["path"],
        "--prompt",
        PROMPTS[prompt_key],
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(spec["frames"][duration]),
        "--seed",
        str(SEED),
        "--save-output",
        "--master-port",
        str(port_base),
        "--scheduler-port",
        str(port_base + 1),
        "--port",
        str(port_base + 2),
    ] + sparse_args(method, method_config)
    if wait_gpu:
        wait_for_free_gpu(gpu)
    started = time.time()
    with open(log, "w") as handle:
        proc = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=base_env(gpu),
            cwd=out_dir,
            timeout=GENERATE_TIMEOUT_S,
        )
    result = parse_log(log)
    result["returncode"] = proc.returncode
    result["wall_s"] = round(time.time() - started, 1)
    result["config"] = method_config
    return result


# --------------------------------------------------------------------------
# LingBot realtime: one server per sparse config, one session per prompt
# --------------------------------------------------------------------------


def condition_frame(prompt_key: str, res: str = "720p") -> pathlib.Path:
    """LingBot's I2V condition image: frame 0 of the Rolling Forcing *dense*
    run for the same prompt, seed and resolution.

    The condition image dominates the scene, so it must match the prompt (see
    the chunk_runtime investigation); a T2V model's own frame 0 is the
    cheapest prompt-faithful source. Run the Rolling Forcing sweeps first.
    """
    target = ROOT / "first_frames" / f"{prompt_key}_{res}.png"
    if target.exists():
        return target
    rf_root = ROOT / "rolling_forcing"
    if prompt_key == MAIN_PROMPT:
        pattern = str(rf_root / "runs" / "dense" / "**" / "*.mp4")
    else:
        pattern = str(rf_root / "runs_prompts" / f"{prompt_key}_dense" / "**" / "*.mp4")
    sources = sorted(glob.glob(pattern, recursive=True))
    if not sources:
        raise RuntimeError(
            f"no Rolling Forcing dense video for {prompt_key} to take the "
            f"condition frame from ({pattern}); run the rolling_forcing "
            "sweeps first"
        )
    import imageio.v2 as imageio_v2

    target.parent.mkdir(parents=True, exist_ok=True)
    imageio_v2.imwrite(target, imageio_v2.get_reader(sources[-1]).get_data(0))
    print(f"wrote LingBot condition frame {target}", flush=True)
    return target


class LingbotServer:
    """`sglang serve` for LingBot with one sparse config, health-checked."""

    def __init__(
        self,
        *,
        gpu: int,
        port_base: int,
        server_dir: pathlib.Path,
        method: str | None,
        method_config: dict | None,
    ) -> None:
        self.port = port_base + 2
        self.server_dir = server_dir
        self.log = server_dir / "server.log"
        server_dir.mkdir(parents=True, exist_ok=True)
        wait_for_free_gpu(gpu)
        args = [
            "sglang",
            "serve",
            "--model-path",
            MODELS["lingbot_world_v2"]["path"],
            "--master-port",
            str(port_base),
            "--scheduler-port",
            str(port_base + 1),
            "--port",
            str(self.port),
        ] + sparse_args(method, method_config)
        self._handle = open(self.log, "w")
        self._proc = subprocess.Popen(
            args,
            stdout=self._handle,
            stderr=subprocess.STDOUT,
            env=base_env(gpu),
            cwd=server_dir,
        )

    def wait_ready(self) -> None:
        import urllib.request

        deadline = time.time() + SERVER_READY_TIMEOUT_S
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"server died, see {self.log}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=5
                ) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(f"server on port {self.port} not healthy, see {self.log}")

    def density_state(self) -> tuple[float, int]:
        """Cumulative (density, calls) the server has logged so far."""
        if not self.log.exists():
            return 1.0, 0
        found = DENSITY_LINE.findall(self.log.read_text())
        if not found:
            return 1.0, 0
        density, calls, _ = found[-1]
        return float(density), int(calls)

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._handle.close()

    def __enter__(self) -> "LingbotServer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_lingbot_session(
    server: LingbotServer,
    *,
    out_dir: pathlib.Path,
    duration: int,
    res: str = "720p",
    prompt_key: str = MAIN_PROMPT,
) -> dict:
    """One realtime session; denoise time is the summed per-chunk forward."""
    spec = MODELS["lingbot_world_v2"]
    width, height = spec["resolutions"][res]
    out_dir.mkdir(parents=True, exist_ok=True)
    density_before, calls_before = server.density_state()
    started = time.time()
    with open(out_dir / "run.log", "w") as handle:
        proc = subprocess.run(
            [
                "python",
                str(WS_CLIENT),
                "--port",
                str(server.port),
                "--model-path",
                spec["path"],
                "--prompt",
                PROMPTS[prompt_key],
                "--first-frame",
                str(condition_frame(prompt_key, res)),
                "--size",
                f"{width}x{height}",
                "--num-frames",
                str(spec["frames"][duration]),
                "--fps",
                str(spec["fps"]),
                "--save-video",
                str(out_dir / "video.mp4"),
                "--out",
                str(out_dir / "session.json"),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=str(REPO / "python")),
            timeout=GENERATE_TIMEOUT_S,
        )
    result: dict = {"returncode": proc.returncode}
    result["wall_s"] = round(time.time() - started, 1)
    session_path = out_dir / "session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text())
        result["session_wall_s"] = round(session["session_wall_s"], 1)
        forward_ms = [
            stats.get("scheduler_forward_ms")
            for stats in session.get("chunk_stats", [])
        ]
        forward_ms = [ms for ms in forward_ms if ms is not None]
        if forward_ms:
            # The per-chunk pipeline forward covers denoising + KV refresh —
            # the part sparse attention can speed up.
            result["denoise_s"] = round(sum(forward_ms) / 1000.0, 1)
            result["chunks"] = len(forward_ms)
    # Density accumulates across the server's sessions; report this session's
    # contribution from the running (density, calls) figure.
    time.sleep(5)  # let the last density report land in the log
    density_after, calls_after = server.density_state()
    if calls_after > calls_before:
        session_sum = density_after * calls_after - density_before * calls_before
        result["density"] = round(session_sum / (calls_after - calls_before), 4)
    # Sessions need pacing; the server rejects a new one mid-dispose.
    time.sleep(15)
    return result


# --------------------------------------------------------------------------
# Frame sheets (same rendering as the Self-Forcing study)
# --------------------------------------------------------------------------


def extract_frames(
    run_dir: pathlib.Path, frame_indices: list[int]
) -> list[np.ndarray] | None:
    """Half-res frames at the given indices from the newest mp4 under run_dir."""
    paths = (
        sorted(glob.glob(str(run_dir / "outputs" / "*.mp4")))
        or sorted(glob.glob(str(run_dir / "*.mp4")))
        or sorted(glob.glob(str(run_dir / "**" / "*.mp4"), recursive=True))
    )
    if not paths:
        return None
    found: dict[int, np.ndarray] = {}
    last: np.ndarray | None = None
    for index, frame in enumerate(imageio.get_reader(paths[-1])):
        last = np.asarray(frame)[::2, ::2]
        if index in frame_indices:
            found[index] = last
        if index >= max(frame_indices):
            break
    if last is None:
        return None
    return [found.get(index, last) for index in frame_indices]


def render_frame_sheet(
    *,
    rows: list[tuple[str, list[np.ndarray]]],
    frame_indices: list[int],
    fps: int,
    out_path: pathlib.Path,
) -> None:
    """Tile sheet with method-name row labels and frame-number column labels."""
    from PIL import Image, ImageDraw, ImageFont

    tile_h, tile_w = rows[0][1][0].shape[:2]
    row_font = ImageFont.truetype(SHEET_FONT, 30)
    col_font = ImageFont.truetype(SHEET_FONT, 26)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    margin = int(max(probe.textlength(label, font=row_font) for label, _ in rows)) + 32
    header = 56
    sheet = Image.new(
        "RGB",
        (margin + tile_w * len(frame_indices), header + tile_h * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for col, frame_index in enumerate(frame_indices):
        text = f"frame {frame_index} ({frame_index / fps:.0f} s)"
        text_w = probe.textlength(text, font=col_font)
        draw.text(
            (margin + col * tile_w + (tile_w - text_w) / 2, 13),
            text,
            fill="black",
            font=col_font,
        )
    for row, (label, frames) in enumerate(rows):
        top = header + row * tile_h
        draw.text((16, top + tile_h / 2 - 18), label, fill="black", font=row_font)
        for col, frame in enumerate(frames):
            sheet.paste(Image.fromarray(frame), (margin + col * tile_w, top))
    sheet.save(out_path)


def sheet_frame_indices(*, fps: int, duration: int, count: int = 7) -> list[int]:
    """`count` frame indices spread over t = 1 .. duration-1 seconds."""
    times = np.linspace(1, duration - 1, count)
    return [int(round(t * fps)) for t in times]
