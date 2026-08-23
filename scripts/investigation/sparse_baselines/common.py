# SPDX-License-Identifier: Apache-2.0
"""Shared runner for the sparse-attention baseline study across five models.

Unifies the Self-Forcing study (scripts/investigation/sparse_osa) and the
capped-window OSA study (scripts/investigation/sparse_osa_models): every
block-causal video model in the fleet runs every sparse-attention method —
dense, OSA, LightForcing, Radial, SVG1, SVG2, XAttention, STA — through the
same calibration / sweep / quality / multi-prompt protocol.

Models:
  self_forcing      1.3B full context     (`sglang generate`)
  causal_forcing    1.3B 21-frame window  (`sglang generate`)
  rolling_forcing   1.3B rolling windows  (`sglang generate`)
  longlive2         5B 8f chunks, 32f cap (`sglang generate`)
  lingbot_world_v2  14B realtime I2V      (`sglang serve` + WebSocket session)

A run's figure of merit is the *achieved* cumulative read density the backend
reports (dense fallbacks counted as 1.0).

GPU discipline (shared box): a run only starts on an idle GPU, a watchdog
kills it if a co-tenant process appears mid-run, and the caller re-queues it —
timings measured through contention are invalid.
"""

import glob
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_baselines")
WS_CLIENT = REPO / "scripts/investigation/runtime_breakdown/lingbot_ws_client.py"
SEED = 42

# Latent-frame geometry per model (frames maps duration seconds -> pixel
# frames; all Wan models are 16 fps with 4*latent-3 pixel frames, LongLive-2
# is 24 fps). window_frames is the attention cap in latent frames (-1 = full
# context) and sink_frames the model's pinned sink block — both feed the
# per-model method configs below.
MODELS = {
    "self_forcing": {
        "path": "/data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers-fullctx-null",
        "frames": {5: 81, 20: 321},
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
        "fps": 16,
        "token_downsample": 16,
        "window_frames": -1,
        "sink_frames": 1,
        "latents_20s": 81,
    },
    "causal_forcing": {
        "path": "frankleeeee/CausalForcing-Wan2.1-T2V-1.3B-Diffusers",
        "frames": {5: 81, 20: 321},
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
        "fps": 16,
        "token_downsample": 16,
        "window_frames": 21,
        "sink_frames": 1,
        "latents_20s": 81,
    },
    "rolling_forcing": {
        "path": "frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers",
        "frames": {5: 81, 20: 321},
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
        "fps": 16,
        "token_downsample": 16,
        "window_frames": 21,
        "sink_frames": 3,
        "latents_20s": 81,
    },
    "longlive2": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "frames": {5: 125, 20: 477},
        "resolutions": {"480p": (832, 480), "720p": (1280, 704)},
        "kind": "generate",
        "fps": 24,
        "token_downsample": 32,
        "window_frames": 32,
        "sink_frames": 8,
        "latents_20s": 120,
    },
    "lingbot_world_v2": {
        "path": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "frames": {5: 81, 20: 321},
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "realtime",
        "fps": 16,
        "token_downsample": 16,
        "window_frames": 18,
        "sink_frames": 9,
        "latents_20s": 81,
    },
}

METHODS = ("osa", "lightforcing", "radial", "svg1", "svg2", "xattention", "sta")

METHOD_LABELS = {
    "dense": "Dense",
    "osa": "OSA",
    "lightforcing": "LightForcing",
    "radial": "Radial",
    "svg1": "SVG1",
    "svg2": "SVG2",
    "xattention": "XAttention",
    "sta": "STA",
}


def method_base_config(method: str, model: str) -> dict:
    """The non-knob part of a method's config, resolved per model.

    LightForcing's schedule needs the run's video length and window cap up
    front (upstream computes them from CLI args); the sink-aware methods
    (SVG1's dense columns, Radial's exempt frames, LightForcing's keep_sink)
    get the model's actual sink block size.
    """
    spec = MODELS[model]
    sink = spec["sink_frames"]
    if method == "osa":
        return {"sink_latent_frames": 1, "num_recent_frames": 1}
    if method == "lightforcing":
        return {
            "num_output_frames": spec["latents_20s"],
            "local_attn_size": spec["window_frames"],
            "keep_sink": sink,
            "keep_near": 2,
            "keep_frames": sink + 2 + 3,
        }
    if method == "svg1":
        return {"dense_sink_frames": sink}
    if method == "radial":
        return {"dense_sink_frames": sink}
    return {}


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

SHEET_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DENSITY_LINE = re.compile(
    r"attention density so far: ([0-9.]+) over (\d+) calls \((\d+) dense\)"
)
STAGE_LINE = re.compile(r"\[(\w+Stage)\] finished in ([0-9.]+) seconds")
E2E_LINE = re.compile(r"Pixel data generated successfully in ([0-9.]+) seconds")

GENERATE_TIMEOUT_S = 7200
SERVER_READY_TIMEOUT_S = 1800
GPU_WAIT_TIMEOUT_S = 28800


class GpuContended(RuntimeError):
    """A co-tenant process appeared on the GPU mid-run; the timing is invalid."""


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


def _gpu_uuid(gpu: int) -> str:
    return subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _descendants(pid: int) -> set[int]:
    """pid plus all its descendants, via one `ps` snapshot."""
    output = subprocess.run(
        ["ps", "-eo", "pid,ppid"], capture_output=True, text=True
    ).stdout
    children: dict[int, list[int]] = {}
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        child, parent = int(parts[0]), int(parts[1])
        children.setdefault(parent, []).append(child)
    seen = {pid}
    stack = [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def foreign_compute_pids(gpu: int, own_root_pid: int | None = None) -> set[int]:
    """Compute-app PIDs on ``gpu`` that are not ours.

    PIDs from other containers are invisible in our /proc, so any compute PID
    that is not a descendant of ``own_root_pid`` counts as foreign.
    """
    uuid = _gpu_uuid(gpu)
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pids = set()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == uuid:
            pids.add(int(parts[1]))
    if own_root_pid is not None:
        pids -= _descendants(own_root_pid)
    return pids


def wait_for_exclusive_gpu(gpu: int) -> None:
    """Block until no other process computes on ``gpu``."""
    deadline = time.time() + GPU_WAIT_TIMEOUT_S
    warned = False
    while time.time() < deadline:
        if not foreign_compute_pids(gpu):
            return
        if not warned:
            print(f"[gpu{gpu}] occupied, waiting for it to go idle", flush=True)
            warned = True
        time.sleep(60)
    raise RuntimeError(f"gpu {gpu} still occupied after {GPU_WAIT_TIMEOUT_S}s")


class GpuPool:
    """Hand out whichever candidate GPU is exclusively idle right now.

    On this shared box GPUs come and go; pinning a worker to a busy GPU
    starves it, so workers acquire dynamically and release when done. A GPU
    is grantable when no foreign compute process runs on it and no other
    worker of this pool holds it.
    """

    def __init__(self, candidates: list[int]):
        self._candidates = candidates
        self._held: set[int] = set()
        self._lock = threading.Lock()

    def acquire(self, *, timeout_s: int = GPU_WAIT_TIMEOUT_S) -> int:
        deadline = time.time() + timeout_s
        warned = False
        while time.time() < deadline:
            for gpu in self._candidates:
                with self._lock:
                    if gpu in self._held:
                        continue
                try:
                    busy = bool(foreign_compute_pids(gpu))
                except Exception:
                    continue
                if busy:
                    continue
                with self._lock:
                    if gpu in self._held:
                        continue
                    self._held.add(gpu)
                return gpu
            if not warned:
                print(
                    f"[pool] all of {self._candidates} occupied, waiting",
                    flush=True,
                )
                warned = True
            time.sleep(60)
        raise RuntimeError(f"no candidate GPU went idle within {timeout_s}s")

    def release(self, gpu: int) -> None:
        with self._lock:
            self._held.discard(gpu)


class GpuWatchdog:
    """Kill the watched process if a co-tenant appears on the GPU."""

    def __init__(self, gpu: int, proc: subprocess.Popen, *, interval_s: int = 30):
        self._gpu = gpu
        self._proc = proc
        self._interval = interval_s
        self.contended = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.wait(self._interval):
            if self._proc.poll() is not None:
                return
            try:
                foreign = foreign_compute_pids(self._gpu, self._proc.pid)
            except Exception:
                continue
            if foreign:
                self.contended = True
                print(
                    f"[gpu{self._gpu}] co-tenant appeared ({sorted(foreign)}), "
                    "killing the run",
                    flush=True,
                )
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


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
    max_retries: int = 20,
) -> dict:
    """One exclusive-GPU `sglang generate` run; re-queues on contention."""
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

    for attempt in range(max_retries):
        wait_for_exclusive_gpu(gpu)
        started = time.time()
        with open(log, "w") as handle:
            proc = subprocess.Popen(
                args,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=base_env(gpu),
                cwd=out_dir,
                start_new_session=True,
            )
            watchdog = GpuWatchdog(gpu, proc)
            try:
                proc.wait(timeout=GENERATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            finally:
                watchdog.stop()
        if watchdog.contended:
            print(
                f"[gpu{gpu}] rerunning {out_dir.name} after contention "
                f"(attempt {attempt + 1})",
                flush=True,
            )
            continue
        result = parse_log(log)
        result["returncode"] = proc.returncode
        result["wall_s"] = round(time.time() - started, 1)
        result["config"] = method_config
        return result
    raise GpuContended(f"gpu {gpu} contended on every attempt for {out_dir}")


# --------------------------------------------------------------------------
# LingBot realtime: one server per sparse config, one session per prompt
# --------------------------------------------------------------------------


def condition_frame(prompt_key: str, res: str = "720p") -> pathlib.Path:
    """LingBot's I2V condition image: frame 0 of the Rolling Forcing *dense*
    run for the same prompt, seed and resolution.

    The condition image dominates the scene, so it must match the prompt; a
    T2V model's own frame 0 is the cheapest prompt-faithful source. Run the
    rolling_forcing sweeps first.
    """
    target = ROOT / "first_frames" / f"{prompt_key}_{res}.png"
    if target.exists():
        return target
    rf_root = ROOT / "rolling_forcing"
    patterns = [
        str(rf_root / "runs" / "dense" / "**" / "*.mp4"),
        str(rf_root / "runs_prompts" / f"{prompt_key}_dense" / "**" / "*.mp4"),
        str(rf_root / "calibration" / "**" / "*.mp4"),
    ]
    if prompt_key != MAIN_PROMPT:
        patterns = patterns[1:2]
    sources = []
    for pattern in patterns:
        sources = sorted(glob.glob(pattern, recursive=True))
        if sources:
            break
    if not sources:
        raise RuntimeError(
            f"no Rolling Forcing dense video for {prompt_key} to take the "
            "condition frame from; run the rolling_forcing sweeps first"
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
        self.gpu = gpu
        self.port = port_base + 2
        self.server_dir = server_dir
        self.log = server_dir / "server.log"
        server_dir.mkdir(parents=True, exist_ok=True)
        wait_for_exclusive_gpu(gpu)
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
            start_new_session=True,
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
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
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
    """One realtime session; denoise time is the summed per-chunk forward.

    Contention policy: if a co-tenant appears during the session, the result
    is flagged ``contended`` — the caller decides whether to redo the session
    (timing runs) or accept it (density-only calibration).
    """
    spec = MODELS["lingbot_world_v2"]
    width, height = spec["resolutions"][res]
    out_dir.mkdir(parents=True, exist_ok=True)
    density_before, calls_before = server.density_state()
    started = time.time()
    with open(out_dir / "run.log", "w") as handle:
        proc = subprocess.Popen(
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
        )
        contended = False
        deadline = time.time() + GENERATE_TIMEOUT_S
        while proc.poll() is None and time.time() < deadline:
            time.sleep(20)
            try:
                if foreign_compute_pids(server.gpu, os.getpid()):
                    contended = True
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
    result: dict = {"returncode": proc.returncode, "contended": contended}
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
# Frame sheets (same rendering as the earlier studies)
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


def newest_video(run_dir: pathlib.Path) -> pathlib.Path | None:
    paths = (
        sorted(glob.glob(str(run_dir / "outputs" / "*.mp4")))
        or sorted(glob.glob(str(run_dir / "*.mp4")))
        or sorted(glob.glob(str(run_dir / "**" / "*.mp4"), recursive=True))
    )
    return pathlib.Path(paths[-1]) if paths else None


def render_frame_sheet(
    *,
    rows: list[tuple[str, list[np.ndarray]]],
    frame_indices: list[int],
    fps: int,
    out_path: pathlib.Path,
) -> None:
    """Tile sheet with method-name row labels and frame-number column labels.

    Row labels must be ASCII — DejaVuSans has no CJK glyphs.
    """
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


def psnr_vs_dense(
    dense_dir: pathlib.Path, other_dir: pathlib.Path, *, first_seconds: int, fps: int
) -> dict | None:
    """Overall and first-N-seconds PSNR of ``other`` against ``dense``."""
    dense_path = newest_video(dense_dir)
    other_path = newest_video(other_dir)
    if dense_path is None or other_path is None:
        return None
    reader_a = imageio.get_reader(dense_path)
    reader_b = imageio.get_reader(other_path)
    total_sq = 0.0
    total_px = 0
    first_sq = 0.0
    first_px = 0
    count = 0
    for frame_a, frame_b in zip(reader_a, reader_b):
        a = np.asarray(frame_a, dtype=np.float64)
        b = np.asarray(frame_b, dtype=np.float64)
        sq = float(((a - b) ** 2).sum())
        total_sq += sq
        total_px += a.size
        if count < first_seconds * fps:
            first_sq += sq
            first_px += a.size
        count += 1
    if total_px == 0:
        return None

    def to_psnr(sq: float, px: int) -> float:
        mse = sq / px
        return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)

    return {
        "psnr": round(to_psnr(total_sq, total_px), 2),
        "psnr_first": round(to_psnr(first_sq, first_px), 2),
        "frames": count,
    }
