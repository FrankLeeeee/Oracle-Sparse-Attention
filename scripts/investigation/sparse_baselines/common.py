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

import fcntl
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
        "pinned_sink_frames": 0,
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
        "pinned_sink_frames": 0,
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
        "pinned_sink_frames": 3,
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
        "pinned_sink_frames": 8,
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
        "pinned_sink_frames": 9,
        "latents_20s": 81,
    },
}

METHODS = ("osa", "osa2", "osa2s", "osa2a", "lightforcing", "radial", "svg1", "svg2", "xattention", "sta")

METHOD_LABELS = {
    "dense": "Dense",
    # OSA labels are additive: bare "OSA" is the fully sparse pattern (no
    # frames kept whole); each "+ ... full" suffix names an enabled
    # keep_*_full switch.
    "osa": "OSA + own chunk full + sink full + recent full",
    "osa2": "OSA",
    "osa2s": "OSA + sink full",
    "osa2a": "OSA + sink full + recent full",
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
        # num_recent_frames is per-model optimal. 2 is the precision
        # campaign's adopted setting (doc EbY9dmoEVoShFEx7KkScf60bnre P5): at
        # matched density it buys full-context Self-Forcing +0.9 dB overall /
        # +2.7 dB first-5s for free (density floor 0.26 -> 0.31). On the
        # capped-window models the same extra dense frame blows the ~0.30
        # budget instead (Causal Forcing floor 0.36, Rolling Forcing 0.48 —
        # the rolling plans compound it), so they stay at 1.
        recent = 2 if spec["window_frames"] < 0 else 1
        return {"sink_latent_frames": 1, "num_recent_frames": recent}
    if method == "osa2":
        # Fully sparse OSA: nothing kept whole, so the density knob has no
        # geometric floor. (OSA itself is 2-D now; osa2 differs only in
        # whole_frames.)
        return {
            "keep_own_chunk_full": False,
            "keep_sink_full": False,
            "keep_recent_frames_full": False,
        }
    if method == "osa2s":
        # Fully sparse except the sink frame — the dense-anchor middle ground.
        return {
            "keep_own_chunk_full": False,
            "keep_sink_full": True,
            "keep_recent_frames_full": False,
            "sink_latent_frames": 1,
        }
    if method == "osa2a":
        # Fully sparse except the two anchor frames (sink + recent): fixes the
        # chunk-periodic camera oscillation of "none" (the model re-anchors
        # global composition on the sink and temporal smoothness on the recent
        # frame; both under-recalled ~0.6/0.8 when patterned).
        return {
            "keep_own_chunk_full": False,
            "keep_sink_full": True,
            "keep_recent_frames_full": True,
            "sink_latent_frames": 1,
            "num_recent_frames": 1,
        }
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
    if method == "sta":
        # Both pinned rather than left to defaults so the calibration cache is
        # keyed by them: the query-block size changes the executed density (a
        # block unions the windows of the tiles it spans) and the kernel
        # throughput, and the sink exemption changes what is kept. The sink
        # here is the model's *explicit pinned block* (0 where the model has
        # none), not the 1-frame default Radial and SVG1 use.
        return {"block": 128, "dense_sink_frames": spec["pinned_sink_frames"]}
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


# Mid-run tolerance only. A run never *starts* on a GPU that has any CUDA
# context on it — that is the standing rule for this box, and a parked
# context is reason enough to pick another GPU. Once a run is under way,
# though, killing it for a context that appears and vanishes within seconds
# only throws away good work, so the watchdog requires a co-tenant to persist
# across two checks and hold at least this much memory before it invalidates
# the timing.
IDLE_CONTEXT_MIB = 2048


def record_result(path: pathlib.Path, key: str, value: dict) -> None:
    """Merge one run into a results file, atomically.

    Sweeps of the same model can overlap (a targeted redo alongside the full
    chain), and each holding the dict it read at startup would let the second
    writer erase the first's rows — the same lost update that once wiped a
    whole model's calibration. Re-read under an flock and write through a
    temp file.
    """
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            rows = json.loads(path.read_text()) if path.exists() else {}
            rows[key] = value
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(rows, indent=2))
            temporary.replace(path)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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


def compute_apps(gpu: int) -> dict[int, int]:
    """{pid: used MiB} of compute apps on ``gpu``, in the *host* PID namespace.

    We run inside a container with its own PID namespace, so these numbers
    never match our /proc — ownership cannot be decided by pid ancestry.
    Callers therefore reason by *sets over time*: a run only starts on an
    empty GPU, pids appearing in its launch window are blessed as its own,
    and later additions are co-tenants — unless they are transient probe
    blips (gone within seconds, tiny memory), which this box produces
    routinely and which do not disturb a multi-minute timing.
    """
    uuid = _gpu_uuid(gpu)
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    apps: dict[int, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            try:
                apps[int(parts[1])] = int(parts[2])
            except ValueError:
                apps[int(parts[1])] = 0
    return apps


def compute_pids(gpu: int) -> set[int]:
    return set(compute_apps(gpu))


def workload_pids(gpu: int) -> set[int]:
    """Compute apps big enough to actually contend for the GPU.

    Only for reasoning about a run already in flight; acquisition uses
    :func:`compute_pids`, because a GPU with any context on it is not ours
    to take.
    """
    return {pid for pid, mib in compute_apps(gpu).items() if mib >= IDLE_CONTEXT_MIB}


def wait_for_exclusive_gpu(gpu: int) -> None:
    """Block until no process computes on ``gpu``."""
    deadline = time.time() + GPU_WAIT_TIMEOUT_S
    warned = False
    while time.time() < deadline:
        if not compute_pids(gpu):
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
    is grantable when no compute process runs on it, no worker of this pool
    holds it, and no *other campaign process* holds its lockfile — two
    concurrently running drivers (one model's timing sweep, the next model's
    calibration) must never double-book a GPU, and nvidia-smi alone cannot
    show a run that is still loading its model.
    """

    _LOCK_DIR = ROOT / "gpu_locks"

    def __init__(self, candidates: list[int]):
        self._candidates = candidates
        self._held: set[int] = set()
        self._lock = threading.Lock()
        self._LOCK_DIR.mkdir(parents=True, exist_ok=True)

    def _lock_path(self, gpu: int) -> pathlib.Path:
        return self._LOCK_DIR / f"gpu{gpu}.lock"

    def _try_file_lock(self, gpu: int) -> bool:
        path = self._lock_path(gpu)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Stale if its owner process is gone (same container, so /proc
            # is authoritative for our own campaign drivers).
            try:
                owner = int(path.read_text().strip())
            except (ValueError, FileNotFoundError):
                owner = -1
            if owner > 0 and pathlib.Path(f"/proc/{owner}").exists():
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return self._try_file_lock(gpu)
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        return True

    def acquire(self, *, timeout_s: int = GPU_WAIT_TIMEOUT_S) -> int:
        deadline = time.time() + timeout_s
        warned = False
        while time.time() < deadline:
            for gpu in self._candidates:
                with self._lock:
                    if gpu in self._held:
                        continue
                try:
                    busy = bool(compute_pids(gpu))
                except Exception:
                    continue
                if busy:
                    continue
                with self._lock:
                    if gpu in self._held:
                        continue
                    if not self._try_file_lock(gpu):
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
            try:
                self._lock_path(gpu).unlink()
            except FileNotFoundError:
                pass


class GpuWatchdog:
    """Kill the watched process if a co-tenant appears on the GPU.

    nvidia-smi pids live in the host namespace, so ownership is inferred from
    timing: the GPU was empty at launch, so pids appearing during the launch
    window are the run's own processes; later additions are co-tenants. Two
    tolerances keep this from livelocking on this box's routine noise: an
    extra pid must persist across two checks (short-lived probe contexts come
    and go in seconds) and hold non-trivial memory (a real job, not an idling
    probe context) before the run is declared contended.
    """

    # Pids first seen within this window of launch are the run's own
    # (covers model load up to the 5B/14B checkpoints).
    _BLESS_WINDOW_S = 120
    # An extra pid below this footprint that merely idles cannot disturb a
    # multi-minute timing measurably.
    _KILL_MIB = 2048
    # An `sglang generate` run registers exactly this many compute processes
    # (worker + scheduler); anything beyond is a co-tenant even inside the
    # bless window — the cycling job on this box re-acquires its GPUs every
    # few minutes and would otherwise get blessed alongside our own run.
    _EXPECTED_OWN = 2

    def __init__(
        self,
        gpu: int,
        proc: subprocess.Popen,
        *,
        interval_s: int = 15,
        preexisting: set[int] | None = None,
    ):
        self._gpu = gpu
        self._proc = proc
        self._interval = interval_s
        # Parked contexts that were already on the GPU at launch. They are not
        # ours, but they are not contention either — and crucially they must
        # not consume the bless slots, or one of our own two processes would
        # look foreign and get the run killed.
        self._preexisting = set(preexisting or ())
        self.contended = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _kill(self, reason: str) -> None:
        self.contended = True
        print(f"[gpu{self._gpu}] {reason}, killing the run", flush=True)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _watch(self) -> None:
        started = time.time()
        blessed: set[int] = set()
        suspect: set[int] = set()
        while not self._stop.wait(self._interval):
            if self._proc.poll() is not None:
                return
            try:
                apps = compute_apps(self._gpu)
            except Exception:
                continue
            in_window = time.time() - started <= self._BLESS_WINDOW_S
            if in_window and len(blessed) < self._EXPECTED_OWN:
                # Accept new pids as our own, oldest-first, up to the count
                # a run actually creates.
                for pid in sorted(set(apps) - blessed - self._preexisting):
                    if len(blessed) >= self._EXPECTED_OWN:
                        break
                    blessed.add(pid)
            extra = set(apps) - blessed - self._preexisting
            # Confirmed co-tenant: seen on two consecutive checks with a
            # real memory footprint.
            confirmed = {
                pid for pid in extra & suspect if apps.get(pid, 0) >= self._KILL_MIB
            }
            if confirmed:
                self._kill(
                    "co-tenant appeared "
                    f"({sorted((pid, apps[pid]) for pid in confirmed)} MiB)"
                )
                return
            if extra - suspect:
                print(
                    f"[gpu{self._gpu}] transient/small extra process "
                    f"{sorted((pid, apps.get(pid, 0)) for pid in extra - suspect)}"
                    " MiB, watching",
                    flush=True,
                )
            suspect = extra

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# Variant method keys: an experiment method that runs an existing backend
# under a different base config. Tags/results keep the variant name.
BACKEND_OF = {"osa2": "osa", "osa2s": "osa", "osa2a": "osa"}


def sparse_args(method: str | None, method_config: dict | None) -> list[str]:
    args = []
    if method is not None:
        args += ["--sparse-attention", BACKEND_OF.get(method, method)]
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
    max_retries: int = 3,
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
        parked = compute_pids(gpu)
        with open(log, "w") as handle:
            proc = subprocess.Popen(
                args,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=base_env(gpu),
                cwd=out_dir,
                start_new_session=True,
            )
            watchdog = GpuWatchdog(gpu, proc, preexisting=parked)
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
        result["gpu"] = gpu
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
    # Only a *dense* Rolling Forcing run is an acceptable source: a sparse
    # run's frame 0 would fold that method's artifacts into every LingBot
    # video, including the dense reference's.
    patterns = [
        str(rf_root / "runs" / "dense" / "**" / "*.mp4"),
        str(rf_root / "runs_prompts" / f"{prompt_key}_dense" / "**" / "*.mp4"),
    ]
    if prompt_key != MAIN_PROMPT:
        patterns = patterns[1:]
    sources: list[str] = []
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
    from PIL import Image

    width, height = MODELS["lingbot_world_v2"]["resolutions"][res]
    frame = imageio_v2.get_reader(sources[-1]).get_data(0)
    image = Image.fromarray(frame)
    if image.size != (width, height):
        # Rolling Forcing's 720p frame is the only dense source; resample it
        # to whatever resolution this session runs at.
        image = image.resize((width, height), Image.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    print(f"wrote LingBot condition frame {target} ({width}x{height})", flush=True)
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
        # The server's own processes are whatever computes on the GPU at
        # session start (the ws client itself never touches the GPU); a pid
        # beyond that set that persists with a real footprint is a co-tenant
        # (probe blips are tolerated, same policy as GpuWatchdog).
        try:
            blessed = compute_pids(server.gpu)
        except Exception:
            blessed = set()
        contended = False
        suspect: set[int] = set()
        deadline = time.time() + GENERATE_TIMEOUT_S
        while proc.poll() is None and time.time() < deadline:
            time.sleep(20)
            try:
                apps = compute_apps(server.gpu)
            except Exception:
                continue
            extra = set(apps) - blessed
            if any(apps.get(pid, 0) >= 2048 for pid in extra & suspect):
                contended = True
            suspect = extra
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
