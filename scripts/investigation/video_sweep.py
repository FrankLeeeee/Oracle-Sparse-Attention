# SPDX-License-Identifier: Apache-2.0
"""Shared sweep driver for the block-causal video-model investigations.

Runs one generation per (model, resolution, duration) with a debugging probe
enabled, and collects whatever that probe wrote next to the config's video and
log. The topics differ only in which probe they switch on, so they all call
:func:`main` with a different ``topic`` and ``probe_env``.

Self-Forcing / Rolling Forcing / LongLive-2 run through one-shot
`sglang generate`. LingBot-World v2 only rolls out multiple chunks inside a
realtime WebSocket session, so its jobs start `sglang serve` once per
resolution and drive one session per duration via lingbot_ws_client.py; being
I2V, it also needs a condition image matching the prompt (see
:func:`condition_frame`).

Results land in results/investigation/<topic>/runs/<model>/<res>_<dur>s/.
"""

import argparse
import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import REPO, results_dir  # noqa: E402

WS_CLIENT = REPO / "scripts/investigation/runtime_breakdown/lingbot_ws_client.py"
PROMPT = (
    "A dynamic and chaotic scene in a dense forest during a heavy rainstorm, "
    "capturing a real girl frantically running through the foliage. Her wild "
    "hair flows behind her as she sprints, her arms flailing and her face "
    "contorted in fear and desperation. Behind her, various animals—rabbits, "
    "deer, and birds—are also running, creating a frenzied atmosphere. The "
    "girl's clothes are soaked, clinging to her body, and she is screaming and "
    "shouting as she tries to escape. The background is a blur of greenery and "
    "rain-drenched trees, with occasional glimpses of the darkening sky. A "
    "wide-angle shot from a low angle, emphasizing the urgency and chaos of "
    "the moment."
)

# 16 fps Wan models: pixel frames = 4 * latent - 3, latent divisible by 3.
WAN_FRAMES = {5: 81, 10: 165, 20: 321}
# 24 fps LongLive-2: latent divisible by 8.
LONGLIVE_FRAMES = {5: 125, 10: 253, 20: 477}

MODELS = {
    "self_forcing": {
        "path": "/data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers-fullctx-null",
        "frames": WAN_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
        "fps": 16,
    },
    "rolling_forcing": {
        "path": "frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers",
        "frames": WAN_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
        "fps": 16,
    },
    "longlive2": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "frames": LONGLIVE_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 704)},
        "kind": "generate",
        "fps": 24,
    },
    "lingbot_world_v2": {
        "path": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "frames": WAN_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "realtime",
        "fps": 16,
    },
}

DURATIONS = [5, 10, 20]
GENERATE_TIMEOUT_S = 7200
SERVER_READY_TIMEOUT_S = 1800
FLUSH_TIMEOUT_S = 300
GPU_WAIT_TIMEOUT_S = 7200


def idle_gpus(max_used_mib: int = 1024) -> list[int]:
    """GPU indices with essentially nothing resident — timing runs want these."""
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = []
    for line in output.strip().splitlines():
        index, used = (part.strip() for part in line.split(","))
        if int(used) <= max_used_mib:
            found.append(int(index))
    return found


def condition_frame(res: str) -> pathlib.Path:
    """The I2V condition image for LingBot, matching this run's prompt.

    LingBot is the only image-conditioned model here, and the condition image
    dominates the scene: point it at an unrelated picture and the video shows
    that picture's world with the prompt only bleeding in. Derive it instead
    from frame 0 of the T2V run for the same prompt, seed and resolution, so
    all four models depict the same thing.
    """
    target = results_dir("first_frames") / f"{res}.png"
    if target.exists():
        return target
    source = results_dir("chunk_runtime") / "runs/self_forcing" / f"{res}_20s/video.mp4"
    if not source.exists():
        raise RuntimeError(
            f"no Self-Forcing video to take LingBot's condition frame from: {source}"
        )
    import imageio.v2 as imageio

    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(target, imageio.get_reader(source).get_data(0))
    print(f"wrote LingBot condition frame {target}", flush=True)
    return target


def wait_for_idle_gpus(*, max_used_mib: int = 1024) -> list[int]:
    """Idle GPU indices, blocking until at least one shows up.

    On a shared box every card can be busy when a sweep is queued; waiting
    beats failing, since these runs are long and unattended.
    """
    deadline = time.time() + GPU_WAIT_TIMEOUT_S
    warned = False
    while time.time() < deadline:
        found = idle_gpus(max_used_mib)
        if found:
            return found
        if not warned:
            print("no idle GPU yet, waiting", flush=True)
            warned = True
        time.sleep(60)
    raise SystemExit(f"no GPU freed up within {GPU_WAIT_TIMEOUT_S}s")


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


def wait_for_free_gpu(gpu: int, *, max_used_mib: int = 2048) -> int:
    """Block until nobody else holds the GPU.

    These are wall-time measurements on a shared box: a co-tenant process
    changes the numbers (or OOMs the run outright), so a contended GPU is
    worth waiting out rather than measuring through.
    """
    deadline = time.time() + GPU_WAIT_TIMEOUT_S
    warned = False
    while time.time() < deadline:
        used = gpu_used_mib(gpu)
        if used <= max_used_mib:
            return used
        if not warned:
            print(f"[gpu{gpu}] busy ({used} MiB resident), waiting", flush=True)
            warned = True
        time.sleep(60)
    raise RuntimeError(f"gpu {gpu} still busy after {GPU_WAIT_TIMEOUT_S}s")


def base_env(
    gpu: int,
    probe_dir: pathlib.Path,
    probe_env: dict,
    extra: dict | None = None,
) -> dict:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        SGLANG_DIFFUSION_SYNC_STAGE_PROFILING="1",
        **{name: str(probe_dir) for name in probe_env},
        **(extra or {}),
    )
    return env


def collect_probe_output(probe_dir: pathlib.Path, out_dir: pathlib.Path) -> bool:
    """Copy the probe's newest dump next to the config's video and log.

    Probes write ``<dir>/<ModelTag>-<timestamp>/`` and flush asynchronously for
    realtime sessions (on session dispose), so this polls for the directory to
    appear rather than assuming it is already there.
    """
    deadline = time.time() + FLUSH_TIMEOUT_S
    while time.time() < deadline:
        dumps = sorted(d for d in probe_dir.iterdir() if d.is_dir())
        if dumps:
            for item in dumps[-1].iterdir():
                shutil.copy(item, out_dir / item.name)
            return True
        time.sleep(2)
    return False


def run_generate_job(
    *,
    model: str,
    res: str,
    duration: int,
    gpu: int,
    port_base: int,
    runs: pathlib.Path,
    probe_env: dict,
    extra_env: dict,
) -> None:
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    frames = spec["frames"][duration]
    out_dir = runs / model / f"{res}_{duration}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = out_dir / "probe_raw"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir()
    log = out_dir / "run.log"
    args = [
        "sglang",
        "generate",
        "--model-path",
        spec["path"],
        "--prompt",
        PROMPT,
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(frames),
        "--seed",
        "42",
        "--save-output",
        "--master-port",
        str(port_base),
        "--scheduler-port",
        str(port_base + 1),
        "--port",
        str(port_base + 2),
    ]
    resident_before = wait_for_free_gpu(gpu)
    print(f"[gpu{gpu}] START {model} {res} {duration}s", flush=True)
    started = time.time()
    with open(log, "w") as handle:
        proc = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=base_env(gpu, probe_dir, probe_env, extra_env),
            cwd=out_dir,
            timeout=GENERATE_TIMEOUT_S,
        )
    ok = collect_probe_output(probe_dir, out_dir)
    (out_dir / "run_env.json").write_text(
        json.dumps({"gpu": gpu, "resident_mib_before": resident_before})
    )
    videos = sorted((out_dir / "outputs").glob("*.mp4"))
    if videos:
        shutil.copy(videos[-1], out_dir / "video.mp4")
    status = "OK" if proc.returncode == 0 and ok else f"RC={proc.returncode} probe={ok}"
    print(
        f"[gpu{gpu}] DONE  {model} {res} {duration}s {status} "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )


def run_realtime_jobs(
    *,
    model: str,
    res: str,
    durations: list[int],
    gpu: int,
    port_base: int,
    runs: pathlib.Path,
    probe_env: dict,
    extra_env: dict,
) -> None:
    """One server per resolution, one realtime session per duration."""
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    port = port_base + 2
    first_frame = condition_frame(res)
    server_dir = runs / model / f"{res}_server"
    server_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = server_dir / "probe_raw"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir()
    env = base_env(gpu, probe_dir, probe_env, extra_env)
    resident_before = wait_for_free_gpu(gpu)
    server_log = server_dir / "server.log"
    print(f"[gpu{gpu}] START server {model} {res}", flush=True)
    with open(server_log, "w") as handle:
        server = subprocess.Popen(
            [
                "sglang",
                "serve",
                "--model-path",
                spec["path"],
                "--master-port",
                str(port_base),
                "--scheduler-port",
                str(port_base + 1),
                "--port",
                str(port),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=server_dir,
        )
    try:
        _wait_for_server(port, server, server_log)
        for duration in durations:
            out_dir = runs / model / f"{res}_{duration}s"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[gpu{gpu}] START {model} {res} {duration}s", flush=True)
            started = time.time()
            with open(out_dir / "run.log", "w") as handle:
                proc = subprocess.run(
                    [
                        "python",
                        str(WS_CLIENT),
                        "--port",
                        str(port),
                        "--model-path",
                        spec["path"],
                        "--prompt",
                        PROMPT,
                        "--first-frame",
                        str(first_frame),
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
                    env=env,
                    timeout=GENERATE_TIMEOUT_S,
                )
            # The probe flushes on session dispose, shortly after the last chunk.
            ok = collect_probe_output(probe_dir, out_dir)
            (out_dir / "run_env.json").write_text(
                json.dumps({"gpu": gpu, "resident_mib_before": resident_before})
            )
            if ok:
                for stale in probe_dir.glob("*/"):
                    shutil.rmtree(stale)
            status = (
                "OK"
                if proc.returncode == 0 and ok
                else f"RC={proc.returncode} probe={ok}"
            )
            print(
                f"[gpu{gpu}] DONE  {model} {res} {duration}s {status} "
                f"in {time.time() - started:.0f}s",
                flush=True,
            )
            # Back-to-back realtime sessions need pacing; the server rejects a
            # new one while the previous is still being disposed.
            time.sleep(20)
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()


def _wait_for_server(port: int, proc: subprocess.Popen, log) -> None:
    import urllib.request

    deadline = time.time() + SERVER_READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died, see {log}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"server on port {port} not healthy in time, see {log}")


def main(
    *,
    topic: str,
    probe_env: dict,
    description: str,
    default_port_base: int = 35000,
    extra_env: Callable[[str, str, int], dict] | None = None,
) -> None:
    """Run one sweep for ``topic`` with ``probe_env`` switched on.

    ``probe_env`` maps env var names to the probe's output directory (the value
    is filled in per config), e.g.
    ``{"SGLANG_DIFFUSION_CHUNK_TIMING_DIR": None}``. ``extra_env`` returns any
    further env vars for a given (model, resolution, duration) — attention
    captures need per-model layer and head selections.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--gpus",
        default="auto",
        help="'auto' waits for and uses every GPU with <1 GiB resident, else e.g. 4,7",
    )
    parser.add_argument("--durations", default=",".join(str(d) for d in DURATIONS))
    parser.add_argument("--resolutions", default="480p,720p")
    # Concurrent invocations must not share a port base.
    parser.add_argument("--port-base", type=int, default=default_port_base)
    args = parser.parse_args()

    runs = results_dir(topic) / "runs"
    if args.gpus == "auto":
        gpus = wait_for_idle_gpus()
    else:
        gpus = [int(g) for g in args.gpus.split(",")]
    print(f"using GPUs {gpus}", flush=True)
    durations = [int(d) for d in args.durations.split(",")]
    resolutions = args.resolutions.split(",")

    # One job per (model, resolution): generate models fan out per duration,
    # realtime models keep the durations together behind a single server.
    jobs: queue.Queue = queue.Queue()
    for model in args.models.split(","):
        for res in resolutions:
            if MODELS[model]["kind"] == "realtime":
                jobs.put((model, res, durations))
            else:
                for duration in durations:
                    jobs.put((model, res, [duration]))

    def worker(index: int, gpu: int) -> None:
        port_base = args.port_base + index * 20
        while True:
            try:
                model, res, job_durations = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                if MODELS[model]["kind"] == "realtime":
                    run_realtime_jobs(
                        model=model,
                        res=res,
                        durations=job_durations,
                        gpu=gpu,
                        port_base=port_base,
                        runs=runs,
                        probe_env=probe_env,
                        extra_env=(
                            extra_env(model, res, job_durations[0]) if extra_env else {}
                        ),
                    )
                else:
                    run_generate_job(
                        model=model,
                        res=res,
                        duration=job_durations[0],
                        gpu=gpu,
                        port_base=port_base,
                        runs=runs,
                        probe_env=probe_env,
                        extra_env=(
                            extra_env(model, res, job_durations[0]) if extra_env else {}
                        ),
                    )
            except Exception as error:
                print(
                    f"[gpu{gpu}] FAIL {model} {res} {job_durations}: {error}",
                    flush=True,
                )
            finally:
                jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(i, g), daemon=True)
        for i, g in enumerate(gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("SWEEP DONE", flush=True)
