# SPDX-License-Identifier: Apache-2.0
"""Per-chunk forward / attention wall-time sweep over four block-causal models.

Models x resolutions {480p, 720p} x durations {5, 10, 20}s, one clean timing
run per config with the chunk-timing probe on
(``SGLANG_DIFFUSION_CHUNK_TIMING_DIR``). The probe brackets every DiT forward
and every attention module with CUDA events and only resolves them at chunk
boundaries, so the run stays close to an unprobed one.

Self-Forcing / Rolling Forcing / LongLive-2 run through one-shot
`sglang generate`. LingBot-World v2 only rolls out multiple chunks inside a
realtime WebSocket session, so its jobs start `sglang serve` once per
resolution and drive one session per duration via lingbot_ws_client.py.

    python run_sweep.py [--models m1,m2] [--gpus auto|4,7] [--durations 5,10,20]

Results land in results/investigation/chunk_runtime/runs/<model>/<res>_<dur>s/
{run.log, chunk_timing.json, video.mp4}.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("chunk_runtime")
RUNS = ROOT / "runs"
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


def lingbot_first_frame(res: str) -> pathlib.Path:
    """The I2V condition image for LingBot, matching this run's prompt.

    LingBot is the only image-conditioned model here, and the condition image
    dominates the scene: point it at an unrelated picture and the video shows
    that picture's world with the prompt only bleeding in. Derive it instead
    from frame 0 of the T2V run for the same prompt, seed and resolution, so
    all four models depict the same thing.
    """
    target = ROOT / "first_frames" / f"{res}.png"
    if target.exists():
        return target
    source = RUNS / "self_forcing" / f"{res}_20s" / "video.mp4"
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


def base_env(gpu: int, timing_dir: pathlib.Path) -> dict:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        SGLANG_DIFFUSION_SYNC_STAGE_PROFILING="1",
        SGLANG_DIFFUSION_CHUNK_TIMING_DIR=str(timing_dir),
    )
    return env


def collect_timing(timing_dir: pathlib.Path, out_dir: pathlib.Path) -> bool:
    """Move the probe's newest run dir next to the config's other artifacts."""
    deadline = time.time() + FLUSH_TIMEOUT_S
    while time.time() < deadline:
        dumps = sorted(timing_dir.glob("*/chunk_timing.json"))
        if dumps:
            shutil.copy(dumps[-1], out_dir / "chunk_timing.json")
            return True
        time.sleep(2)
    return False


def run_generate_job(
    *, model: str, res: str, duration: int, gpu: int, port_base: int
) -> None:
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    frames = spec["frames"][duration]
    out_dir = RUNS / model / f"{res}_{duration}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = out_dir / "timing_raw"
    if timing_dir.exists():
        shutil.rmtree(timing_dir)
    timing_dir.mkdir()
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
            env=base_env(gpu, timing_dir),
            cwd=out_dir,
            timeout=GENERATE_TIMEOUT_S,
        )
    ok = collect_timing(timing_dir, out_dir)
    (out_dir / "run_env.json").write_text(
        json.dumps({"gpu": gpu, "resident_mib_before": resident_before})
    )
    videos = sorted((out_dir / "outputs").glob("*.mp4"))
    if videos:
        shutil.copy(videos[-1], out_dir / "video.mp4")
    status = (
        "OK" if proc.returncode == 0 and ok else f"RC={proc.returncode} timing={ok}"
    )
    print(
        f"[gpu{gpu}] DONE  {model} {res} {duration}s {status} "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )


def run_realtime_jobs(
    *, model: str, res: str, durations: list[int], gpu: int, port_base: int
) -> None:
    """One server per resolution, one realtime session per duration."""
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    port = port_base + 2
    first_frame = lingbot_first_frame(res)
    server_dir = RUNS / model / f"{res}_server"
    server_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = server_dir / "timing_raw"
    if timing_dir.exists():
        shutil.rmtree(timing_dir)
    timing_dir.mkdir()
    env = base_env(gpu, timing_dir)
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
            out_dir = RUNS / model / f"{res}_{duration}s"
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
            ok = collect_timing(timing_dir, out_dir)
            (out_dir / "run_env.json").write_text(
                json.dumps({"gpu": gpu, "resident_mib_before": resident_before})
            )
            if ok:
                for stale in timing_dir.glob("*/"):
                    shutil.rmtree(stale)
            status = (
                "OK"
                if proc.returncode == 0 and ok
                else f"RC={proc.returncode} timing={ok}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument(
        "--gpus",
        default="auto",
        help="'auto' waits for and uses every GPU with <1 GiB resident, else e.g. 4,7",
    )
    parser.add_argument("--durations", default=",".join(str(d) for d in DURATIONS))
    parser.add_argument("--resolutions", default="480p,720p")
    # Concurrent invocations must not share a port base.
    parser.add_argument("--port-base", type=int, default=35000)
    args = parser.parse_args()

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
                    )
                else:
                    run_generate_job(
                        model=model,
                        res=res,
                        duration=job_durations[0],
                        gpu=gpu,
                        port_base=port_base,
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


if __name__ == "__main__":
    main()
