# SPDX-License-Identifier: Apache-2.0
"""Runtime-breakdown sweep over four block-causal video models.

Models x resolutions {480p, 720p} x durations {5, 10, 20, 30}s, two runs per
config: a `timing` run (clean stage walltimes) and a `profile` run
(torch-profiler trace over 40 denoising steps, used only for GPU-kernel
*shares*; profiling changes latency, so its walltimes are never reported).

Self-Forcing / Rolling Forcing / LongLive-2 run through one-shot
`sglang generate`. LingBot-World v2 only rolls out multiple chunks inside a
realtime WebSocket session, so its jobs start `sglang serve` once per
resolution and drive sessions via lingbot_ws_client.py (timing sessions for
every duration + one profiled 20s session whose kernel shares stand in for
all durations — the working window is capped, so steady-state per-chunk work
is duration-independent).

    python run_sweep.py [--models m1,m2] [--gpus 0,1,6,7] [--modes timing,profile]

Results land in results/investigation/runtime_breakdown/runs/<model>/
<res>_<dur>s/{timing.log, profile.log, trace/}.
"""

import argparse
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = results_dir("runtime_breakdown")
RUNS = ROOT / "runs"
PROMPT = "A red fox trotting across a snowy field, camera slowly tracking sideways"
FIRST_FRAME = REPO / "inputs/uploads/a816103ba740450f9ded724ea1bf11e7_first_frame"

# 16 fps Wan models: pixel frames = 4 * latent - 3, latent divisible by 3.
WAN_FRAMES = {5: 81, 10: 165, 20: 321, 30: 477}
# 24 fps LongLive-2: latent divisible by 8.
LONGLIVE_FRAMES = {5: 125, 10: 253, 20: 477, 30: 733}

MODELS = {
    "self_forcing": {
        "path": "/data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers-fullctx-null",
        "frames": WAN_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
    },
    "rolling_forcing": {
        "path": "frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers",
        "frames": WAN_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "generate",
    },
    "longlive2": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "frames": LONGLIVE_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 704)},
        "kind": "generate",
    },
    "lingbot_world_v2": {
        "path": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "frames": WAN_FRAMES,
        "resolutions": {"480p": (832, 480), "720p": (1280, 720)},
        "kind": "realtime",
    },
}

DURATIONS = [5, 10, 20, 30]
PROFILED_TIMESTEPS = 40
GENERATE_TIMEOUT_S = 7200
SERVER_READY_TIMEOUT_S = 1800


def base_env(gpu: int) -> dict:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        SGLANG_DIFFUSION_SYNC_STAGE_PROFILING="1",
    )
    return env


def run_generate_job(
    *, model: str, res: str, duration: int, mode: str, gpu: int, port_base: int
) -> None:
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    frames = spec["frames"][duration]
    out_dir = RUNS / model / f"{res}_{duration}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"{mode}.log"
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
        "--master-port",
        str(port_base),
        "--scheduler-port",
        str(port_base + 1),
        "--port",
        str(port_base + 2),
    ]
    env = base_env(gpu)
    if mode == "profile":
        args += ["--profile", "--num-profiled-timesteps", str(PROFILED_TIMESTEPS)]
        trace_dir = out_dir / "trace"
        trace_dir.mkdir(exist_ok=True)
        env["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"] = str(trace_dir)
    else:
        args += ["--save-output"]
    print(f"[gpu{gpu}] START {model} {res} {duration}s {mode}", flush=True)
    started = time.time()
    with open(log, "w") as handle:
        proc = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=out_dir,
            timeout=GENERATE_TIMEOUT_S,
        )
    status = "OK" if proc.returncode == 0 else f"RC={proc.returncode}"
    print(
        f"[gpu{gpu}] DONE  {model} {res} {duration}s {mode} "
        f"{status} in {time.time() - started:.0f}s",
        flush=True,
    )


def wait_for_server(port: int, proc: subprocess.Popen, log: pathlib.Path) -> None:
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


def run_realtime_jobs(
    *, model: str, res: str, modes: list[str], gpu: int, port_base: int
) -> None:
    """One server per resolution; sequential ws sessions per duration."""
    spec = MODELS[model]
    width, height = spec["resolutions"][res]
    res_dir = RUNS / model
    res_dir.mkdir(parents=True, exist_ok=True)
    server_log = res_dir / f"server_{res}.log"
    trace_dir = res_dir / f"server_{res}_trace"
    trace_dir.mkdir(exist_ok=True)
    env = base_env(gpu)
    env["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"] = str(trace_dir)
    port = port_base + 2
    args = [
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
    ]
    print(f"[gpu{gpu}] START {model} server {res}", flush=True)
    with open(server_log, "w") as handle:
        server = subprocess.Popen(
            args, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=res_dir
        )
    try:
        wait_for_server(port, server, server_log)
        session_specs = []
        if "timing" in modes:
            session_specs += [(d, False) for d in DURATIONS]
        if "profile" in modes:
            session_specs += [(20, True)]
        for duration, profiled in session_specs:
            out_dir = RUNS / model / f"{res}_{duration}s"
            out_dir.mkdir(parents=True, exist_ok=True)
            mode = "profile" if profiled else "timing"
            client_args = [
                "python",
                str(HERE / "lingbot_ws_client.py"),
                "--port",
                str(port),
                "--model-path",
                spec["path"],
                "--prompt",
                PROMPT,
                "--first-frame",
                str(FIRST_FRAME),
                "--size",
                f"{width}x{height}",
                "--num-frames",
                str(spec["frames"][duration]),
                "--out",
                str(out_dir / f"{mode}_session.json"),
            ]
            if profiled:
                client_args += [
                    "--profile",
                    "--num-profiled-timesteps",
                    str(PROFILED_TIMESTEPS),
                ]
            print(f"[gpu{gpu}] START {model} {res} {duration}s {mode} (ws)", flush=True)
            # The server releases a disposed session asynchronously; connecting
            # too soon is rejected with "another realtime session is already
            # active", so pace the sessions and retry.
            for attempt in range(3):
                time.sleep(15)
                with open(out_dir / f"{mode}.log", "w") as handle:
                    proc = subprocess.run(
                        client_args,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=GENERATE_TIMEOUT_S,
                    )
                if (
                    proc.returncode == 0
                    or "already active" not in (out_dir / f"{mode}.log").read_text()
                ):
                    break
            status = "OK" if proc.returncode == 0 else f"RC={proc.returncode}"
            print(
                f"[gpu{gpu}] DONE  {model} {res} {duration}s {mode} {status}",
                flush=True,
            )
            if profiled:
                # The server writes traces into the shared per-resolution dir;
                # claim the ones this session just produced.
                session_trace = out_dir / "trace"
                session_trace.mkdir(exist_ok=True)
                time.sleep(10)
                for trace in trace_dir.glob("*.json.gz"):
                    trace.rename(session_trace / trace.name)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    print(f"[gpu{gpu}] STOP  {model} server {res}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--gpus", default="0,1,6,7")
    parser.add_argument("--modes", default="timing,profile")
    parser.add_argument("--durations", default=",".join(str(d) for d in DURATIONS))
    parser.add_argument("--resolutions", default="480p,720p")
    # Concurrent invocations must not share a port base.
    parser.add_argument("--port-base", type=int, default=32000)
    args = parser.parse_args()

    models = args.models.split(",")
    gpus = [int(g) for g in args.gpus.split(",")]
    modes = args.modes.split(",")
    durations = [int(d) for d in args.durations.split(",")]
    resolutions = args.resolutions.split(",")

    jobs: queue.Queue = queue.Queue()
    # Realtime jobs are long-lived (server per resolution); enqueue them first
    # so they start immediately on the first free workers.
    for model in models:
        if MODELS[model]["kind"] != "realtime":
            continue
        for res in resolutions:
            jobs.put(("realtime", model, res, None, None))
    # Big configs first so the queue drains evenly.
    for duration in sorted(durations, reverse=True):
        for res in sorted(resolutions, reverse=True):
            for model in models:
                if MODELS[model]["kind"] != "generate":
                    continue
                for mode in modes:
                    jobs.put(("generate", model, res, duration, mode))

    def worker(worker_index: int, gpu: int) -> None:
        port_base = args.port_base + worker_index * 20
        while True:
            try:
                kind, model, res, duration, mode = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                if kind == "realtime":
                    run_realtime_jobs(
                        model=model,
                        res=res,
                        modes=modes,
                        gpu=gpu,
                        port_base=port_base,
                    )
                else:
                    run_generate_job(
                        model=model,
                        res=res,
                        duration=duration,
                        mode=mode,
                        gpu=gpu,
                        port_base=port_base,
                    )
            except Exception as error:
                print(
                    f"[gpu{gpu}] FAIL {model} {res} {duration} {mode}: {error}",
                    flush=True,
                )
            finally:
                jobs.task_done()

    threads = [
        threading.Thread(target=worker, args=(index, gpu), daemon=True)
        for index, gpu in enumerate(gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
