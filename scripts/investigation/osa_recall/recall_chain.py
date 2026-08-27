# SPDX-License-Identifier: Apache-2.0
"""Run the recall-study captures under the exclusive-GPU policy.

    python recall_chain.py

Each job waits for a fully idle GPU (GpuPool), runs `run.py` on it under a
GpuWatchdog that kills and requeues the job if a co-tenant appears, and
releases the GPU when done. Jobs run concurrently when several GPUs are free.
"""

import concurrent.futures
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sparse_baselines"))
sys.path.insert(0, str(HERE.parent))
from common import GpuPool, GpuWatchdog, compute_apps  # noqa: E402

# replan_each_chunk final timing at the shipped default (stride 32).
JOBS = [
    {
        "tag": "sf20_replan32_time",
        "args": [
            "--no-hook",
            "--seconds", "20",
            "--osa-extra", '{"replan_each_chunk": true}',
            "--tag", "sf20_replan32_time",
            "--port-base", "29785",
        ],
    },
]
COMMON = ["--model", "self_forcing", "--density", "0.3", "--seconds", "10"]
MAX_ATTEMPTS = 3


def run_job(pool: GpuPool, job: dict) -> str:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        gpu = pool.acquire()
        preexisting = set(compute_apps(gpu))
        print(f"[chain] {job['tag']}: attempt {attempt} on gpu {gpu}", flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "run.py"), *COMMON, *job["args"],
             "--gpu", str(gpu)],
            start_new_session=True,
        )
        watchdog = GpuWatchdog(gpu, proc, preexisting=preexisting)
        rc = proc.wait()
        watchdog.stop()
        pool.release(gpu)
        if watchdog.contended:
            print(f"[chain] {job['tag']}: contended, requeueing", flush=True)
            continue
        return f"{job['tag']}: rc={rc}"
    return f"{job['tag']}: gave up after {MAX_ATTEMPTS} contended attempts"


def main() -> None:
    pool = GpuPool(list(range(8)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda j: run_job(pool, j), JOBS))
    for line in results:
        print(f"[chain] {line}", flush=True)


if __name__ == "__main__":
    main()
