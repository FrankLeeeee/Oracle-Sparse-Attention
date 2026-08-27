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

# Lever experiments: exact plan recall + free oracle for the allocation knobs.
JOBS = [
    {
        "tag": "sf_exact_base",
        "args": ["--exact", "--tag", "sf_exact_base", "--port-base", "29760"],
    },
    {
        "tag": "sf_exact_dw",
        "args": [
            "--exact",
            "--osa-extra", '{"demand_weighted": true}',
            "--tag", "sf_exact_dw",
            "--port-base", "29770",
        ],
    },
    {
        "tag": "sf_exact_sched",
        "args": [
            "--exact",
            "--osa-extra",
            '{"chunk_schedule": "flops_matched", "schedule_num_frames": 39}',
            "--tag", "sf_exact_sched",
            "--port-base", "29775",
        ],
    },
    {
        "tag": "sf_exact_dw_sched",
        "args": [
            "--exact",
            "--osa-extra",
            '{"demand_weighted": true, "chunk_schedule": "flops_matched",'
            ' "schedule_num_frames": 39}',
            "--tag", "sf_exact_dw_sched",
            "--port-base", "29780",
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
