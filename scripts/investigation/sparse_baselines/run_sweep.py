# SPDX-License-Identifier: Apache-2.0
"""Main sweep: 720p / 20 s per model, dense + every calibrated method tier.

    python run_sweep.py --model self_forcing [--gpus 4,5] [--methods ...]

Reads configs.json (written by calibrate.py); tiers whose calibration bottomed
out (``floored``) run once at the floor config and are labeled by their floor
density in the results. Results -> <model>/results.json, videos under
<model>/runs/<tag>/. Timing runs run on exclusive GPUs: each worker waits for
its GPU to go idle and re-queues a run if a co-tenant appears mid-run.
"""

import argparse
import json
import pathlib
import queue
import sys
import threading

from common import (
    MODELS,
    ROOT,
    GpuPool,
    LingbotServer,
    run_generate,
    run_lingbot_session,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_METHODS = ["osa", "lightforcing", "radial", "svg1", "svg2", "xattention", "sta"]


def sweep_jobs(model: str, methods: list[str], skip_dense: bool) -> list[tuple]:
    """(tag, method, config) list from configs.json; floored tiers dedup."""
    configs = json.loads((ROOT / "configs.json").read_text())[model]
    jobs: list[tuple] = [] if skip_dense else [("dense", None, None)]
    for method in methods:
        seen_configs: list = []
        for tier, entry in configs[method].items():
            if entry["config"] in seen_configs:
                continue  # a floored tier repeats the floor config
            seen_configs.append(entry["config"])
            jobs.append((f"{method}_{tier}", method, entry["config"]))
    return jobs


def sweep_generate(args, jobs: list[tuple]) -> None:
    model_root = ROOT / args.model
    results_path = model_root / args.out
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()
    pool = GpuPool([int(g) for g in args.gpus.split(",")])

    job_queue: queue.Queue = queue.Queue()
    for job in jobs:
        if job[0] in results and results[job[0]].get("returncode") == 0:
            print(f"skip {job[0]} (already done)", flush=True)
            continue
        job_queue.put(job)

    def worker(index: int) -> None:
        port_base = args.port_base + index * 20
        while True:
            try:
                tag, method, config = job_queue.get_nowait()
            except queue.Empty:
                return
            gpu = pool.acquire()
            print(f"[gpu{gpu}] START {args.model} {tag}", flush=True)
            try:
                result = run_generate(
                    model=args.model,
                    out_dir=model_root / "runs" / tag,
                    gpu=gpu,
                    port_base=port_base,
                    duration=args.duration,
                    res=args.res,
                    method=method,
                    method_config=config,
                )
                with lock:
                    results[tag] = result
                    results_path.write_text(json.dumps(results, indent=2))
                print(
                    f"[gpu{gpu}] DONE  {tag} rc={result['returncode']} "
                    f"e2e={result.get('e2e_s')} denoise={result.get('denoise_s')} "
                    f"density={result.get('density')}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[gpu{gpu}] FAIL {tag}: {error}", flush=True)
            finally:
                pool.release(gpu)
                job_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("SWEEP DONE ->", results_path)


def sweep_lingbot(args, jobs: list[tuple]) -> None:
    """One server per sparse config (server args carry the method), serial.

    A session whose timing was contended is redone (same server) up to three
    times; density is unaffected by contention but walltime is.
    """
    model_root = ROOT / args.model
    results_path = model_root / args.out
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    gpu = int(args.gpus.split(",")[0])

    for tag, method, config in jobs:
        if tag in results and results[tag].get("returncode") == 0:
            print(f"skip {tag} (already done)", flush=True)
            continue
        print(f"[gpu{gpu}] START server {args.model} {tag}", flush=True)
        with LingbotServer(
            gpu=gpu,
            port_base=args.port_base,
            server_dir=model_root / "runs" / tag,
            method=method,
            method_config=config,
        ) as server:
            server.wait_ready()
            for attempt in range(3):
                result = run_lingbot_session(
                    server,
                    out_dir=model_root / "runs" / tag,
                    duration=args.duration,
                    res=args.res,
                )
                if not result.get("contended"):
                    break
                print(f"[gpu{gpu}] {tag} contended, redoing session", flush=True)
        result["config"] = config
        results[tag] = result
        results_path.write_text(json.dumps(results, indent=2))
        print(
            f"[gpu{gpu}] DONE  {tag} rc={result['returncode']} "
            f"denoise={result.get('denoise_s')} density={result.get('density')}",
            flush=True,
        )
    print("SWEEP DONE ->", results_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--res", default="720p")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--port-base", type=int, default=36000)
    args = parser.parse_args()
    jobs = sweep_jobs(args.model, args.methods, args.skip_dense)
    if MODELS[args.model]["kind"] == "realtime":
        sweep_lingbot(args, jobs)
    else:
        sweep_generate(args, jobs)


if __name__ == "__main__":
    main()
