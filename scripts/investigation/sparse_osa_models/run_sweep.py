# SPDX-License-Identifier: Apache-2.0
"""Main sweep: 720p / 20 s per model, dense vs OSA at several density knobs.

    python run_sweep.py --model rolling_forcing [--gpus 4,5] [--targets 0.5,0.4,0.3,0.2]
    python run_sweep.py --model lingbot_world_v2 --gpus 4

Results -> <model>/results.json, videos under <model>/runs/<tag>/.
Timing runs want an otherwise idle GPU; each worker waits for its GPU to free
up before launching.
"""

import argparse
import json
import pathlib
import queue
import sys
import threading

from common import (
    MAIN_PROMPT,
    MODELS,
    ROOT,
    LingbotServer,
    osa_config,
    run_generate,
    run_lingbot_session,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

TARGETS = [0.5, 0.4, 0.3, 0.2]


def sweep_generate(args, targets: list[float]) -> None:
    model_root = ROOT / args.model
    results_path = model_root / args.out
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()
    gpus = [int(g) for g in args.gpus.split(",")]

    jobs: queue.Queue = queue.Queue()
    if not args.skip_dense:
        jobs.put(("dense", None, None))
    for target in targets:
        jobs.put((f"osa_{target:g}", "osa", osa_config(target)))

    def worker(index: int, gpu: int) -> None:
        port_base = args.port_base + index * 20
        while True:
            try:
                tag, method, config = jobs.get_nowait()
            except queue.Empty:
                return
            print(f"[gpu{gpu}] START {args.model} {tag}", flush=True)
            try:
                result = run_generate(
                    model=args.model,
                    out_dir=model_root / "runs" / tag,
                    gpu=gpu,
                    port_base=port_base,
                    duration=args.duration,
                    res=args.res,
                    prompt_key=MAIN_PROMPT,
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
            except Exception as error:
                print(f"[gpu{gpu}] FAIL {tag}: {error}", flush=True)
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
    print("SWEEP DONE ->", results_path)


def sweep_lingbot(args, targets: list[float]) -> None:
    """One server per sparse config (server args carry the method), serial."""
    model_root = ROOT / args.model
    results_path = model_root / args.out
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    gpu = int(args.gpus.split(",")[0])

    configs = [] if args.skip_dense else [("dense", None, None)]
    configs += [(f"osa_{t:g}", "osa", osa_config(t)) for t in targets]
    for tag, method, config in configs:
        print(f"[gpu{gpu}] START server {args.model} {tag}", flush=True)
        with LingbotServer(
            gpu=gpu,
            port_base=args.port_base,
            server_dir=model_root / "runs" / tag,
            method=method,
            method_config=config,
        ) as server:
            server.wait_ready()
            result = run_lingbot_session(
                server,
                out_dir=model_root / "runs" / tag,
                duration=args.duration,
                res=args.res,
                prompt_key=MAIN_PROMPT,
            )
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
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--targets", default=",".join(str(t) for t in TARGETS))
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--res", default="720p")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--port-base", type=int, default=36000)
    args = parser.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    if MODELS[args.model]["kind"] == "realtime":
        sweep_lingbot(args, targets)
    else:
        sweep_generate(args, targets)


if __name__ == "__main__":
    main()
