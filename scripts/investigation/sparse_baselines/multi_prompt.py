# SPDX-License-Identifier: Apache-2.0
"""Multi-prompt validation: p1-p5 at 720p / 20 s, dense + every method's ~0.30
tier config.

Checks that the calibrated density is content-independent and that the
single-prompt quality conclusions hold across scenes. Results ->
<model>/results_prompts.json, videos under <model>/runs_prompts/<prompt>_<tag>/,
sheets <model>/prompt_sheet_<prompt>.png.

    python multi_prompt.py --model self_forcing [--gpus 4,5]
    python multi_prompt.py --model self_forcing --sheets-only
"""

import argparse
import json
import pathlib
import queue
import sys
import threading

from common import (
    METHOD_LABELS,
    METHODS,
    MODELS,
    PROMPTS,
    ROOT,
    GpuPool,
    LingbotServer,
    extract_frames,
    render_frame_sheet,
    run_generate,
    run_lingbot_session,
    sheet_frame_indices,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

PROMPT_KEYS = ["p1_forest", "p2_plating", "p3_raccoon", "p4_teacup", "p5_tsunami"]
TIER = "0.3"


def tier_configs(model: str, methods: list[str]) -> list[tuple[str, str, dict]]:
    """(tag, method, config) for each method's ~0.30 calibrated config."""
    configs = json.loads((ROOT / "configs.json").read_text())[model]
    out = []
    for method in methods:
        entry = configs[method].get(TIER)
        if entry is None:
            tiers = sorted(configs[method], key=lambda t: abs(float(t) - float(TIER)))
            entry = configs[method][tiers[0]]
        out.append((f"{method}_{TIER}", method, entry["config"]))
    return out


def run_prompts_generate(args, jobs) -> None:
    model_root = ROOT / args.model
    results_path = model_root / "results_prompts.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()
    pool = GpuPool([int(g) for g in args.gpus.split(",")])

    job_queue: queue.Queue = queue.Queue()
    for prompt_key in args.prompts:
        for tag, method, config in jobs:
            run_key = f"{prompt_key}_{tag}"
            if run_key in results and results[run_key].get("returncode") == 0:
                continue
            job_queue.put((prompt_key, tag, method, config))

    def worker(index: int) -> None:
        port_base = args.port_base + index * 20
        while True:
            try:
                prompt_key, tag, method, config = job_queue.get_nowait()
            except queue.Empty:
                return
            gpu = pool.acquire()
            run_key = f"{prompt_key}_{tag}"
            print(f"[gpu{gpu}] START {run_key}", flush=True)
            try:
                result = run_generate(
                    model=args.model,
                    out_dir=model_root / "runs_prompts" / run_key,
                    gpu=gpu,
                    port_base=port_base,
                    duration=20,
                    res="720p",
                    prompt_key=prompt_key,
                    method=method,
                    method_config=config,
                )
                with lock:
                    results[run_key] = result
                    results_path.write_text(json.dumps(results, indent=2))
                print(
                    f"[gpu{gpu}] DONE {run_key} rc={result['returncode']} "
                    f"density={result.get('density')}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                print(f"[gpu{gpu}] FAIL {run_key}: {error}", flush=True)
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


def run_prompts_lingbot(args, jobs) -> None:
    """One server per method config; all prompts' sessions share it."""
    model_root = ROOT / args.model
    results_path = model_root / "results_prompts.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    gpu = int(args.gpus.split(",")[0])
    for tag, method, config in jobs:
        pending = [
            prompt_key
            for prompt_key in args.prompts
            if not results.get(f"{prompt_key}_{tag}", {}).get("returncode") == 0
        ]
        if not pending:
            continue
        print(f"[gpu{gpu}] START server {tag} for {pending}", flush=True)
        with LingbotServer(
            gpu=gpu,
            port_base=args.port_base,
            server_dir=model_root / "runs_prompts" / f"server_{tag}",
            method=method,
            method_config=config,
        ) as server:
            server.wait_ready()
            for prompt_key in pending:
                run_key = f"{prompt_key}_{tag}"
                result = run_lingbot_session(
                    server,
                    out_dir=model_root / "runs_prompts" / run_key,
                    duration=20,
                    res="720p",
                    prompt_key=prompt_key,
                )
                result["config"] = config
                results[run_key] = result
                results_path.write_text(json.dumps(results, indent=2))
                print(
                    f"[gpu{gpu}] DONE {run_key} rc={result['returncode']} "
                    f"density={result.get('density')}",
                    flush=True,
                )


def build_sheets(args) -> None:
    model_root = ROOT / args.model
    results = json.loads((model_root / "results_prompts.json").read_text())
    fps = MODELS[args.model]["fps"]
    frame_indices = sheet_frame_indices(fps=fps, duration=20)
    for prompt_key in args.prompts:
        rows = []
        for tag in ["dense"] + [f"{method}_{TIER}" for method in args.methods]:
            run_key = f"{prompt_key}_{tag}"
            frames = extract_frames(
                model_root / "runs_prompts" / run_key, frame_indices
            )
            if frames is None:
                print(f"missing video: {run_key}", flush=True)
                continue
            if tag == "dense":
                label = "Dense"
            else:
                method = tag.rsplit("_", 1)[0]
                density = results.get(run_key, {}).get("density")
                label = METHOD_LABELS.get(method, method)
                if density:
                    label = f"{label} d={density:.2f}"
            rows.append((label, frames))
        out = model_root / f"prompt_sheet_{prompt_key}.png"
        render_frame_sheet(
            rows=rows, frame_indices=frame_indices, fps=fps, out_path=out
        )
        print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["osa", "lightforcing", "radial", "svg1", "svg2", "xattention", "sta"],
    )
    parser.add_argument("--prompts", nargs="*", default=PROMPT_KEYS)
    parser.add_argument("--sheets-only", action="store_true")
    parser.add_argument("--port-base", type=int, default=37000)
    args = parser.parse_args()
    assert all(p in PROMPTS for p in args.prompts)
    if not args.sheets_only:
        jobs = [("dense", None, None)] + tier_configs(args.model, args.methods)
        if MODELS[args.model]["kind"] == "realtime":
            run_prompts_lingbot(args, jobs)
        else:
            run_prompts_generate(args, jobs)
    build_sheets(args)


if __name__ == "__main__":
    main()
