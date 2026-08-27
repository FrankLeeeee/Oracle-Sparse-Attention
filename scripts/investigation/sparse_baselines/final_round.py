# SPDX-License-Identifier: Apache-2.0
"""Final-round runner: chosen method configs over chosen prompts, any duration.

One tool for the 2026-08-27 doc-final campaign: runs dense plus a list of
``method:tier`` jobs (configs resolved from configs.json, nearest tier if the
exact one is absent) for every requested prompt, then computes PSNR against
each prompt's own dense run and renders one labeled frame sheet per prompt.

    python final_round.py --model self_forcing --duration 5 \
        --prompts p1_forest p2_plating \
        --methods osasched:0.2 lightforcing:0.2 \
        --out results_prompts_5s.json --runs-dir runs_prompts_5s \
        --sheet-prefix prompt_sheet_5s_ --gpus 0,4,5,6

Results -> <model>/<out>; videos under <model>/<runs-dir>/<prompt>_<tag>/;
sheets  -> <model>/<sheet-prefix><prompt>.png with full method labels.
"""

import argparse
import glob
import json
import pathlib
import queue
import sys
import threading

import imageio
import numpy as np

from common import (
    METHOD_LABELS,
    MODELS,
    PROMPTS,
    ROOT,
    GpuContended,
    GpuPool,
    extract_frames,
    record_result,
    render_frame_sheet,
    run_generate,
    sheet_frame_indices,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def load_video(run_dir: pathlib.Path) -> np.ndarray | None:
    paths = sorted(glob.glob(str(run_dir / "**" / "*.mp4"), recursive=True))
    if not paths:
        return None
    reader = imageio.get_reader(paths[-1])
    return np.stack([frame for frame in reader])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255**2 / max(mse, 1e-9)))


def resolve_jobs(model: str, specs: list[str]) -> list[tuple[str, str, dict]]:
    """``method:tier`` specs -> (tag, method, config) via configs.json."""
    configs = json.loads((ROOT / "configs.json").read_text())[model]
    jobs = []
    for spec in specs:
        method, tier = spec.split(":")
        entry = configs[method].get(tier)
        if entry is None:
            nearest = min(configs[method], key=lambda t: abs(float(t) - float(tier)))
            entry = configs[method][nearest]
        jobs.append((f"{method}_{tier}", method, entry["config"]))
    return jobs


def run_all(args, jobs) -> None:
    model_root = ROOT / args.model
    results_path = model_root / args.out
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()
    pool = GpuPool([int(g) for g in args.gpus.split(",")])
    job_queue: queue.Queue = queue.Queue()
    all_jobs = ([("dense", None, None)] if not args.skip_dense else []) + jobs
    for prompt_key in args.prompts:
        for tag, method, config in all_jobs:
            run_key = f"{prompt_key}_{tag}"
            if results.get(run_key, {}).get("returncode") == 0:
                print(f"skip {run_key} (already done)", flush=True)
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
            print(f"[gpu{gpu}] START {args.model} {run_key}", flush=True)
            try:
                result = run_generate(
                    model=args.model,
                    out_dir=model_root / args.runs_dir / run_key,
                    gpu=gpu,
                    port_base=port_base,
                    duration=args.duration,
                    res=args.res,
                    prompt_key=prompt_key,
                    method=method,
                    method_config=config,
                )
                with lock:
                    results[run_key] = result
                    record_result(results_path, run_key, result)
                print(
                    f"[gpu{gpu}] DONE  {run_key} rc={result['returncode']} "
                    f"denoise={result.get('denoise_s')} "
                    f"density={result.get('density')}",
                    flush=True,
                )
            except GpuContended:
                print(f"[gpu{gpu}] {run_key} contended, requeueing", flush=True)
                job_queue.put((prompt_key, tag, method, config))
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
    print("RUNS DONE ->", results_path)


def quality_and_sheets(args, jobs) -> None:
    """PSNR vs each prompt's dense + one labeled frame sheet per prompt."""
    model_root = ROOT / args.model
    results_path = model_root / args.out
    results = json.loads(results_path.read_text())
    fps = MODELS[args.model]["fps"]
    frame_indices = sheet_frame_indices(fps=fps, duration=args.duration)
    for prompt_key in args.prompts:
        dense_dir = model_root / args.runs_dir / f"{prompt_key}_dense"
        dense = load_video(dense_dir)
        if dense is None:
            print(f"no dense video for {prompt_key}; skipping quality")
            continue
        rows = [("Dense", extract_frames(dense_dir, frame_indices))]
        for tag, method, _config in jobs:
            run_key = f"{prompt_key}_{tag}"
            run_dir = model_root / args.runs_dir / run_key
            video = load_video(run_dir)
            if video is None:
                continue
            n = min(len(video), len(dense))
            first = min(n, 5 * fps)
            entry = results.get(run_key, {})
            entry["psnr_overall_db"] = round(psnr(video[:n], dense[:n]), 2)
            entry["psnr_first5s_db"] = round(psnr(video[:first], dense[:first]), 2)
            results[run_key] = entry
            label = (
                f"{METHOD_LABELS[method]} "
                f"density={entry.get('density', 0):.2f}"
            )
            rows.append((label, extract_frames(run_dir, frame_indices)))
        rows = [(label, frames) for label, frames in rows if frames is not None]
        sheet = model_root / f"{args.sheet_prefix}{prompt_key}.png"
        render_frame_sheet(
            rows=rows, frame_indices=frame_indices, fps=fps, out_path=sheet
        )
        print(f"wrote {sheet}")
    results_path.write_text(json.dumps(results, indent=2))
    print("PSNR ->", results_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--res", default="720p")
    parser.add_argument("--prompts", nargs="+", required=True, choices=sorted(PROMPTS))
    parser.add_argument("--methods", nargs="+", required=True,
                        help="method:tier specs, e.g. osasched:0.2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--sheet-prefix", required=True)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--gpus", default="0,4,5,6")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--port-base", type=int, default=56000)
    args = parser.parse_args()
    jobs = resolve_jobs(args.model, args.methods)
    if not args.quality_only:
        run_all(args, jobs)
    quality_and_sheets(args, jobs)


if __name__ == "__main__":
    main()
