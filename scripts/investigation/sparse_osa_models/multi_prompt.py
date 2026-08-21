# SPDX-License-Identifier: Apache-2.0
"""Multi-prompt validation: 5 prompts x {dense, OSA at the ~0.30 knob}.

One 720p / 20 s video per (prompt, method); then one frame-comparison sheet
per prompt (rows: Dense / OSA; labeled columns: 7 frames). LingBot reuses one
server per method across all prompts (the sparse config is a server arg).

    python multi_prompt.py --model rolling_forcing [--gpus 4,5]
Videos -> <model>/runs_prompts/<prompt>_<method>/, sheets ->
<model>/prompt_sheet_<p>.png, timings/densities -> <model>/results_prompts.json.
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
    PROMPTS,
    ROOT,
    LingbotServer,
    extract_frames,
    osa_config,
    render_frame_sheet,
    run_generate,
    run_lingbot_session,
    sheet_frame_indices,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

TARGET = 0.3
PROMPT_KEYS = [key for key in PROMPTS if key != MAIN_PROMPT]


def run_generate_jobs(args) -> None:
    model_root = ROOT / args.model
    results_path = model_root / "results_prompts.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()
    gpus = [int(g) for g in args.gpus.split(",")]

    jobs: queue.Queue = queue.Queue()
    for prompt_key in PROMPT_KEYS:
        jobs.put((f"{prompt_key}_dense", prompt_key, None, None))
        jobs.put((f"{prompt_key}_osa", prompt_key, "osa", osa_config(TARGET)))

    def worker(index: int, gpu: int) -> None:
        port_base = args.port_base + index * 20
        while True:
            try:
                tag, prompt_key, method, config = jobs.get_nowait()
            except queue.Empty:
                return
            # `sglang generate` exits 0 even when the run fails; a parsed e2e
            # time is the real success marker.
            if (
                tag in results
                and results[tag].get("returncode") == 0
                and results[tag].get("e2e_s") is not None
            ):
                print(f"SKIP {tag}: already done", flush=True)
                jobs.task_done()
                continue
            print(f"[gpu{gpu}] START {args.model} {tag}", flush=True)
            try:
                result = run_generate(
                    model=args.model,
                    out_dir=model_root / "runs_prompts" / tag,
                    gpu=gpu,
                    port_base=port_base,
                    duration=args.duration,
                    res=args.res,
                    prompt_key=prompt_key,
                    method=method,
                    method_config=config,
                    wait_gpu=not args.no_gpu_wait,
                )
                with lock:
                    results[tag] = result
                    results_path.write_text(json.dumps(results, indent=2))
                print(
                    f"[gpu{gpu}] DONE  {tag} rc={result['returncode']} "
                    f"denoise={result.get('denoise_s')} "
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


def run_lingbot_jobs(args) -> None:
    model_root = ROOT / args.model
    results_path = model_root / "results_prompts.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    gpu = int(args.gpus.split(",")[0])
    for method, config, suffix in (
        (None, None, "dense"),
        ("osa", osa_config(TARGET), "osa"),
    ):
        todo = [
            key
            for key in PROMPT_KEYS
            if results.get(f"{key}_{suffix}", {}).get("denoise_s") is None
        ]
        if not todo:
            continue
        print(f"[gpu{gpu}] START server {args.model} {suffix}", flush=True)
        with LingbotServer(
            gpu=gpu,
            port_base=args.port_base,
            server_dir=model_root / "runs_prompts" / f"server_{suffix}",
            method=method,
            method_config=config,
            wait_gpu=not args.no_gpu_wait,
        ) as server:
            server.wait_ready()
            for prompt_key in todo:
                tag = f"{prompt_key}_{suffix}"
                print(f"[gpu{gpu}] START {tag}", flush=True)
                result = run_lingbot_session(
                    server,
                    out_dir=model_root / "runs_prompts" / tag,
                    duration=args.duration,
                    res=args.res,
                    prompt_key=prompt_key,
                )
                result["config"] = config
                results[tag] = result
                results_path.write_text(json.dumps(results, indent=2))
                print(
                    f"[gpu{gpu}] DONE  {tag} rc={result['returncode']} "
                    f"denoise={result.get('denoise_s')} "
                    f"density={result.get('density')}",
                    flush=True,
                )


def build_sheets(args) -> None:
    model_root = ROOT / args.model
    fps = MODELS[args.model]["fps"]
    frame_indices = sheet_frame_indices(fps=fps, duration=args.duration)
    for prompt_key in PROMPT_KEYS:
        rows = []
        for label, suffix in (("Dense", "dense"), ("OSA", "osa")):
            frames = extract_frames(
                model_root / "runs_prompts" / f"{prompt_key}_{suffix}", frame_indices
            )
            if frames is None:
                print(f"missing video: {prompt_key}_{suffix}", flush=True)
                continue
            rows.append((label, frames))
        if not rows:
            continue
        out = model_root / f"prompt_sheet_{prompt_key}.png"
        render_frame_sheet(
            rows=rows, frame_indices=frame_indices, fps=fps, out_path=out
        )
        print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--res", default="720p")
    parser.add_argument("--sheets-only", action="store_true")
    # Multi-prompt runs feed the quality comparison, not the headline timing
    # table, so they may share a GPU with a resident co-tenant.
    parser.add_argument("--no-gpu-wait", action="store_true")
    parser.add_argument("--port-base", type=int, default=36500)
    args = parser.parse_args()
    if not args.sheets_only:
        if MODELS[args.model]["kind"] == "realtime":
            run_lingbot_jobs(args)
        else:
            run_generate_jobs(args)
    build_sheets(args)


if __name__ == "__main__":
    main()
