# SPDX-License-Identifier: Apache-2.0
"""Benchmark dense vs MSA vs LightForcing on the five held-out prompts.

    python run_bench.py [--methods dense,msa,lightforcing] [--seconds 5]

The head taxonomy was calibrated on the study's five prompts (p1-p5); this
benchmark uses five *new* prompts (bench_prompts.json) so the comparison is
out-of-calibration. Per (method, prompt): one exclusive-GPU serial
``sglang generate`` run at 720p, timing from the stage logs, achieved
cumulative density from the backend's log line, and PSNR against the same
prompt's dense output.

- MSA: --sparse-attention msa with the exported taxonomy
  (qk_map_similarity/msa_taxonomy_self_forcing.json), content_density 0.2.
- LightForcing: the sparse-baselines study's calibrated 0.2-tier config
  (configs.json), i.e. exactly the setting behind the published 5s numbers.

Output: results/investigation/msa_bench/{runs/<tag>/, results.json}
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import REPO, results_dir  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/investigation/sparse_baselines"))
from common import (  # noqa: E402
    MODELS,
    SEED,
    GpuPool,
    GpuWatchdog,
    base_env,
    compute_pids,
    parse_log,
    psnr_vs_dense,
)
from common import ROOT as SPARSE_ROOT  # noqa: E402

ROOT = results_dir("msa_bench")
PROMPTS = json.loads((HERE / "bench_prompts.json").read_text())
MODEL = "self_forcing"
RES = "720p"
TAXONOMY = str(HERE.parent / "qk_map_similarity" / "msa_taxonomy_self_forcing.json")


def method_flags(method: str) -> list[str]:
    if method == "dense":
        return []
    if method == "msa":
        config = {"taxonomy_path": TAXONOMY, "content_density": 0.2}
    elif method == "lightforcing":
        config = json.loads((SPARSE_ROOT / "configs.json").read_text())[MODEL][
            "lightforcing"
        ]["0.2"]["config"]
    else:
        raise ValueError(method)
    return ["--sparse-attention", method, "--sparse-attention-config", json.dumps(config)]


def run_one(
    *, tag: str, prompt: str, method: str, gpu: int, port_base: int, seconds: int
) -> dict:
    spec = MODELS[MODEL]
    width, height = spec["resolutions"][RES]
    out_dir = ROOT / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sglang", "generate",
        "--model-path", spec["path"],
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(spec["frames"][seconds]),
        "--seed", str(SEED),
        "--save-output",
        "--master-port", str(port_base),
        "--scheduler-port", str(port_base + 1),
        "--port", str(port_base + 2),
    ] + method_flags(method)
    log = out_dir / "run.log"
    started = time.time()
    parked = compute_pids(gpu)
    with open(log, "w") as handle:
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=base_env(gpu),
            cwd=out_dir,
            start_new_session=True,
        )
        watchdog = GpuWatchdog(gpu, proc, preexisting=parked)
        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        finally:
            watchdog.stop()
    result = parse_log(log)
    result.update(
        returncode=proc.returncode,
        contended=watchdog.contended,
        wall_s=round(time.time() - started, 1),
        gpu=gpu,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--methods", default="dense,msa,lightforcing")
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--port-base", type=int, default=29960)
    args = parser.parse_args()

    pool = GpuPool([int(g) for g in args.gpus.split(",")])
    results_path = ROOT / f"results_{args.seconds}s.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    index = 0
    for prompt_id in args.prompts.split(","):
        for method in args.methods.split(","):
            tag = f"{prompt_id}_{method}_{args.seconds}s"
            if results.get(tag, {}).get("returncode") == 0:
                print(f"skip {tag} (already done)", flush=True)
                continue
            for attempt in range(3):
                gpu = pool.acquire()
                print(f"[{tag}] attempt {attempt + 1} on gpu {gpu}", flush=True)
                try:
                    result = run_one(
                        tag=tag,
                        prompt=PROMPTS[prompt_id]["prompt"],
                        method=method,
                        gpu=gpu,
                        port_base=args.port_base + 10 * (index % 8),
                        seconds=args.seconds,
                    )
                finally:
                    pool.release(gpu)
                index += 1
                if not result["contended"]:
                    break
                print(f"[{tag}] contended, re-queueing", flush=True)
            results[tag] = result
            results_path.write_text(json.dumps(results, indent=2))
            print(f"[{tag}] {result}", flush=True)
            if result["returncode"] != 0:
                print((ROOT / "runs" / tag / "run.log").read_text()[-2000:])
                raise SystemExit(f"{tag} failed")

    # PSNR of every sparse run against its prompt's dense output.
    fps = MODELS[MODEL]["fps"]
    for prompt_id in args.prompts.split(","):
        dense_dir = ROOT / "runs" / f"{prompt_id}_dense_{args.seconds}s"
        for method in args.methods.split(","):
            if method == "dense":
                continue
            tag = f"{prompt_id}_{method}_{args.seconds}s"
            quality = psnr_vs_dense(
                dense_dir, ROOT / "runs" / tag, first_seconds=2, fps=fps
            )
            if quality:
                results[tag].update(quality)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[done] -> {results_path}")


if __name__ == "__main__":
    main()
