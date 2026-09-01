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
MODEL = "self_forcing"  # default; --model switches (CF / RF / CF-long share
# the 30x12 Wan geometry, each with its own calibrated taxonomy + LF tier)
RES = "720p"
TAXONOMY = str(HERE.parent / "qk_map_similarity" / "msa_taxonomy_self_forcing.json")


def model_taxonomy(model: str) -> str:
    return str(HERE.parent / "qk_map_similarity" / f"msa_taxonomy_{model}.json")
# The official fine-tuned Light-Forcing checkpoint (retrained WITH its sparse
# attention; converted upload of mack-williams/Light-Forcing). Same 5s
# inference geometry as Self-Forcing. Quality for these rows must be scored
# against THIS model's own dense output, never the Self-Forcing dense.
LF_CHECKPOINT = "/data/projects/vision-gen/models/LightForcing-Wan2.1-T2V-1.3B-Diffusers"
METHOD_MODEL_PATH = {"lfofficial": LF_CHECKPOINT, "lfofficialdense": LF_CHECKPOINT}


def method_flags(method: str, *, seconds: int = 5, model: str = MODEL) -> list[str]:
    if method == "dense":
        return []
    lf_tiers = json.loads((SPARSE_ROOT / "configs.json").read_text())[model][
        "lightforcing"
    ]
    if method.startswith("msa"):
        config = {
            "taxonomy_path": model_taxonomy(model),
            "content_density": {"msa25": 0.25, "msa10": 0.1}.get(method, 0.2),
        }
        # The content stage's two-stage keeps are per-model (pinned sinks,
        # window sizes) — take them from the model's calibrated LF config.
        lf_config = lf_tiers["0.2"]["config"]
        config.update(
            keep_sink=lf_config["keep_sink"],
            keep_near=lf_config["keep_near"],
            keep_frames=lf_config["keep_frames"],
        )
        if "turbo" in method or "mild" in method:
            # turbo/mild ride on the schedule at the density-parity means.
            config["content_density"] = 0.22 if "22" in method else 0.14
            method = "msasched" + ("22" if "22" in method else "14") + (
                "turbo" if "turbo" in method else "mild"
            )
        if method.startswith("msalf"):
            # Content heads on LightForcing's own calibrated front-loaded
            # schedule — the capped-window models' large LF latency lead is
            # this schedule shape, which flops_matched cannot express there.
            lf = dict(lf_tiers["0.2"]["config"])
            config.update(
                content_schedule="lightforcing",
                lf_sparsity=lf["sparsity"],
                lf_sparsity_base=lf.get("sparsity_base", 0.98),
                lf_num_output_frames=MODELS[model]["frames"][seconds],
                lf_local_attn_size=lf.get("local_attn_size", -1),
            )
            if method == "msalf3":
                # Late-window density ~2.2x LightForcing's (0.013 -> 0.028):
                # those calls are planning-bound, so the extra keys cost
                # almost nothing while buying back fidelity.
                config["lf_sparsity"] = 0.972
        elif "sched" in method:
            config.update(
                content_schedule="flops_matched",
                schedule_num_frames=(MODELS[model]["frames"][seconds] + 3) // 4,
                schedule_window_frames=MODELS[model]["window_frames"],
            )
        if "15" in method:
            config["content_density"] = 0.15
        elif "14" in method:
            config["content_density"] = 0.14
        elif "13" in method:
            config["content_density"] = 0.13
        elif "22" in method:
            config["content_density"] = 0.22
        if method.endswith("r2"):
            config["replan_interval"] = 2
        if "turbo" in method:
            # Step-aware density: late denoise steps read a shrinking prefix
            # of the chunk's ranked blocks (attention concentrates through
            # denoising, so the same mass needs fewer blocks).
            config["step_density_scale"] = (1.0, 0.85, 0.65, 0.45)
        elif "mild" in method:
            config["step_density_scale"] = (1.0, 0.9, 0.75, 0.6)
        method = "msa"
    elif method == "lfofficialdense":
        return []
    elif method in ("lightforcing", "lf10", "lfofficial"):
        tier = "0.1" if method == "lf10" else "0.2"
        config = dict(lf_tiers[tier]["config"])
        if seconds != 5 and "num_output_frames" in config:
            # LF's schedule solve needs the run's actual video length.
            config["num_output_frames"] = MODELS[model]["frames"][seconds]
        method = "lightforcing"
    else:
        raise ValueError(method)
    return ["--sparse-attention", method, "--sparse-attention-config", json.dumps(config)]


def run_one(
    *, tag: str, prompt: str, method: str, gpu: int, port_base: int, seconds: int,
    model: str = MODEL,
) -> dict:
    spec = MODELS[model]
    width, height = spec["resolutions"][RES]
    out_dir = ROOT / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sglang", "generate",
        "--model-path", METHOD_MODEL_PATH.get(method, spec["path"]),
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(spec["frames"][seconds]),
        "--seed", str(SEED),
        "--save-output",
        "--master-port", str(port_base),
        "--scheduler-port", str(port_base + 1),
        "--port", str(port_base + 2),
    ] + method_flags(method, seconds=seconds, model=model)
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
    parser.add_argument("--model", default=MODEL, choices=sorted(MODELS))
    parser.add_argument("--port-base", type=int, default=29960)
    args = parser.parse_args()

    pool = GpuPool([int(g) for g in args.gpus.split(",")])
    prefix = "" if args.model == MODEL else f"{args.model}_"
    results_path = ROOT / f"results_{prefix}{args.seconds}s.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    index = 0
    for prompt_id in args.prompts.split(","):
        for method in args.methods.split(","):
            tag = f"{prefix}{prompt_id}_{method}_{args.seconds}s"
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
                        model=args.model,
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
    fps = MODELS[args.model]["fps"]
    for prompt_id in args.prompts.split(","):
        for method in args.methods.split(","):
            if method in ("dense", "lfofficialdense"):
                continue
            # Official-checkpoint rows score against that checkpoint's own
            # dense; everything else against the base model's dense.
            reference = (
                "lfofficialdense" if method.startswith("lfofficial") else "dense"
            )
            dense_dir = ROOT / "runs" / f"{prefix}{prompt_id}_{reference}_{args.seconds}s"
            tag = f"{prefix}{prompt_id}_{method}_{args.seconds}s"
            quality = psnr_vs_dense(
                dense_dir, ROOT / "runs" / tag, first_seconds=2, fps=fps
            )
            if quality:
                results[tag].update(quality)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[done] -> {results_path}")


if __name__ == "__main__":
    main()
