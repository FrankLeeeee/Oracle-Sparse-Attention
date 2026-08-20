# SPDX-License-Identifier: Apache-2.0
"""720p / 20 s Self-Forcing: dense vs six sparse methods at matched densities.

Reads configs.json (from calibrate.py) for the baselines; OSA-replicate takes
the density directly. One timing run per (method, target density), plus one
dense reference; every run saves its video for the quality comparison.

    python run_sweep.py [--gpus 0,1] [--methods ...] [--targets 0.5,0.3]
Results -> results.json, videos under runs/<method>_<target>/.
"""

import argparse
import json
import pathlib
import queue
import sys
import threading

from common import run_generate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa")
TARGETS = [0.5, 0.4, 0.3, 0.2, 0.1]
BASELINES = ["xattention", "svg1", "svg2", "radial", "lightforcing"]


def method_config(method: str, target: float, configs: dict) -> dict | None:
    entry = configs.get(method, {}).get(f"{target:g}")
    if entry is not None:
        return entry["config"]
    if method == "osa":  # fallback: direct steady-state knob
        return {
            "density": target,
            "sink_latent_frames": 1,
            "num_recent_frames": 1,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--methods", default="osa," + ",".join(BASELINES))
    parser.add_argument("--targets", default=",".join(str(t) for t in TARGETS))
    parser.add_argument("--skip-dense", action="store_true")
    # Separate output files let two sweep processes run concurrently on
    # different GPUs; analysis merges every results*.json.
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    targets = [float(t) for t in args.targets.split(",")]
    configs_path = ROOT / "configs.json"
    configs = json.loads(configs_path.read_text()) if configs_path.exists() else {}

    jobs: queue.Queue = queue.Queue()
    if not args.skip_dense:
        jobs.put(("dense", None, None))
    for method in args.methods.split(","):
        seen: list[dict] = []
        for target in targets:
            config = method_config(method, target, configs)
            if config is None:
                print(f"SKIP {method}@{target:g}: no calibrated config", flush=True)
                continue
            if config in seen:  # saturated knob — same run, skip duplicates
                print(f"SKIP {method}@{target:g}: saturated (same config)", flush=True)
                continue
            seen.append(config)
            jobs.put((f"{method}_{target:g}", method, config))

    results_path = ROOT / args.out
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    lock = threading.Lock()

    def worker(index: int, gpu: int) -> None:
        port_base = 39000 + index * 20
        while True:
            try:
                tag, method, config = jobs.get_nowait()
            except queue.Empty:
                return
            print(f"[gpu{gpu}] START {tag}", flush=True)
            try:
                result = run_generate(
                    out_dir=ROOT / "runs" / tag,
                    log_name="timing.log",
                    gpu=gpu,
                    port_base=port_base,
                    width=1280,
                    height=720,
                    num_frames=321,
                    method=method,
                    method_config=config,
                    save_output=True,
                    timeout_s=2400,
                )
                result["config"] = config
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


if __name__ == "__main__":
    main()
