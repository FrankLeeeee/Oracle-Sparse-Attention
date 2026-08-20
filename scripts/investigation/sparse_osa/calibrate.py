# SPDX-License-Identifier: Apache-2.0
"""Calibrate every baseline's knob to hit the target read densities.

OSA-replicate takes `density` directly, so only the baselines are searched.
Each (method, target) does a monotone secant search on the method's knob,
measured as the run-final cumulative density of a 480p / 20 s (321-frame)
generation — the same 27-chunk trajectory as the 720p target runs, so the
frame-granular densities transfer.

    python calibrate.py [--gpus 0,1] -> configs.json
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
MAX_ITERS = 5
TOLERANCE = 0.02

# knob -> config; two seed values bracket each search. All knobs are monotone
# in density (band width, kept-mass thresholds, decay length, 1 - sparsity).
METHODS = {
    # OSA's density knob targets the steady-state per-chunk density; the
    # cumulative run density sits above it because early chunks are dense
    # (calibration + fixed frames dominating a short history), so it is
    # searched like the others to match the baselines' cumulative figure.
    "osa": {
        "knob": "density",
        "seeds": (0.45, 0.12),
        "bounds": (0.02, 0.95),
        # Minimal fixed keeps (own chunk + 1 sink + 1 recent frame): the
        # replicated tile pattern carries the budget, and the cumulative
        # density floor drops from ~0.34 to ~0.2.
        "base": {
            "sink_latent_frames": 1,
            "num_recent_frames": 1,
        },
    },
    "xattention": {
        "knob": "threshold",
        "seeds": (0.95, 0.5),
        "bounds": (0.01, 0.9999),
        "base": {},
    },
    "svg1": {
        "knob": "band_frames",
        "seeds": (40.0, 8.0),
        "bounds": (0.5, 80.0),
        "base": {},
    },
    "svg2": {
        "knob": "top_p",
        "seeds": (0.95, 0.5),
        "bounds": (0.01, 0.9999),
        "base": {},
    },
    "radial": {
        "knob": "decay_factor",
        "seeds": (2.0, 0.25),
        "bounds": (0.01, 64.0),
        "base": {},
    },
    "lightforcing": {
        "knob": "sparsity",
        "seeds": (0.5, 0.9),
        "bounds": (0.0, 0.995),
        "base": {"num_output_frames": 81, "local_attn_size": -1},
    },
}


def measure(method: str, config: dict, gpu: int, port_base: int, tag: str) -> float:
    result = run_generate(
        out_dir=ROOT / "calibration" / method,
        log_name=f"{tag}.log",
        gpu=gpu,
        port_base=port_base,
        width=832,
        height=480,
        num_frames=321,
        method=method,
        method_config=config,
        timeout_s=1200,
    )
    if result["returncode"] != 0 or "density" not in result:
        raise RuntimeError(f"{method} {config} failed (rc={result['returncode']})")
    return result["density"]


def calibrate_method(method: str, gpu: int, port_base: int) -> dict[str, dict]:
    spec = METHODS[method]
    knob, (seed_hi, seed_lo) = spec["knob"], spec["seeds"]
    lo_bound, hi_bound = spec["bounds"]
    # Two seed measurements shared across every target of the method.
    observations: list[tuple[float, float]] = []  # (knob value, density)
    for seed in (seed_hi, seed_lo):
        config = dict(spec["base"], **{knob: seed})
        density = measure(method, config, gpu, port_base, f"seed_{seed:g}")
        observations.append((seed, density))
        print(f"[{method}] {knob}={seed:g} -> density {density:.3f}", flush=True)

    chosen: dict[str, dict] = {}
    for target in TARGETS:
        points = sorted(observations, key=lambda p: abs(p[1] - target))[:2]
        best = min(observations, key=lambda p: abs(p[1] - target))
        for iteration in range(MAX_ITERS):
            if abs(best[1] - target) <= TOLERANCE:
                break
            (x0, y0), (x1, y1) = points
            if abs(y1 - y0) < 1e-4:
                value = x1 * (0.5 if y1 > target else 2.0)
            else:
                value = x1 + (target - y1) * (x1 - x0) / (y1 - y0)
            value = min(max(value, lo_bound), hi_bound)
            if any(abs(value - seen) < 1e-6 for seen, _ in observations):
                break  # clamped onto an already-measured value (saturated)
            config = dict(spec["base"], **{knob: round(value, 4)})
            density = measure(
                method,
                config,
                gpu,
                port_base,
                f"t{target:g}_i{iteration}",
            )
            print(
                f"[{method}] target {target:.2f}: {knob}={value:.4f} -> "
                f"density {density:.3f}",
                flush=True,
            )
            observations.append((value, density))
            points = sorted(observations, key=lambda p: abs(p[1] - target))[:2]
            best = min(observations, key=lambda p: abs(p[1] - target))
        chosen[f"{target:g}"] = {
            "config": dict(spec["base"], **{knob: round(best[0], 4)}),
            "calibrated_density_480p": round(best[1], 3),
        }
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    jobs: queue.Queue = queue.Queue()
    for method in args.methods.split(","):
        jobs.put(method)
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def worker(index: int, gpu: int) -> None:
        port_base = 38000 + index * 20
        while True:
            try:
                method = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                chosen = calibrate_method(method, gpu, port_base)
                with lock:
                    results[method] = chosen
            except Exception as error:
                print(f"[{method}] FAILED: {error}", flush=True)
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

    out = ROOT / "configs.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(results)
    out.write_text(json.dumps(merged, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
