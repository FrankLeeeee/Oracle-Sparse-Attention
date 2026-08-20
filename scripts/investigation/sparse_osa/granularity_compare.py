# SPDX-License-Identifier: Apache-2.0
"""Compare OSA granularities (chunk / frame / replicate) at matched density.

Chunk and frame granularity are retention-driven (density is emergent), so
each is first calibrated to the ~0.30 cumulative-density tier with the same
480p / 20 s secant search as calibrate.py, then run at 720p / 20 s on the fox
prompt. Replicate and dense reuse the existing runs. Both retention-driven
configs use the docstring-recommended unbounded-KV setup (reference_chunk 8);
frame granularity keeps the classic 1-frame sink.

    python granularity_compare.py [--gpus 4,5] [--sheet-only]
Videos -> runs_granularity/<granularity>/, sheet ->
granularity_sheet_p0_fox.png, numbers -> results_granularity.json.
"""

import argparse
import json
import pathlib
import sys
import threading

from common import extract_frames, render_frame_sheet, run_generate

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa")
TARGET = 0.30
TOLERANCE = 0.02
MAX_ITERS = 5
SEEDS = (0.9, 0.5)
BOUNDS = (0.05, 0.995)

BASES = {
    "chunk": {"granularity": "chunk", "reference_chunk": 8},
    "frame": {
        "granularity": "frame",
        "reference_chunk": 8,
        "sink_latent_frames": 1,
    },
}


def measure_480p(granularity: str, config: dict, gpu: int, tag: str) -> float:
    result = run_generate(
        out_dir=ROOT / "calibration" / f"osa_{granularity}",
        log_name=f"{tag}.log",
        gpu=gpu,
        port_base=43000 + gpu * 20,
        width=832,
        height=480,
        num_frames=321,
        method="osa",
        method_config=config,
        timeout_s=1200,
    )
    if result["returncode"] != 0 or "density" not in result:
        raise RuntimeError(
            f"osa/{granularity} {config} failed (rc={result['returncode']})"
        )
    return result["density"]


def calibrate(granularity: str, gpu: int) -> dict:
    base = BASES[granularity]
    observations: list[tuple[float, float]] = []
    for seed in SEEDS:
        density = measure_480p(
            granularity, dict(base, retention=seed), gpu, f"seed_{seed:g}"
        )
        observations.append((seed, density))
        print(
            f"[{granularity}] retention={seed:g} -> density {density:.3f}", flush=True
        )
    points = sorted(observations, key=lambda p: abs(p[1] - TARGET))[:2]
    best = min(observations, key=lambda p: abs(p[1] - TARGET))
    for iteration in range(MAX_ITERS):
        if abs(best[1] - TARGET) <= TOLERANCE:
            break
        (x0, y0), (x1, y1) = points
        if abs(y1 - y0) < 1e-4:
            value = x1 * (0.5 if y1 > TARGET else 2.0)
        else:
            value = x1 + (TARGET - y1) * (x1 - x0) / (y1 - y0)
        value = min(max(value, BOUNDS[0]), BOUNDS[1])
        if any(abs(value - seen) < 1e-6 for seen, _ in observations):
            break
        density = measure_480p(
            granularity, dict(base, retention=round(value, 4)), gpu, f"i{iteration}"
        )
        print(
            f"[{granularity}] retention={value:.4f} -> density {density:.3f}",
            flush=True,
        )
        observations.append((value, density))
        points = sorted(observations, key=lambda p: abs(p[1] - TARGET))[:2]
        best = min(observations, key=lambda p: abs(p[1] - TARGET))
    return {
        "config": dict(base, retention=round(best[0], 4)),
        "calibrated_density_480p": round(best[1], 3),
    }


def calibrate_and_run(granularity: str, gpu: int, results: dict, lock) -> None:
    chosen = calibrate(granularity, gpu)
    print(f"[{granularity}] calibrated: {chosen}", flush=True)
    result = run_generate(
        out_dir=ROOT / "runs_granularity" / granularity,
        log_name="timing.log",
        gpu=gpu,
        port_base=43000 + gpu * 20,
        width=1280,
        height=720,
        num_frames=321,
        method="osa",
        method_config=chosen["config"],
        save_output=True,
        timeout_s=2400,
    )
    result["calibration"] = chosen
    with lock:
        results[granularity] = result
        (ROOT / "results_granularity.json").write_text(json.dumps(results, indent=2))
    print(
        f"DONE {granularity} rc={result['returncode']} "
        f"denoise={result.get('denoise_s')} density={result.get('density')}",
        flush=True,
    )


def run_at_retention(
    granularity: str, retention: float, gpu: int, results: dict, lock
) -> None:
    """720p run at a fixed retention (no calibration) for the matched-tier rows."""
    run_id = f"{granularity}_r{retention:g}"
    result = run_generate(
        out_dir=ROOT / "runs_granularity" / run_id,
        log_name="timing.log",
        gpu=gpu,
        port_base=43000 + gpu * 20,
        width=1280,
        height=720,
        num_frames=321,
        method="osa",
        method_config=dict(BASES[granularity], retention=retention),
        save_output=True,
        timeout_s=2400,
    )
    with lock:
        results[run_id] = result
        (ROOT / "results_granularity.json").write_text(json.dumps(results, indent=2))
    print(
        f"DONE {run_id} rc={result['returncode']} "
        f"denoise={result.get('denoise_s')} density={result.get('density')}",
        flush=True,
    )


def build_sheet() -> None:
    fps = 16
    frame_indices = [second * fps for second in (1, 4, 7, 10, 13, 16, 19)]
    results_path = ROOT / "results_granularity.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    merged = json.loads((ROOT / "results_merged.json").read_text())

    def density_of(entry: dict) -> str:
        density = entry.get("density")
        return f"d={density:.2f}" if density is not None else "d=?"

    # Top tier: the three granularities at a comfortable matched density
    # (~0.46-0.50); bottom tier: the closest each can get to the 0.30 tier —
    # replicate hits it exactly, chunk/frame saturate at their floor.
    sources = [
        ("Dense", ROOT / "runs" / "dense", None),
        (
            "chunk r=0.5",
            ROOT / "runs_granularity" / "chunk_r0.5",
            results.get("chunk_r0.5", {}),
        ),
        (
            "frame r=0.5",
            ROOT / "runs_granularity" / "frame_r0.5",
            results.get("frame_r0.5", {}),
        ),
        ("replicate", ROOT / "runs" / "osa_0.5", merged.get("osa_0.5", {})),
        ("chunk r=0.05", ROOT / "runs_granularity" / "chunk", results.get("chunk", {})),
        ("frame r=0.05", ROOT / "runs_granularity" / "frame", results.get("frame", {})),
        ("replicate", ROOT / "runs" / "osa_0.3", merged.get("osa_0.3", {})),
    ]
    rows = []
    for label, run_dir, entry in sources:
        frames = extract_frames(run_dir, frame_indices)
        if frames is None:
            print(f"missing video: {run_dir}", flush=True)
            continue
        if entry is not None:
            label = f"{label} ({density_of(entry)})"
        rows.append((label, frames))
    out = ROOT / "granularity_sheet_p0_fox.png"
    render_frame_sheet(rows=rows, frame_indices=frame_indices, fps=fps, out_path=out)
    print(f"wrote {out} rows={[label for label, _ in rows]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--sheet-only", action="store_true")
    parser.add_argument("--phase", default="all", choices=("all", "calibrated", "r0.5"))
    args = parser.parse_args()
    if not args.sheet_only:
        gpus = [int(g) for g in args.gpus.split(",")]
        results_path = ROOT / "results_granularity.json"
        results: dict = (
            json.loads(results_path.read_text()) if results_path.exists() else {}
        )
        lock = threading.Lock()
        if args.phase in ("all", "calibrated"):
            threads = [
                threading.Thread(
                    target=calibrate_and_run,
                    args=(granularity, gpu, results, lock),
                    daemon=True,
                )
                for granularity, gpu in zip(BASES, gpus)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if args.phase in ("all", "r0.5"):
            threads = [
                threading.Thread(
                    target=run_at_retention,
                    args=(granularity, 0.5, gpu, results, lock),
                    daemon=True,
                )
                for granularity, gpu in zip(BASES, gpus)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    build_sheet()


if __name__ == "__main__":
    main()
