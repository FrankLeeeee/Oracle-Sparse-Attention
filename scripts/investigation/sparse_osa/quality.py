# SPDX-License-Identifier: Apache-2.0
"""Quality of every sweep run against the dense reference.

PSNR against the dense output is a *divergence* measure: in an autoregressive
rollout any perturbation compounds into content drift, so the early segments
(before trajectories separate) carry the signal and the absolute value decays
with time for every method. Reports overall + first-5s PSNR per run into
results.json, and renders a frame sheet (dense + each method at one target)
for the perceptual check.

    python quality.py [--sheet-target 0.2]
"""

import argparse
import glob
import json
import pathlib
import sys

import imageio
import numpy as np
from common import METHOD_LABELS, extract_frames, render_frame_sheet

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa")


def load_video(run_dir: pathlib.Path) -> np.ndarray | None:
    paths = sorted(glob.glob(str(run_dir / "outputs" / "*.mp4"))) or sorted(
        glob.glob(str(run_dir / "*.mp4"))
    )
    if not paths:
        return None
    reader = imageio.get_reader(paths[-1])
    return np.stack([frame for frame in reader])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255**2 / max(mse, 1e-9)))


def build_sheet(target: str) -> None:
    """Labeled frame sheet: dense + every method at one target, 7 frames."""
    fps = 16
    frame_indices = [second * fps for second in (1, 4, 7, 10, 13, 16, 19)]
    rows = []
    for method in METHOD_LABELS:
        run_dir = (
            ROOT / "runs" / ("dense" if method == "dense" else f"{method}_{target}")
        )
        frames = extract_frames(run_dir, frame_indices)
        if frames is None:
            print(f"missing video: {run_dir.name}", flush=True)
            continue
        rows.append((METHOD_LABELS[method], frames))
    out = ROOT / f"quality_sheet_target{target}.png"
    render_frame_sheet(rows=rows, frame_indices=frame_indices, fps=fps, out_path=out)
    print(f"wrote {out} rows={[label for label, _ in rows]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-target", default="0.2")
    parser.add_argument("--sheet-only", action="store_true")
    args = parser.parse_args()

    if args.sheet_only:
        build_sheet(args.sheet_target)
        return

    results: dict = {}
    for path in sorted(ROOT.glob("results*.json")):
        results.update(json.loads(path.read_text()))
    results_path = ROOT / "results_merged.json"
    dense = load_video(ROOT / "runs" / "dense")
    if dense is None:
        raise SystemExit("no dense reference video under runs/dense")

    fps = 16
    first = 5 * fps
    for tag, entry in sorted(results.items()):
        if tag == "dense":
            continue
        video = load_video(ROOT / "runs" / tag)
        if video is None:
            continue
        n = min(len(dense), len(video))
        entry["psnr_overall_db"] = round(psnr(dense[:n], video[:n]), 2)
        entry["psnr_first5s_db"] = round(psnr(dense[:first], video[:first]), 2)
        print(
            f"{tag:22s} density={entry.get('density')} "
            f"PSNR overall={entry['psnr_overall_db']} "
            f"first5s={entry['psnr_first5s_db']}",
            flush=True,
        )
    results_path.write_text(json.dumps(results, indent=2))
    build_sheet(args.sheet_target)


if __name__ == "__main__":
    main()
