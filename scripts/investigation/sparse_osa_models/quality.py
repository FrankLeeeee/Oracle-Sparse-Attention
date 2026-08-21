# SPDX-License-Identifier: Apache-2.0
"""Quality of every sweep run against the model's own dense reference.

PSNR against the dense output is a *divergence* measure: in an autoregressive
rollout any perturbation compounds into content drift, so the early segments
(before trajectories separate) carry the signal. Reports overall + first-5s
PSNR per run into <model>/results.json, and renders one labeled frame sheet
(dense + OSA at every knob) for the perceptual check.

    python quality.py --model rolling_forcing
"""

import argparse
import glob
import json
import pathlib
import sys

import imageio
import numpy as np
from common import (
    MODELS,
    ROOT,
    extract_frames,
    render_frame_sheet,
    sheet_frame_indices,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def load_video(run_dir: pathlib.Path) -> np.ndarray | None:
    paths = (
        sorted(glob.glob(str(run_dir / "outputs" / "*.mp4")))
        or sorted(glob.glob(str(run_dir / "*.mp4")))
        or sorted(glob.glob(str(run_dir / "**" / "*.mp4"), recursive=True))
    )
    if not paths:
        return None
    reader = imageio.get_reader(paths[-1])
    return np.stack([frame for frame in reader])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255**2 / max(mse, 1e-9)))


def build_sheet(model: str, results: dict, *, duration: int) -> None:
    fps = MODELS[model]["fps"]
    frame_indices = sheet_frame_indices(fps=fps, duration=duration)
    model_root = ROOT / model
    rows = []
    for tag in sorted(results, key=lambda t: (t != "dense", t), reverse=False):
        frames = extract_frames(model_root / "runs" / tag, frame_indices)
        if frames is None:
            print(f"missing video: {tag}", flush=True)
            continue
        if tag == "dense":
            label = "Dense"
        else:
            density = results[tag].get("density")
            label = f"OSA d={density:.2f}" if density else tag
        rows.append((label, frames))
    out = model_root / "quality_sheet.png"
    render_frame_sheet(rows=rows, frame_indices=frame_indices, fps=fps, out_path=out)
    print(f"wrote {out} rows={[label for label, _ in rows]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--sheet-only", action="store_true")
    args = parser.parse_args()
    model_root = ROOT / args.model
    results_path = model_root / "results.json"
    results = json.loads(results_path.read_text())

    if not args.sheet_only:
        dense = load_video(model_root / "runs" / "dense")
        if dense is None:
            raise SystemExit(f"no dense reference video under {model_root}/runs/dense")
        first = 5 * MODELS[args.model]["fps"]
        for tag, entry in sorted(results.items()):
            if tag == "dense":
                continue
            video = load_video(model_root / "runs" / tag)
            if video is None:
                continue
            n = min(len(dense), len(video))
            entry["psnr_overall_db"] = round(psnr(dense[:n], video[:n]), 2)
            entry["psnr_first5s_db"] = round(psnr(dense[:first], video[:first]), 2)
            print(
                f"{tag:12s} density={entry.get('density')} "
                f"PSNR overall={entry['psnr_overall_db']} "
                f"first5s={entry['psnr_first5s_db']}",
                flush=True,
            )
        results_path.write_text(json.dumps(results, indent=2))

    build_sheet(args.model, results, duration=args.duration)


if __name__ == "__main__":
    main()
