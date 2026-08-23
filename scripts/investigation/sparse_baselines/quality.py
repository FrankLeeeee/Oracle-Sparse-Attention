# SPDX-License-Identifier: Apache-2.0
"""Quality of every sweep run against the model's own dense reference.

PSNR against the dense output is a *divergence* measure: in an autoregressive
rollout any perturbation compounds into content drift, so the early segments
(before trajectories separate) carry the signal. Reports overall + first-5s
PSNR per run into <model>/results.json, and renders one labeled frame sheet
(dense + each method at the ~0.30 tier) for the perceptual check.

    python quality.py --model self_forcing
"""

import argparse
import glob
import json
import pathlib
import sys

import imageio
import numpy as np
from common import (
    METHOD_LABELS,
    METHODS,
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


def tier_tag(results: dict, method: str, tier: str = "0.3") -> str | None:
    """The sweep tag of a method's ~tier run (floored tiers reuse a config)."""
    exact = f"{method}_{tier}"
    if exact in results:
        return exact
    candidates = [tag for tag in results if tag.startswith(f"{method}_")]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda tag: abs(results[tag].get("density", 1.0) - float(tier)),
    )


def build_sheet(model: str, results: dict, *, duration: int, tier: str) -> None:
    fps = MODELS[model]["fps"]
    frame_indices = sheet_frame_indices(fps=fps, duration=duration)
    model_root = ROOT / model
    rows = []
    tags = ["dense"] + [
        tag for tag in (tier_tag(results, method, tier) for method in METHODS) if tag
    ]
    for tag in tags:
        frames = extract_frames(model_root / "runs" / tag, frame_indices)
        if frames is None:
            print(f"missing video: {tag}", flush=True)
            continue
        if tag == "dense":
            label = "Dense"
        else:
            method = tag.rsplit("_", 1)[0]
            density = results[tag].get("density")
            label = METHOD_LABELS.get(method, method)
            if density:
                label = f"{label} d={density:.2f}"
        rows.append((label, frames))
    out = model_root / f"quality_sheet_target{tier}.png"
    render_frame_sheet(rows=rows, frame_indices=frame_indices, fps=fps, out_path=out)
    print(f"wrote {out} rows={[label for label, _ in rows]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--tier", default="0.3")
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
                f"{tag:20s} density={entry.get('density')} "
                f"PSNR overall={entry['psnr_overall_db']} "
                f"first5s={entry['psnr_first5s_db']}",
                flush=True,
            )
        results_path.write_text(json.dumps(results, indent=2))

    build_sheet(args.model, results, duration=args.duration, tier=args.tier)


if __name__ == "__main__":
    main()
