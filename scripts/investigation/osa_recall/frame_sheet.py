# SPDX-License-Identifier: Apache-2.0
"""Labeled frame sheet: dense + frozen/replan at each density tier.

    python frame_sheet.py [--out replan_sheet_lowd.png]

Rows are labeled on the left with the method name (ASCII only — the sheet
font has no CJK glyphs); columns are t = 1/5/10/15/19 s at 16 fps.
"""

import argparse
import glob
import pathlib
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("osa_recall")
DENSE = (
    REPO
    / "results/investigation/sparse_baselines/self_forcing/runs/dense"
)
CAMPAIGN = REPO / "results/investigation/sparse_baselines/self_forcing/runs"
ROWS = [
    ("Dense", DENSE),
    ("OSA frozen d=0.2", ROOT / "sf20t_frozen_d02"),
    ("OSA replan d=0.2", ROOT / "sf20t_replan_d02"),
    ("LF constant d=0.2", ROOT / "sf20_lf_d02"),
    ("LF front-loaded d=0.2", CAMPAIGN / "lightforcing_0.2"),
    ("OSA frozen d=0.1", ROOT / "sf20t_frozen_d01"),
    ("OSA replan d=0.1", ROOT / "sf20t_replan_d01"),
    ("LF constant d=0.1", ROOT / "sf20_lf_d01"),
    ("LF front-loaded d=0.1", CAMPAIGN / "lightforcing_0.1"),
]
FRAME_IDX = [16, 80, 160, 240, 304]  # t = 1, 5, 10, 15, 19 s @ 16 fps
LABEL_WIDTH = 330


def video_path(run_dir: pathlib.Path) -> str:
    paths = sorted(glob.glob(str(run_dir / "**" / "*.mp4"), recursive=True))
    if not paths:
        raise FileNotFoundError(run_dir)
    return paths[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="replan_sheet_lowd.png")
    args = parser.parse_args()

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30
    )
    rows = []
    for label, run_dir in ROWS:
        reader = imageio.get_reader(video_path(run_dir))
        frames = [reader.get_data(i)[::2, ::2] for i in FRAME_IDX]
        strip = np.concatenate(frames, axis=1)
        panel = Image.new("RGB", (LABEL_WIDTH + strip.shape[1], strip.shape[0]), "black")
        panel.paste(Image.fromarray(strip), (LABEL_WIDTH, 0))
        draw = ImageDraw.Draw(panel)
        for offset, line in enumerate(label.split(" d=")):
            text = line if offset == 0 else f"d={line}"
            draw.text((14, strip.shape[0] // 2 - 40 + offset * 40), text,
                      fill="white", font=font)
        rows.append(np.asarray(panel))
    sheet = np.concatenate(rows, axis=0)
    out = ROOT / args.out
    imageio.imwrite(out, sheet)
    print(f"wrote {out} {sheet.shape}")


if __name__ == "__main__":
    main()
