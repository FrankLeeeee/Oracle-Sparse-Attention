# SPDX-License-Identifier: Apache-2.0
"""Walltime vs achieved density figure for one model's sweep, all methods.

    python plot.py --model self_forcing
-> <model>/walltime_vs_density.png
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import METHOD_LABELS, METHODS, MODELS, ROOT

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

COLORS = {
    "osa": "#1f77b4",
    "osa2": "#0b3d91",
    "osa2s": "#17becf",
    "osa2a": "#bcbd22",
    "osa1d": "#9edae5",
    "lightforcing": "#ff7f0e",
    "radial": "#2ca02c",
    "svg1": "#d62728",
    "svg2": "#9467bd",
    "xattention": "#8c564b",
    "sta": "#e377c2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    args = parser.parse_args()
    model_root = ROOT / args.model
    results = json.loads((model_root / "results.json").read_text())
    dense = results["dense"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in METHODS:
        runs = sorted(
            (
                entry
                for tag, entry in results.items()
                if tag.startswith(f"{method}_")
                and entry.get("returncode") == 0
                and "density" in entry
                and "denoise_s" in entry
            ),
            key=lambda entry: entry["density"],
        )
        if not runs:
            continue
        ax.plot(
            [entry["density"] for entry in runs],
            [entry["denoise_s"] for entry in runs],
            "o-",
            color=COLORS[method],
            label=METHOD_LABELS[method],
            markersize=4,
        )
    ax.axhline(
        dense["denoise_s"],
        color="black",
        linestyle="--",
        alpha=0.6,
        label=f"dense ({dense['denoise_s']:.1f} s)",
    )
    ax.set_xlabel("achieved cumulative read density")
    ax.set_ylabel("denoise seconds (720p / 20 s)")
    ax.set_title(f"{args.model}: denoise walltime vs read density")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = model_root / "walltime_vs_density.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
