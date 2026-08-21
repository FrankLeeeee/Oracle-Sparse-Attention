# SPDX-License-Identifier: Apache-2.0
"""Walltime vs achieved density figure for one model's dense-vs-OSA sweep.

    python plot.py --model rolling_forcing
-> <model>/walltime_vs_density.png
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import MODELS, ROOT

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    args = parser.parse_args()
    model_root = ROOT / args.model
    results = json.loads((model_root / "results.json").read_text())

    dense = results["dense"]
    runs = sorted(
        (entry for tag, entry in results.items() if tag != "dense"),
        key=lambda entry: entry.get("density", 1.0),
    )
    densities = [entry["density"] for entry in runs]
    denoise = [entry["denoise_s"] for entry in runs]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(densities, denoise, "o-", color="#1f77b4", label="OSA denoise")
    ax.axhline(
        dense["denoise_s"],
        color="#1f77b4",
        linestyle="--",
        alpha=0.6,
        label=f"dense denoise ({dense['denoise_s']:.1f} s)",
    )
    if all("e2e_s" in entry for entry in runs) and "e2e_s" in dense:
        e2e = [entry["e2e_s"] for entry in runs]
        ax.plot(densities, e2e, "s-", color="#d62728", label="OSA end-to-end")
        ax.axhline(
            dense["e2e_s"],
            color="#d62728",
            linestyle="--",
            alpha=0.6,
            label=f"dense end-to-end ({dense['e2e_s']:.1f} s)",
        )
    for x, y, entry in zip(densities, denoise, runs):
        ax.annotate(
            f"{dense['denoise_s'] / y:.2f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("achieved cumulative read density")
    ax.set_ylabel("seconds (720p / 20 s)")
    ax.set_title(f"{args.model}: dense vs OSA")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = model_root / "walltime_vs_density.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
