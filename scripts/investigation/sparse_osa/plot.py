# SPDX-License-Identifier: Apache-2.0
"""Plots of the matched-density sparse-attention sweep.

- walltime_vs_density.png: denoise + e2e walltime against *achieved* density,
  one line per method, dense as a reference line.
- speedup_bars.png: denoise speedup over dense at each target level.

Palette follows the dataviz reference instance (light mode); the categorical
order passes adjacent-pair validation.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa")
DATA: dict = {}
for _path in sorted(ROOT.glob("results*.json")):
    if _path.name != "results_merged.json":
        DATA.update(json.loads(_path.read_text()))
_merged = ROOT / "results_merged.json"
if _merged.exists():  # quality.py enriches the merged file with PSNR
    DATA.update(json.loads(_merged.read_text()))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

METHODS = [
    ("osa", "OSA", "#2a78d6"),
    ("lightforcing", "LightForcing", "#eb6834"),
    ("svg1", "SVG1", "#1baf7a"),
    ("svg2", "SVG2", "#eda100"),
    ("xattention", "XAttention", "#e87ba4"),
    ("radial", "Radial", "#008300"),
]
TARGETS = [0.5, 0.4, 0.3, 0.2, 0.1]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def series(method: str, metric: str):
    xs, ys = [], []
    for target in TARGETS:
        entry = DATA.get(f"{method}_{target:g}")
        if entry and metric in entry and "density" in entry:
            xs.append(entry["density"])
            ys.append(entry[metric])
    return xs, ys


def walltime_vs_density() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    dense = DATA.get("dense", {})
    for ax, metric, label in (
        (axes[0], "denoise_s", "denoise walltime (s)"),
        (axes[1], "e2e_s", "end-to-end walltime (s)"),
    ):
        style_axes(ax)
        if metric in dense:
            ax.axhline(dense[metric], color=INK2, linewidth=1.2, linestyle="--")
            ax.text(
                0.99,
                dense[metric],
                f" dense {dense[metric]:.0f}s",
                fontsize=8,
                color=INK2,
                va="bottom",
                ha="right",
            )
        for method, label_m, color in METHODS:
            xs, ys = series(method, metric)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label=label_m,
                )
        ax.set_xlabel("achieved read density", color=INK2, fontsize=9.5)
        ax.set_ylabel(label, color=INK2, fontsize=9.5)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle(
        "Self-Forcing 1.3B full context · 720p / 20s · walltime vs achieved density",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "walltime_vs_density.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def speedup_bars() -> None:
    dense = DATA.get("dense", {}).get("denoise_s")
    if not dense:
        return
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    width = 0.13
    for index, (method, label, color) in enumerate(METHODS):
        xs, ys = [], []
        for t_index, target in enumerate(TARGETS):
            entry = DATA.get(f"{method}_{target:g}")
            if entry and "denoise_s" in entry:
                xs.append(t_index + (index - len(METHODS) / 2 + 0.5) * width)
                ys.append(dense / entry["denoise_s"])
        ax.bar(
            xs,
            ys,
            width=width,
            color=color,
            label=label,
            edgecolor=SURFACE,
            linewidth=0.8,
        )
        for x, y in zip(xs, ys):
            ax.text(
                x,
                y + 0.02,
                f"{y:.2f}",
                ha="center",
                fontsize=6.2,
                color=INK2,
                rotation=90,
            )
    ax.axhline(1.0, color=INK2, linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(
        [f"target density {t:g}\n(sparsity {1-t:.0%})" for t in TARGETS], fontsize=8.5
    )
    ax.set_ylabel("denoise speedup vs dense (×)", color=INK2, fontsize=9.5)
    ax.set_title(
        "Denoise speedup over dense — 720p / 20s Self-Forcing",
        color=INK,
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2, ncol=3)
    fig.tight_layout()
    out = ROOT / "speedup_bars.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    walltime_vs_density()
    speedup_bars()
