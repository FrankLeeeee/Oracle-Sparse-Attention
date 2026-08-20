# SPDX-License-Identifier: Apache-2.0
"""Plot the runtime-breakdown sweep: e2e stages, denoise components, scaling.

Reads breakdown.json (from analyze.py), writes PNGs next to it:
- e2e_stages_<res>.png       stacked walltime by pipeline stage
- denoise_components_<res>.png  denoise walltime split by GPU-kernel category
- denoise_scaling.png        denoise walltime vs video duration per model

Palette/marks follow the dataviz reference instance (light mode); the
categorical order below passes adjacent-pair validation.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("runtime_breakdown")
DATA = json.loads((ROOT / "breakdown.json").read_text())

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"

MODELS = [
    ("self_forcing", "Self-Forcing 1.3B (full ctx)"),
    ("rolling_forcing", "Rolling Forcing 1.3B"),
    ("longlive2", "LongLive-2.0 5B"),
    ("lingbot_world_v2", "LingBot-World v2 14B"),
]
DURATIONS = [5, 10, 20, 30]

STAGES = [
    ("denoise_s", "denoise", "#2a78d6"),
    ("vae_decode_s", "VAE decode", "#eb6834"),
    ("text_encode_s", "text encode", "#1baf7a"),
    ("input_prep_s", "input prep", "#eda100"),
    ("other_stages_s", "other stages", "#e87ba4"),
    ("_overhead", "launch/stream overhead", "#4a3aa7"),
]
COMPONENTS = [
    ("attention", "#2a78d6"),
    ("GEMM", "#eb6834"),
    ("elementwise/norm/other", "#eda100"),
    ("memcpy", "#008300"),
    ("host/launch gap", "#4a3aa7"),
]
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _rows(res: str) -> list[tuple[str, str, dict]]:
    rows = []
    for model, label in MODELS:
        for duration in DURATIONS:
            entry = DATA.get(model, {}).get(f"{res}_{duration}s")
            if entry and "e2e_s" in entry:
                rows.append((model, f"{label} · {duration}s", entry))
    return rows


def e2e_stages(res: str) -> None:
    rows = _rows(res)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 2.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    y_positions = list(range(len(rows)))[::-1]
    for y, (model, label, entry) in zip(y_positions, rows):
        known = sum(entry.get(key, 0.0) for key, _, _ in STAGES[:-1])
        overhead = max(entry.get("e2e_s", known) - known, 0.0)
        left = 0.0
        for key, _, color in STAGES:
            value = overhead if key == "_overhead" else entry.get(key, 0.0)
            if value <= 0:
                continue
            ax.barh(
                y,
                value,
                left=left,
                height=0.62,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
            )
            if value >= 0.04 * entry["e2e_s"] and value >= 1.5:
                ax.text(
                    left + value / 2,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=SURFACE,
                    fontweight="bold",
                )
            left += value
        ax.text(
            left + 0.01 * ax.get_xlim()[1],
            y,
            f"  {entry['e2e_s']:.0f}s",
            ha="left",
            va="center",
            fontsize=8,
            color=INK2,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label, _ in rows], color=INK, fontsize=8.5)
    ax.set_xlim(0, max(e["e2e_s"] for _, _, e in rows) * 1.14)
    ax.set_xlabel("end-to-end walltime (s)", color=INK2, fontsize=9.5)
    ax.set_title(
        f"End-to-end walltime by stage — {res}",
        color=INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in STAGES]
    ax.legend(
        handles,
        [n for _, n, _ in STAGES],
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=INK2,
    )
    fig.tight_layout()
    out = ROOT / f"e2e_stages_{res}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def denoise_components(res: str) -> None:
    rows = [
        (model, label, entry)
        for model, label, entry in _rows(res)
        if "denoise_split_s" in entry
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows) + 2.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    y_positions = list(range(len(rows)))[::-1]
    max_total = max(sum(e["denoise_split_s"].values()) for _, _, e in rows)
    for y, (model, label, entry) in zip(y_positions, rows):
        split = entry["denoise_split_s"]
        left = 0.0
        for name, color in COMPONENTS:
            value = split.get(name, 0.0)
            if value <= 0:
                continue
            ax.barh(
                y,
                value,
                left=left,
                height=0.62,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
            )
            if value >= 0.05 * max_total:
                ax.text(
                    left + value / 2,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=SURFACE,
                    fontweight="bold",
                )
            left += value
        ax.text(
            left + 0.01 * max_total,
            y,
            f"  {left:.0f}s",
            ha="left",
            va="center",
            fontsize=8,
            color=INK2,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label, _ in rows], color=INK, fontsize=8.5)
    ax.set_xlim(0, max_total * 1.14)
    ax.set_xlabel(
        "denoise walltime (s), split by profiled GPU-kernel shares",
        color=INK2,
        fontsize=9.5,
    )
    ax.set_title(
        f"Denoise walltime by component — {res}",
        color=INK,
        fontsize=12,
        loc="left",
        pad=14,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in COMPONENTS]
    ax.legend(
        handles,
        [n for n, _ in COMPONENTS],
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=INK2,
    )
    fig.tight_layout()
    out = ROOT / f"denoise_components_{res}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def denoise_scaling() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, res in zip(axes, ["480p", "720p"]):
        style_axes(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        for (model, label), color in zip(MODELS, MODEL_COLORS):
            xs, ys = [], []
            for duration in DURATIONS:
                entry = DATA.get(model, {}).get(f"{res}_{duration}s")
                if entry and "denoise_s" in entry:
                    xs.append(duration)
                    ys.append(entry["denoise_s"])
            if not xs:
                continue
            ax.plot(
                xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label
            )
        ax.set_xticks(DURATIONS)
        ax.set_xlabel("video duration (s)", color=INK2, fontsize=9.5)
        ax.set_title(res, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("denoise walltime (s)", color=INK2, fontsize=9.5)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle(
        "Denoise walltime vs video duration", color=INK, fontsize=12, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "denoise_scaling.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for res in ["480p", "720p"]:
        e2e_stages(res)
        denoise_components(res)
    denoise_scaling()
