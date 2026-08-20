# SPDX-License-Identifier: Apache-2.0
"""Per-chunk forward / attention wall-time figures from the chunk_runtime sweep.

Reads runs/<model>/<res>_<dur>s/chunk_timing.json (written by the probe behind
``SGLANG_DIFFUSION_CHUNK_TIMING_DIR``) and renders, per resolution, a
models x durations grid of "wall time vs chunk index" panels: total DiT
forward and the attention modules inside it (self + cross), both summed over
the chunk's denoising steps. A second figure summarizes the attention share.

    python plot.py [--runs runs] [--out .]

Writes chunk_walltime_<res>.png, attention_share.png and summary.json under
results/investigation/chunk_runtime/.
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("chunk_runtime")

MODEL_LABELS = {
    "self_forcing": "Self-Forcing 1.3B",
    "rolling_forcing": "Rolling Forcing 1.3B",
    "longlive2": "LongLive-2.0 5B",
    "lingbot_world_v2": "LingBot-World v2 14B",
}
MODEL_ORDER = list(MODEL_LABELS)
DURATIONS = [5, 10, 20]
RESOLUTIONS = ["480p", "720p"]

FORWARD_COLOR = "#1f77b4"
ATTENTION_COLOR = "#d62728"
CACHE_COLOR = "#999999"


def load_config(runs: pathlib.Path, model: str, res: str, duration: int) -> dict | None:
    path = runs / model / f"{res}_{duration}s" / "chunk_timing.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def series(payload: dict) -> dict:
    """Per-chunk millisecond series, keyed by what the panel plots."""
    chunks, forward, attention, cache, steps = [], [], [], [], []
    for entry in payload["chunks"]:
        denoise = entry.get("denoise")
        if denoise is None:
            continue
        chunks.append(entry["chunk"])
        forward.append(denoise["forward_ms"])
        attention.append(denoise["self_attn_ms"] + denoise["cross_attn_ms"])
        steps.append(denoise["steps"])
        update = entry.get("cache_update")
        cache.append(update["forward_ms"] if update else 0.0)
    return {
        "chunks": chunks,
        "forward_ms": forward,
        "attention_ms": attention,
        "cache_update_ms": cache,
        "steps": steps,
    }


def warmup_offset(values: dict) -> int:
    """Whether chunk 0 is a warmup outlier and should be left out of the plot.

    A one-shot `sglang generate` pays CUDA kernel autotuning on its very first
    forward (seconds, against a steady state of a few hundred ms), which would
    set the y range for the whole panel. A realtime server warms up before the
    session, so its chunk 0 is a genuine data point and is kept.
    """
    forward = values["forward_ms"]
    if len(forward) < 2:
        return 0
    # Chunk 0 always has the *smallest* context, so a healthy chunk 0 is never
    # more expensive than chunk 1; anything above it is autotuning.
    return 1 if forward[0] > 1.5 * forward[1] else 0


def plot_resolution(data: dict, res: str, out_dir: pathlib.Path) -> None:
    models = [m for m in MODEL_ORDER if any((m, res, d) in data for d in DURATIONS)]
    if not models:
        return
    fig, axes = plt.subplots(
        len(models),
        len(DURATIONS),
        figsize=(4.4 * len(DURATIONS), 2.7 * len(models)),
        dpi=170,
        squeeze=False,
    )
    for row, model in enumerate(models):
        for col, duration in enumerate(DURATIONS):
            ax = axes[row][col]
            entry = data.get((model, res, duration))
            if entry is None:
                ax.set_axis_off()
                continue
            values = entry["series"]
            start = warmup_offset(values)
            chunks = values["chunks"][start:]
            ax.plot(
                chunks,
                values["forward_ms"][start:],
                color=FORWARD_COLOR,
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                label="DiT forward",
            )
            ax.plot(
                chunks,
                values["attention_ms"][start:],
                color=ATTENTION_COLOR,
                marker="s",
                markersize=2.5,
                linewidth=1.3,
                label="attention (self+cross)",
            )
            if any(values["cache_update_ms"][start:]):
                ax.plot(
                    chunks,
                    values["cache_update_ms"][start:],
                    color=CACHE_COLOR,
                    linewidth=1.0,
                    linestyle="--",
                    label="KV cache-update forward",
                )
            ax.set_ylim(bottom=0)
            if start:
                ax.annotate(
                    f"chunk 0: {values['forward_ms'][0] / 1000:.1f}s (warmup)",
                    xy=(0.98, 0.06),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=6.5,
                    color="#666666",
                )
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.25, which="both", linewidth=0.4)
            ax.set_title(f"{MODEL_LABELS[model]} · {duration}s", fontsize=9)
            if row == len(models) - 1:
                ax.set_xlabel("chunk index", fontsize=8)
            if col == 0:
                ax.set_ylabel("wall time per chunk (ms)", fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"Per-chunk wall time · {res} · summed over the chunk's denoising steps "
        "(chunk 0 omitted where it carries one-off kernel warmup)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    path = out_dir / f"chunk_walltime_{res}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_attention_share(data: dict, out_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, len(RESOLUTIONS), figsize=(11, 3.6), dpi=170)
    for ax, res in zip(axes, RESOLUTIONS):
        for model in MODEL_ORDER:
            entry = data.get((model, res, 20))
            if entry is None:
                continue
            values = entry["series"]
            start = warmup_offset(values)
            share = [
                100.0 * a / f if f else 0.0
                for a, f in zip(
                    values["attention_ms"][start:], values["forward_ms"][start:]
                )
            ]
            ax.plot(
                values["chunks"][start:],
                share,
                linewidth=1.4,
                marker="o",
                markersize=2.5,
                label=MODEL_LABELS[model],
            )
        ax.set_title(f"{res} · 20s", fontsize=10)
        ax.set_xlabel("chunk index", fontsize=9)
        ax.set_ylabel("attention share of forward (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Attention share of the DiT forward, per chunk", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "attention_share.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=pathlib.Path, default=ROOT / "runs")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    runs, out_dir = args.runs, args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    data: dict[tuple[str, str, int], dict] = {}
    for model in MODEL_ORDER:
        for res in RESOLUTIONS:
            for duration in DURATIONS:
                payload = load_config(runs, model, res, duration)
                if payload is None:
                    continue
                data[(model, res, duration)] = {
                    "payload": payload,
                    "series": series(payload),
                }

    for res in RESOLUTIONS:
        plot_resolution(data, res, out_dir)
    plot_attention_share(data, out_dir)

    summary = []
    for (model, res, duration), entry in sorted(data.items()):
        values = entry["series"]
        # Steady state deliberately drops chunk 0, whose first forward pays the
        # process's one-time kernel autotuning.
        start = warmup_offset(values)
        steady_forward = values["forward_ms"][start:]
        steady_attention = values["attention_ms"][start:]
        summary.append(
            {
                "model": model,
                "resolution": res,
                "duration_s": duration,
                "num_chunks": len(values["chunks"]),
                "steps_per_chunk": max(values["steps"]),
                "chunk0_forward_ms": values["forward_ms"][0],
                "chunk0_is_warmup": bool(start),
                "first_steady_forward_ms": steady_forward[0],
                "last_forward_ms": values["forward_ms"][-1],
                "mean_steady_forward_ms": sum(steady_forward) / len(steady_forward),
                "mean_steady_attention_ms": (
                    sum(steady_attention) / len(steady_attention)
                ),
                "attention_share_pct": (
                    100.0 * sum(steady_attention) / sum(steady_forward)
                    if sum(steady_forward)
                    else 0.0
                ),
                "total_forward_s": sum(values["forward_ms"]) / 1000.0,
                "total_cache_update_s": sum(values["cache_update_ms"]) / 1000.0,
            }
        )
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
