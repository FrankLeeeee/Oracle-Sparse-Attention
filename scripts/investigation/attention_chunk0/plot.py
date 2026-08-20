# SPDX-License-Identifier: Apache-2.0
"""Task 3.1 figures — chunk 0's attention pattern forming over its denoising steps.

One contact sheet per (config, layer): rows are chunk 0's denoising steps,
columns are the four sampled heads, each cell the query x key attention matrix
on a log colour scale. Reading a sheet top to bottom shows what the layer's
attention pattern looks like when it starts from pure noise and what it settles
into.

Per-figure maps with the coverage panel and full axis labelling are still
available from the generic tool:

    python -m sglang.multimodal_gen.tools.plot_attention_token_maps <run_dir> \\
        --steps 0,3 --layers 29

    python plot.py [--configs self_forcing/720p_20s] [--layers 0,29] [--out .]

Writes sheets/<model>_<res>_<dur>s_layer<L>.png.
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from geometry import GEOMETRY  # noqa: E402
from paths import results_dir  # noqa: E402

ROOT = results_dir("attention_chunk0")
MODEL_LABELS = {
    "self_forcing": "Self-Forcing 1.3B",
    "rolling_forcing": "Rolling Forcing 1.3B",
    "longlive2": "LongLive-2.0 5B",
    "lingbot_world_v2": "LingBot-World v2 14B",
}
MODEL_ORDER = list(MODEL_LABELS)
RESOLUTIONS = ["480p", "720p"]
DURATIONS = [5, 10, 20]
CMAP = "magma"


def load(model: str, res: str, duration: int) -> dict | None:
    run = ROOT / "runs" / model / f"{res}_{duration}s"
    dumps = sorted(run.glob("qk_chunk_000_step_*.npz"))
    if not dumps or not (run / "meta.json").exists():
        return None
    steps = {}
    for dump in dumps:
        step = int(dump.stem.split("_")[-1])
        data = np.load(dump)
        steps[step] = {
            "scores": data["scores"],  # [layers, heads, queries, keys]
            "layer_ids": [int(i) for i in data["layer_ids"]],
        }
    return {"steps": steps, "meta": json.loads((run / "meta.json").read_text())}


def plot_sheet(
    entry: dict, model: str, res: str, duration: int, layer: int, out_dir: pathlib.Path
) -> None:
    meta = entry["meta"]
    head_ids = [int(h) for h in (meta.get("qk_head_ids") or {}).get(str(layer), [])]
    steps = sorted(entry["steps"])
    panels = []
    for step in steps:
        payload = entry["steps"][step]
        if layer not in payload["layer_ids"]:
            return
        panels.append(payload["scores"][payload["layer_ids"].index(layer)])
    if not panels or not head_ids:
        return

    # One log colour scale for the whole sheet so steps really are comparable.
    # float32 on purpose: np.percentile over a large float16 array overflows
    # its own index arithmetic (n-1 exceeds float16 range) and returns NaN.
    positive = np.concatenate(
        [p[p > 0].ravel()[:200000].astype(np.float32) for p in panels]
    )
    vmax = float(max(p.max() for p in panels))
    if not positive.size or not np.isfinite(vmax) or vmax <= 0:
        return
    # Float16 dumps underflow to 0, so the low end comes from a percentile
    # rather than the minimum; keep at least a decade of range for LogNorm.
    vmin = float(np.percentile(positive, 1))
    vmin = min(max(vmin, 1e-8), vmax / 10.0)

    # One shared norm instance, not one per panel: the colorbar tracks a single
    # image and matplotlib re-derives its ticks from that image's norm.
    norm = LogNorm(vmin=vmin, vmax=vmax)
    # float16 underflows small probabilities to exactly 0, which LogNorm masks
    # and would draw as white -- i.e. looking like the *high* end. Paint the
    # masked and under-range cells with the colormap's low colour instead.
    colormap = matplotlib.colormaps[CMAP].with_extremes(
        bad=matplotlib.colormaps[CMAP](0.0), under=matplotlib.colormaps[CMAP](0.0)
    )

    rows, columns = len(steps), len(head_ids)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns + 1.1, 2.45 * rows + 0.8),
        dpi=165,
        squeeze=False,
        gridspec_kw={"hspace": 0.16, "wspace": 0.08},
    )
    image = None
    for row, step in enumerate(steps):
        for column, head in enumerate(head_ids):
            ax = axes[row][column]
            image = ax.imshow(
                panels[row][column].astype(np.float32),
                aspect="auto",
                cmap=colormap,
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"head {head}", fontsize=9)
            if column == 0:
                ax.set_ylabel(f"step {step + 1}", fontsize=9)
    bar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    bar.set_label("attention probability (log)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{MODEL_LABELS[model]} · {res} · {duration}s · chunk 0 · layer {layer}"
        f"/{GEOMETRY[model][0] - 1} — attention forming over the denoising steps "
        "(query x key, both axes are chunk 0's own tokens)",
        fontsize=11,
    )
    path = out_dir / f"{model}_{res}_{duration}s_layer{layer:02d}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def concentration(entry: dict, layer: int) -> list[float]:
    """Median share of keys needed for 90% of the mass, per denoising step."""
    shares = []
    for step in sorted(entry["steps"]):
        payload = entry["steps"][step]
        if layer not in payload["layer_ids"]:
            continue
        scores = payload["scores"][payload["layer_ids"].index(layer)]
        # scores are a strided view of a normalized row; rank and accumulate
        ranked = np.sort(scores.astype(np.float32), axis=-1)[..., ::-1]
        total = ranked.sum(axis=-1, keepdims=True)
        cumulative = np.cumsum(ranked, axis=-1) / np.maximum(total, 1e-12)
        needed = (cumulative < 0.9).sum(axis=-1) + 1
        shares.append(float(np.median(needed / scores.shape[-1])))
    return shares


def step_correlation(entry: dict, layer: int) -> list[float]:
    """How close each denoising step's map already is to the final one.

    Pearson correlation in log space (attention spans decades, so a linear
    correlation would only see the few brightest cells), averaged over heads.
    1.0 means the pattern was already in place at that step.
    """
    steps = sorted(entry["steps"])
    maps = []
    for step in steps:
        payload = entry["steps"][step]
        if layer not in payload["layer_ids"]:
            return []
        scores = payload["scores"][payload["layer_ids"].index(layer)]
        maps.append(np.log10(np.maximum(scores.astype(np.float32), 1e-9)))
    final = maps[-1]
    if final.shape != maps[0].shape:
        # Rolling Forcing's ramp-up windows grow, so the maps are not aligned
        return []
    values = []
    for current in maps:
        per_head = []
        for head in range(final.shape[0]):
            a, b = current[head].ravel(), final[head].ravel()
            a = a - a.mean()
            b = b - b.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            per_head.append(float(a @ b / denominator) if denominator else float("nan"))
        values.append(float(np.nanmean(per_head)))
    return values


def plot_formation(data: dict, res: str, duration: int, out_dir: pathlib.Path) -> None:
    """How early each depth's pattern settles, per model."""
    models = [m for m in MODEL_ORDER if (m, res, duration) in data]
    if not models:
        return
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.2 * len(models), 3.2), dpi=170, squeeze=False
    )
    for column, model in enumerate(models):
        ax = axes[0][column]
        entry = data[(model, res, duration)]
        layers = entry["steps"][sorted(entry["steps"])[0]]["layer_ids"]
        drawn = 0
        for layer in layers:
            values = step_correlation(entry, layer)
            if not values:
                continue
            drawn += 1
            ax.plot(
                range(1, len(values) + 1),
                values,
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=f"layer {layer}",
            )
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        ax.set_xlabel("denoising step", fontsize=8)
        if column == 0:
            ax.set_ylabel("correlation with the final step", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
        if drawn:
            ax.legend(fontsize=6, frameon=False, loc="lower right")
        else:
            # Rolling Forcing ramps up by *growing* the joint window, so its
            # chunk-0 maps change shape from step to step and cannot be
            # correlated against the last one.
            ax.text(
                0.5,
                0.5,
                "not comparable:\nthe ramp-up window grows\n1→5 blocks across the steps",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#777777",
            )
            ax.set_xlim(1, 5)
    fig.suptitle(
        f"How early chunk 0's attention pattern settles · {res} · {duration}s "
        "(log-space correlation of each step's map with the last step's)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out_dir / f"formation_{res}_{duration}s.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--layers", default=None, help="default: every dumped layer")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT)
    parser.add_argument(
        "--no-sheets",
        action="store_true",
        help="refresh only summary.json and the formation figures",
    )
    args = parser.parse_args()
    out_dir = args.out
    sheets = out_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    wanted = set(args.configs.split(",")) if args.configs else None
    summary = []
    loaded: dict[tuple[str, str, int], dict] = {}
    for model in MODEL_ORDER:
        for res in RESOLUTIONS:
            for duration in DURATIONS:
                if wanted is not None and f"{model}/{res}_{duration}s" not in wanted:
                    continue
                entry = load(model, res, duration)
                if entry is None:
                    continue
                loaded[(model, res, duration)] = entry
                dumped = entry["steps"][sorted(entry["steps"])[0]]["layer_ids"]
                layers = (
                    [int(x) for x in args.layers.split(",")] if args.layers else dumped
                )
                for layer in layers:
                    if layer not in dumped:
                        continue
                    if not args.no_sheets:
                        plot_sheet(entry, model, res, duration, layer, sheets)
                    summary.append(
                        {
                            "model": model,
                            "resolution": res,
                            "duration_s": duration,
                            "layer": layer,
                            "num_steps": len(entry["steps"]),
                            "key_share_for_90pct_by_step": concentration(entry, layer),
                            "correlation_with_final_step": step_correlation(
                                entry, layer
                            ),
                        }
                    )
    for res in RESOLUTIONS:
        plot_formation(loaded, res, 20, out_dir)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
