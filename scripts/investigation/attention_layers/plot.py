# SPDX-License-Identifier: Apache-2.0
"""Tasks 3.2 + 3.3 figures — attention across the video and across depth.

One contact sheet per (config, layer): rows are the sampled chunks, spread over
the whole video, columns are the four sampled heads. Each cell is that chunk's
query x key attention at its last denoising step, so reading a sheet top to
bottom shows how a layer's attention changes as the video grows -- and reading
across sheets of the same config shows how it changes with depth.

The key axis grows with the chunk index for a full-context model (a late chunk
sees every earlier one), so cells in a column are not the same width; that is
the point rather than an artefact.

    python plot.py [--configs self_forcing/720p_20s] [--layers 0,29] [--out .]

Writes sheets/<model>_<res>_<dur>s_layer<L>.png and summary.json.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from geometry import GEOMETRY, chunk_ids  # noqa: E402
from paths import results_dir  # noqa: E402

ROOT = results_dir("attention_layers")
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
    dumps = sorted(run.glob("qk_chunk_*_step_*.npz"))
    if not dumps or not (run / "meta.json").exists():
        return None
    # A realtime model dumps the union of every duration's chunk percentiles
    # (one server serves them all), so keep only this duration's own set.
    wanted = set(chunk_ids(model, duration))
    chunks = {}
    for dump in dumps:
        chunk = int(dump.stem.split("_")[2])
        if chunk not in wanted:
            continue
        data = np.load(dump)
        chunks[chunk] = {
            "scores": data["scores"],
            "layer_ids": [int(i) for i in data["layer_ids"]],
        }
    if not chunks:
        return None
    return {"chunks": chunks, "meta": json.loads((run / "meta.json").read_text())}


def key_share(scores: np.ndarray) -> float:
    """Median share of keys a query row needs for 90% of its mass."""
    ranked = np.sort(scores.astype(np.float32), axis=-1)[..., ::-1]
    total = ranked.sum(axis=-1, keepdims=True)
    cumulative = np.cumsum(ranked, axis=-1) / np.maximum(total, 1e-12)
    needed = (cumulative < 0.9).sum(axis=-1) + 1
    return float(np.median(needed / scores.shape[-1]))


def plot_sheet(
    entry: dict, model: str, res: str, duration: int, layer: int, out_dir: pathlib.Path
) -> None:
    head_ids = [
        int(h) for h in (entry["meta"].get("qk_head_ids") or {}).get(str(layer), [])
    ]
    chunks = sorted(entry["chunks"])
    panels = []
    for chunk in chunks:
        payload = entry["chunks"][chunk]
        if layer not in payload["layer_ids"]:
            return
        panels.append(payload["scores"][payload["layer_ids"].index(layer)])
    if not panels or not head_ids:
        return

    positive = np.concatenate(
        [p[p > 0].ravel()[:200000].astype(np.float32) for p in panels]
    )
    vmax = float(max(p.max() for p in panels))
    if not positive.size or not np.isfinite(vmax) or vmax <= 0:
        return
    vmin = min(max(float(np.percentile(positive, 1)), 1e-8), vmax / 10.0)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    colormap = matplotlib.colormaps[CMAP].with_extremes(
        bad=matplotlib.colormaps[CMAP](0.0), under=matplotlib.colormaps[CMAP](0.0)
    )

    rows, columns = len(chunks), len(head_ids)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns + 1.1, 2.2 * rows + 0.8),
        dpi=165,
        squeeze=False,
        gridspec_kw={"hspace": 0.2, "wspace": 0.1},
    )
    image = None
    for row, chunk in enumerate(chunks):
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
                ax.set_ylabel(
                    f"chunk {chunk}\n({panels[row].shape[-1]} keys)", fontsize=8
                )
    bar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    bar.set_label("attention probability (log)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{MODEL_LABELS[model]} · {res} · {duration}s · layer {layer}"
        f"/{GEOMETRY[model][0] - 1} — attention over the video "
        "(rows = chunks, last denoising step; x = visible keys, y = the chunk's queries)",
        fontsize=11,
    )
    path = out_dir / f"{model}_{res}_{duration}s_layer{layer:02d}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_depth_trend(data: dict, res: str, duration: int, out_dir: pathlib.Path):
    """Key concentration by depth and chunk -- the numeric side of the sheets."""
    models = [m for m in MODEL_ORDER if (m, res, duration) in data]
    if not models:
        return
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.2 * len(models), 3.2), dpi=170, squeeze=False
    )
    for column, model in enumerate(models):
        ax = axes[0][column]
        entry = data[(model, res, duration)]
        chunks = sorted(entry["chunks"])
        layers = entry["chunks"][chunks[0]]["layer_ids"]
        for layer in layers:
            shares = []
            for chunk in chunks:
                payload = entry["chunks"][chunk]
                index = payload["layer_ids"].index(layer)
                shares.append(100 * key_share(payload["scores"][index]))
            ax.plot(
                chunks,
                shares,
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=f"layer {layer}",
            )
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        ax.set_xlabel("chunk index", fontsize=8)
        if column == 0:
            ax.set_ylabel("% of visible keys for 90% mass", fontsize=8)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both", linewidth=0.4)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle(
        f"Attention concentration by depth and position · {res} · {duration}s "
        "(median over query rows; lower = sparser)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out_dir / f"concentration_{res}_{duration}s.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()
    sheets = args.out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    wanted = set(args.configs.split(",")) if args.configs else None
    data, summary = {}, []
    for model in MODEL_ORDER:
        for res in RESOLUTIONS:
            for duration in DURATIONS:
                if wanted is not None and f"{model}/{res}_{duration}s" not in wanted:
                    continue
                entry = load(model, res, duration)
                if entry is None:
                    continue
                data[(model, res, duration)] = entry
                chunks = sorted(entry["chunks"])
                dumped = entry["chunks"][chunks[0]]["layer_ids"]
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
                            "chunks": chunks,
                            "visible_keys": [
                                int(entry["chunks"][c]["scores"].shape[-1])
                                for c in chunks
                            ],
                            "key_share_for_90pct": [
                                key_share(
                                    entry["chunks"][c]["scores"][
                                        entry["chunks"][c]["layer_ids"].index(layer)
                                    ]
                                )
                                for c in chunks
                            ],
                        }
                    )
    for res in RESOLUTIONS:
        plot_depth_trend(data, res, 20, args.out)
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
