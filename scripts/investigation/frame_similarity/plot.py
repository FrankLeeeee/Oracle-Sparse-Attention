# SPDX-License-Identifier: Apache-2.0
"""Intra-chunk frame-similarity figures from the frame_similarity sweep.

Reads runs/<model>/<res>_<dur>s/frame_similarity.npz (written by the probe
behind ``SGLANG_DIFFUSION_FRAME_SIMILARITY_DIR``) and renders two things:

* ``grid``  — one figure per (config, denoising step): a subplot per frame
  pair, each a chunk-index x layer-index heat map of the pair's cosine
  similarity. This is the per-video view.
* ``summary`` — one figure per resolution: the similarity profile across the
  network depth, one panel per model, one line per denoising step. This is the
  cross-model view.

    python plot.py [--kind grid,summary] [--configs self_forcing/720p_20s] [--out .]

Writes grids/<model>_<res>_<dur>s_step<k>.png and layer_profile_<res>.png.
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("frame_similarity")

MODEL_LABELS = {
    "self_forcing": "Self-Forcing 1.3B",
    "rolling_forcing": "Rolling Forcing 1.3B",
    "longlive2": "LongLive-2.0 5B",
    "lingbot_world_v2": "LingBot-World v2 14B",
}
MODEL_ORDER = list(MODEL_LABELS)
RESOLUTIONS = ["480p", "720p"]
DURATIONS = [5, 10, 20]
CMAP = "viridis"


def load(model: str, res: str, duration: int) -> dict | None:
    run = ROOT / "runs" / model / f"{res}_{duration}s"
    payload = run / "frame_similarity.npz"
    if not payload.exists():
        return None
    data = np.load(payload)
    return {
        "sim": data["sim"],  # [chunks, steps, layers, pairs]
        "pairs": [tuple(p) for p in data["pairs"]],
        "meta": json.loads((run / "meta.json").read_text()),
    }


def _grid_shape(count: int) -> tuple[int, int]:
    """Rows x columns for ``count`` pair subplots, kept close to 2:1."""
    if count <= 4:
        return 1, count
    columns = int(np.ceil(np.sqrt(count * 2)))
    return int(np.ceil(count / columns)), columns


def plot_grid(entry: dict, model: str, res: str, duration: int, out_dir: pathlib.Path):
    sim = entry["sim"]
    pairs = entry["pairs"]
    num_chunks, num_steps, num_layers, num_pairs = sim.shape
    # One colour scale for the whole config so the steps are comparable.
    vmin, vmax = float(np.nanmin(sim)), float(np.nanmax(sim))
    rows, columns = _grid_shape(num_pairs)

    for step in range(num_steps):
        if np.isnan(sim[:, step]).all():
            continue
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(2.6 * columns + 1.2, 2.5 * rows + 0.9),
            dpi=170,
            squeeze=False,
        )
        image = None
        for index in range(rows * columns):
            ax = axes[index // columns][index % columns]
            if index >= num_pairs:
                ax.set_axis_off()
                continue
            # [layers, chunks] so that y is depth and x is time
            panel = sim[:, step, :, index].T
            image = ax.imshow(
                panel,
                aspect="auto",
                origin="lower",
                cmap=CMAP,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            first, second = pairs[index]
            ax.set_title(f"frame {first} vs {second}", fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.tick_params(labelsize=7)
            if index % columns == 0:
                ax.set_ylabel("layer index", fontsize=8)
            if index // columns == rows - 1:
                ax.set_xlabel("chunk index", fontsize=8)
        if image is not None:
            bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.015)
            bar.set_label("cosine similarity", fontsize=8)
            bar.ax.tick_params(labelsize=7)
        fig.suptitle(
            f"{MODEL_LABELS[model]} · {res} · {duration}s · denoising step "
            f"{step + 1}/{num_steps} — intra-chunk frame similarity "
            f"(layer {num_layers - 1} = final output)",
            fontsize=11,
        )
        path = out_dir / f"{model}_{res}_{duration}s_step{step}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


def plot_layer_profile(data: dict, res: str, out_dir: pathlib.Path) -> None:
    models = [m for m in MODEL_ORDER if (m, res, 20) in data]
    if not models:
        return
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.3 * len(models), 3.4), dpi=170, squeeze=False
    )
    for column, model in enumerate(models):
        ax = axes[0][column]
        sim = data[(model, res, 20)]["sim"]
        num_steps = sim.shape[1]
        for step in range(num_steps):
            # average over chunks and pairs: the depth profile of the chunk
            profile = np.nanmean(sim[:, step], axis=(0, 2))
            ax.plot(
                np.arange(profile.shape[0]),
                profile,
                linewidth=1.3,
                label=f"step {step + 1}",
            )
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        ax.set_xlabel("layer index", fontsize=8)
        if column == 0:
            ax.set_ylabel("mean cosine similarity", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"Intra-chunk frame similarity across network depth · {res} · 20s "
        "(mean over chunks and frame pairs)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / f"layer_profile_{res}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def summarize(data: dict) -> list[dict]:
    rows = []
    for (model, res, duration), entry in sorted(data.items()):
        sim = entry["sim"]
        last = sim[:, -1]  # [chunks, layers, pairs] at the final denoising step
        rows.append(
            {
                "model": model,
                "resolution": res,
                "duration_s": duration,
                "num_chunks": int(sim.shape[0]),
                "num_steps": int(sim.shape[1]),
                "num_layers": int(sim.shape[2]),
                "num_pairs": int(sim.shape[3]),
                "input_layer_mean": float(np.nanmean(last[:, 0])),
                "first_block_mean": float(np.nanmean(last[:, 1])),
                "final_output_mean": float(np.nanmean(last[:, -1])),
                "body_mean": float(np.nanmean(last[:, 1:-1])),
                "first_step_body_mean": float(np.nanmean(sim[:, 0, 1:-1])),
                "min": float(np.nanmin(sim)),
                "max": float(np.nanmax(sim)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="grid,summary")
    parser.add_argument(
        "--configs",
        default=None,
        help="limit the grids, e.g. self_forcing/720p_20s,longlive2/720p_20s",
    )
    parser.add_argument("--out", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    kinds = args.kind.split(",")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for model in MODEL_ORDER:
        for res in RESOLUTIONS:
            for duration in DURATIONS:
                entry = load(model, res, duration)
                if entry is not None:
                    data[(model, res, duration)] = entry

    if "grid" in kinds:
        wanted = set(args.configs.split(",")) if args.configs else None
        grids = out_dir / "grids"
        grids.mkdir(exist_ok=True)
        for (model, res, duration), entry in sorted(data.items()):
            if wanted is not None and f"{model}/{res}_{duration}s" not in wanted:
                continue
            plot_grid(entry, model, res, duration, grids)

    if "summary" in kinds:
        for res in RESOLUTIONS:
            plot_layer_profile(data, res, out_dir)

    path = out_dir / "summary.json"
    path.write_text(json.dumps(summarize(data), indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
