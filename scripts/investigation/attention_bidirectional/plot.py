# SPDX-License-Identifier: Apache-2.0
"""Task 4 figures — the bidirectional model's attention over its whole video.

One contact sheet per (config, layer): rows are the sampled denoising steps,
columns are the four sampled heads. Each cell is the full token x token
attention of Wan2.1-T2V-1.3B — no chunks, no causal mask, no KV cache — so the
diagonal band and any frame-periodic structure are the model's own, not a
consequence of a block-causal schedule.

    python plot.py [--configs 480p_20s] [--layers 0,29] [--out .]

Writes sheets/<res>_<dur>s_layer<L>.png, concentration.png and summary.json.
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
from geometry import GEOMETRY, coverage_share  # noqa: E402
from paths import results_dir  # noqa: E402

ROOT = results_dir("attention_bidirectional")
MODEL = "wan2_1_t2v_1_3b"
LABEL = "Wan2.1-T2V-1.3B (bidirectional)"
RESOLUTIONS = ["480p", "720p"]
DURATIONS = [5, 10, 20]
CMAP = "magma"


def load(res: str, duration: int) -> dict | None:
    run = ROOT / "runs" / MODEL / f"{res}_{duration}s"
    dumps = sorted(run.glob("qk_chunk_000_step_*.npz"))
    if not dumps or not (run / "meta.json").exists():
        return None
    steps = {}
    for dump in dumps:
        data = np.load(dump)
        steps[int(dump.stem.split("_")[-1])] = {
            "scores": data["scores"],
            "coverage": data["coverage"],  # full un-strided key axis
            "layer_ids": [int(i) for i in data["layer_ids"]],
        }
    return {"steps": steps, "meta": json.loads((run / "meta.json").read_text())}


def key_share(payload: dict, layer: int, stride: int) -> float:
    """Median share of all tokens a query row needs for 90% of its mass.

    From the probe's ``coverage``, measured on the full un-strided key axis;
    the strided ``scores`` would badly misjudge the narrow diagonal bands that
    dominate this model's shallowest layer.
    """
    index = payload["layer_ids"].index(layer)
    return coverage_share(
        payload["coverage"][index], payload["scores"].shape[-1] * stride
    )


def plot_sheet(entry: dict, res: str, duration: int, layer: int, out_dir: pathlib.Path):
    head_ids = [
        int(h) for h in (entry["meta"].get("qk_head_ids") or {}).get(str(layer), [])
    ]
    steps = sorted(entry["steps"])
    panels = []
    for step in steps:
        payload = entry["steps"][step]
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
    total_steps = entry["meta"].get("num_inference_steps", 50)
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
                ax.set_ylabel(f"step {step}/{total_steps - 1}", fontsize=9)
    bar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    bar.set_label("attention probability (log)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{LABEL} · {res} · {duration}s · layer {layer}/{GEOMETRY[MODEL][0] - 1} — "
        "full attention over the whole video (both axes are every token)",
        fontsize=11,
    )
    path = out_dir / f"{res}_{duration}s_layer{layer:02d}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_concentration(data: dict, out_dir: pathlib.Path) -> None:
    configs = sorted(data)
    if not configs:
        return
    fig, axes = plt.subplots(
        1, len(configs), figsize=(2.7 * len(configs), 3.2), dpi=170, squeeze=False
    )
    for column, key in enumerate(configs):
        ax = axes[0][column]
        entry = data[key]
        stride = entry["meta"]["qk_key_stride"]
        steps = sorted(entry["steps"])
        layers = entry["steps"][steps[0]]["layer_ids"]
        for layer in layers:
            shares = []
            for step in steps:
                shares.append(100 * key_share(entry["steps"][step], layer, stride))
            ax.plot(
                steps,
                shares,
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=f"layer {layer}",
            )
        ax.set_title(f"{key[0]} · {key[1]}s", fontsize=9)
        ax.set_xlabel("denoising step", fontsize=8)
        if column == 0:
            ax.set_ylabel("% of tokens for 90% mass", fontsize=8)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both", linewidth=0.4)
        ax.tick_params(labelsize=7)
        if column == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.suptitle(
        f"{LABEL} — attention concentration over the denoising schedule "
        "(median over query rows; lower = sparser)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out_dir / "concentration.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None, help="e.g. 480p_20s,720p_20s")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()
    sheets = args.out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    wanted = set(args.configs.split(",")) if args.configs else None
    data, summary = {}, []
    for res in RESOLUTIONS:
        for duration in DURATIONS:
            if wanted is not None and f"{res}_{duration}s" not in wanted:
                continue
            entry = load(res, duration)
            if entry is None:
                continue
            data[(res, duration)] = entry
            steps = sorted(entry["steps"])
            dumped = entry["steps"][steps[0]]["layer_ids"]
            layers = [int(x) for x in args.layers.split(",")] if args.layers else dumped
            for layer in layers:
                if layer not in dumped:
                    continue
                if not args.no_sheets:
                    plot_sheet(entry, res, duration, layer, sheets)
                summary.append(
                    {
                        "resolution": res,
                        "duration_s": duration,
                        "layer": layer,
                        "steps": steps,
                        "sampled_tokens": int(
                            entry["steps"][steps[0]]["scores"].shape[-1]
                        ),
                        "stride": entry["meta"].get("qk_key_stride"),
                        "key_share_for_90pct": [
                            key_share(
                                entry["steps"][s],
                                layer,
                                entry["meta"]["qk_key_stride"],
                            )
                            for s in steps
                        ],
                    }
                )
    plot_concentration(data, args.out)
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
