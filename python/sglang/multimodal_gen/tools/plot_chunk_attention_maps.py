# SPDX-License-Identifier: Apache-2.0
"""Render the per-chunk attention dumps of block-causal video DiTs.

Reads a run directory produced by the attention-map probe (see
``runtime/utils/attention_map_probe.py``; enable it with
``SGLANG_DIFFUSION_ATTENTION_MAP_DIR``) and writes one figure per generated
chunk showing, per transformer layer, how the chunk's attention mass is spread
over all chunks of the video — plus a ``summary.png`` chunk-to-chunk map.

Example:

    SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/attn sglang generate \\
        --model-path /data/models/RollingForcing-Wan2.1-T2V-1.3B-Diffusers \\
        --prompt "..." --num-frames 81 --save-output

    python -m sglang.multimodal_gen.tools.plot_chunk_attention_maps \\
        /data/attn/RollingForcingWanTransformer3DModel-20260721-101500
"""

import argparse
import json
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

# Okabe-Ito, the canonical colorblind-safe qualitative set, in fixed order.
STEP_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
HEATMAP_CMAP = "magma"
MASKED_COLOR = "#e8e8e8"
GRID_COLOR = "#c9c9c9"
DENOISE_PASS = "denoise"


def to_chunk_scores(scores: np.ndarray, *, frames_per_block: int) -> np.ndarray:
    """``[steps, layers, heads, frames]`` → ``[steps, layers, chunks]``.

    Heads are averaged (each head's row is already a distribution) and the
    frames of a block are summed, so a row still sums to 1 over chunks.
    """
    steps, layers, heads, frames = scores.shape
    padded = frames_per_block * int(np.ceil(frames / frames_per_block))
    if padded != frames:
        scores = np.concatenate(
            [scores, np.full((steps, layers, heads, padded - frames), np.nan)], axis=3
        )
    by_chunk = scores.reshape(steps, layers, heads, -1, frames_per_block)
    summed = np.nansum(by_chunk, axis=4)
    # a block whose frames were all invisible stays NaN, not 0 — the two mean
    # different things (never co-visible vs. evicted from the cache)
    summed[np.isnan(by_chunk).all(axis=4)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(summed, axis=2)


def load_run(
    run_dir: pathlib.Path, pass_kind: str
) -> tuple[dict, dict[int, np.ndarray]]:
    """Load ``meta.json`` and ``{chunk index: [steps, layers, chunks]}``."""
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    frames_per_block = meta.get("num_frames_per_block", 1)
    chunks: dict[int, np.ndarray] = {}
    for path in sorted(run_dir.glob("chunk_*.npz")):
        with np.load(path) as data:
            if pass_kind not in data:
                continue
            scores = data[pass_kind]
            if scores.ndim == 4:
                scores = to_chunk_scores(scores, frames_per_block=frames_per_block)
            chunks[int(path.stem.split("_")[1])] = scores
    if not chunks:
        raise SystemExit(f"no '{pass_kind}' attention dumps found in {run_dir}")
    return meta, chunks


def _nanmean(values: np.ndarray, axis) -> np.ndarray:
    """``np.nanmean`` without the all-NaN-slice warning (chunks not yet seen)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _norm(values: np.ndarray, color_scale: str):
    finite = values[np.isfinite(values) & (values > 0)]
    vmax = float(finite.max()) if finite.size else 1.0
    if color_scale == "log":
        vmin = max(float(finite.min()) if finite.size else 1e-6, vmax * 1e-4)
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=0.0, vmax=vmax)


def _for_display(values: np.ndarray, norm) -> np.ndarray:
    """Lift exact zeros onto the log floor so they read as "no attention".

    A chunk that has been evicted from the KV cache scores exactly 0, which a
    log scale would mask and paint like the "not part of this video" cells; the
    floor keeps it dark instead. ``NaN`` stays ``NaN``.
    """
    if not isinstance(norm, LogNorm):
        return values
    return np.where(np.isnan(values) | (values > norm.vmin), values, norm.vmin)


def _frame_range(chunk_index: int, meta: dict) -> str:
    block = meta.get("num_frames_per_block")
    if not block:
        return ""
    return f" (latent frames {chunk_index * block}-{(chunk_index + 1) * block - 1})"


def plot_chunk(
    chunk_index: int,
    scores: np.ndarray,
    *,
    meta: dict,
    out_path: pathlib.Path,
    color_scale: str,
    dpi: int,
) -> None:
    """One figure: layer x chunk heatmap, plus per-step layer-mean curves."""
    num_steps, num_layers, num_chunks = scores.shape
    per_layer = _nanmean(scores, axis=0)  # [layers, chunks]
    per_step = _nanmean(scores, axis=1)  # [steps, chunks]

    figure, (heat_ax, line_ax) = plt.subplots(
        2,
        1,
        figsize=(max(6.0, 0.42 * num_chunks + 3.2), 6.6),
        height_ratios=(3.0, 1.25),
        sharex=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap(HEATMAP_CMAP).with_extremes(bad=MASKED_COLOR)
    norm = _norm(per_layer, color_scale)
    image = heat_ax.imshow(
        np.ma.masked_invalid(_for_display(per_layer, norm)),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(-0.5, num_chunks - 0.5, -0.5, num_layers - 0.5),
    )
    heat_ax.add_patch(
        Rectangle(
            (chunk_index - 0.5, -0.5),
            1.0,
            num_layers,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
            linestyle="--",
        )
    )
    heat_ax.set_ylabel("transformer layer")
    heat_ax.set_title(
        f"Chunk {chunk_index} — attention over all chunks"
        f"{_frame_range(chunk_index, meta)}\n"
        f"{meta.get('model_tag', '')} · mean of {num_steps} steps · "
        "dashed = this chunk",
        fontsize=9.5,
    )
    colorbar = figure.colorbar(
        image, ax=heat_ax, pad=0.01, extend="min" if color_scale == "log" else "neither"
    )
    colorbar.set_label(
        "attention mass" + (" (log scale)" if color_scale == "log" else ""),
        fontsize=9,
    )

    for step in range(num_steps):
        line_ax.plot(
            np.arange(num_chunks),
            per_step[step],
            color=STEP_COLORS[step % len(STEP_COLORS)],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"step {step}",
        )
    if color_scale == "log":
        line_ax.set_yscale("log")
    line_ax.set_xlabel("target chunk")
    line_ax.set_ylabel("mass\n(layer mean)", fontsize=9)
    line_ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
    line_ax.set_axisbelow(True)
    for spine in ("top", "right"):
        line_ax.spines[spine].set_visible(False)
    line_ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    line_ax.set_xlim(-0.5, num_chunks - 0.5)
    line_ax.set_xticks(np.arange(num_chunks))
    line_ax.get_xticklabels()[chunk_index].set_fontweight("bold")

    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def plot_summary(
    chunks: dict[int, np.ndarray],
    *,
    meta: dict,
    out_path: pathlib.Path,
    color_scale: str,
    dpi: int,
) -> None:
    """Chunk-to-chunk map: every chunk's layer- and step-averaged attention."""
    num_chunks = max(scores.shape[2] for scores in chunks.values())
    matrix = np.full((num_chunks, num_chunks), np.nan, dtype=np.float32)
    for chunk_index, scores in chunks.items():
        row = _nanmean(scores, axis=(0, 1))
        matrix[chunk_index, : row.shape[0]] = row

    figure, axes = plt.subplots(
        figsize=(max(5.0, 0.4 * num_chunks + 2.6),) * 2, constrained_layout=True
    )
    cmap = plt.get_cmap(HEATMAP_CMAP).with_extremes(bad=MASKED_COLOR)
    norm = _norm(matrix, color_scale)
    image = axes.imshow(
        np.ma.masked_invalid(_for_display(matrix, norm)),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    axes.set_xlabel("target chunk")
    axes.set_ylabel("generating chunk")
    axes.set_title(
        f"Chunk-to-chunk attention · {meta.get('model_tag', '')}\n"
        "mean over layers and denoising steps",
        fontsize=10,
    )
    colorbar = figure.colorbar(
        image, ax=axes, pad=0.01, extend="min" if color_scale == "log" else "neither"
    )
    colorbar.set_label(
        "attention mass" + (" (log scale)" if color_scale == "log" else ""),
        fontsize=9,
    )
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path, help="probe output directory")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="where to write the figures (default: <run_dir>/plots)",
    )
    parser.add_argument(
        "--pass-kind",
        default=DENOISE_PASS,
        help="'denoise' (default) or 'cache_update'",
    )
    parser.add_argument("--color-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    meta, chunks = load_run(args.run_dir, args.pass_kind)
    out_dir = args.out_dir or args.run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, scores in sorted(chunks.items()):
        out_path = out_dir / f"chunk_{chunk_index:03d}.png"
        plot_chunk(
            chunk_index,
            scores,
            meta=meta,
            out_path=out_path,
            color_scale=args.color_scale,
            dpi=args.dpi,
        )
    plot_summary(
        chunks,
        meta=meta,
        out_path=out_dir / "summary.png",
        color_scale=args.color_scale,
        dpi=args.dpi,
    )
    print(f"wrote {len(chunks)} chunk figures + summary.png to {out_dir}")


if __name__ == "__main__":
    main()
