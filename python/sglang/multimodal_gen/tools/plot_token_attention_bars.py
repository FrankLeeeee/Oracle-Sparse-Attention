# SPDX-License-Identifier: Apache-2.0
"""Per-token attention bars: one figure per (chunk, layer), one panel per head.

Reads ``token_scores.npz`` (written by the attention-map probe with
``SGLANG_DIFFUSION_ATTENTION_MAP_TOKEN_SCORES=true``) and renders, for a given
generated chunk and transformer layer, every key token's attention score as its
own bar — no binning, no aggregation. The highest-scoring ``--top-k`` bars per
head are drawn in a second colour so the sparse tail is visible against the
dense background.

Companion to ``plot_token_attention_maps``, which folds the same array back onto
the latent grid and averages heads: this view keeps the head axis and the global
token index, so per-head divergence and chunk-boundary structure read directly
off the x-axis, at the cost of any spatial reading.

    python -m sglang.multimodal_gen.tools.plot_token_attention_bars <run_dir> \\
        [--chunks 3,6] [--layers 0,15] [--top-k 2048] [--log-y]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BASE_COLOR = "#B9C6D6"  # the bulk of the distribution, recessive
TOP_COLOR = "#D55E00"  # the top-k bars
BOUNDARY_COLOR = "#7a7a7a"
QUERY_COLOR = "#0072B2"
GRID_COLOR = "#c9c9c9"


def parse_selection(raw: str | None, limit: int) -> list[int]:
    if raw is None:
        return list(range(limit))
    picked = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            picked.extend(range(int(lo), int(hi) + 1))
        elif part:
            picked.append(int(part))
    return [i for i in picked if 0 <= i < limit]


def frame_and_chunk_tokens(
    meta: dict, *, num_tokens: int, grid: tuple[int, int] | None
) -> tuple[int, int]:
    """Tokens per latent frame and per generated chunk.

    Causal runs record both directly. Full-attention runs record neither — the
    whole video is one chunk — and dumps written before the probe recorded the
    latent grid carry only ``num_token_per_frame``.
    """
    if grid is not None:
        frame_seqlen = grid[0] * grid[1]
    elif "num_token_per_frame" in meta:
        frame_seqlen = int(meta["num_token_per_frame"])
    elif "grid_height" in meta:
        frame_seqlen = int(meta["grid_height"]) * int(meta["grid_width"])
    else:
        raise SystemExit(
            f"cannot tell how many tokens a frame holds from {meta}; pass --grid HxW"
        )
    frames_per_block = int(meta.get("num_frames_per_block", 0))
    chunk_tokens = frames_per_block * frame_seqlen or num_tokens
    return frame_seqlen, chunk_tokens


def plot_layer(
    scores, *, chunk, layer, meta, frame_seqlen, chunk_tokens, top_k, log_y, out_path
):
    """``scores`` is ``[heads, tokens]`` for one (chunk, layer)."""
    heads, num_tokens = scores.shape
    visible = ~np.isnan(scores[0])
    last = int(np.where(visible)[0].max()) + 1 if visible.any() else num_tokens
    chunked = chunk_tokens < num_tokens

    fig, axes = plt.subplots(
        heads, 1, figsize=(19.0, 1.15 * heads + 1.4), sharex=True, squeeze=False
    )
    axes = [row[0] for row in axes]
    positions = np.arange(last)

    for head in range(heads):
        ax = axes[head]
        row = np.nan_to_num(scores[head, :last])
        cut = min(top_k, row.size)
        top_index = np.argpartition(row, -cut)[-cut:] if cut else np.array([], int)
        is_top = np.zeros(row.size, bool)
        is_top[top_index] = True

        # the bars are drawn from the axis floor, so fix the limits first
        if log_y:
            ax.set_yscale("log")
            positive = row[row > 0]
            floor = max(positive.min(), row.max() * 1e-6) if positive.size else 1e-12
            ax.set_ylim(floor, max(row.max() * 1.6, floor * 10))
        else:
            floor = 0.0
            ax.set_ylim(0, max(row.max() * 1.08, 1e-12))

        # One vertical bar per token, drawn as two LineCollections. `ax.bar`
        # would allocate a Rectangle patch per token (~400k per figure) and
        # takes minutes; vlines keeps every value its own mark and is ~100x
        # faster. Nothing is binned or merged.
        ax.vlines(positions[~is_top], floor, row[~is_top], color=BASE_COLOR, lw=0.5)
        ax.vlines(positions[is_top], floor, row[is_top], color=TOP_COLOR, lw=0.5)

        # A full-attention run is one chunk covering the whole sequence, so the
        # boundaries and the "this is the query chunk" band would span every
        # panel and say nothing; they only mean something when chunks exist.
        if chunked:
            for boundary in range(chunk_tokens, last, chunk_tokens):
                ax.axvline(boundary, color=BOUNDARY_COLOR, lw=0.5, alpha=0.5)
            ax.axvspan(
                chunk * chunk_tokens,
                min((chunk + 1) * chunk_tokens, last),
                color=QUERY_COLOR,
                alpha=0.07,
                lw=0,
            )
        ax.set_ylabel(f"h{head}", fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color=GRID_COLOR, lw=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        share = row[is_top].sum() / max(row.sum(), 1e-12)
        ax.text(
            0.997,
            0.86,
            f"top-{cut} = {share:.1%} of mass",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=TOP_COLOR,
        )

    axes[-1].set_xlabel(
        f"key token index (global latent token; {frame_seqlen} tokens per frame"
        + (f", {chunk_tokens} per chunk)" if chunked else ")"),
        fontsize=9,
    )
    axes[-1].set_xlim(0, last)
    fig.suptitle(
        f"{meta.get('model_tag', '')} — "
        + (f"chunk {chunk}, " if chunked else "")
        + f"layer {layer}: attention score of every key token, per head\n"
        + ("blue band = the query chunk itself; " if chunked else "")
        + f"orange = top-{top_k} tokens of that head"
        + ("  (log y)" if log_y else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--chunks", default=None, help="e.g. '3,6' or '0-2'")
    parser.add_argument("--layers", default=None, help="e.g. '0,15,29' or '0-5'")
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--grid",
        default=None,
        help="latent grid as HxW; only needed for dumps whose meta.json records "
        "neither the grid nor the tokens per frame",
    )
    args = parser.parse_args()

    meta = json.loads((args.run_dir / "meta.json").read_text())
    scores = np.load(args.run_dir / "token_scores.npz")["token_scores"]
    num_chunks, num_layers = scores.shape[0], scores.shape[1]
    chunks = parse_selection(args.chunks, num_chunks)
    layers = parse_selection(args.layers, num_layers)
    grid = None
    if args.grid is not None:
        height, width = args.grid.lower().split("x")
        grid = (int(height), int(width))
    frame_seqlen, chunk_tokens = frame_and_chunk_tokens(
        meta, num_tokens=scores.shape[3], grid=grid
    )

    out_dir = args.out_dir or args.run_dir / "token_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        for layer in layers:
            plot_layer(
                scores[chunk, layer],
                chunk=chunk,
                layer=layer,
                meta=meta,
                frame_seqlen=frame_seqlen,
                chunk_tokens=chunk_tokens,
                top_k=args.top_k,
                log_y=args.log_y,
                out_path=out_dir / f"chunk_{chunk:03d}_layer_{layer:02d}.png",
            )
    print(f"wrote {len(chunks) * len(layers)} figures to {out_dir}")


if __name__ == "__main__":
    main()
