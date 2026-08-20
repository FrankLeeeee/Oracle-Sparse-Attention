# SPDX-License-Identifier: Apache-2.0
"""Per-head query x key attention token maps from a probe run's QK dumps.

One PNG per (chunk, layer, head, step): the raw attention matrix of the
chunk's queries (y axis, global token index) against every visible key
(x axis, column index with global-token tick labels — robust to disjoint
sink/window segments), color = softmax probability (log scale). Latent
*frame* boundaries are drawn in green on both axes, *chunk* boundaries in
white, and the query's own chunk is bracketed in cyan. A right-hand panel
annotates each query row with
the minimum number of top-ranked key tokens whose probabilities sum past
the coverage threshold (0.9 by default) — computed by the probe on the
full, unstrided key axis.

Requires a run recorded with SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS
(and optionally ..._QK_STEPS / ..._QK_LAYERS, see envs.py): the probe
writes one ``qk_chunk_<c>_step_<s>.npz`` per (chunk, denoising step) with
``scores [layers, heads, queries, keys]`` and ``coverage [layers, heads,
queries]``. For Rolling Forcing, chunk *c* means the window whose oldest
block is chunk *c*: its query axis spans the five staggered-noise blocks
of that window, and only step 0 exists for post-ramp-up chunks.

    python -m sglang.multimodal_gen.tools.plot_attention_token_maps <run_dir> \
        [--chunks 8,26] [--steps 0,1,3] [--layers 15] [--heads 0-4] [--out-dir ...]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_MAX_ROW_LABELS = 24


def _parse_ids(spec: str | None, upper: int) -> list[int]:
    if not spec:
        return list(range(upper))
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        elif part:
            ids.append(int(part))
    return [i for i in ids if 0 <= i < upper]


def plot_head(
    *,
    scores: np.ndarray,  # [queries, keys] float
    coverage: np.ndarray,  # [queries] int — full-key top-k counts per row
    coverage_threshold: float,
    query_positions: np.ndarray,
    key_positions: np.ndarray,
    frame_seqlen: int,
    chunk_tokens: int,
    title: str,
    out_path: pathlib.Path,
) -> None:
    values = scores.astype(np.float64)
    positive = values[values > 0]
    vmin = max(positive.min() if positive.size else 1e-8, 1e-8)
    vmax = max(values.max(), vmin * 10)

    fig, (ax, ax_cov) = plt.subplots(
        1,
        2,
        figsize=(13.6, 4.2),
        dpi=170,
        sharey=True,
        gridspec_kw={"width_ratios": (10, 1.6), "wspace": 0.02},
    )
    # The x axis is *column index* into the visible (strided) key set, with
    # tick labels giving the global token index. Position-space extents would
    # silently stretch across the gap between disjoint visible segments
    # (attention-sink caches), putting boundary lines in the wrong place.
    num_keys = len(key_positions)
    image = ax.imshow(
        np.clip(values, vmin, None),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
        extent=(
            0.0,
            float(num_keys),
            float(query_positions[-1]),
            float(query_positions[0]),
        ),
    )
    # Frame boundaries in green, chunk boundaries in white, the query's own
    # chunk bracketed in cyan — all located where the *global* frame/chunk id
    # of adjacent kept columns changes, so they stay correct across segment
    # gaps and key striding.
    key_frames = key_positions // frame_seqlen
    key_chunks = key_positions // chunk_tokens
    chunk_columns = set((np.nonzero(np.diff(key_chunks))[0] + 1).tolist())
    for column in np.nonzero(np.diff(key_frames))[0] + 1:
        if int(column) in chunk_columns:
            continue
        ax.axvline(column, color="lime", linewidth=0.5, alpha=0.6)
    for column in sorted(chunk_columns):
        ax.axvline(column, color="w", linewidth=1.0, alpha=0.8)
    own = np.nonzero(
        (key_positions >= query_positions[0]) & (key_positions <= query_positions[-1])
    )[0]
    if own.size:
        ax.axvline(float(own[0]), color="c", linewidth=0.9, alpha=0.9)
        ax.axvline(float(own[-1]) + 1, color="c", linewidth=0.9, alpha=0.9)
    tick_columns = np.linspace(0, num_keys - 1, 8).astype(int)
    ax.set_xticks(tick_columns.astype(np.float64) + 0.5)
    ax.set_xticklabels([str(int(key_positions[c])) for c in tick_columns], fontsize=7)
    ax.set_xlim(0.0, float(num_keys))
    ax.set_xlabel(
        "key token (labels = global index; green = frame, white = chunk, "
        "cyan = own chunk)"
    )
    # Frame boundaries along the query axis (contiguous, position space).
    first_row_boundary = (int(query_positions[0]) // frame_seqlen + 1) * frame_seqlen
    for boundary in range(
        first_row_boundary, int(query_positions[-1]) + 1, frame_seqlen
    ):
        ax.axhline(boundary, color="lime", linewidth=0.5, alpha=0.6)
    ax.set_ylabel("query token index")
    ax.set_title(title, fontsize=10, loc="left")

    # Right panel: per-row top-k count on a log axis, numeric labels on a
    # subsample of rows (every row would overlap into an unreadable smear).
    counts = coverage.astype(np.float64)
    ax_cov.plot(
        counts, query_positions.astype(np.float64), color="#2a78d6", linewidth=0.7
    )
    ax_cov.set_xscale("log")
    ax_cov.set_xlim(1, max(counts.max() * 4.0, 10.0))
    label_stride = max(1, len(counts) // _MAX_ROW_LABELS)
    for row in range(0, len(counts), label_stride):
        ax_cov.text(
            counts[row] * 1.25,
            query_positions[row],
            f"{int(coverage[row])}",
            fontsize=5.5,
            va="center",
            color="#0b0b0b",
        )
    ax_cov.set_xlabel(f"top-k for >{coverage_threshold:g}")
    ax_cov.tick_params(labelsize=7)
    ax_cov.grid(True, which="both", axis="x", color="#e5e4e0", linewidth=0.5)
    ax_cov.set_axisbelow(True)

    fig.colorbar(image, ax=ax_cov, label="attention probability (log)", pad=0.08)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "--chunks", default=None, help="e.g. 8,26 (default: all dumped)"
    )
    parser.add_argument(
        "--steps", default=None, help="e.g. 0,1,3 (default: all dumped)"
    )
    parser.add_argument(
        "--layers", default=None, help="e.g. 0,15,29 (default: all dumped)"
    )
    parser.add_argument("--heads", default=None, help="e.g. 0-3,7 (default: all)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.run_dir / "meta.json").read_text())
    frame_seqlen = meta["grid_height"] * meta["grid_width"]
    chunk_tokens = frame_seqlen * meta["num_frames_per_block"]
    coverage_threshold = meta.get("qk_coverage_threshold", 0.9)
    # When the probe dumped only a few heads, the dump's head axis is dense but
    # the real head ids are not; label figures with the real ones.
    dumped_head_ids = {
        int(layer): list(ids) for layer, ids in (meta.get("qk_head_ids") or {}).items()
    }

    dumps = sorted(args.run_dir.glob("qk_chunk_*_step_*.npz"))
    if not dumps:
        raise SystemExit(
            f"no qk_chunk_*_step_*.npz in {args.run_dir} — record with "
            "SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS set"
        )
    wanted_chunks = {int(c) for c in args.chunks.split(",")} if args.chunks else None
    wanted_steps = {int(s) for s in args.steps.split(",")} if args.steps else None
    out_dir = args.out_dir or (args.run_dir / "token_map_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for dump in dumps:
        parts = dump.stem.split("_")  # qk chunk <c> step <s>
        chunk, step = int(parts[2]), int(parts[4])
        if wanted_chunks is not None and chunk not in wanted_chunks:
            continue
        if wanted_steps is not None and step not in wanted_steps:
            continue
        data = np.load(dump)
        scores = data["scores"]  # [layers, heads, queries, keys]
        coverage = data["coverage"]  # [layers, heads, queries]
        layer_ids = data["layer_ids"]
        query_positions = data["query_positions"]
        key_positions = data["key_positions"]
        layer_lookup = {int(l): i for i, l in enumerate(layer_ids)}
        for layer in _parse_ids(args.layers, int(layer_ids.max()) + 1):
            if layer not in layer_lookup:
                continue
            per_layer = scores[layer_lookup[layer]]
            head_ids = dumped_head_ids.get(layer) or list(range(per_layer.shape[0]))
            # --heads selects real head ids; map them onto the dumped axis
            selected = _parse_ids(args.heads, max(head_ids) + 1)
            for column, head in enumerate(head_ids):
                if args.heads is not None and head not in selected:
                    continue
                plot_head(
                    scores=per_layer[column],
                    coverage=coverage[layer_lookup[layer], column],
                    coverage_threshold=coverage_threshold,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    frame_seqlen=frame_seqlen,
                    chunk_tokens=chunk_tokens,
                    title=(
                        f"chunk {chunk} · step {step} · layer {layer} · "
                        f"head {head} — query x key attention "
                        f"({meta['model_tag']})"
                    ),
                    out_path=out_dir
                    / (
                        f"chunk_{chunk:03d}_step_{step}_layer_{layer:02d}"
                        f"_head_{head:02d}.png"
                    ),
                )
                written += 1
    print(f"wrote {written} figures to {out_dir}")


if __name__ == "__main__":
    main()
