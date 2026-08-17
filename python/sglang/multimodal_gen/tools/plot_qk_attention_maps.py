# SPDX-License-Identifier: Apache-2.0
"""Per-head query x key attention matrices from a probe run's QK dumps.

One PNG per (chunk, layer, head): the raw attention matrix of the chunk's
queries (y axis, global token index) against every visible key (x axis,
global token index), color = softmax probability (log scale). Chunk
boundaries on the key axis are drawn as thin lines and the query's own chunk
is bracketed, so sink columns, recency bands and register tokens are readable
directly off the plot.

Requires a run recorded with SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS
(see envs.py): the probe then writes ``qk_chunk_<c>.npz`` with
``scores [layers, heads, queries, keys]`` (strided both ways, softmax over
the full key axis, first denoising step only).

    python -m sglang.multimodal_gen.tools.plot_qk_attention_maps <run_dir> \
        [--chunks 8,26] [--layers 0,10,20,29] [--heads 0-11] [--out-dir ...]
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


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
    query_positions: np.ndarray,
    key_positions: np.ndarray,
    chunk_tokens: int,
    title: str,
    out_path: pathlib.Path,
) -> None:
    values = scores.astype(np.float64)
    positive = values[values > 0]
    vmin = max(positive.min() if positive.size else 1e-8, 1e-8)
    vmax = max(values.max(), vmin * 10)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=170)
    image = ax.imshow(
        np.clip(values, vmin, None),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(key_positions[0]),
            float(key_positions[-1]),
            float(query_positions[-1]),
            float(query_positions[0]),
        ),
    )
    # Chunk boundaries on the key axis; the query's own chunk bracketed.
    first_boundary = (int(key_positions[0]) // chunk_tokens + 1) * chunk_tokens
    for boundary in range(first_boundary, int(key_positions[-1]) + 1, chunk_tokens):
        ax.axvline(boundary, color="w", linewidth=0.25, alpha=0.35)
    ax.axvline(float(query_positions[0]), color="c", linewidth=0.9, alpha=0.9)
    ax.axvline(
        float(query_positions[-1]) + 1, color="c", linewidth=0.9, alpha=0.9
    )
    ax.set_xlabel("key token index (global; cyan = own chunk)")
    ax.set_ylabel("query token index")
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(image, ax=ax, label="attention probability (log)", pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--chunks", default=None, help="e.g. 8,26 (default: all dumped)")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--heads", default=None, help="e.g. 0-3,7 (default: all)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.run_dir / "meta.json").read_text())
    frame_seqlen = meta["grid_height"] * meta["grid_width"]
    chunk_tokens = frame_seqlen * meta["num_frames_per_block"]

    dumps = sorted(args.run_dir.glob("qk_chunk_*.npz"))
    if not dumps:
        raise SystemExit(
            f"no qk_chunk_*.npz in {args.run_dir} — record with "
            "SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS set"
        )
    wanted_chunks = (
        {int(c) for c in args.chunks.split(",")} if args.chunks else None
    )
    out_dir = args.out_dir or (args.run_dir / "qk_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for dump in dumps:
        chunk = int(dump.stem.split("_")[-1])
        if wanted_chunks is not None and chunk not in wanted_chunks:
            continue
        data = np.load(dump)
        scores = data["scores"]  # [layers, heads, queries, keys]
        layer_ids = data["layer_ids"]
        query_positions = data["query_positions"]
        key_positions = data["key_positions"]
        layer_lookup = {int(l): i for i, l in enumerate(layer_ids)}
        for layer in _parse_ids(args.layers, int(layer_ids.max()) + 1):
            if layer not in layer_lookup:
                continue
            per_layer = scores[layer_lookup[layer]]
            for head in _parse_ids(args.heads, per_layer.shape[0]):
                plot_head(
                    scores=per_layer[head],
                    query_positions=query_positions,
                    key_positions=key_positions,
                    chunk_tokens=chunk_tokens,
                    title=(
                        f"chunk {chunk} · layer {layer} · head {head} — "
                        f"query x key attention ({meta['model_tag']})"
                    ),
                    out_path=out_dir
                    / f"chunk_{chunk:03d}_layer_{layer:02d}_head_{head:02d}.png",
                )
                written += 1
    print(f"wrote {written} figures to {out_dir}")


if __name__ == "__main__":
    main()
