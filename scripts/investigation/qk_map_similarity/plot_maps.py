# SPDX-License-Identifier: Apache-2.0
"""Render the QK attention maps from the raw Q/K dumps of run.py.

    python plot_maps.py [--run p1] [--device cuda]

For each of the study's four (layer, head) picks x captured chunk x denoising
step, recomputes the exact ``softmax(q k^T / sqrt(d))`` over the *full* visible
key axis and renders it as a token-by-token heat map:

- y axis: query token index (the 3 latent frames of the chunk being denoised),
- x axis: key token index (every visible latent frame, oldest first),
- color: post-softmax attention probability, log color scale (a linear scale
  is a black rectangle: with up to 75k visible keys the mean probability is
  ~1e-5), mean-pooled to display resolution,
- green lines: latent-frame boundaries on both axes,
- white vertical line: where the current chunk's own keys start.

Output: results/investigation/qk_map_similarity/plots/<run>/L{l}_h{h}_c{c}_s{s}.png
"""

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from run import CHUNK_IDS, HEAD_SPECS, STEP_IDS  # noqa: E402

ROOT = results_dir("qk_map_similarity")
# Display pooling (mean) of the raw token axes. 15 query tokens/pixel keeps
# 240 rows per latent frame-triple; 45 key tokens/pixel is exactly one latent
# row, so frame boundaries stay pixel-aligned (3600 tokens = 80 pixels).
QUERY_POOL = 15
KEY_POOL = 45


def load_qk(run_dir: pathlib.Path, layer: int, head: int, chunk: int, step: int):
    data = np.load(run_dir / "qk" / f"qk_L{layer:02d}_c{chunk}_s{step}.npz")
    column = list(data["head_ids"]).index(head)
    return (
        torch.from_numpy(data["query"][:, column]).float(),
        torch.from_numpy(data["key"][:, column]).float(),
        int(data["frame_seqlen"]),
    )


@torch.no_grad()
def pooled_probs(
    query: torch.Tensor, key: torch.Tensor, *, device: str, tile: int = 3600
) -> np.ndarray:
    # tile must stay a multiple of QUERY_POOL so each tile pools cleanly.
    """Full-key-axis softmax map, mean-pooled to display resolution."""
    query, key = query.to(device), key.to(device)
    scale = query.shape[-1] ** -0.5
    num_q, num_k = query.shape[0], key.shape[0]
    assert num_q % QUERY_POOL == 0 and num_k % KEY_POOL == 0
    out = torch.empty(num_q // QUERY_POOL, num_k // KEY_POOL, dtype=torch.float32)
    for start in range(0, num_q, tile):
        probs = torch.softmax((query[start : start + tile] @ key.T) * scale, dim=-1)
        pooled = probs.view(-1, QUERY_POOL, num_k // KEY_POOL, KEY_POOL).mean((1, 3))
        out[start // QUERY_POOL : (start + tile) // QUERY_POOL] = pooled.cpu()
    return out.numpy()


def render(
    pooled: np.ndarray, *, frame_seqlen: int, num_q: int, num_k: int, path: pathlib.Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=170)
    positive = pooled[pooled > 0]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(pooled.max()))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(cmap(0.0))
    ax.imshow(
        pooled,
        norm=norm,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        extent=(0, num_k, num_q, 0),
    )
    for boundary in range(frame_seqlen, num_k, frame_seqlen):
        ax.axvline(boundary, color="limegreen", linewidth=1.6)
    for boundary in range(frame_seqlen, num_q, frame_seqlen):
        ax.axhline(boundary, color="limegreen", linewidth=1.6)
    ax.axvline(num_k - num_q, color="white", linewidth=2.2)
    ax.set_xlabel("key token index", fontsize=8)
    ax.set_ylabel("query token index", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run
    plot_dir = ROOT / "plots" / args.run
    plot_dir.mkdir(parents=True, exist_ok=True)
    for spec in HEAD_SPECS:
        for chunk in CHUNK_IDS:
            for step in STEP_IDS:
                query, key, frame_seqlen = load_qk(
                    run_dir, spec["layer"], spec["head"], chunk, step
                )
                pooled = pooled_probs(query, key, device=args.device)
                path = (
                    plot_dir / f"L{spec['layer']:02d}_h{spec['head']}_c{chunk}_s{step}.png"
                )
                render(
                    pooled,
                    frame_seqlen=frame_seqlen,
                    num_q=query.shape[0],
                    num_k=key.shape[0],
                    path=path,
                )
            print(f"[plot] L{spec['layer']} h{spec['head']} c{chunk}: done", flush=True)
    print(f"[plot] wrote {len(list(plot_dir.glob('*.png')))} maps -> {plot_dir}")


if __name__ == "__main__":
    main()
