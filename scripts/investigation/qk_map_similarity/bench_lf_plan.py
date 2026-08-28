# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark LightForcing's per-call planning cost at the study shapes.

    python bench_lf_plan.py [--iters 50]

Times the exact planning pipeline of ``LightForcingAttention.prepare`` —
query pooling, own-chunk key pooling, ``lightforcing_block_mask``,
``plan_from_segment_mask`` — on synthetic bf16 tensors at the 720p/5s shapes
(what stock LightForcing pays on EVERY attention call of every denoising
step), next to a dense SDPA attention call at the same shape for scale.
History-key pooling is cached across steps in the real backend and is
reported separately. This is measurement (c)'s microbenchmark: the hybrid
proposal moves this planning off the denoise path entirely (once per chunk,
in the cache-update forward), so the per-call number x the call count is the
amortizable budget.

Output: deep_dive/bench_lf_plan.json
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "python"))
from paths import results_dir  # noqa: E402

from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (  # noqa: E402
    frame_aligned_block_bounds,
    lightforcing_block_mask,
    mean_pool_blocks,
    plan_from_segment_mask,
)

ROOT = results_dir("qk_map_similarity")
HEADS, DIM, T = 12, 128, 3600
BLOCK = 128
# (name, visible frames): the last chunk and a mid-video chunk of 720p/5s.
SHAPES = (("c6", 21), ("c3", 12))
DENSITY = 0.2


def timed(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_shape(num_frames: int, iters: int) -> dict:
    device = "cuda"
    q_len, kv_len = 3 * T, num_frames * T
    history_len = kv_len - q_len
    query = torch.randn(q_len, HEADS, DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(kv_len, HEADS, DIM, device=device, dtype=torch.bfloat16)
    blocks_per_frame = -(-T // BLOCK)
    kv_blocks = num_frames * blocks_per_frame
    topk = max(1, int(DENSITY * kv_blocks))
    pooled_history = mean_pool_blocks(key[:history_len], block=BLOCK, group=T)
    block_lo, block_hi = frame_aligned_block_bounds(
        num_frames=num_frames, frame_seqlen=T, block=BLOCK, device=device
    )

    def plan_call():
        pooled_query = mean_pool_blocks(query, block=BLOCK)
        pooled_own = mean_pool_blocks(key[history_len:], block=BLOCK, group=T)
        pooled_key = torch.cat([pooled_history, pooled_own], dim=0)
        mask = lightforcing_block_mask(
            pooled_query=pooled_query.permute(1, 0, 2),
            pooled_key=pooled_key.permute(1, 0, 2),
            blocks_per_frame=blocks_per_frame,
            past_frames=num_frames - 3,
            keep_frames=num_frames,
            keep_sink=1,
            keep_near=1,
            topk=topk,
        )
        return plan_from_segment_mask(
            mask, segment_starts=block_lo, segment_ends=block_hi, block_m=BLOCK
        )

    def history_pool():
        mean_pool_blocks(key[:history_len], block=BLOCK, group=T)

    sdpa_q = query.permute(1, 0, 2)[None]
    sdpa_k = key.permute(1, 0, 2)[None]
    sdpa_v = torch.randn_like(sdpa_k)

    def dense_attention():
        torch.nn.functional.scaled_dot_product_attention(sdpa_q, sdpa_k, sdpa_v)

    return {
        "visible_frames": num_frames,
        "plan_ms": round(timed(plan_call, iters), 4),
        "history_pool_ms_first_call": round(timed(history_pool, iters), 4),
        "dense_attention_ms": round(timed(dense_attention, iters), 4),
        "kv_blocks": kv_blocks,
        "topk": topk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    results = {name: bench_shape(frames, args.iters) for name, frames in SHAPES}
    # Per-video planning budget: 30 layers x 4 steps x 7 chunks, sized by the
    # average chunk (~c3's shape).
    per_call = results["c3"]["plan_ms"]
    results["per_video"] = {
        "plan_calls": 30 * 4 * 7,
        "estimated_total_plan_s": round(per_call * 30 * 4 * 7 / 1000, 3),
        "note": "sized at the mid-video (c3) shape; hybrid moves this off the "
        "denoise path (once per chunk in the cache-update forward)",
    }
    out = ROOT / "deep_dive" / "bench_lf_plan.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
