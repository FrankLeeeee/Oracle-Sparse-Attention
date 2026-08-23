# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark of the sparse-attention methods at Self-Forcing shapes.

Reports, per method: the density it selects, the time the kernel spends on that
selection, the time the *planning* costs (score estimation, clustering, mask
choice — all of which the dense path does not pay), and the resulting speedup
over dense flash attention on the same tensors.

    python -m sglang.multimodal_gen.tools.benchmark_sparse_attention

The default shape is Self-Forcing's steady state: a 3-frame chunk of queries
(4680 tokens) against a 21-frame visible KV window (32760 tokens), 12 heads of
128 channels, bf16.
"""

import argparse
import json

import torch

from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen.runtime.layers.attention.sparse import (
    SPARSE_ATTENTION_METHODS,
    build_sparse_attention_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    SparseAttentionCall,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    sparse_attention,
)


def timed(function, *, warmup: int = 5, iterations: int = 20) -> float:
    """Median-ish wall-clock milliseconds of ``function``."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def dense_attention(query, key, value, softmax_scale):
    """The dense path the model actually runs, so the speedups are the real ones."""
    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=query.shape[1],
        max_seqlen_k=key.shape[1],
        softmax_scale=softmax_scale,
        causal=False,
        ver=3,
    )


def make_call(
    *,
    layer_index: int,
    chunk_index: int,
    geometry: ChunkGeometry,
    visible_frames: int,
    heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SparseAttentionCall:
    """A synthetic steady-state call with a rolling visible window."""
    frame_seqlen = geometry.frame_seqlen
    q_len = geometry.chunk_tokens
    kv_len = visible_frames * frame_seqlen
    first_visible_token = geometry.query_token_start + q_len - kv_len
    generator = torch.Generator(device=device).manual_seed(
        1000 * layer_index + chunk_index
    )
    shape_q = (1, q_len, heads, head_dim)
    shape_kv = (1, kv_len, heads, head_dim)
    return SparseAttentionCall(
        layer_index=layer_index,
        query=torch.randn(shape_q, device=device, dtype=dtype, generator=generator),
        key=torch.randn(shape_kv, device=device, dtype=dtype, generator=generator),
        value=torch.randn(shape_kv, device=device, dtype=dtype, generator=generator),
        key_segments=((first_visible_token, kv_len),),
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
    )


def benchmark_method(
    method: str,
    *,
    config: dict,
    frame_seqlen: int,
    frames_per_block: int,
    visible_frames: int,
    heads: int,
    head_dim: int,
    steady_chunk: int,
    denoise_steps: int,
    grid_height: int,
    grid_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    backend = build_sparse_attention_backend(method, config)

    def geometry_for(chunk_index: int) -> ChunkGeometry:
        return ChunkGeometry(
            frame_seqlen=frame_seqlen,
            frames_per_block=frames_per_block,
            query_token_start=chunk_index * frames_per_block * frame_seqlen,
            grid_height=grid_height,
            grid_width=grid_width,
        )

    # Walk the earlier chunks, denoising steps included, so anything that
    # calibrates (OSA) or counts steps (FAST-AR) is in the state it would be in
    # mid-video.
    execution = None
    call = None
    for chunk_index in range(steady_chunk + 1):
        visible = min(visible_frames, (chunk_index + 1) * frames_per_block)
        geometry = geometry_for(chunk_index)
        call = make_call(
            layer_index=0,
            chunk_index=chunk_index,
            geometry=geometry,
            visible_frames=visible,
            heads=heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        layout = visible_layout(
            call.key_segments, geometry=geometry, query_tokens=call.query.shape[1]
        )
        for _ in range(denoise_steps):
            backend.begin_forward(geometry)
            execution = backend.prepare(call, layout)

    if execution is None:
        return {"method": method, "status": "declined (dense)"}

    kv_len = execution.key.shape[1]
    plan = execution.plan
    q_blocks = plan.range_counts.shape[1]
    # kept_tokens instead of plan_key_mask: the mask materializes
    # [heads, q_blocks, max_ranges, kv] and OOMs at 720p/20s shapes. Ranges
    # never overlap within a row, so the token count is exact.
    density = (
        plan.kept_tokens().sum() / float(plan.range_counts.shape[0] * q_blocks * kv_len)
    ).item()

    geometry = geometry_for(steady_chunk)
    layout = visible_layout(
        call.key_segments, geometry=geometry, query_tokens=call.query.shape[1]
    )
    kernel_ms = timed(
        lambda: sparse_attention(
            query=execution.query,
            key=execution.key,
            value=execution.value,
            plan=execution.plan,
            softmax_scale=call.softmax_scale,
        )
    )
    total_ms = timed(lambda: backend.attend(call))
    dense_ms = timed(
        lambda: dense_attention(call.query, call.key, call.value, call.softmax_scale)
    )
    return {
        "method": method,
        "density": round(density, 4),
        "dense_ms": round(dense_ms, 3),
        "kernel_ms": round(kernel_ms, 3),
        "total_ms": round(total_ms, 3),
        "planning_ms": round(total_ms - kernel_ms, 3),
        "kernel_speedup": round(dense_ms / kernel_ms, 2),
        "end_to_end_speedup": round(dense_ms / total_ms, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-seqlen", type=int, default=1560)
    # Post-patch grid of one latent frame; must multiply to --frame-seqlen
    # (Wan 480p: 30x52, 720p: 45x80).
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--frames-per-block", type=int, default=3)
    parser.add_argument("--visible-frames", type=int, default=21)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--steady-chunk", type=int, default=10)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--methods", nargs="*", default=list(SPARSE_ATTENTION_METHODS))
    parser.add_argument(
        "--config", type=str, default="{}", help="JSON config shared by all methods"
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    shared_config = json.loads(args.config)
    rows = []
    for method in args.methods:
        rows.append(
            benchmark_method(
                method,
                config=shared_config,
                frame_seqlen=args.frame_seqlen,
                frames_per_block=args.frames_per_block,
                visible_frames=args.visible_frames,
                heads=args.heads,
                head_dim=args.head_dim,
                steady_chunk=args.steady_chunk,
                denoise_steps=args.denoise_steps,
                grid_height=args.grid_height,
                grid_width=args.grid_width,
                device=device,
                dtype=torch.bfloat16,
            )
        )

    header = (
        f"{'method':<12}{'density':>9}{'dense ms':>10}{'kernel ms':>11}"
        f"{'plan ms':>9}{'total ms':>10}{'kernel x':>10}{'e2e x':>8}"
    )
    print(
        f"\nq={args.frames_per_block * args.frame_seqlen} "
        f"kv={args.visible_frames * args.frame_seqlen} "
        f"heads={args.heads} head_dim={args.head_dim} bf16\n"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if "density" not in row:
            print(f"{row['method']:<12}{row['status']:>57}")
            continue
        print(
            f"{row['method']:<12}{row['density']:>9.3f}{row['dense_ms']:>10.3f}"
            f"{row['kernel_ms']:>11.3f}{row['planning_ms']:>9.3f}"
            f"{row['total_ms']:>10.3f}{row['kernel_speedup']:>10.2f}"
            f"{row['end_to_end_speedup']:>8.2f}"
        )


if __name__ == "__main__":
    main()
