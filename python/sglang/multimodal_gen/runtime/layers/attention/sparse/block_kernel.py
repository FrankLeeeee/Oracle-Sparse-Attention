# SPDX-License-Identifier: Apache-2.0
"""Block-sparse attention for a replicated frame-to-frame pattern.

This is the execution path for OSA's 2-D pattern: every query tile of a latent
frame keeps its own set of key tiles. The shared range kernel (``kernel.py``)
can express that — its ``range_starts`` is ``[heads, q_blocks, max_ranges]`` —
but its inner loop has a data-dependent trip count, so Triton cannot software
pipeline the K/V loads, and it cannot tile queries per frame when
``frame_seqlen`` is not a multiple of the query block.

Design decisions, each measured at the 720p Self-Forcing shapes:

* **Uniform trip count.** Every ``(head, query tile)`` keeps exactly
  ``n_blocks`` key blocks of exactly ``key_tile`` tokens — the OSA budget
  guarantees the count, and whole frames are expanded tile-by-tile with the
  frame's short raster tail (``frame_seqlen % key_tile``) excluded, so no
  block is ever partial. That drops all masking from the inner loop; a masked
  variant measured 416 TFLOP/s against this kernel's ~480-505.
* **Two interleaved online-softmax chains**, merged by an exact two-way LSE
  combine at the end. The standard flash loop carries a serial rescale
  dependency; halving the chain length bought ~9%. Wider fused blocks, more
  chains, and multi-frame programs all lost to register pressure (measured:
  318-348 TFLOP/s from spills).
* **Ascending tile walk.** The walk order is free (online softmax is exact in
  any order), so the plan sorts each frame's tiles by position for the
  DRAM-friendliest access.
* ``num_warps=8, num_stages=5`` from a sweep; ``BLOCK_M=128`` (64 halves
  tensor-core shapes, 256 spills).

The kernel remains memory-bound at large KV (per-program K/V traffic times
program count tracks the runtime), which caps it near ~500 TFLOP/s against
FA3's ~650 on dense shapes; the remaining reuse (the query frames share every
key block) exceeds what L2 captures and would need a persistent-CTA or CUDA
rewrite to claim.
"""

import msgspec
import torch
import triton
import triton.language as tl

_LOG2_E = tl.constexpr(1.4426950408889634)

DEFAULT_QUERY_TILE = 128
DEFAULT_KEY_TILE = 64


@triton.jit
def _block_sparse_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    starts_ptr,  # [heads, q_tiles, n_blocks] int32, absolute token starts
    stride_qb,
    stride_qm,
    stride_qh,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_ob,
    stride_om,
    stride_oh,
    stride_th,
    stride_tq,
    num_heads,
    q_tiles_per_frame,
    num_q_frames,
    frame_seqlen,
    n_blocks,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program owns one (query frame, query tile, head).

    Every block spans exactly ``BLOCK_N`` tokens, so there is no masking in
    the loop; the two interleaved chains halve the online-softmax dependency
    and are merged exactly at the end.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // num_heads
    head = pid_bh % num_heads

    # q_frame varies fastest: the programs sharing one plan row (identical
    # key blocks, different query frames) are adjacent in the grid and
    # co-scheduled, so their K/V reads coalesce in L2.
    q_tile = pid_m // num_q_frames
    q_frame = pid_m % num_q_frames

    tile_base = q_tile * BLOCK_M
    rows = tl.arange(0, BLOCK_M)
    offs_m = q_frame * frame_seqlen + tile_base + rows
    row_mask = tile_base + rows < frame_seqlen
    offs_d = tl.arange(0, HEAD_DIM)
    cols = tl.arange(0, BLOCK_N)

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )

    m0 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m1 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = softmax_scale * _LOG2_E

    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh
    plan_base = head * stride_th + q_tile * stride_tq

    half = n_blocks // 2
    for index in range(half):
        s0 = tl.load(starts_ptr + plan_base + 2 * index)
        s1 = tl.load(starts_ptr + plan_base + 2 * index + 1)
        n0 = s0 + cols
        n1 = s1 + cols
        k0 = tl.load(k_base + n0[:, None] * stride_kn + offs_d[None, :])
        k1 = tl.load(k_base + n1[:, None] * stride_kn + offs_d[None, :])
        qk0 = tl.dot(q, tl.trans(k0)).to(tl.float32) * qk_scale
        qk1 = tl.dot(q, tl.trans(k1)).to(tl.float32) * qk_scale
        new0 = tl.maximum(m0, tl.max(qk0, 1))
        new1 = tl.maximum(m1, tl.max(qk1, 1))
        p0 = tl.math.exp2(qk0 - new0[:, None])
        p1 = tl.math.exp2(qk1 - new1[:, None])
        r0 = tl.math.exp2(m0 - new0)
        r1 = tl.math.exp2(m1 - new1)
        l0 = l0 * r0 + tl.sum(p0, 1)
        l1 = l1 * r1 + tl.sum(p1, 1)
        acc0 = acc0 * r0[:, None]
        acc1 = acc1 * r1[:, None]
        v0 = tl.load(v_base + n0[:, None] * stride_vn + offs_d[None, :])
        v1 = tl.load(v_base + n1[:, None] * stride_vn + offs_d[None, :])
        acc0 += tl.dot(p0.to(v0.dtype), v0)
        acc1 += tl.dot(p1.to(v1.dtype), v1)
        m0 = new0
        m1 = new1

    # Odd tail block on chain 0.
    if n_blocks % 2 == 1:
        s0 = tl.load(starts_ptr + plan_base + n_blocks - 1)
        n0 = s0 + cols
        k0 = tl.load(k_base + n0[:, None] * stride_kn + offs_d[None, :])
        qk0 = tl.dot(q, tl.trans(k0)).to(tl.float32) * qk_scale
        new0 = tl.maximum(m0, tl.max(qk0, 1))
        p0 = tl.math.exp2(qk0 - new0[:, None])
        r0 = tl.math.exp2(m0 - new0)
        l0 = l0 * r0 + tl.sum(p0, 1)
        acc0 = acc0 * r0[:, None]
        v0 = tl.load(v_base + n0[:, None] * stride_vn + offs_d[None, :])
        acc0 += tl.dot(p0.to(v0.dtype), v0)
        m0 = new0

    # Exact two-way LSE merge of the chains.
    m = tl.maximum(m0, m1)
    w0 = tl.math.exp2(m0 - m)
    w1 = tl.math.exp2(m1 - m)
    l = l0 * w0 + l1 * w1
    acc = acc0 * w0[:, None] + acc1 * w1[:, None]

    acc = acc / tl.where(l == 0.0, 1.0, l)[:, None]
    out_base = out_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


class BlockSparsePlan(msgspec.Struct, frozen=True):
    """One chunk's 2-D pattern, expanded to a uniform per-query-tile block list.

    ``starts[h, t, :]`` are absolute token starts of the key blocks query tile
    ``t`` of head ``h`` reads; every block spans exactly ``key_tile`` tokens
    and every ``(head, query tile)`` has the same count, which is what lets
    the kernel run a uniform, maskless, pipelined loop. One plan row serves
    every query frame of the chunk, since the pattern is section-invariant.
    """

    starts: torch.Tensor  # [heads, q_tiles_per_frame, n_blocks] int32
    q_tiles_per_frame: int
    frame_seqlen: int
    query_tile: int
    key_tile: int
    kept_tokens: int
    density: float


def build_block_plan(
    *,
    tiles: torch.Tensor,  # [heads, q_tiles, band] int32, key tiles of a section
    hist_offsets: torch.Tensor,  # [n_hist] int32, token offset per replicated frame
    whole_offsets: torch.Tensor,  # [n_whole] int32, token offset per whole frame
    frame_seqlen: int,
    query_tile: int,
    key_tile: int,
    kv_len: int,
    hist_tile_counts: torch.Tensor | None = None,  # [n_hist] tiles kept per frame
) -> BlockSparsePlan:
    """Expand the section-relative pattern into the kernel's flat block list.

    Whole frames are expanded over their ``frame_seqlen // key_tile`` full
    tiles — the short raster tail (``frame_seqlen % key_tile`` tokens, <0.5%
    of a 720p frame) is excluded so every block is full and the kernel needs
    no masking.

    ``hist_tile_counts`` lets replicated frames keep different tile counts
    (demand-weighted allocation): frame j keeps the first
    ``hist_tile_counts[j]`` tiles of its by-mass order in ``tiles``. The
    per-(head, query tile) total stays identical across the grid either way,
    which is the kernel's uniform trip count.
    """
    heads, q_tiles, band = tiles.shape
    device = tiles.device
    full_tiles = frame_seqlen // key_tile
    within = torch.arange(full_tiles, device=device, dtype=torch.int32) * key_tile

    # Whole frames: every full tile, identical for all heads and query tiles.
    whole_starts = (whole_offsets[:, None] + within[None, :]).reshape(-1)
    # Replicated frames: this query tile's key tiles at each frame offset.
    # Walk order is free (online softmax is exact in any order); ascending
    # tile position gives the DRAM-friendliest walk within each frame.
    if hist_tile_counts is None:
        sorted_tiles = tiles.sort(dim=-1).values
        hist_starts = (
            hist_offsets[None, None, :, None]
            + (sorted_tiles.to(torch.int32) * key_tile)[:, :, None, :]
        ).reshape(heads, q_tiles, -1)
    else:
        pieces = []
        for index, offset in enumerate(hist_offsets.tolist()):
            kept = tiles[:, :, : int(hist_tile_counts[index])].sort(dim=-1).values
            pieces.append(offset + kept.to(torch.int32) * key_tile)
        hist_starts = (
            torch.cat(pieces, dim=2)
            if pieces
            else tiles.new_zeros((heads, q_tiles, 0), dtype=torch.int32)
        )

    starts = torch.cat(
        [whole_starts[None, None, :].expand(heads, q_tiles, -1), hist_starts], dim=2
    ).contiguous()
    kept = starts.shape[2] * key_tile
    return BlockSparsePlan(
        starts=starts.to(torch.int32),
        q_tiles_per_frame=q_tiles,
        frame_seqlen=frame_seqlen,
        query_tile=query_tile,
        key_tile=key_tile,
        kept_tokens=kept,
        density=kept / kv_len,
    )


def block_sparse_attention(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    value: torch.Tensor,
    plan: BlockSparsePlan,
    softmax_scale: float,
    num_warps: int = 8,
    num_stages: int = 5,
) -> torch.Tensor:
    batch, q_len, heads, head_dim = query.shape
    query_frames, ragged = divmod(q_len, plan.frame_seqlen)
    if ragged:
        raise ValueError(
            f"query of {q_len} tokens is not a whole number of "
            f"{plan.frame_seqlen}-token frames"
        )
    out = torch.empty_like(query)
    grid = (query_frames * plan.q_tiles_per_frame, batch * heads)
    _block_sparse_attn_kernel[grid](
        query,
        key,
        value,
        out,
        plan.starts,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        plan.starts.stride(0),
        plan.starts.stride(1),
        heads,
        plan.q_tiles_per_frame,
        query_frames,
        plan.frame_seqlen,
        plan.starts.shape[2],
        softmax_scale,
        BLOCK_M=plan.query_tile,
        BLOCK_N=plan.key_tile,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
