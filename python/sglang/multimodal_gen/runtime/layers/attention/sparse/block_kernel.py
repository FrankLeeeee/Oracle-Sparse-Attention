# SPDX-License-Identifier: Apache-2.0
"""Block-sparse attention for a replicated frame-to-frame pattern.

This is the execution path for a 2-D pattern: every query tile of a latent
frame keeps its own set of key tiles, instead of the whole chunk sharing one
set. The shared range kernel (``kernel.py``) can express that — its
``range_starts`` is ``[heads, q_blocks, max_ranges]`` — but its inner loop has
a data-dependent trip count, so Triton cannot software pipeline the K/V loads,
and it cannot tile queries per frame when ``frame_seqlen`` is not a multiple of
the query block.

The design here follows the one structural fact that makes OSA work: **the
pattern is the same in every (query frame, key frame) section**. So the plan
stores, per ``(head, query tile)``, only the *relative* key tiles of one
section — a handful of indices — and the kernel sweeps those over the frame
offsets. Two consequences:

so the plan is *authored* once per section, in ``build_block_plan``, and
expanded there into the flat per-query-tile block list the kernel walks. The
expansion is deliberate: a single flat loop with a uniform trip count
pipelines better than the equivalent nested frame/tile loops (6.5 ms vs 8.1 ms
at the 720p / 39-frame shape).

Frames kept whole — the query's own chunk, the sink, the recent band — enter
the same list as every one of their key tiles, so they stay off the pattern.
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
    ends_ptr,  # [heads, q_tiles, n_blocks] int32, half-open ends
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
    frame_seqlen,
    n_blocks,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program owns one (query frame, query tile, head).

    The block list has a *uniform* length across the whole grid, so the trip
    count is the same everywhere: the K/V loads pipeline and no program is left
    doing more work than its neighbours.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // num_heads
    head = pid_bh % num_heads

    # pid_m enumerates (query frame, tile within that frame), so a tile never
    # straddles a frame boundary and the key list depends only on the tile.
    q_frame = pid_m // q_tiles_per_frame
    q_tile = pid_m % q_tiles_per_frame

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

    row_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = softmax_scale * _LOG2_E

    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh
    plan_base = head * stride_th + q_tile * stride_tq

    for index in range(n_blocks):
        key_start = tl.load(starts_ptr + plan_base + index)
        key_end = tl.load(ends_ptr + plan_base + index)
        offs_n = key_start + cols
        col_mask = offs_n < key_end
        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        qk = tl.where(col_mask[None, :], qk, float("-inf"))
        new_max = tl.maximum(row_max, tl.max(qk, 1))
        rescale = tl.math.exp2(row_max - new_max)
        probs = tl.math.exp2(qk - new_max[:, None])
        row_sum = row_sum * rescale + tl.sum(probs, 1)
        acc = acc * rescale[:, None]
        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=col_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(probs.to(v.dtype), v)
        row_max = new_max

    acc = acc / tl.where(row_sum == 0.0, 1.0, row_sum)[:, None]
    out_base = out_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


class BlockSparsePlan(msgspec.Struct, frozen=True):
    """One chunk's 2-D pattern, expanded to a uniform per-query-tile block list.

    ``starts``/``ends`` are ``[heads, q_tiles_per_frame, n_blocks]`` half-open
    token ranges into the KV view, each covering at most ``key_tile`` tokens.
    Every ``(head, query tile)`` has exactly ``n_blocks`` of them, which is what
    lets the kernel run a uniform, pipelined loop. One plan row serves every
    query frame of the chunk, since the pattern is section-invariant.
    """

    starts: torch.Tensor
    ends: torch.Tensor
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
) -> BlockSparsePlan:
    """Expand the section-relative pattern into the kernel's flat block list.

    The pattern is authored once per section — ``tiles[h, q]`` are the key
    tiles query tile ``q`` keeps in *every* replicated frame — and expanded
    here because a single flat loop pipelines better in Triton than the
    equivalent nested frame/tile loops (measured: 6.5 ms vs 8.1 ms at the
    720p / 39-frame shape).
    """
    heads, q_tiles, band = tiles.shape
    device = tiles.device
    k_tiles = (frame_seqlen + key_tile - 1) // key_tile
    within = torch.arange(k_tiles, device=device, dtype=torch.int32) * key_tile

    # Whole frames: every key tile, identical for all heads and query tiles.
    whole_starts = (whole_offsets[:, None] + within[None, :]).reshape(-1)
    # Replicated frames: this query tile's key tiles at each frame offset.
    hist_starts = (
        hist_offsets[None, None, :, None] + (tiles.to(torch.int32) * key_tile)[:, :, None, :]
    ).reshape(heads, q_tiles, -1)

    starts = torch.cat(
        [whole_starts[None, None, :].expand(heads, q_tiles, -1), hist_starts], dim=2
    ).contiguous()
    # A tile never runs past the end of its own frame.
    frame_of = starts // frame_seqlen
    ends = torch.minimum(starts + key_tile, (frame_of + 1) * frame_seqlen)
    kept = int((ends[0, 0] - starts[0, 0]).sum())
    return BlockSparsePlan(
        starts=starts.to(torch.int32),
        ends=ends.to(torch.int32).contiguous(),
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
    num_stages: int = 3,
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
        plan.ends,
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
