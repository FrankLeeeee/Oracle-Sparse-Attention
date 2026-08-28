# SPDX-License-Identifier: Apache-2.0
"""The two Triton kernels of Mixed Sparse Attention (MSA).

MSA splits a layer's heads into content-independent ("static") heads, whose
key pattern is the same within-frame structure replicated over every visible
frame, and content-dependent heads, which keep a per-call top-k of pooled key
blocks. One attention call therefore launches (up to) two kernels, each
covering its head subset via an explicit head-id list and writing its slice of
the shared output tensor:

``_msa_static_kernel``
    Frame-replicated flash attention. A static head's plan is a tiny
    per-(head, query block) list of *within-frame* token ranges plus a first
    visible frame; the kernel walks ``frame x range x tile`` and rebuilds the
    global key offsets on the fly. Frame replication lives in the loop
    structure instead of in materialized range lists, so a static head's plan
    is a few hundred bytes, costs nothing to build per call (it is cached per
    layout), and its key walks are contiguous runs.

``_msa_content_kernel``
    The package's range-sparse flash attention (see ``kernel.py``) with head
    indirection, so a plan covering only the content-head subset executes
    against the full Q/KV tensors without materializing per-subset copies.

Both keep the shared kernel's tiling (64-token key tiles, method-chosen query
blocks) and online-softmax recurrence.
"""

import torch
import triton
import triton.language as tl

_LOG2_E = tl.constexpr(1.4426950408889634)
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 3


@triton.jit
def _msa_static_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    head_ids_ptr,
    frame_lo_ptr,
    range_ptr,  # [groups, q_blocks, max_ranges, 2] within-frame token bounds
    count_ptr,  # [groups, q_blocks]
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
    stride_range_g,
    stride_range_q,
    stride_count_g,
    num_groups,
    num_frames,
    frame_seqlen,
    q_len,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bg = tl.program_id(1)
    batch = pid_bg // num_groups
    group = pid_bg % num_groups
    head = tl.load(head_ids_ptr + group)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_row_mask = offs_m < q_len

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    q_tile = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=q_row_mask[:, None],
        other=0.0,
    )

    row_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = softmax_scale * _LOG2_E
    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh

    frame_lo = tl.load(frame_lo_ptr + group)
    range_base = group * stride_range_g + pid_m * stride_range_q
    num_ranges = tl.load(count_ptr + group * stride_count_g + pid_m)

    for frame in range(frame_lo, num_frames):
        frame_base = frame * frame_seqlen
        for range_index in range(num_ranges):
            range_lo = tl.load(range_ptr + range_base + range_index * 2)
            range_hi = tl.load(range_ptr + range_base + range_index * 2 + 1)
            for tile_start in range(range_lo, range_hi, BLOCK_N):
                offs_n = frame_base + tile_start + tl.arange(0, BLOCK_N)
                tile_mask = tile_start + tl.arange(0, BLOCK_N) < range_hi
                k_tile = tl.load(
                    k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=tile_mask[:, None],
                    other=0.0,
                )
                qk = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * qk_scale
                qk = tl.where(tile_mask[None, :], qk, float("-inf"))
                new_max = tl.maximum(row_max, tl.max(qk, 1))
                rescale = tl.math.exp2(row_max - new_max)
                probs = tl.math.exp2(qk - new_max[:, None])
                row_sum = row_sum * rescale + tl.sum(probs, 1)
                acc = acc * rescale[:, None]
                v_tile = tl.load(
                    v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=tile_mask[:, None],
                    other=0.0,
                )
                acc += tl.dot(probs.to(v_tile.dtype), v_tile)
                row_max = new_max

    acc = acc / tl.where(row_sum == 0.0, 1.0, row_sum)[:, None]
    out_base = out_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=q_row_mask[:, None],
    )


@triton.jit
def _msa_content_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    head_ids_ptr,
    starts_ptr,
    ends_ptr,
    counts_ptr,
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
    stride_range_g,
    stride_range_q,
    stride_count_g,
    num_groups,
    q_len,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bg = tl.program_id(1)
    batch = pid_bg // num_groups
    group = pid_bg % num_groups
    head = tl.load(head_ids_ptr + group)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_row_mask = offs_m < q_len

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    q_tile = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=q_row_mask[:, None],
        other=0.0,
    )

    row_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = softmax_scale * _LOG2_E
    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh

    range_base = group * stride_range_g + pid_m * stride_range_q
    num_ranges = tl.load(counts_ptr + group * stride_count_g + pid_m)

    for range_index in range(num_ranges):
        range_start = tl.load(starts_ptr + range_base + range_index)
        range_end = tl.load(ends_ptr + range_base + range_index)
        for tile_start in range(range_start, range_end, BLOCK_N):
            offs_n = tile_start + tl.arange(0, BLOCK_N)
            tile_mask = offs_n < range_end
            k_tile = tl.load(
                k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
                mask=tile_mask[:, None],
                other=0.0,
            )
            qk = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * qk_scale
            qk = tl.where(tile_mask[None, :], qk, float("-inf"))
            new_max = tl.maximum(row_max, tl.max(qk, 1))
            rescale = tl.math.exp2(row_max - new_max)
            probs = tl.math.exp2(qk - new_max[:, None])
            row_sum = row_sum * rescale + tl.sum(probs, 1)
            acc = acc * rescale[:, None]
            v_tile = tl.load(
                v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                mask=tile_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(probs.to(v_tile.dtype), v_tile)
            row_max = new_max

    acc = acc / tl.where(row_sum == 0.0, 1.0, row_sum)[:, None]
    out_base = out_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=q_row_mask[:, None],
    )


def msa_static_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    head_ids: torch.Tensor,  # [groups] int32
    frame_lo: torch.Tensor,  # [groups] int32
    ranges: torch.Tensor,  # [groups, q_blocks, max_ranges, 2] int32
    counts: torch.Tensor,  # [groups, q_blocks] int32
    num_frames: int,
    frame_seqlen: int,
    block_m: int,
    softmax_scale: float,
) -> None:
    """Run the static heads' frame-replicated attention into ``out``."""
    batch, q_len, _, head_dim = query.shape
    num_groups = head_ids.shape[0]
    grid = (triton.cdiv(q_len, block_m), batch * num_groups)
    _msa_static_kernel[grid](
        query,
        key,
        value,
        out,
        head_ids,
        frame_lo,
        ranges,
        counts,
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
        ranges.stride(0),
        ranges.stride(1),
        counts.stride(0),
        num_groups,
        num_frames,
        frame_seqlen,
        q_len,
        softmax_scale,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=head_dim,
        num_warps=_NUM_WARPS if head_dim >= 128 else 4,
        num_stages=_NUM_STAGES,
    )


def msa_content_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    head_ids: torch.Tensor,  # [groups] int32
    range_starts: torch.Tensor,  # [groups, q_blocks, max_ranges] int32
    range_ends: torch.Tensor,
    range_counts: torch.Tensor,  # [groups, q_blocks] int32
    block_m: int,
    softmax_scale: float,
) -> None:
    """Run the content heads' range-sparse attention into ``out``."""
    batch, q_len, _, head_dim = query.shape
    num_groups = head_ids.shape[0]
    grid = (triton.cdiv(q_len, block_m), batch * num_groups)
    _msa_content_kernel[grid](
        query,
        key,
        value,
        out,
        head_ids,
        range_starts,
        range_ends,
        range_counts,
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
        range_starts.stride(0),
        range_starts.stride(1),
        range_counts.stride(0),
        num_groups,
        q_len,
        softmax_scale,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=head_dim,
        num_warps=_NUM_WARPS if head_dim >= 128 else 4,
        num_stages=_NUM_STAGES,
    )
