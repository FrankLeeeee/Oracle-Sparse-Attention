# SPDX-License-Identifier: Apache-2.0
"""The block-sparse attention kernel shared by every sparse-attention method.

Every method in this package — Oracle Sparse Attention and the four published
baselines — ultimately says the same thing: *for this query block, this head
only needs these keys*. So they all produce the same object, a
:class:`SparseAttentionPlan`, and run through one Triton kernel.

A plan describes the kept keys as **half-open token ranges** rather than a
bitmap of fixed-size blocks. Ranges are strictly more general (a latent frame
of 1560 tokens is not a whole number of 64-token tiles, so a frame-granular
method like OSA cannot be expressed as an aligned block mask at all) and they
are strictly faster: adjacent selected blocks merge into one range, so a head
that keeps a contiguous window of eight latent frames walks a single range of
12480 tokens instead of 195 separate tile descriptors.

Layout: ``range_starts``/``range_ends`` are ``[heads, q_blocks, max_ranges]``
and ``range_counts`` is ``[heads, q_blocks]``. ``q_blocks == 1`` is the
broadcast case — every query block of the call shares one key set, which is
what the chunk-level methods (OSA, Radial's frame band) produce.
"""

import msgspec
import torch
import triton
import triton.language as tl

_LOG2_E = tl.constexpr(1.4426950408889634)

# Key tile of the kernel's inner loop, and the warp/stage shape that goes with
# it. Swept on H200 at Self-Forcing shapes (q=4680, kv=32760, 12 heads of 128):
# this reaches ~376 TFLOP/s both at full density and at the ~40% density OSA
# selects, which is 57% of what FA3 gets on the dense problem. Query tiling
# (``block_m``) is chosen by each method, because it is also the granularity at
# which that method's pattern is defined.
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 3
DEFAULT_BLOCK_M = 128


@triton.jit
def _range_sparse_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    counts_ptr,
    bias_ptr,
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
    stride_range_h,
    stride_range_q,
    stride_count_h,
    stride_bias_h,
    num_heads,
    q_len,
    softmax_scale,
    HAS_BIAS: tl.constexpr,
    SHARED_Q_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Flash attention restricted to each ``(head, query block)``'s key ranges.

    One program owns one ``BLOCK_M`` tile of queries for one ``(batch, head)``
    and walks that tile's ranges with the standard online-softmax recurrence.
    Within a chunk the DiT's attention is bidirectional and the KV view already
    contains only visible keys, so the only masking needed is the tail of each
    range.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // num_heads
    head = pid_bh % num_heads

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

    q_block = 0 if SHARED_Q_BLOCK else pid_m
    range_base = head * stride_range_h + q_block * stride_range_q
    num_ranges = tl.load(counts_ptr + head * stride_count_h + q_block)

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
            if HAS_BIAS:
                bias = tl.load(
                    bias_ptr + head * stride_bias_h + offs_n,
                    mask=tile_mask,
                    other=0.0,
                ).to(tl.float32)
                qk += bias[None, :] * _LOG2_E
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

    # A query block with no ranges attends to nothing; emit zeros rather than
    # NaN. Callers never build such plans, but a wrong answer here would be
    # silent while a zero row is not.
    acc = acc / tl.where(row_sum == 0.0, 1.0, row_sum)[:, None]
    out_base = out_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        out_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=q_row_mask[:, None],
    )


class SparseAttentionPlan(msgspec.Struct, frozen=True):
    """The kept key ranges of one attention call, per head and query block.

    ``range_starts[h, b, :range_counts[h, b]]`` and the matching ``range_ends``
    are half-open token ranges into the KV view. ``q_blocks == 1`` broadcasts
    one key set over every query block.
    """

    range_starts: torch.Tensor  # [heads, q_blocks, max_ranges] int32
    range_ends: torch.Tensor  # [heads, q_blocks, max_ranges] int32
    range_counts: torch.Tensor  # [heads, q_blocks] int32
    block_m: int
    # Optional per-key additive logit bias, [heads, kv_len]; used by FAST-AR's
    # TempCache to keep merged-key attention exact.
    logit_bias: torch.Tensor | None = None

    @property
    def shared_q_block(self) -> bool:
        return self.range_starts.shape[1] == 1

    def kept_tokens(self) -> torch.Tensor:
        """``[heads, q_blocks]`` count of keys each query block visits."""
        lengths = (self.range_ends - self.range_starts).clamp(min=0)
        valid = torch.arange(
            lengths.shape[-1], device=lengths.device
        ) < self.range_counts.unsqueeze(-1)
        return (lengths * valid).sum(-1)


def sparse_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: SparseAttentionPlan,
    softmax_scale: float,
) -> torch.Tensor:
    """Attention over the plan's kept ranges; layouts match ``LocalAttention``.

    ``query`` is ``[batch, q_len, heads, head_dim]``, ``key``/``value`` are
    ``[batch, kv_len, heads, head_dim]`` views of the KV cache (any token
    stride, contiguous head_dim).
    """
    batch, q_len, num_heads, head_dim = query.shape
    if query.stride(-1) != 1:
        query = query.contiguous()
    if key.stride(-1) != 1:
        key = key.contiguous()
    if value.stride(-1) != 1:
        value = value.contiguous()

    block_m = plan.block_m
    q_blocks = plan.range_starts.shape[1]
    shared = plan.shared_q_block
    if not shared and q_blocks != triton.cdiv(q_len, block_m):
        raise ValueError(
            f"plan has {q_blocks} query blocks but q_len={q_len} at block_m="
            f"{block_m} needs {triton.cdiv(q_len, block_m)}"
        )

    out = torch.empty(
        (batch, q_len, num_heads, head_dim), dtype=query.dtype, device=query.device
    )
    bias = plan.logit_bias
    grid = (triton.cdiv(q_len, block_m), batch * num_heads)
    _range_sparse_attn_kernel[grid](
        query,
        key,
        value,
        out,
        plan.range_starts,
        plan.range_ends,
        plan.range_counts,
        bias if bias is not None else query,
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
        plan.range_starts.stride(0),
        plan.range_starts.stride(1),
        plan.range_counts.stride(0),
        bias.stride(0) if bias is not None else 0,
        num_heads,
        q_len,
        softmax_scale,
        HAS_BIAS=bias is not None,
        SHARED_Q_BLOCK=shared,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=head_dim,
        num_warps=_NUM_WARPS if head_dim >= 128 else 4,
        num_stages=_NUM_STAGES,
    )
    return out


def plan_from_shared_ranges(
    ranges: list[list[tuple[int, int]]],
    *,
    block_m: int,
    device: torch.device,
) -> SparseAttentionPlan:
    """Plan from one key-range list per head, shared by all query blocks."""
    num_heads = len(ranges)
    max_ranges = max(1, max(len(r) for r in ranges))
    starts = torch.zeros((num_heads, 1, max_ranges), dtype=torch.int32)
    ends = torch.zeros((num_heads, 1, max_ranges), dtype=torch.int32)
    counts = torch.zeros((num_heads, 1), dtype=torch.int32)
    for head, head_ranges in enumerate(ranges):
        counts[head, 0] = len(head_ranges)
        for index, (start, end) in enumerate(head_ranges):
            starts[head, 0, index] = start
            ends[head, 0, index] = end
    return SparseAttentionPlan(
        range_starts=starts.to(device, non_blocking=True),
        range_ends=ends.to(device, non_blocking=True),
        range_counts=counts.to(device, non_blocking=True),
        block_m=block_m,
    )


@triton.jit
def _mask_to_ranges_kernel(
    mask_ptr,
    segment_start_ptr,
    segment_end_ptr,
    starts_ptr,
    ends_ptr,
    counts_ptr,
    stride_mask_row,
    stride_segment_head,
    stride_out_row,
    q_blocks,
    num_segments,
    BLOCK: tl.constexpr,
):
    """Turn one ``[segments]`` mask row into merged ``(start, end)`` ranges.

    One program per ``(head, query block)``. Run boundaries, their slot numbers
    and the run count all come out of a single vectorized pass: a run's start and
    its end land in the same slot because no other run starts in between, so one
    ``cumsum`` numbers both.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < num_segments

    mask_row = mask_ptr + row * stride_mask_row
    selected = tl.load(mask_row + offs, mask=valid, other=0) != 0
    previous = tl.load(mask_row + offs - 1, mask=valid & (offs > 0), other=0) != 0
    following = tl.load(
        mask_row + offs + 1, mask=valid & (offs < num_segments - 1), other=0
    ) != 0

    is_start = selected & ~previous
    is_end = selected & ~following
    slot = tl.cumsum(is_start.to(tl.int32), 0) - 1

    segment_base = segment_start_ptr + (row // q_blocks) * stride_segment_head
    segment_end_base = segment_end_ptr + (row // q_blocks) * stride_segment_head
    range_start = tl.load(segment_base + offs, mask=valid, other=0)
    range_end = tl.load(segment_end_base + offs, mask=valid, other=0)

    out_row = row * stride_out_row
    tl.store(starts_ptr + out_row + slot, range_start, mask=is_start & valid)
    tl.store(ends_ptr + out_row + slot, range_end, mask=is_end & valid)
    tl.store(counts_ptr + row, tl.sum(is_start.to(tl.int32), 0))


def plan_from_segment_mask(
    segment_mask: torch.Tensor,
    *,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    block_m: int,
    logit_bias: torch.Tensor | None = None,
) -> SparseAttentionPlan:
    """Plan from a ``[heads, q_blocks, segments]`` bool mask over key segments.

    Segments partition the key axis in order — fixed-size blocks for the
    block-granular baselines, variable-size semantic clusters for SVG2, LSH
    buckets for FAST-AR. Runs of adjacent selected segments merge into single
    ranges, which is what makes the block-granular methods competitive: a band
    of 40 consecutive key blocks costs one range, not 40.

    This runs as one Triton kernel rather than a dozen tensor ops. It is called
    once per attention call by every method whose pattern depends on the current
    Q/K — about 4000 times per 20-second video — on a mask of only ~100k
    elements, so it is entirely launch-bound: the tensor-op version cost 0.32 ms
    per call in ~18 launches plus a mid-sequence device sync, which was more
    than a third of those methods' whole planning budget. ``max_ranges`` is
    bounded statically (a boolean row of length *n* holds at most
    ``(n + 1) // 2`` runs) so no sync is needed to size the output.
    """
    if segment_mask.dtype != torch.bool:
        segment_mask = segment_mask.bool()
    segment_mask = segment_mask.contiguous()
    num_heads, q_blocks, num_segments = segment_mask.shape
    device = segment_mask.device
    max_ranges = max(1, (num_segments + 1) // 2)

    starts = torch.zeros(
        (num_heads, q_blocks, max_ranges), dtype=torch.int32, device=device
    )
    ends = torch.zeros_like(starts)
    counts = torch.empty((num_heads, q_blocks), dtype=torch.int32, device=device)

    segment_starts = segment_starts.to(torch.int32).contiguous()
    segment_ends = segment_ends.to(torch.int32).contiguous()
    # Segment bounds are either shared by every head ([segments]) or per-head
    # ([heads, 1, segments], SVG2's clusters and FAST-AR's buckets).
    stride_segment_head = 0 if segment_starts.dim() == 1 else num_segments

    _mask_to_ranges_kernel[(num_heads * q_blocks,)](
        segment_mask,
        segment_starts,
        segment_ends,
        starts,
        ends,
        counts,
        num_segments,
        stride_segment_head,
        max_ranges,
        q_blocks,
        num_segments,
        BLOCK=triton.next_power_of_2(num_segments),
        num_warps=4,
    )
    return SparseAttentionPlan(
        range_starts=starts,
        range_ends=ends,
        range_counts=counts,
        block_m=block_m,
        logit_bias=logit_bias,
    )


def plan_from_block_mask(
    block_mask: torch.Tensor,
    *,
    block_n: int,
    kv_len: int,
    block_m: int,
    logit_bias: torch.Tensor | None = None,
) -> SparseAttentionPlan:
    """Plan from a ``[heads, q_blocks, kv_blocks]`` bool mask of fixed-size blocks."""
    kv_blocks = block_mask.shape[-1]
    block_ids = torch.arange(kv_blocks, device=block_mask.device, dtype=torch.int32)
    return plan_from_segment_mask(
        block_mask,
        segment_starts=block_ids * block_n,
        segment_ends=((block_ids + 1) * block_n).clamp(max=kv_len),
        block_m=block_m,
        logit_bias=logit_bias,
    )


def plan_key_mask(plan: SparseAttentionPlan, *, kv_len: int) -> torch.Tensor:
    """``[heads, q_blocks, kv_len]`` bool mask of the plan — for tests/analysis."""
    heads, q_blocks, max_ranges = plan.range_starts.shape
    positions = torch.arange(kv_len, device=plan.range_starts.device)
    valid = (
        torch.arange(max_ranges, device=plan.range_starts.device)
        < plan.range_counts.unsqueeze(-1)
    )
    inside = (positions[None, None, None, :] >= plan.range_starts.unsqueeze(-1)) & (
        positions[None, None, None, :] < plan.range_ends.unsqueeze(-1)
    )
    return (inside & valid.unsqueeze(-1)).any(dim=2)
