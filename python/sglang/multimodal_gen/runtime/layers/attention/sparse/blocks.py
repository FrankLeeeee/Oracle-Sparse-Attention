# SPDX-License-Identifier: Apache-2.0
"""Block-level geometry shared by the block-granular baselines.

XAttention, SVG and Radial Attention all select ``(query block, key block)``
pairs, and three of the four patterns they build are stated in terms of *where
a block sits inside its latent frame* — "the same spatial position across
frames" (SVG temporal), "a band around the same spatial position, narrowing
with temporal distance" (Radial). A latent frame of a Wan 480p video is 1560
tokens, which is not a whole number of 128-token blocks, so a block can
straddle two frames and cover two disjoint runs of intra-frame offsets. These
helpers keep that bookkeeping exact and in one place.
"""

import torch


def block_bounds(length: int, block: int, *, device: torch.device) -> tuple[
    torch.Tensor, torch.Tensor
]:
    """Half-open ``(lo, hi)`` token bounds of each block covering ``length``."""
    num_blocks = -(-length // block)
    lo = torch.arange(num_blocks, device=device) * block
    hi = (lo + block).clamp(max=length)
    return lo, hi


def frame_span(
    lo: torch.Tensor, hi: torch.Tensor, *, frame_seqlen: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """First and last latent frame each block touches (inclusive)."""
    return lo // frame_seqlen, (hi - 1) // frame_seqlen


def intra_frame_coverage(
    lo: torch.Tensor, hi: torch.Tensor, *, frame_seqlen: int
) -> torch.Tensor:
    """``[num_blocks, frame_seqlen]`` bool: which intra-frame offsets a block covers.

    A block that spans a whole frame or more covers every offset; otherwise it
    covers one wrap-around run ``[lo % F, hi % F)``.
    """
    offsets = torch.arange(frame_seqlen, device=lo.device)
    span = hi - lo
    start = lo % frame_seqlen
    end = start + span
    within = (offsets[None, :] >= start[:, None]) & (offsets[None, :] < end[:, None])
    wrapped = offsets[None, :] < (end - frame_seqlen)[:, None]
    return (within | wrapped) | (span >= frame_seqlen)[:, None]


def dilate_coverage(coverage: torch.Tensor, radius: int) -> torch.Tensor:
    """Widen each block's offset coverage by ``radius`` offsets on both sides."""
    if radius <= 0:
        return coverage
    widened = torch.nn.functional.max_pool1d(
        coverage.float().unsqueeze(1), kernel_size=2 * radius + 1, stride=1,
        padding=radius,
    )
    return widened.squeeze(1) > 0


def coverage_overlap(
    query_coverage: torch.Tensor, key_coverage: torch.Tensor
) -> torch.Tensor:
    """``[q_blocks, k_blocks]`` bool: do the two offset coverages intersect?"""
    return (query_coverage.float() @ key_coverage.float().T) > 0


def own_block_mask(
    *,
    q_lo: torch.Tensor,
    q_hi: torch.Tensor,
    k_lo: torch.Tensor,
    k_hi: torch.Tensor,
    query_offset_in_view: int,
) -> torch.Tensor:
    """``[q_blocks, k_blocks]``: key blocks holding the query block's own tokens.

    The current chunk's queries are the last ``q_len`` keys of the view, so
    query token ``t`` is key token ``query_offset_in_view + t``. Every method
    keeps this diagonal: a token that cannot see itself is not attention.
    """
    q_start = q_lo + query_offset_in_view
    q_end = q_hi + query_offset_in_view
    return (q_start[:, None] < k_hi[None, :]) & (q_end[:, None] > k_lo[None, :])
