# SPDX-License-Identifier: Apache-2.0
"""Reference block selection for Light Forcing.

From https://github.com/chengtao-lv/LightForcing @ d1e6333 (2026-08-05), files
``wan/modules/sparse_attention.py`` (``calculate_chunk_sparsities``) and
``wan/modules/kernel.py`` (``get_sm_80_120_block_map_1stage``,
``get_sm_80_120_block_map_2stage``, ``_select_2stage_middle_frames``).

The four functions above are VERBATIM COPIES. Their two helpers are Triton
kernels upstream (``mean_pool_blhd``, ``score_full_selected_blocks_from_frames``);
they are transcribed here as PyTorch line for line from the kernel bodies —
``block_mean_kernel`` accumulates a block's rows in float32, divides by the
*actual* row count of the (possibly short) last block and stores back in the
input dtype; ``full_selected_block_score_from_frames_kernel`` marks a key block
eligible when it is in the tail (``frame_id >= F_past``), the sink
(``frame_id < KEEP_SINK``), the near window
(``F_past - KEEP_NEAR <= frame_id < F_past``) or a stage-1 kept frame
(``frame_id == keep[i] + KEEP_OFFSET``), scores eligible blocks by the pooled
dot product and ineligible ones by ``-inf``.

Do not "improve" anything in this file. If upstream changes, re-copy it.
"""

import math

import torch


def mean_pool_blhd(x, BLK):
    """Transcription of ``block_mean_kernel``: (B, L, H, D) -> (B, L_BLOCKS, H, D)."""
    B, L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    pad = L_BLOCKS * BLK - L
    padded = torch.nn.functional.pad(x.float(), (0, 0, 0, 0, 0, pad))
    sums = padded.view(B, L_BLOCKS, BLK, H, D).sum(dim=2)
    counts = torch.full((L_BLOCKS,), BLK, device=x.device, dtype=torch.float32)
    counts[-1] = L - (L_BLOCKS - 1) * BLK
    return (sums / counts[None, :, None, None]).to(x.dtype)


def score_full_selected_blocks_from_frames(
    pooled_qblocks,
    pooled_kblocks,
    keep_idx,
    frame_blk,
    f_past,
    keep_offset=0,
    keep_sink=0,
    keep_near=0,
):
    """Transcription of ``full_selected_block_score_from_frames_kernel``.

    Returns ``(B, H, Q_BLOCKS, K_BLOCKS)`` scores with ineligible blocks at
    ``-inf``. ``keep_idx`` is ``(B, H, Q_BLOCKS, KEEP_FRAMES)``.
    """
    B, Q, H, D = pooled_qblocks.shape
    K = pooled_kblocks.shape[1]
    device = pooled_qblocks.device

    offs_k = torch.arange(K, device=device)
    past_blocks = f_past * frame_blk
    is_tail = offs_k >= past_blocks
    frame_id = offs_k // frame_blk

    is_sink = frame_id < keep_sink
    is_near = (frame_id >= f_past - keep_near) & (frame_id < f_past)
    is_keep = (is_tail | is_sink | is_near)[None, None, None, :].expand(B, H, Q, K)
    keep_frame = keep_idx + keep_offset  # (B, H, Q, KEEP_FRAMES)
    is_keep = is_keep | (frame_id[None, None, None, None, :] == keep_frame[..., None]).any(
        dim=-2
    )

    score = torch.einsum(
        "bqhd,bkhd->bhqk", pooled_qblocks.float(), pooled_kblocks.float()
    ).to(pooled_qblocks.dtype)
    return torch.where(is_keep, score, torch.full_like(score, -float("inf")))


# ---------------------------------------------------------------------------
# VERBATIM from wan/modules/sparse_attention.py
# ---------------------------------------------------------------------------


def calculate_chunk_sparsities(num_output_frames, num_frame_per_block, local_attn_size=21, sparse_config=None):
    sparse_config = sparse_config or {}
    target_sparsity = sparse_config.get("sparsity", None)
    base_sparsity = sparse_config.get("sparsity_base", target_sparsity)
    if target_sparsity is None:
        return []

    target_sparsity = float(target_sparsity)
    base_sparsity = float(base_sparsity)
    chunk_frame_counts = range(
        2 * num_frame_per_block,
        num_output_frames + 1,
        num_frame_per_block,
    )
    kv_lengths = [
        frame_count if local_attn_size == -1 else min(frame_count, local_attn_size)
        for frame_count in chunk_frame_counts
    ]
    alphas = [1 / math.sqrt(frame_count) for frame_count in chunk_frame_counts]

    target_flops = sum((1 - target_sparsity) * kv_length for kv_length in kv_lengths)
    base_flops = sum((1 - base_sparsity) * kv_length for kv_length in kv_lengths)
    alpha_weighted_flops = sum(
        alpha * kv_length
        for alpha, kv_length in zip(alphas, kv_lengths)
    )
    if alpha_weighted_flops == 0:
        return [base_sparsity] * len(alphas)

    beta = (target_flops - base_flops) / alpha_weighted_flops
    return  [0.0] + [
        base_sparsity - alpha * beta
        for alpha in alphas
    ]


# ---------------------------------------------------------------------------
# VERBATIM from wan/modules/kernel.py
# ---------------------------------------------------------------------------


def get_sm_80_120_block_map_1stage(q, k, topk_ratio, BLKQ=64, BLKK=64):
    # q, k: (B, L, H, D)
    pooled_qblocks = mean_pool_blhd(q, BLKQ)       # (B, M_BLOCKS, H, D)
    pooled_kblocks = mean_pool_blhd(k, BLKK)    # (B, N_BLOCKS, H, D)

    pooled_score = pooled_qblocks.transpose(1, 2) @ pooled_kblocks.permute(0, 2, 3, 1)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def _select_2stage_middle_frames(pooled_qblocks, pooled_kblocks, frame_blk, f_past, keep_frames, keep_sink, keep_near):
    if keep_sink < 0 or keep_near < 0:
        raise ValueError("keep_sink and keep_near must be non-negative.")
    if keep_sink + keep_near > keep_frames:
        raise ValueError("keep_sink + keep_near must be <= keep_frames.")

    B, Q, H, _ = pooled_qblocks.shape
    middle_start = keep_sink
    middle_end = f_past - keep_near
    middle_frames = middle_end - middle_start
    keep_middle = keep_frames - keep_sink - keep_near

    if keep_middle == 0:
        return torch.empty((B, H, Q, 0), device=pooled_qblocks.device, dtype=torch.int64)

    pooled_middle_frames = (
        pooled_kblocks[:, middle_start * frame_blk:middle_end * frame_blk]
        .reshape(B, middle_frames, frame_blk, H, -1)
        .mean(dim=2)
    )
    pooled_frame_score = pooled_qblocks.transpose(1, 2) @ pooled_middle_frames.permute(0, 2, 3, 1)
    return torch.topk(pooled_frame_score, keep_middle, dim=-1, largest=True, sorted=False).indices


def get_sm_80_120_block_map_2stage(q, k, topk_ratio, BLKQ=64, BLKK=64, frame_seq=1536, keep_frames=6, keep_sink=0, keep_near=0):
    # q, k: (B, L, H, D)
    pooled_qblocks = mean_pool_blhd(q, BLKQ)       # (B, M_BLOCKS, H, D)
    pooled_kblocks = mean_pool_blhd(k, BLKK)       # (B, N_BLOCKS, H, D)

    K = pooled_kblocks.shape[1]
    frame_blk = frame_seq // BLKK
    F = K // frame_blk
    num_frame_per_block = q.shape[1] // frame_seq
    F_past = F - num_frame_per_block

    keep_idx = _select_2stage_middle_frames(pooled_qblocks, pooled_kblocks, frame_blk, F_past, keep_frames, keep_sink, keep_near)
    pooled_score = score_full_selected_blocks_from_frames(pooled_qblocks, pooled_kblocks, keep_idx, frame_blk, F_past, keep_sink, keep_sink, keep_near)

    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk
