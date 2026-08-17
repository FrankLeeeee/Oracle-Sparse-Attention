# SPDX-License-Identifier: Apache-2.0
"""Reference spatial/temporal masks for Sparse VideoGen (Wan).

VERBATIM COPY from https://github.com/svg-project/Sparse-VideoGen @ f89aeda (2026-07-03),
files ``svg/models/wan/utils.py`` lines 63-112 (``get_attention_mask``) and
``svg/kmeans_utils.py`` lines 854-862, 866-897 (SVG2's cluster selection). Only the module-level imports are omitted (upstream
imports CUDA extensions we do not have installed); every function body below is
byte-for-byte upstream so that a test against it is a test against upstream.

Do not "improve" anything in this file. If upstream changes, re-copy it.
"""

import math

import torch


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size):
    """
    Generate the emulated attention mask. Used for online profiling.
    """


    attention_mask = torch.zeros(
        (context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu"
    )

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = (
            pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
            .permute(1, 0, 3, 2)
            .reshape(frame_size * num_frame, frame_size * num_frame)
        )
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


def weighted_softmax(scores, weights):
    input_dtype = scores.dtype
    scores = scores.float()
    weights = weights.float()
    max_score = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_score)
    weighted_exp = weights * exp_scores
    softmax_out = weighted_exp / torch.sum(weighted_exp, dim=-1, keepdim=True).clamp(min=1e-12)
    return softmax_out.to(input_dtype)


def identify_dynamic_map(
    query_centroids,
    key_centroids,
    q_cluster_sizes,
    k_cluster_sizes,
    p,
    min_kc_ratio=0,
):
    B, H, qc_num, D = query_centroids.shape
    kc_num = key_centroids.shape[2]
    device = query_centroids.device

    attn_scores = torch.matmul(query_centroids, key_centroids.transpose(-2, -1)) / (D**0.5)
    k_weights = k_cluster_sizes.unsqueeze(-2).float()

    weighted_attn_probs = weighted_softmax(attn_scores, k_weights)
    sorted_probs, sorted_indices = torch.sort(weighted_attn_probs, dim=-1, descending=True)

    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    remove_indices = cumsum_probs > p
    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
    remove_indices[..., 0] = False

    if min_kc_ratio > 0:
        preserve_length = int(min_kc_ratio * kc_num)
        remove_indices[..., :preserve_length] = False

    sorted_clusters_to_keep = ~remove_indices

    dynamic_map = torch.zeros(B, H, qc_num, kc_num, dtype=torch.bool, device=device)
    dynamic_map.scatter_(-1, sorted_indices, sorted_clusters_to_keep)
    return dynamic_map
