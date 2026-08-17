# SPDX-License-Identifier: Apache-2.0
"""Light Forcing: parity against the upstream schedule and block selection.

Compared against ``reference/lightforcing_reference.py``, from
chengtao-lv/LightForcing @ d1e6333:

* ``calculate_chunk_sparsities`` — the chunk-aware sparsity schedule
* ``get_sm_80_120_block_map_1stage`` / ``_2stage`` — the block top-k, plain and
  hierarchical

One structural difference bounds what can be compared exactly. Upstream's key
blocks are a fixed grid over the flat key axis and its frame-level stage
requires ``frame_seq % BLKK == 0`` (its geometry is 1536-token frames);
production blocks are frame-aligned so the method survives Wan 480p's
1560-token frames. The two partitions coincide exactly when the frame length
divides by the block, so exact-equality tests run at a 256-token frame and the
Wan-geometry tests assert properties (per-row budget, frame alignment) rather
than upstream equality — upstream's own code cannot run there.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse import (
    build_sparse_attention_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import plan_key_mask
from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (
    LightForcingConfig,
    calculate_chunk_sparsities,
    frame_aligned_block_bounds,
    lightforcing_block_mask,
    mean_pool_blocks,
)

from .reference import lightforcing_reference

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

BLOCK = 64
FRAMES_PER_BLOCK = 3
SMALL_FRAME = 256  # divides by BLOCK: upstream's partition and ours coincide
WAN_FRAME = 1560
HEADS = 4
HEAD_DIM = 64


def _layout(*, frame_seqlen, chunk_index):
    chunk_tokens = FRAMES_PER_BLOCK * frame_seqlen
    geometry = ChunkGeometry(
        frame_seqlen=frame_seqlen,
        frames_per_block=FRAMES_PER_BLOCK,
        query_token_start=chunk_index * chunk_tokens,
        grid_height=1,
        grid_width=frame_seqlen,
    )
    return visible_layout(
        ((0, (chunk_index + 1) * chunk_tokens),),
        geometry=geometry,
        query_tokens=chunk_tokens,
    )


def _pooled(tensor, *, block, frame_seqlen=None):
    """Production pooling of ``[1, len, heads, dim]`` → ``[heads, blocks, dim]``."""
    return mean_pool_blocks(
        tensor[0], block=block, group=frame_seqlen
    ).permute(1, 0, 2)


# --------------------------------------------------------------------------
# The chunk-aware sparsity schedule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_output_frames,frames_per_block,local_attn_size,sparsity,sparsity_base",
    [
        (21, 3, 21, 0.88, 0.98),  # short model's shipped config
        (63, 3, 12, 0.85, 0.95),  # long model's shipped config
        (21, 3, -1, 0.88, 0.98),
        (9, 3, -1, 0.5, 0.5),
        (40, 4, 16, 0.9, 0.9),
    ],
)
def test_chunk_sparsities_match_upstream(
    num_output_frames, frames_per_block, local_attn_size, sparsity, sparsity_base
):
    ours = calculate_chunk_sparsities(
        num_output_frames=num_output_frames,
        frames_per_block=frames_per_block,
        local_attn_size=local_attn_size,
        sparsity=sparsity,
        sparsity_base=sparsity_base,
    )
    theirs = lightforcing_reference.calculate_chunk_sparsities(
        num_output_frames,
        frames_per_block,
        local_attn_size,
        {"sparsity": sparsity, "sparsity_base": sparsity_base},
    )
    assert ours == pytest.approx(theirs)
    assert ours[0] == 0.0
    # Later chunks carry more sparsity: the schedule must be non-decreasing
    # past the dense first chunk whenever base >= target.
    if sparsity_base >= sparsity:
        assert all(a <= b + 1e-9 for a, b in zip(ours[1:], ours[2:]))


def test_chunk_sparsities_flat_when_no_chunks():
    """One-chunk videos have no past-KV chunks; upstream returns the base."""
    ours = calculate_chunk_sparsities(
        num_output_frames=3,
        frames_per_block=3,
        local_attn_size=-1,
        sparsity=0.9,
        sparsity_base=0.98,
    )
    assert ours == [0.0]


# --------------------------------------------------------------------------
# Block selection against upstream, at upstream-compatible geometry
# --------------------------------------------------------------------------


def _mask_pair(*, chunk_index, topk_ratio, keep_frames, keep_sink, keep_near):
    """(production mask, upstream map) for the same random Q/K, fp32."""
    torch.manual_seed(chunk_index)
    device = torch.device("cuda")
    frame_seqlen = SMALL_FRAME
    q_len = FRAMES_PER_BLOCK * frame_seqlen
    kv_len = (chunk_index + 1) * q_len
    query = torch.randn(1, q_len, HEADS, HEAD_DIM, device=device)
    key = torch.randn(1, kv_len, HEADS, HEAD_DIM, device=device)

    num_frames = kv_len // frame_seqlen
    blocks_per_frame = frame_seqlen // BLOCK
    kv_blocks = num_frames * blocks_per_frame
    topk = min(kv_blocks, int(topk_ratio * kv_blocks))
    ours = lightforcing_block_mask(
        pooled_query=_pooled(query, block=BLOCK),
        pooled_key=_pooled(key, block=BLOCK, frame_seqlen=frame_seqlen),
        blocks_per_frame=blocks_per_frame,
        past_frames=num_frames - FRAMES_PER_BLOCK,
        keep_frames=keep_frames,
        keep_sink=keep_sink,
        keep_near=keep_near,
        topk=topk,
    )

    past_frames = num_frames - FRAMES_PER_BLOCK
    if past_frames > keep_frames:
        theirs, _, ref_topk = lightforcing_reference.get_sm_80_120_block_map_2stage(
            query,
            key,
            topk_ratio,
            BLKQ=BLOCK,
            BLKK=BLOCK,
            frame_seq=frame_seqlen,
            keep_frames=keep_frames,
            keep_sink=keep_sink,
            keep_near=keep_near,
        )
    else:
        theirs, _, ref_topk = lightforcing_reference.get_sm_80_120_block_map_1stage(
            query, key, topk_ratio, BLKQ=BLOCK, BLKK=BLOCK
        )
    assert ref_topk == topk
    return ours, theirs[0].bool()


@requires_cuda
@pytest.mark.parametrize("chunk_index", [1, 2])
def test_1stage_mask_matches_upstream(chunk_index):
    """``past_frames <= keep_frames``: plain top-k over every block."""
    ours, theirs = _mask_pair(
        chunk_index=chunk_index,
        topk_ratio=0.3,
        keep_frames=6,
        keep_sink=1,
        keep_near=2,
    )
    torch.testing.assert_close(ours.int(), theirs.int())


@requires_cuda
@pytest.mark.parametrize("chunk_index", [3, 5])
@pytest.mark.parametrize("keep_sink,keep_near", [(1, 1), (0, 0), (1, 2)])
def test_2stage_mask_matches_upstream(chunk_index, keep_sink, keep_near):
    """``past_frames > keep_frames``: hierarchical frame stage, then top-k.

    ``topk_ratio`` is chosen so the per-row budget fits inside the eligible
    blocks — beyond that both sides pad the selection with arbitrary ``-inf``
    blocks and equality is not well-defined.
    """
    keep_frames = 3
    blocks_per_frame = SMALL_FRAME // BLOCK
    eligible_frames = keep_frames + FRAMES_PER_BLOCK
    kv_blocks = (chunk_index + 1) * FRAMES_PER_BLOCK * blocks_per_frame
    topk_ratio = 0.8 * eligible_frames * blocks_per_frame / kv_blocks
    ours, theirs = _mask_pair(
        chunk_index=chunk_index,
        topk_ratio=topk_ratio,
        keep_frames=keep_frames,
        keep_sink=keep_sink,
        keep_near=keep_near,
    )
    torch.testing.assert_close(ours.int(), theirs.int())


@requires_cuda
def test_frame_aligned_bounds_match_fixed_grid_when_divisible():
    device = torch.device("cuda")
    lo, hi = frame_aligned_block_bounds(
        num_frames=4, frame_seqlen=SMALL_FRAME, block=BLOCK, device=device
    )
    expected = torch.arange(4 * SMALL_FRAME // BLOCK, device=device) * BLOCK
    torch.testing.assert_close(lo, expected)
    torch.testing.assert_close(hi, expected + BLOCK)


def test_frame_aligned_bounds_never_straddle_frames():
    lo, hi = frame_aligned_block_bounds(
        num_frames=5, frame_seqlen=WAN_FRAME, block=BLOCK, device=torch.device("cpu")
    )
    assert int(hi[-1]) == 5 * WAN_FRAME
    assert bool((lo < hi).all())
    assert bool((lo // WAN_FRAME == (hi - 1) // WAN_FRAME).all())


# --------------------------------------------------------------------------
# The backend end to end, at real Wan geometry
# --------------------------------------------------------------------------


def _wan_call_and_layout(*, chunk_index, device):
    from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
        SparseAttentionCall,
    )

    layout = _layout(frame_seqlen=WAN_FRAME, chunk_index=chunk_index)
    q_len = FRAMES_PER_BLOCK * WAN_FRAME
    kv_len = (chunk_index + 1) * q_len
    query = torch.randn(1, q_len, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, kv_len, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    call = SparseAttentionCall(
        layer_index=0,
        query=query,
        key=key,
        value=torch.randn_like(key),
        key_segments=((0, kv_len),),
        head_start=0,
        num_local_heads=HEADS,
        softmax_scale=HEAD_DIM**-0.5,
    )
    return call, layout


@requires_cuda
def test_backend_budget_and_alignment_at_wan_geometry():
    """Each query block keeps exactly its top-k budget of frame-aligned blocks."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(
        "lightforcing", {"block_q": BLOCK, "block_k": BLOCK}
    )
    chunk_index = 4
    call, layout = _wan_call_and_layout(chunk_index=chunk_index, device=device)
    execution = backend.prepare(call, layout)
    assert execution is not None

    kv_len = call.key.shape[1]
    key_mask = plan_key_mask(execution.plan, kv_len=kv_len)
    schedule = calculate_chunk_sparsities(
        num_output_frames=21,
        frames_per_block=FRAMES_PER_BLOCK,
        local_attn_size=-1,
        sparsity=0.88,
        sparsity_base=0.98,
    )
    blocks_per_frame = -(-WAN_FRAME // BLOCK)
    kv_blocks = layout.num_frames * blocks_per_frame
    topk = int((1.0 - schedule[chunk_index]) * kv_blocks)

    lo, hi = frame_aligned_block_bounds(
        num_frames=layout.num_frames,
        frame_seqlen=WAN_FRAME,
        block=BLOCK,
        device=device,
    )
    block_kept = key_mask[:, :, lo]  # every block is kept or dropped whole
    sizes = (hi - lo)[None, None, :]
    torch.testing.assert_close(
        key_mask.long().sum(-1), (block_kept.long() * sizes).sum(-1)
    )
    assert bool((block_kept.long().sum(-1) == topk).all())


@requires_cuda
def test_backend_declines_dense_and_first_chunks():
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend("lightforcing", {})
    # Chunk 0 is scheduled dense; chunk index past the schedule clamps to the
    # last entry instead of upstream's IndexError.
    call, layout = _wan_call_and_layout(chunk_index=0, device=device)
    assert backend.prepare(call, layout) is None
    call, layout = _wan_call_and_layout(chunk_index=11, device=device)
    assert backend.prepare(call, layout) is not None


def test_config_rejects_bad_budgets():
    with pytest.raises(ValueError, match="keep_sink"):
        build_sparse_attention_backend(
            "lightforcing", {"keep_frames": 2, "keep_sink": 2, "keep_near": 1}
        )
    with pytest.raises(ValueError, match="power of two"):
        build_sparse_attention_backend("lightforcing", {"block_q": 48})
