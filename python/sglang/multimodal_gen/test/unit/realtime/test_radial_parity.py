# SPDX-License-Identifier: Apache-2.0
"""Radial Attention: parity against the upstream mask construction.

Compared against ``reference/radial_reference.py``, which is a verbatim copy of
mit-han-lab/radial-attention @ 72788d4. The comparison is exact where it can be
and property-based where it cannot, and the boundary between the two is the
point of this file:

*Exact* equality is only well defined when the block grid of the two settings
coincides. Upstream blocks one whole clip starting at token 0; we block the
current chunk, whose first token is at ``chunk_index × frames_per_block ×
frame_seqlen``. When ``frame_seqlen`` is a multiple of the block size both grids
line up and every block of our mask must equal the corresponding block of
upstream's square mask. At Wan's real 1560-token frame they do not line up, so
there the tests pin the *properties* that define the method instead — band
widths, frame decimation, and the sink.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.radial import (
    RadialConfig,
    build_radial_block_mask,
    radial_frame_is_kept,
    radial_window_width,
    shrink_mask_strict,
)

from .reference import radial_reference

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

BLOCK = 128
# Aligned geometry: 256-token frames are exactly two blocks, so our per-chunk
# block grid coincides with upstream's whole-clip grid.
ALIGNED_FRAME = 256
FRAMES_PER_BLOCK = 3
WAN_FRAME = 1560  # 480x832 Wan latent frame; deliberately not block-aligned


def _layout(*, frame_seqlen, chunk_index, visible_frames, frames_per_block=FRAMES_PER_BLOCK):
    chunk_tokens = frames_per_block * frame_seqlen
    query_start = chunk_index * chunk_tokens
    first_visible = query_start + chunk_tokens - visible_frames * frame_seqlen
    geometry = ChunkGeometry(
        frame_seqlen=frame_seqlen,
        frames_per_block=frames_per_block,
        query_token_start=query_start,
        grid_height=1,
        grid_width=frame_seqlen,
    )
    return visible_layout(
        ((first_visible, visible_frames * frame_seqlen),),
        geometry=geometry,
        query_tokens=chunk_tokens,
    )


# --------------------------------------------------------------------------
# Piece-by-piece equality with the upstream helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("frame_seqlen", [256, 1560, 3600])
@pytest.mark.parametrize("decay_factor", [1.0, 0.5, 0.25])
def test_window_width_matches_upstream(frame_seqlen, decay_factor):
    for distance in range(0, 40):
        ours = radial_window_width(
            distance,
            frame_seqlen=frame_seqlen,
            decay_factor=decay_factor,
            block=BLOCK,
        )
        theirs = radial_reference.get_window_width(
            0,
            distance,
            frame_seqlen,
            "radial",
            num_frame=64,
            decay_factor=decay_factor,
            block_size=BLOCK,
            model_type="wan",
        )
        assert ours == pytest.approx(float(theirs)), distance


@requires_cuda
@pytest.mark.parametrize("frame_seqlen", [256, 1560])
def test_frame_decimation_matches_upstream(frame_seqlen):
    """`get_diagonal_split_mask` returns all-ones or all-zeros; we return a bool.

    Deliberately not parametrized over decay_factor: upstream's decimation does
    not take one, and discovering that was the point of the whole-mask test.
    """
    device = torch.device("cuda")
    for distance in range(0, 40):
        ours = radial_frame_is_kept(distance, frame_seqlen=frame_seqlen, block=BLOCK)
        theirs = radial_reference.get_diagonal_split_mask(
            0, distance, frame_seqlen, "radial", torch.zeros(1, device=device)
        )
        assert ours == bool(theirs.all()), distance


@requires_cuda
def test_shrink_mask_strict_matches_upstream():
    torch.manual_seed(0)
    device = torch.device("cuda")
    for density in (0.05, 0.3, 0.9):
        mask = torch.rand(4 * BLOCK, 4 * BLOCK, device=device) < density
        torch.testing.assert_close(
            shrink_mask_strict(mask, block=BLOCK).int(),
            radial_reference.shrinkMaskStrict(mask, block_size=BLOCK).int(),
        )
    # And on a realistic banded mask, which is what it is actually fed.
    rows = torch.arange(4 * BLOCK, device=device).view(-1, 1)
    columns = torch.arange(4 * BLOCK, device=device).view(1, -1)
    banded = (columns - rows).abs() <= 100
    torch.testing.assert_close(
        shrink_mask_strict(banded, block=BLOCK).int(),
        radial_reference.shrinkMaskStrict(banded, block_size=BLOCK).int(),
    )


# --------------------------------------------------------------------------
# Whole-mask equality on the aligned geometry
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("visible_frames", [6, 12, 21])
@pytest.mark.parametrize("decay_factor", [1.0, 0.5])
def test_block_mask_matches_upstream_slice_when_grids_align(
    visible_frames, decay_factor
):
    """Our per-chunk mask must equal the matching block-rows of upstream's clip mask.

    Upstream is bidirectional over the whole clip; the causal setting needs only
    the block-rows belonging to the chunk being generated, against the columns of
    the visible window. With a block-aligned frame the two are the same object.
    """
    device = torch.device("cuda")
    frame_seqlen = ALIGNED_FRAME
    # Put the query chunk at the end of the visible window, which is what the
    # sliding KV cache always hands us.
    chunk_index = visible_frames // FRAMES_PER_BLOCK - 1
    layout = _layout(
        frame_seqlen=frame_seqlen,
        chunk_index=chunk_index,
        visible_frames=visible_frames,
    )
    config = RadialConfig(block=BLOCK, decay_factor=decay_factor)
    ours = build_radial_block_mask(
        layout=layout,
        q_len=layout.query_frames * frame_seqlen,
        kv_len=layout.kv_len,
        config=config,
        device=device,
    )

    total_tokens = visible_frames * frame_seqlen
    theirs = radial_reference.gen_log_mask_shrinked(
        torch.zeros(1, total_tokens, 1, 1, device=device),
        total_tokens,
        total_tokens,
        visible_frames,
        block_size=BLOCK,
        sparse_type="radial",
        decay_factor=decay_factor,
        model_type="wan",
    )
    query_block_start = (
        layout.num_frames - layout.query_frames
    ) * frame_seqlen // BLOCK
    expected = theirs[query_block_start:]
    # Our mask additionally pins the query block's own diagonal; upstream gets
    # the same blocks from its d=0 full-frame case, so on an aligned grid the two
    # must already agree.
    torch.testing.assert_close(ours.int(), expected.int())


# --------------------------------------------------------------------------
# Properties at the real Wan geometry, where the grids cannot align
# --------------------------------------------------------------------------


@requires_cuda
def test_wan_geometry_band_narrows_and_frames_decimate():
    device = torch.device("cuda")
    layout = _layout(frame_seqlen=WAN_FRAME, chunk_index=6, visible_frames=21)
    mask = build_radial_block_mask(
        layout=layout,
        q_len=layout.query_frames * WAN_FRAME,
        kv_len=layout.kv_len,
        config=RadialConfig(block=BLOCK, decay_factor=1.0),
        device=device,
    )
    key_frame_of_block = (
        torch.arange(mask.shape[1], device=device) * BLOCK // WAN_FRAME
    )
    query_frame = layout.num_frames - layout.query_frames

    density = {}
    for distance in (1, 2, 4, 8):
        columns = key_frame_of_block == (query_frame - distance)
        if columns.any():
            density[distance] = mask[:, columns].float().mean().item()
    assert density[1] > density[2] >= density[4] >= density[8], density
    # The sink frame (video frame 0) is not in a 21-frame window at chunk 6, so
    # nothing should be fully dense at long range.
    assert density[8] < 0.5, density


@requires_cuda
def test_sink_frame_is_dense_while_it_is_still_visible():
    device = torch.device("cuda")
    layout = _layout(frame_seqlen=WAN_FRAME, chunk_index=4, visible_frames=15)
    assert layout.global_frame_ids[0] == 0  # frame 0 still in the window
    mask = build_radial_block_mask(
        layout=layout,
        q_len=layout.query_frames * WAN_FRAME,
        kv_len=layout.kv_len,
        config=RadialConfig(block=BLOCK, decay_factor=1.0),
        device=device,
    )
    sink_columns = torch.arange(mask.shape[1], device=device) * BLOCK < WAN_FRAME
    assert mask[:, sink_columns].all()


@requires_cuda
def test_decay_factor_is_monotone_in_density():
    device = torch.device("cuda")
    layout = _layout(frame_seqlen=WAN_FRAME, chunk_index=8, visible_frames=21)
    densities = []
    for decay_factor in (2.0, 1.0, 0.5, 0.25):
        mask = build_radial_block_mask(
            layout=layout,
            q_len=layout.query_frames * WAN_FRAME,
            kv_len=layout.kv_len,
            config=RadialConfig(block=BLOCK, decay_factor=decay_factor),
            device=device,
        )
        densities.append(mask.float().mean().item())
    assert densities == sorted(densities, reverse=True), densities
