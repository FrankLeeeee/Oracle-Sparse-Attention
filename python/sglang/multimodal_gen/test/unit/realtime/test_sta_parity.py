# SPDX-License-Identifier: Apache-2.0
"""Sliding Tile Attention: parity against the upstream mask construction.

Compared against ``reference/sta_reference.py``, a verbatim copy of the flex
reference mask from hao-ai-lab/fastvideo @ 98f761e. Upstream's canvas is one
bidirectional clip in tile-major token order; our causal adaptation takes the
visible frame axis as the temporal canvas and the current chunk's frames as
the query rows. When the visible view is a contiguous prefix of the video the
two settings describe the same canvas, so our tile mask must equal the
query-row slice of upstream's square mask exactly — tile for tile, with
``tile_t_size = 1`` and no text tokens.

The executed plan additionally quantizes query rows on ``config.block``; that
reduction (union over the tiles a block covers) is checked separately, and the
kernel-level numerics are covered by the generic
``test_method_output_matches_its_own_plan`` in ``test_sparse_attention.py``.
"""

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.sta import (
    StaConfig,
    build_sta_tile_mask,
    pick_tile,
    tile_major_permutation,
)

from .reference import sta_reference


def _layout(
    *,
    grid_h: int,
    grid_w: int,
    chunk_index: int,
    visible_frames: int,
    frames_per_block: int = 3,
):
    frame_seqlen = grid_h * grid_w
    chunk_tokens = frames_per_block * frame_seqlen
    query_start = chunk_index * chunk_tokens
    first_visible = query_start + chunk_tokens - visible_frames * frame_seqlen
    geometry = ChunkGeometry(
        frame_seqlen=frame_seqlen,
        frames_per_block=frames_per_block,
        query_token_start=query_start,
        grid_height=grid_h,
        grid_width=grid_w,
    )
    layout = visible_layout(
        ((first_visible, visible_frames * frame_seqlen),),
        geometry=geometry,
        query_tokens=chunk_tokens,
    )
    assert layout is not None
    return layout


def _upstream_token_mask(
    *, canvas_t, grid_h, grid_w, kernel, tile_h, tile_w
) -> torch.Tensor:
    """Upstream's full ``[tokens, tokens]`` mask in tile-major order."""
    mask_fn = sta_reference.generate_sta_mask(
        (canvas_t, grid_h, grid_w),
        kernel,
        (1, tile_h, tile_w),
        text_length=0,
    )
    total = canvas_t * grid_h * grid_w
    idx = torch.arange(total)
    zero = torch.zeros((), dtype=torch.long)
    return mask_fn(zero, zero, idx[:, None], idx[None, :])


def _tile_mask_to_token_mask(tile_mask: np.ndarray, tile_tokens: int) -> torch.Tensor:
    """Expand a ``[q_tiles, k_tiles]`` mask to tile-major token resolution."""
    return torch.from_numpy(
        np.repeat(np.repeat(tile_mask, tile_tokens, axis=0), tile_tokens, axis=1)
    )


@pytest.mark.parametrize(
    "kernel",
    [(3, 3, 3), (1, 1, 1), (5, 3, 1), (7, 5, 5), (4, 2, 3)],
)
def test_tile_mask_matches_upstream_on_full_history(kernel):
    """Contiguous-prefix view: our mask == the row slice of upstream's mask.

    Canvas 9 frames of an 8x8 grid in 2x2 tiles, query = the last 3-frame
    chunk. Includes even kernels, whose effective window is the next odd one
    in upstream's ``abs(center - x) <= k // 2`` arithmetic — the copy must
    reproduce that too.
    """
    grid_h = grid_w = 8
    tile_h = tile_w = 2
    visible_frames = 9
    layout = _layout(
        grid_h=grid_h, grid_w=grid_w, chunk_index=2, visible_frames=visible_frames
    )
    config = StaConfig(
        kernel_t=kernel[0],
        kernel_h=kernel[1],
        kernel_w=kernel[2],
        tile_h=tile_h,
        tile_w=tile_w,
    )
    ours = build_sta_tile_mask(
        layout=layout,
        grid_h=grid_h,
        grid_w=grid_w,
        tile_h=tile_h,
        tile_w=tile_w,
        config=config,
    )
    ours_tokens = _tile_mask_to_token_mask(ours, tile_h * tile_w)

    upstream = _upstream_token_mask(
        canvas_t=visible_frames,
        grid_h=grid_h,
        grid_w=grid_w,
        kernel=kernel,
        tile_h=tile_h,
        tile_w=tile_w,
    )
    q_len = 3 * grid_h * grid_w
    torch.testing.assert_close(ours_tokens, upstream[-q_len:], atol=0, rtol=0)


def test_sliding_window_view_shifts_the_canvas():
    """A sliding-window view is upstream's mask on the *visible* canvas.

    With a 21-frame cap the visible axis no longer starts at video frame 0;
    the canvas is the window itself, so the mask must equal upstream's built
    for a 21-frame canvas — not for the whole video.
    """
    grid_h = grid_w = 4
    tile_h = tile_w = 2
    layout = _layout(grid_h=grid_h, grid_w=grid_w, chunk_index=10, visible_frames=21)
    assert layout.global_frame_ids[0] > 0  # genuinely mid-video
    config = StaConfig(kernel_t=5, kernel_h=3, kernel_w=3, tile_h=2, tile_w=2)
    ours = build_sta_tile_mask(
        layout=layout,
        grid_h=grid_h,
        grid_w=grid_w,
        tile_h=tile_h,
        tile_w=tile_w,
        config=config,
    )
    upstream = _upstream_token_mask(
        canvas_t=21,
        grid_h=grid_h,
        grid_w=grid_w,
        kernel=(5, 3, 3),
        tile_h=tile_h,
        tile_w=tile_w,
    )
    q_len = 3 * grid_h * grid_w
    torch.testing.assert_close(
        _tile_mask_to_token_mask(ours, 4), upstream[-q_len:], atol=0, rtol=0
    )


def test_multi_chunk_query_rows_match_upstream():
    """Rolling-Forcing-shaped call: each query frame centers its own window.

    A 5-chunk (15-frame) query over a 21-frame view must produce exactly the
    last 15 frame-rows of upstream's square mask — the newest frame clamps at
    the canvas end while the middle frames get centered windows.
    """
    grid_h = grid_w = 4
    frame_seqlen = grid_h * grid_w
    frames_per_block = 3
    query_frames = 15
    visible_frames = 21
    geometry = ChunkGeometry(
        frame_seqlen=frame_seqlen,
        frames_per_block=frames_per_block,
        query_token_start=6 * frame_seqlen,
        grid_height=grid_h,
        grid_width=grid_w,
    )
    layout = visible_layout(
        ((0, visible_frames * frame_seqlen),),
        geometry=geometry,
        query_tokens=query_frames * frame_seqlen,
    )
    assert layout is not None
    config = StaConfig(kernel_t=5, kernel_h=3, kernel_w=3, tile_h=2, tile_w=2)
    ours = build_sta_tile_mask(
        layout=layout,
        grid_h=grid_h,
        grid_w=grid_w,
        tile_h=2,
        tile_w=2,
        config=config,
    )
    upstream = _upstream_token_mask(
        canvas_t=visible_frames,
        grid_h=grid_h,
        grid_w=grid_w,
        kernel=(5, 3, 3),
        tile_h=2,
        tile_w=2,
    )
    q_len = query_frames * frame_seqlen
    torch.testing.assert_close(
        _tile_mask_to_token_mask(ours, 4), upstream[-q_len:], atol=0, rtol=0
    )


def test_kernel_covering_the_canvas_keeps_everything():
    grid_h = grid_w = 4
    layout = _layout(grid_h=grid_h, grid_w=grid_w, chunk_index=1, visible_frames=6)
    config = StaConfig(kernel_t=7, kernel_h=3, kernel_w=3, tile_h=2, tile_w=2)
    ours = build_sta_tile_mask(
        layout=layout, grid_h=grid_h, grid_w=grid_w, tile_h=2, tile_w=2, config=config
    )
    assert ours.all()


def test_tile_major_permutation_matches_upstream_indexing():
    """``perm[new] = old`` must land every token in its upstream canvas slot.

    Upstream token index ``i`` decomposes as tile ``i // tile_tokens`` (tiles
    row-major over the tile grid) and offset ``i % tile_tokens`` (row-major
    within the tile); the permutation must map exactly that order onto the
    natural row-major frame.
    """
    grid_h, grid_w, tile_h, tile_w = 6, 8, 3, 2
    perm = tile_major_permutation(
        grid_h=grid_h, grid_w=grid_w, tile_h=tile_h, tile_w=tile_w
    )
    tiles_w = grid_w // tile_w
    tile_tokens = tile_h * tile_w
    for new_index, old_index in enumerate(perm):
        row, col = divmod(int(old_index), grid_w)
        tile_id = (row // tile_h) * tiles_w + col // tile_w
        within = (row % tile_h) * tile_w + col % tile_w
        assert new_index == tile_id * tile_tokens + within


def test_pick_tile_prefers_upstream_area_and_square():
    assert pick_tile(45, 80) == (9, 8)
    assert pick_tile(30, 52) == (5, 13)
    assert pick_tile(22, 40) == (11, 5)
    th, tw = pick_tile(1, 256)
    assert th == 1 and tw == 64
