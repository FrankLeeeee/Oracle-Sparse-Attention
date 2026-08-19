# SPDX-License-Identifier: Apache-2.0
"""Sparse VideoGen: parity against the upstream masks and cluster selection.

Compared against ``reference/svg_reference.py``, verbatim from
svg-project/Sparse-VideoGen @ f89aeda:

* ``svg/models/wan/utils.py::get_attention_mask`` — SVG1's two candidate masks
* ``svg/kmeans_utils.py::weighted_softmax`` and ``identify_dynamic_map`` — SVG2's
  top-p cluster selection

Two structural differences between the settings bound what can be compared, and
both are asserted rather than assumed:

**SVG1's masks are defined at token level over a whole clip.** At Wan's geometry
that is a 32760² bool tensor (1 GB), so production builds the same two bands
directly at block granularity. The tests therefore check token-level equality
with upstream on a small clip, and check that the block-level builder is the
any-overlap reduction of exactly those token masks.

**SVG2 groups queries by cluster; our kernel tiles them at 128.** Upstream scores
each query *cluster* centroid against the key centroids; we score the mean of
each 128-query block of the cluster-sorted queries. The selection *rule* is
compared exactly (same scores in, same clusters out); the grouping is a
documented adaptation, and the test pins that a cluster-homogeneous block gives
the same answer as upstream's cluster.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.svg import (
    Svg1Config,
    build_svg1_segment_masks,
    select_clusters_by_top_p,
    svg1_token_masks,
)

from .reference import svg_reference

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

BLOCK = 128
FRAMES_PER_BLOCK = 3
SMALL_FRAME = 256  # token-level masks are tractable and block-aligned
WAN_FRAME = 1560


def _layout(*, frame_seqlen, visible_frames, chunk_index):
    chunk_tokens = FRAMES_PER_BLOCK * frame_seqlen
    query_start = chunk_index * chunk_tokens
    first_visible = query_start + chunk_tokens - visible_frames * frame_seqlen
    geometry = ChunkGeometry(
        frame_seqlen=frame_seqlen,
        frames_per_block=FRAMES_PER_BLOCK,
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
# SVG1: token-level masks against upstream
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("num_frames", [4, 6, 9])
def test_svg1_token_masks_match_upstream(num_frames):
    """Both candidate masks, token for token, against ``get_attention_mask``."""
    device = torch.device("cuda")
    frame_seqlen = SMALL_FRAME
    total = num_frames * frame_seqlen
    ours_spatial, ours_temporal = svg1_token_masks(
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        band_frames=2.0,
        dense_sink_frames=1,
        device=device,
    )
    for name, ours in (("spatial", ours_spatial), ("temporal", ours_temporal)):
        theirs = svg_reference.get_attention_mask(
            name,
            total,  # sample_mse_max_row: keep every row
            0,  # context_length: Wan self-attention carries no text tokens
            num_frames,
            frame_seqlen,
        ).bool()
        torch.testing.assert_close(ours.int(), theirs.int(), msg=name)


@requires_cuda
def test_svg1_sink_columns_land_where_upstream_puts_them():
    """Upstream's sink is applied *before* the temporal permutation.

    ``get_attention_mask`` sets ``mask[:, :frame_size] = 1`` and only then
    permutes for the temporal head, so the two masks get different sinks:

    * spatial — the first frame, as one would expect;
    * temporal — the columns whose *spatial-major* index is below
      ``frame_seqlen``, i.e. the lowest ``frame_seqlen / num_frames`` spatial
      cells of every frame.

    Pinned because it is easy to "fix" into a first-frame sink and because
    omitting the sink entirely is what made this baseline's video collapse into
    rainbow artifacts by chunk 8.
    """
    device = torch.device("cuda")
    num_frames = 6
    spatial, temporal = svg1_token_masks(
        num_frames=num_frames,
        frame_seqlen=SMALL_FRAME,
        band_frames=2.0,
        dense_sink_frames=1,
        device=device,
    )
    assert spatial[:, :SMALL_FRAME].all()

    tokens = torch.arange(num_frames * SMALL_FRAME, device=device)
    spatial_major = (tokens % SMALL_FRAME) * num_frames + tokens // SMALL_FRAME
    assert temporal[:, spatial_major < SMALL_FRAME].all()
    # ... and that is genuinely not the first frame.
    assert not temporal[:, :SMALL_FRAME].all()


@requires_cuda
@pytest.mark.parametrize("visible_frames", [6, 9])
def test_svg1_segment_masks_are_the_any_overlap_reduction(visible_frames):
    """The segment builder must equal the tile reduction of the token masks.

    This is what licenses using the segment builder at Wan's geometry, where
    the token masks cannot be materialized. The temporal mask's executed rows
    are the *spatial-major permuted* queries (upstream's head placement,
    applied to the query side), so its reduction pools the token mask's rows
    in that order.
    """
    device = torch.device("cuda")
    frame_seqlen = SMALL_FRAME
    chunk_index = visible_frames // FRAMES_PER_BLOCK - 1
    layout = _layout(
        frame_seqlen=frame_seqlen,
        visible_frames=visible_frames,
        chunk_index=chunk_index,
    )
    q_len = FRAMES_PER_BLOCK * frame_seqlen
    kv_len = layout.kv_len
    config = Svg1Config(block=BLOCK, band_frames=2.0, dense_sink_frames=1)
    ours_spatial, ours_temporal, permutation = build_svg1_segment_masks(
        layout=layout, q_len=q_len, kv_len=kv_len, config=config, device=device
    )

    token_masks = svg1_token_masks(
        num_frames=visible_frames,
        frame_seqlen=frame_seqlen,
        band_frames=2.0,
        dense_sink_frames=1,
        device=device,
    )
    natural = torch.arange(q_len, device=device)
    tile = config.key_tile
    for ours_mask, token_mask, order in zip(
        (ours_spatial, ours_temporal),
        token_masks,
        (natural, permutation),
        strict=True,
    ):
        # Query rows of the current chunk, in the executed order, max-pooled
        # onto the frame-aligned tile grid (aligned here: 256 % 32 == 0).
        rows = token_mask[kv_len - q_len :][order]
        pooled = (
            rows.view(q_len // BLOCK, BLOCK, kv_len // tile, tile)
            .amax(dim=1)
            .amax(dim=-1)
        )
        torch.testing.assert_close(ours_mask.int(), pooled.int())


@requires_cuda
def test_svg1_band_frames_is_monotone_in_density_at_wan_geometry():
    device = torch.device("cuda")
    layout = _layout(frame_seqlen=WAN_FRAME, visible_frames=21, chunk_index=10)
    previous = None
    for band_frames in (4.0, 2.0, 1.0, 0.5):
        spatial, temporal, _ = build_svg1_segment_masks(
            layout=layout,
            q_len=FRAMES_PER_BLOCK * WAN_FRAME,
            kv_len=layout.kv_len,
            config=Svg1Config(block=BLOCK, band_frames=band_frames),
            device=device,
        )
        density = (spatial.float().mean() + temporal.float().mean()).item() / 2
        if previous is not None:
            assert density < previous, band_frames
        previous = density


# --------------------------------------------------------------------------
# SVG2: cluster selection against upstream
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("top_p", [0.3, 0.6, 0.9, 0.99])
def test_svg2_cluster_selection_matches_upstream(top_p):
    """Same centroids and cluster sizes in, same clusters out.

    Upstream weights the centroid softmax by cluster size (``weighted_softmax``);
    we add ``log(size)`` to the logits, which is the same distribution. The
    boundary convention matters too: upstream keeps the cluster that *crosses*
    ``top_p`` (it shifts its removal mask right by one), so the kept set is the
    descending prefix whose exclusive cumulative mass is ``<= top_p``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, query_clusters, key_clusters, dim = 3, 5, 32, 64
    query_centroids = torch.randn(1, heads, query_clusters, dim, device=device)
    key_centroids = torch.randn(1, heads, key_clusters, dim, device=device)
    key_sizes = torch.randint(
        1, 400, (1, heads, key_clusters), device=device, dtype=torch.float32
    )

    theirs = svg_reference.identify_dynamic_map(
        query_centroids,
        key_centroids,
        None,  # q_cluster_sizes is unused by the function
        key_sizes,
        top_p,
    )[0]

    logits = (
        query_centroids[0] @ key_centroids[0].transpose(-2, -1)
    ) / dim**0.5 + key_sizes[0].log()[:, None, :]
    ours = select_clusters_by_top_p(logits, top_p=top_p)
    torch.testing.assert_close(ours.int(), theirs.int())


@requires_cuda
def test_svg2_selection_always_keeps_the_strongest_cluster():
    """Upstream forces ``remove_indices[..., 0] = False``; nothing is ever empty."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    logits = torch.randn(2, 4, 16, device=device) * 10
    for top_p in (0.0, 0.01, 0.5):
        keep = select_clusters_by_top_p(logits, top_p=top_p)
        assert (keep.sum(-1) >= 1).all(), top_p
