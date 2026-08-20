# SPDX-License-Identifier: Apache-2.0

import json
import pathlib

import numpy as np
import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    visible_key_segments as causal_key_segments,
)
from sglang.multimodal_gen.runtime.models.dits.rolling_forcing_wanvideo import (
    compute_rolling_cache_layout,
)
from sglang.multimodal_gen.runtime.models.dits.rolling_forcing_wanvideo import (
    visible_key_segments as rolling_key_segments,
)
from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
    TEMPORAL_BUCKETS,
    ChunkAttentionEvent,
    attention_mass_by_frame,
    dense_chunk_scores,
    segment_frame_ids,
    spatial_displacement_mass,
    temporal_bucket_ids,
)

FRAME_SEQLEN = 8
BLOCK_FRAMES = 3
BLOCK_TOKENS = BLOCK_FRAMES * FRAME_SEQLEN
CACHE_SIZE = 24 * FRAME_SEQLEN
MAX_ATTENTION_TOKENS = 21 * FRAME_SEQLEN
WINDOW_BLOCKS = 5


def test_segment_frame_ids_maps_disjoint_ranges():
    ids = segment_frame_ids(
        [(0, BLOCK_TOKENS), (5 * BLOCK_TOKENS, 2 * BLOCK_TOKENS)],
        frame_seqlen=FRAME_SEQLEN,
        device=torch.device("cpu"),
    )
    assert ids.shape == (3 * BLOCK_TOKENS,)
    # first block -> latent frames 0,1,2; the far segment -> frames 15..20
    assert (
        ids[:BLOCK_TOKENS].tolist()
        == [0] * FRAME_SEQLEN + [1] * FRAME_SEQLEN + [2] * FRAME_SEQLEN
    )
    assert ids[BLOCK_TOKENS:].min().item() == 15
    assert ids[BLOCK_TOKENS:].max().item() == 20


def test_attention_mass_by_frame_matches_reference_softmax():
    torch.manual_seed(0)
    num_chunks, num_frames, heads, head_dim = 3, 9, 4, 16
    query = torch.randn(2 * BLOCK_TOKENS, heads, head_dim)
    key = torch.randn(num_frames * FRAME_SEQLEN, heads, head_dim)
    query_chunk_ids = torch.arange(query.shape[0]) // BLOCK_TOKENS
    key_frame_ids = torch.arange(key.shape[0]) // FRAME_SEQLEN

    mass, counts = attention_mass_by_frame(
        query=query,
        key=key,
        query_chunk_ids=query_chunk_ids,
        key_frame_ids=key_frame_ids,
        num_chunks=num_chunks,
        num_frames=num_frames,
        query_tile=7,  # deliberately not a divisor of the query length
    )

    probs = torch.softmax(
        torch.einsum("qhd,khd->hqk", query, key) * head_dim**-0.5, dim=-1
    )  # [heads, queries, keys]
    per_frame = probs.reshape(heads, query.shape[0], num_frames, FRAME_SEQLEN).sum(-1)
    expected = torch.stack(
        [per_frame[:, query_chunk_ids == chunk].sum(dim=1) for chunk in range(2)]
    )

    assert mass.shape == (num_chunks, heads, num_frames)
    assert counts[:2].tolist() == [BLOCK_TOKENS, BLOCK_TOKENS]
    assert counts[2] == 0
    torch.testing.assert_close(mass[:2], expected, atol=1e-4, rtol=1e-4)
    # every head of every query distributes exactly one unit of attention mass
    torch.testing.assert_close(
        mass[:2].sum(dim=2), counts[:2, None].expand(2, heads), atol=1e-3, rtol=1e-3
    )


def test_dense_chunk_scores_pads_growing_rows():
    events = [
        ChunkAttentionEvent(
            pass_kind="denoise",
            query_chunk=1,
            layer_index=layer,
            step_index=step,
            scores=torch.tensor([[0.4, 0.6], [0.5, 0.5]]),  # [heads, frames]
        )
        for layer in range(2)
        for step in range(3)
    ]
    dense = dense_chunk_scores(events, num_frames=4)
    assert dense.shape == (3, 2, 2, 4)  # steps, layers, heads, frames
    assert np.allclose(dense[..., 0, :2], np.array([0.4, 0.6]))
    assert np.allclose(dense[..., 1, :2], np.array([0.5, 0.5]))
    assert np.isnan(dense[..., 2:]).all()


def _position_tagged_kv(start: int, length: int) -> torch.Tensor:
    """K/V whose leading feature is the global token index."""
    tokens = torch.arange(start, start + length, dtype=torch.float32)
    return tokens.reshape(1, length, 1, 1).expand(1, length, 2, 4).contiguous()


def test_causal_key_segments_track_global_positions_across_eviction():
    cache_tokens = 6 * BLOCK_TOKENS
    cache = CausalSelfAttentionKVCache(
        k=torch.zeros(1, cache_tokens, 2, 4),
        v=torch.zeros(1, cache_tokens, 2, 4),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        cache_size=cache_tokens,
    )
    for block in range(10):  # more blocks than the cache holds -> eviction
        start = block * BLOCK_TOKENS
        tagged = _position_tagged_kv(start, BLOCK_TOKENS)
        view = cache.update_and_get_attention_kv(
            key=tagged,
            value=tagged,
            current_chunk_start=start,
        )
        segments = causal_key_segments(cache, view)
        assert segments is not None
        positions = torch.cat(
            [
                torch.arange(seg_start, seg_start + seg_len, dtype=torch.float32)
                for seg_start, seg_len in segments
            ]
        )
        # the probe's reconstructed global positions must equal the tags the
        # cache actually returns for this chunk's attention
        torch.testing.assert_close(positions, view.k[0, :, 0, 0])


def test_causal_key_segments_track_sink_cache_across_eviction():
    """A sink cache is locally contiguous but globally disjoint once it evicts."""
    cache_tokens = 6 * BLOCK_TOKENS
    cache = CausalSelfAttentionKVCache(
        k=torch.zeros(1, cache_tokens, 2, 4),
        v=torch.zeros(1, cache_tokens, 2, 4),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        cache_size=cache_tokens,
        sink_tokens=2 * BLOCK_TOKENS,
    )
    saw_disjoint = False
    for block in range(10):  # more blocks than the cache holds -> eviction
        start = block * BLOCK_TOKENS
        tagged = _position_tagged_kv(start, BLOCK_TOKENS)
        view = cache.update_and_get_attention_kv(
            key=tagged,
            value=tagged,
            current_chunk_start=start,
        )
        segments = causal_key_segments(cache, view)
        assert segments is not None
        saw_disjoint |= len(segments) > 1
        positions = torch.cat(
            [
                torch.arange(seg_start, seg_start + seg_len, dtype=torch.float32)
                for seg_start, seg_len in segments
            ]
        )
        # the probe's reconstructed global positions must equal the tags the
        # cache actually returns for this chunk's attention
        torch.testing.assert_close(positions, view.k[0, :, 0, 0])
        # the sink must still be the very first tokens of the video
        assert positions[0].item() == 0.0
    assert saw_disjoint, "eviction should have split the sink from the rolled window"


def test_causal_key_segments_skip_pinned_and_global_sink_caches():
    """Layouts carrying a third, history-dependent slice stay unmapped."""
    cache = CausalSelfAttentionKVCache(
        k=torch.zeros(1, 4 * BLOCK_TOKENS, 2, 4),
        v=torch.zeros(1, 4 * BLOCK_TOKENS, 2, 4),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        cache_size=4 * BLOCK_TOKENS,
        global_sink_tokens=BLOCK_TOKENS,
    )
    tagged = _position_tagged_kv(0, BLOCK_TOKENS)
    view = cache.update_and_get_attention_kv(
        key=tagged, value=tagged, current_chunk_start=0
    )
    assert causal_key_segments(cache, view) is None


def _rolling_schedule(num_blocks: int):
    """Layouts of one denoise + one cache-update pass per rolling window."""
    global_end = local_end = 0
    for window_index in range(num_blocks + WINDOW_BLOCKS - 1):
        start_block = max(0, window_index - WINDOW_BLOCKS + 1)
        end_block = min(num_blocks - 1, window_index)
        current_start = start_block * BLOCK_TOKENS
        window_tokens = (end_block + 1 - start_block) * BLOCK_TOKENS
        for updating in (False, True):
            q_tokens = BLOCK_TOKENS if updating else window_tokens
            layout = compute_rolling_cache_layout(
                global_end_index=global_end,
                local_end_index=local_end,
                cache_size=CACHE_SIZE,
                current_start=current_start,
                block_tokens=BLOCK_TOKENS,
                sink_tokens=BLOCK_TOKENS,
                max_attention_tokens=MAX_ATTENTION_TOKENS,
                q_tokens=q_tokens,
                frame_seqlen=FRAME_SEQLEN,
                num_frames_per_block=BLOCK_FRAMES,
                updating_cache=updating,
            )
            global_end = layout.global_end_after
            local_end = layout.local_end_after
            yield current_start, q_tokens, layout


def _expected_key_tokens(layout, *, q_tokens: int) -> int:
    """Key length the rolling attention actually builds for this layout."""
    if layout.local_start_index == 0:
        return q_tokens
    working = layout.working_end - layout.working_start
    if layout.updating_cache:
        return working
    return BLOCK_TOKENS + working + q_tokens


def test_rolling_key_segments_cover_exactly_the_attended_keys():
    num_blocks = 20
    for current_start, q_tokens, layout in _rolling_schedule(num_blocks):
        segments = rolling_key_segments(
            layout,
            sink_tokens=BLOCK_TOKENS,
            block_tokens=BLOCK_TOKENS,
            current_start=current_start,
            num_query_tokens=q_tokens,
        )
        assert sum(length for _, length in segments) == _expected_key_tokens(
            layout, q_tokens=q_tokens
        )
        for seg_start, seg_len in segments:
            assert seg_start >= 0
            # never claims to see tokens beyond the end of the current window
            assert seg_start + seg_len <= current_start + q_tokens
        # the denoising pass always prepends the re-roped sink (block 0); the
        # cache-update pass drops it as soon as the cache buffer is full
        if layout.local_start_index > 0 and not layout.updating_cache:
            assert segments[0] == (0, BLOCK_TOKENS)


def test_rolling_key_segments_frame_ids_are_ordered_and_bounded():
    num_blocks = 20
    for current_start, q_tokens, layout in _rolling_schedule(num_blocks):
        segments = rolling_key_segments(
            layout,
            sink_tokens=BLOCK_TOKENS,
            block_tokens=BLOCK_TOKENS,
            current_start=current_start,
            num_query_tokens=q_tokens,
        )
        ids = segment_frame_ids(
            segments, frame_seqlen=FRAME_SEQLEN, device=torch.device("cpu")
        )
        assert ids.min() >= 0
        assert ids.max() < num_blocks * BLOCK_FRAMES
        # the working cache is contiguous in chunk space and strictly newer
        # than the sink block it follows
        assert (ids[1:] >= ids[:-1]).all()


def test_spatial_displacement_matches_reference_softmax():
    """The tiled (dt, dy, dx) scatter must equal a per-(query, key) reference."""
    torch.manual_seed(0)
    grid_height, grid_width = 4, 5
    frame_seqlen = grid_height * grid_width
    heads, head_dim = 3, 8
    query_positions = torch.tensor(
        [
            2 * frame_seqlen,
            2 * frame_seqlen + 7,
            2 * frame_seqlen + 13,
            3 * frame_seqlen + 1,
            3 * frame_seqlen + 19,
        ]
    )
    key_positions = torch.arange(4 * frame_seqlen)
    query = torch.randn(query_positions.numel(), heads, head_dim)
    key = torch.randn(key_positions.numel(), heads, head_dim)

    accumulator = torch.zeros(
        heads, len(TEMPORAL_BUCKETS), 2 * grid_height - 1, 2 * grid_width - 1
    )
    recorded = spatial_displacement_mass(
        query=query,
        key=key,
        query_positions=query_positions,
        key_positions=key_positions,
        frame_seqlen=frame_seqlen,
        grid_height=grid_height,
        grid_width=grid_width,
        accumulator=accumulator,
        # a tile smaller than the per-frame query count exercises the tiling
        query_tile=2,
    )

    expected = torch.zeros_like(accumulator)
    probs = torch.softmax(
        torch.einsum("qhd,khd->hqk", query, key).float() * head_dim**-0.5, dim=-1
    )
    for q_index, q_position in enumerate(query_positions.tolist()):
        q_frame, q_spatial = divmod(q_position, frame_seqlen)
        q_y, q_x = divmod(q_spatial, grid_width)
        for k_index, k_position in enumerate(key_positions.tolist()):
            k_frame, k_spatial = divmod(k_position, frame_seqlen)
            k_y, k_x = divmod(k_spatial, grid_width)
            bucket = int(temporal_bucket_ids(torch.tensor(q_frame - k_frame)))
            expected[
                :, bucket, k_y - q_y + grid_height - 1, k_x - q_x + grid_width - 1
            ] += probs[:, q_index, k_index]

    assert recorded == query_positions.numel()
    torch.testing.assert_close(accumulator, expected, atol=1e-5, rtol=1e-4)
    # every query row is a distribution, so the buckets together sum to one per head
    torch.testing.assert_close(
        accumulator.sum(dim=(1, 2, 3)),
        torch.full((heads,), float(query_positions.numel())),
        atol=1e-4,
        rtol=1e-4,
    )


def test_temporal_bucket_ids_separate_past_present_and_future():
    deltas = torch.tensor([-3, -1, 0, 1, 2, 3, 9])
    buckets = temporal_bucket_ids(deltas)
    assert buckets.tolist() == [4, 4, 0, 1, 2, 3, 3]


def test_attention_mass_by_token_matches_reference_and_frame_reduction():
    """Per-token mass must equal the reference and reduce to the frame mass."""
    torch.manual_seed(0)
    frame_seqlen, heads, head_dim = 20, 3, 8
    num_queries, num_keys = 7, 60
    query = torch.randn(num_queries, heads, head_dim)
    key = torch.randn(num_keys, heads, head_dim)
    query_chunk_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    key_frame_ids = torch.arange(num_keys) // frame_seqlen

    token_mass = torch.zeros(2, heads, num_keys)
    mass, counts = attention_mass_by_frame(
        query=query,
        key=key,
        query_chunk_ids=query_chunk_ids,
        key_frame_ids=key_frame_ids,
        num_chunks=2,
        num_frames=3,
        token_mass=token_mass,
        # a tile smaller than a chunk exercises the tiled accumulation
        query_tile=3,
    )

    probs = torch.softmax(
        torch.einsum("qhd,khd->hqk", query, key).float() * head_dim**-0.5, dim=-1
    )
    expected = torch.zeros_like(token_mass)
    for query_index in range(num_queries):
        expected[query_chunk_ids[query_index]] += probs[:, query_index]
    torch.testing.assert_close(token_mass, expected, atol=1e-6, rtol=1e-5)

    # summing the token axis by frame must reproduce the coarse frame mass
    reduced = torch.zeros_like(mass)
    reduced.index_add_(2, key_frame_ids, token_mass)
    torch.testing.assert_close(reduced, mass, atol=1e-5, rtol=1e-4)
    # every query row is a distribution, so a chunk's rows sum to its query count
    torch.testing.assert_close(token_mass.sum(-1)[:, 0], counts, atol=1e-4, rtol=1e-4)


def test_parse_head_spec_flat_and_per_layer():
    from sglang.multimodal_gen.runtime.utils.attention_map_probe import parse_head_spec

    assert parse_head_spec(None) is None
    assert parse_head_spec("") is None
    # a flat list applies to every layer, stored under the -1 wildcard
    assert parse_head_spec("0,3,7") == {-1: (0, 3, 7)}
    assert parse_head_spec("0:1,2;29:9,11") == {0: (1, 2), 29: (9, 11)}


def test_selected_heads_falls_back_to_the_wildcard_and_clamps():
    from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
        ChunkAttentionRecorder,
    )

    recorder = ChunkAttentionRecorder(
        output_dir="/tmp", qk_heads={-1: (0, 5), 3: (1, 99)}
    )
    assert recorder.selected_heads(0, num_heads=12) == (0, 5)
    # a per-layer entry wins over the wildcard, and out-of-range ids are dropped
    assert recorder.selected_heads(3, num_heads=12) == (1,)
    assert ChunkAttentionRecorder(output_dir="/tmp").selected_heads(0, 12) is None


def test_qk_dump_records_only_the_selected_heads(tmp_path):
    from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
        ChunkAttentionRecorder,
    )

    heads, head_dim, queries = 8, 4, 2 * FRAME_SEQLEN
    recorder = ChunkAttentionRecorder(
        output_dir=str(tmp_path),
        query_stride=1,
        qk_chunks=frozenset({0}),
        qk_key_stride=1,
        qk_steps=frozenset({0}),
        qk_heads={-1: (1, 6)},
        qk_only=True,
    )
    recorder.begin_forward(
        frame_seqlen=FRAME_SEQLEN,
        num_frames_per_block=2,
        query_token_start=0,
        grid_height=2,
        grid_width=FRAME_SEQLEN // 2,
    )
    torch.manual_seed(0)
    recorder.record(
        layer_index=0,
        query=torch.randn(1, queries, heads, head_dim),
        key=torch.randn(1, queries, heads, head_dim),
        key_segments=((0, queries),),
    )
    run_dir = pathlib.Path(recorder.flush(model_tag="Fake"))

    dump = np.load(run_dir / "qk_chunk_000_step_0.npz")
    # [layers, heads, queries, keys] -- only the two selected heads
    assert dump["scores"].shape == (1, 2, queries, queries)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["qk_head_ids"] == {"0": [1, 6]}
    assert meta["qk_only"] is True
    # qk-only mode skips the per-frame mass pass, so no chunk_*.npz is written
    assert not list(run_dir.glob("chunk_*.npz"))


def test_qk_only_still_matches_a_reference_softmax(tmp_path):
    from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
        ChunkAttentionRecorder,
    )

    heads, head_dim, queries = 4, 6, FRAME_SEQLEN
    torch.manual_seed(1)
    query = torch.randn(1, queries, heads, head_dim)
    key = torch.randn(1, queries, heads, head_dim)
    recorder = ChunkAttentionRecorder(
        output_dir=str(tmp_path),
        query_stride=1,
        qk_chunks=frozenset({0}),
        qk_key_stride=1,
        qk_steps=frozenset({0}),
        qk_heads={-1: (2,)},
        qk_only=True,
    )
    recorder.begin_forward(
        frame_seqlen=FRAME_SEQLEN,
        num_frames_per_block=1,
        query_token_start=0,
        grid_height=1,
        grid_width=FRAME_SEQLEN,
    )
    recorder.record(layer_index=0, query=query, key=key, key_segments=((0, queries),))
    run_dir = pathlib.Path(recorder.flush(model_tag="Fake"))
    dumped = np.load(run_dir / "qk_chunk_000_step_0.npz")["scores"][0, 0]

    scale = head_dim**-0.5
    expected = torch.softmax((query[0, :, 2] @ key[0, :, 2].T).float() * scale, dim=-1)
    np.testing.assert_allclose(dumped, expected.numpy(), atol=2e-3)
