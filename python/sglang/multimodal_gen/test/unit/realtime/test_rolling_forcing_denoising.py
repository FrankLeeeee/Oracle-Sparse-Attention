# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.models.dits.rolling_forcing_wanvideo import (
    compute_rolling_cache_layout,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.rolling_forcing_denoising import (
    build_rolling_window_bounds,
    build_staggered_timesteps,
)

FRAME_SEQLEN = 8
BLOCK_FRAMES = 3
BLOCK_TOKENS = BLOCK_FRAMES * FRAME_SEQLEN
CACHE_SIZE = 24 * FRAME_SEQLEN
MAX_ATTENTION_TOKENS = 21 * FRAME_SEQLEN
WINDOW_BLOCKS = 5


def test_window_bounds_ramp_full_drain():
    bounds = build_rolling_window_bounds(num_blocks=7, window_length_blocks=5)
    assert len(bounds) == 7 + 5 - 1
    # ramp-up: window grows from one block anchored at block 0
    assert bounds[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    # full windows slide by one block
    assert bounds[5] == (1, 5)
    assert bounds[6] == (2, 6)
    # draining: start keeps advancing, end pinned at the last block
    assert bounds[7:] == [(3, 6), (4, 6), (5, 6), (6, 6)]


def test_staggered_timesteps_oldest_block_cleanest():
    timesteps = torch.tensor([1000.0, 800.0, 600.0, 400.0, 200.0])
    shared = build_staggered_timesteps(timesteps, batch_size=2, num_frames_per_block=3)
    assert shared.shape == (2, 15)
    # oldest window position holds the last (cleanest) denoising step
    assert shared[0, :3].eq(200.0).all()
    assert shared[0, -3:].eq(1000.0).all()
    per_block = shared[1].reshape(5, 3)[:, 0]
    assert per_block.tolist() == [200.0, 400.0, 600.0, 800.0, 1000.0]


def _simulate_schedule(num_blocks: int):
    """Run the full window schedule through the layout math, mirroring one
    denoise pass + one cache-update pass per window."""
    global_end = local_end = 0
    layouts = []
    for window_index in range(num_blocks + WINDOW_BLOCKS - 1):
        start_block = max(0, window_index - WINDOW_BLOCKS + 1)
        end_block = min(num_blocks - 1, window_index)
        current_start = start_block * BLOCK_TOKENS
        q_tokens = (end_block + 1 - start_block) * BLOCK_TOKENS
        for updating in (False, True):
            pass_q_tokens = BLOCK_TOKENS if updating else q_tokens
            layout = compute_rolling_cache_layout(
                global_end_index=global_end,
                local_end_index=local_end,
                cache_size=CACHE_SIZE,
                current_start=current_start,
                block_tokens=BLOCK_TOKENS,
                sink_tokens=BLOCK_TOKENS,
                max_attention_tokens=MAX_ATTENTION_TOKENS,
                q_tokens=pass_q_tokens,
                frame_seqlen=FRAME_SEQLEN,
                num_frames_per_block=BLOCK_FRAMES,
                updating_cache=updating,
            )
            global_end = layout.global_end_after
            local_end = layout.local_end_after
            layouts.append((window_index, updating, pass_q_tokens, layout))
    return layouts


def test_layout_schedule_invariants():
    layouts = _simulate_schedule(num_blocks=12)
    for window_index, updating, _, layout in layouts:
        # the write range is exactly one block inside the buffer
        assert layout.local_end_index - layout.local_start_index == BLOCK_TOKENS
        assert 0 <= layout.local_start_index
        assert layout.local_end_index <= CACHE_SIZE
        # only the denoise pass advances the cache, once per window
        if updating:
            assert layout.num_new_tokens == 0
        # the sink block region is never part of the rolled range
        if layout.num_evicted_tokens > 0:
            assert layout.num_rolled_tokens >= 0
            assert layout.local_end_index == CACHE_SIZE

    # the final global end covers the whole video
    final_layout = layouts[-1][3]
    assert final_layout.global_end_after == 12 * BLOCK_TOKENS


def test_layout_ramp_up_uses_plain_window_attention():
    layouts = _simulate_schedule(num_blocks=8)
    for window_index, _, _, layout in layouts:
        if window_index < WINDOW_BLOCKS:
            # windows still anchored at block 0 rewrite the sink slots and
            # attend purely within the window
            assert layout.local_start_index == 0
            assert layout.anchor_start_frame == -1
        elif not layout.updating_cache:
            # once the window moves past block 0 the sink is re-roped to sit
            # immediately before the working cache
            assert layout.local_start_index > 0
            assert layout.anchor_start_frame >= 0
            working_frames = (layout.working_end - layout.working_start) // FRAME_SEQLEN
            current_start_frame = (
                layout.global_end_after - BLOCK_TOKENS
            ) // FRAME_SEQLEN
            assert (
                layout.anchor_start_frame
                == current_start_frame - working_frames - BLOCK_FRAMES
            )


def test_layout_attention_context_is_bounded():
    layouts = _simulate_schedule(num_blocks=20)
    for _, updating, q_tokens, layout in layouts:
        if layout.local_start_index == 0:
            continue
        if updating:
            context = layout.working_end - layout.working_start
            assert context <= MAX_ATTENTION_TOKENS
        else:
            working = layout.working_end - layout.working_start
            # sink + working cache + current window <= max attention context
            assert BLOCK_TOKENS + working + q_tokens <= MAX_ATTENTION_TOKENS
