# SPDX-License-Identifier: Apache-2.0
"""Correctness of the sparse-attention package.

Two things have to be true for any of these methods to be usable, and they are
tested separately:

1. the shared kernel computes *exactly* the attention its plan describes —
   checked against masked SDPA in float32 for every method's real plan, so a
   permutation, a merged key range or a logit bias that is wrong shows up here
   rather than as a subtly worse video;
2. each method's selection logic is the pattern it claims — OSA's frozen
   per-head tile set replicates across every history frame, X-Attention's
   estimator matches a brute-force antidiagonal sum, and so on.
"""

import msgspec
import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse import (
    build_sparse_attention_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    SparseAttentionCall,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.blocks import (
    block_bounds,
    intra_frame_coverage,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    frame_mask_to_ranges,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.fastar import (
    temporal_merge,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_block_mask,
    plan_from_shared_ranges,
    plan_key_mask,
    sparse_attention,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.radial import (
    RadialConfig,
    build_radial_block_mask,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.xattention import (
    select_blocks_by_cumulative_mass,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# Deliberately not a multiple of the kernel's 128-token key tile, and larger
# than it: a Wan 480p latent frame is 1560 tokens, and the intra-frame patterns
# (SVG temporal, Radial) only mean anything when a frame spans several blocks.
FRAME_SEQLEN = 390
FRAMES_PER_BLOCK = 3


def masked_reference(query, key, value, key_mask, *, block_m, softmax_scale):
    """float32 SDPA under a ``[heads, q_blocks, kv_len]`` key mask."""
    q_len = query.shape[1]
    rows = torch.arange(q_len, device=query.device) // block_m
    per_row = key_mask[:, rows.clamp(max=key_mask.shape[1] - 1), :]
    out = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2).float(),
        key.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
        attn_mask=per_row[None],
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


# --------------------------------------------------------------------------
# The shared kernel
# --------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_m", [64, 128])
def test_kernel_matches_masked_sdpa_shared_ranges(dtype, block_m):
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, heads, head_dim, num_frames = 2, 4, 64, 7
    q_len, kv_len = 2 * FRAME_SEQLEN, num_frames * FRAME_SEQLEN
    query = torch.randn(batch, q_len, heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, kv_len, heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, kv_len, heads, head_dim, device=device, dtype=dtype)

    rng = np.random.default_rng(1)
    keep = rng.random((heads, num_frames)) < 0.5
    keep[:, -1] = True  # the own chunk is always kept in practice
    plan = plan_from_shared_ranges(
        frame_mask_to_ranges(keep, frame_seqlen=FRAME_SEQLEN),
        block_m=block_m,
        device=device,
    )
    scale = head_dim**-0.5
    out = sparse_attention(
        query=query, key=key, value=value, plan=plan, softmax_scale=scale
    )
    reference = masked_reference(
        query,
        key,
        value,
        plan_key_mask(plan, kv_len=kv_len),
        block_m=block_m,
        softmax_scale=scale,
    )
    torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)


@requires_cuda
def test_kernel_matches_masked_sdpa_per_query_block_and_bias():
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, heads, head_dim, block = 1, 3, 64, 128
    q_len, kv_len = 3 * block + 17, 9 * block - 5
    query = torch.randn(
        batch, q_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(
        batch, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)

    rng = torch.Generator(device=device).manual_seed(3)
    block_mask = torch.rand(heads, 4, 9, device=device, generator=rng) < 0.4
    block_mask[..., -1] = True
    bias = torch.randn(heads, kv_len, device=device, generator=rng)
    plan = plan_from_block_mask(
        block_mask, block_n=block, kv_len=kv_len, block_m=block, logit_bias=bias
    )
    scale = head_dim**-0.5
    out = sparse_attention(
        query=query, key=key, value=value, plan=plan, softmax_scale=scale
    )

    key_mask = plan_key_mask(plan, kv_len=kv_len)
    rows = torch.arange(q_len, device=device) // block
    scores = torch.einsum("qhd,khd->hqk", query[0].float(), key[0].float()) * scale
    scores = scores + bias[:, None, :]
    scores = scores.masked_fill(~key_mask[:, rows, :], float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ value[0].float().permute(1, 0, 2)
    torch.testing.assert_close(
        out[0].float(), reference.permute(1, 0, 2), atol=2e-2, rtol=2e-2
    )


@requires_cuda
def test_kernel_accepts_strided_cache_views():
    """KV views sliced out of a larger cache buffer are not contiguous in dim 1."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, heads, head_dim, num_frames = 1, 2, 64, 4
    kv_len = num_frames * FRAME_SEQLEN
    cache = torch.randn(
        batch,
        kv_len + 2 * FRAME_SEQLEN,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = cache[:, FRAME_SEQLEN : FRAME_SEQLEN + kv_len]
    value = cache[:, 2 * FRAME_SEQLEN : 2 * FRAME_SEQLEN + kv_len]
    query = torch.randn(
        batch, FRAME_SEQLEN, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    keep = np.array([[True, False, True, True], [False, True, False, True]])
    plan = plan_from_shared_ranges(
        frame_mask_to_ranges(keep, frame_seqlen=FRAME_SEQLEN),
        block_m=128,
        device=device,
    )
    scale = head_dim**-0.5
    out = sparse_attention(
        query=query, key=key, value=value, plan=plan, softmax_scale=scale
    )
    reference = masked_reference(
        query,
        key,
        value,
        plan_key_mask(plan, kv_len=kv_len),
        block_m=128,
        softmax_scale=scale,
    )
    torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)


@requires_cuda
def test_block_mask_merges_adjacent_ranges():
    mask = torch.tensor([[[True, True, True, False, True]]], device="cuda")
    plan = plan_from_block_mask(mask, block_n=8, kv_len=37, block_m=8)
    assert plan.range_counts.tolist() == [[2]]
    assert plan.range_starts[0, 0, :2].tolist() == [0, 32]
    assert plan.range_ends[0, 0, :2].tolist() == [24, 37]  # clamped to kv_len


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _geometry(chunk_index: int) -> ChunkGeometry:
    return ChunkGeometry(
        frame_seqlen=FRAME_SEQLEN,
        frames_per_block=FRAMES_PER_BLOCK,
        query_token_start=chunk_index * FRAMES_PER_BLOCK * FRAME_SEQLEN,
        grid_height=15,
        grid_width=26,
    )


def _layout(chunk_index: int, *, first_visible_chunk: int = 0):
    chunk_tokens = FRAMES_PER_BLOCK * FRAME_SEQLEN
    visible = chunk_index - first_visible_chunk + 1
    return visible_layout(
        ((first_visible_chunk * chunk_tokens, visible * chunk_tokens),),
        geometry=_geometry(chunk_index),
        query_tokens=chunk_tokens,
    )


def test_visible_layout_rejects_frame_misaligned_views():
    assert (
        visible_layout(
            ((0, FRAME_SEQLEN + 1),), geometry=_geometry(0), query_tokens=FRAME_SEQLEN
        )
        is None
    )


def test_frame_mask_to_ranges_merges_runs():
    keep = np.array([[True, True, False, True]])
    assert frame_mask_to_ranges(keep, frame_seqlen=10) == [[(0, 20), (30, 40)]]


def test_intra_frame_coverage_handles_blocks_that_straddle_frames():
    """A 1560-token frame is not a whole number of blocks; blocks wrap."""
    device = torch.device("cpu")
    lo, hi = block_bounds(35, 4, device=device)
    coverage = intra_frame_coverage(lo, hi, frame_seqlen=10)
    for index in range(len(lo)):
        expected = sorted(
            {token % 10 for token in range(int(lo[index]), int(hi[index]))}
        )
        assert torch.nonzero(coverage[index]).flatten().tolist() == expected
    # A block at least a frame long covers every offset.
    wide_lo, wide_hi = block_bounds(24, 12, device=device)
    assert intra_frame_coverage(wide_lo, wide_hi, frame_seqlen=10).all()


def test_radial_band_narrows_with_temporal_distance():
    layout = _layout(6)
    q_len = FRAMES_PER_BLOCK * FRAME_SEQLEN
    mask = build_radial_block_mask(
        layout=layout,
        q_len=q_len,
        kv_len=layout.kv_len,
        config=RadialConfig(block=64, dense_sink_frames=0, decay_factor=1.0),
        device=torch.device("cpu"),
    )
    block = 64
    key_frame = torch.arange(mask.shape[1]) * block // FRAME_SEQLEN
    query_frame = layout.num_frames - layout.query_frames
    widths = {}
    for distance in (1, 2, 4):
        columns = key_frame == (query_frame - distance)
        widths[distance] = mask[:, columns].float().mean().item()
    assert widths[1] > widths[2] > widths[4]


# --------------------------------------------------------------------------
# X-Attention
# --------------------------------------------------------------------------


# The estimator itself is covered against upstream in
# test_xattention_parity.py; only the selection rule is exercised here.


@requires_cuda
def test_cumulative_selection_takes_the_smallest_sufficient_prefix():
    scores = torch.tensor([[[0.5, 0.3, 0.15, 0.05]]], device="cuda")
    keep = select_blocks_by_cumulative_mass(scores, threshold=0.9)
    assert keep.tolist() == [[[True, True, True, False]]]


# --------------------------------------------------------------------------
# Every method, end to end against masked SDPA
# --------------------------------------------------------------------------


def _self_forcing_call(device, *, chunk_index, layer_index=0, heads=4, head_dim=64):
    """A synthetic attention call with Self-Forcing's chunk structure."""
    frames_per_block = FRAMES_PER_BLOCK
    chunk_tokens = frames_per_block * FRAME_SEQLEN
    kv_len = (chunk_index + 1) * chunk_tokens
    query = torch.randn(
        1, chunk_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(1, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    return SparseAttentionCall(
        layer_index=layer_index,
        query=query,
        key=key,
        value=value,
        key_segments=((0, kv_len),),
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
    )


@requires_cuda
@pytest.mark.parametrize(
    "method,config",
    [
        ("xattention", {"block": 128, "stride": 8, "threshold": 0.6}),
        ("svg1", {"block": 128, "num_sampled_blocks": 2, "band_frames": 0.5}),
        ("svg2", {"block": 128, "cluster_size": 32, "kmeans_iters": 2, "top_p": 0.5}),
        ("radial", {"block": 128}),
        ("sta", {"kernel_t": 3, "kernel_h": 3, "kernel_w": 3}),
        ("fastar", {"block": 128, "hash_bits": 4, "dense_steps": 0}),
        ("lightforcing", {"block_q": 64, "block_k": 64}),
    ],
)
def test_method_output_matches_its_own_plan(method, config):
    """Whatever a method selects, the kernel must compute exactly that.

    The reference is built from the plan the method produced, so this catches
    kernel bugs, permutation bugs and bias bugs without asserting anything
    about the selection policy itself. OSA is absent because it executes via
    gather + FA3 rather than the shared kernel; its execution is checked
    against a masked reference in its own test below.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(method, config)

    execution = None
    for chunk_index in range(2, 7):
        call = _self_forcing_call(device, chunk_index=chunk_index)
        backend.begin_forward(_geometry(chunk_index))
        layout = visible_layout(
            call.key_segments,
            geometry=_geometry(chunk_index),
            query_tokens=call.query.shape[1],
        )
        execution = backend.prepare(call, layout)
        if execution is not None:
            break
    assert execution is not None, f"{method} never produced a plan"

    plan = execution.plan
    kv_len = execution.key.shape[1]
    key_mask = plan_key_mask(plan, kv_len=kv_len)
    out = sparse_attention(
        query=execution.query,
        key=execution.key,
        value=execution.value,
        plan=plan,
        softmax_scale=call.softmax_scale,
    )
    q_len = execution.query.shape[1]
    rows = (torch.arange(q_len, device=device) // plan.block_m).clamp(
        max=key_mask.shape[1] - 1
    )
    scores = (
        torch.einsum(
            "qhd,khd->hqk", execution.query[0].float(), execution.key[0].float()
        )
        * call.softmax_scale
    )
    if plan.logit_bias is not None:
        scores = scores + plan.logit_bias[:, None, :]
    scores = scores.masked_fill(~key_mask[:, rows, :], float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ execution.value[0].float().permute(
        1, 0, 2
    )
    torch.testing.assert_close(
        out[0].float(), reference.permute(1, 0, 2), atol=3e-2, rtol=3e-2
    )
    # The selection must actually be sparse, or the comparison proves nothing.
    density = key_mask.float().mean().item()
    assert density < 0.99, f"{method} kept everything ({density:.3f})"


@requires_cuda
@pytest.mark.parametrize(
    "method,config",
    [
        ("osa", {"density": 0.4, "num_recent_frames": 1, "sink_latent_frames": 1}),
        ("xattention", {"threshold": 0.6}),
        ("svg1", {"num_sampled_blocks": 2, "band_frames": 0.5}),
        ("svg2", {"cluster_size": 32, "kmeans_iters": 2, "top_p": 0.5}),
        ("radial", {}),
        ("sta", {"kernel_t": 3}),
        ("fastar", {"hash_bits": 4, "dense_steps": 0}),
        ("lightforcing", {}),
    ],
)
def test_current_chunk_keys_are_never_cached_across_denoising_steps(method, config):
    """The chunk's own K/V are rewritten every step; caching them corrupts the run.

    Anything a method memoizes per chunk (a compacted cache, a clustering, a
    mask) has to be derived from the *history*, which is stable within a chunk.
    Memoizing the current chunk's keys as well is invisible in a single forward
    and destroys an autoregressive rollout, so it is checked directly: change
    only the own-chunk keys and the attention output must change.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(method, config)
    chunk_index = 5
    # Walk the earlier chunks so anything that calibrates has done so.
    for earlier in range(chunk_index):
        backend.begin_forward(_geometry(earlier))
        backend.attend(_self_forcing_call(device, chunk_index=earlier))
    geometry = _geometry(chunk_index)
    call = _self_forcing_call(device, chunk_index=chunk_index)

    outputs = []
    for step in range(2):
        if step == 1:
            own = call.query.shape[1]
            key = call.key.clone()
            value = call.value.clone()
            key[:, -own:] = torch.randn_like(key[:, -own:])
            value[:, -own:] = torch.randn_like(value[:, -own:])
            call = msgspec.structs.replace(call, key=key, value=value)
        backend.begin_forward(geometry)
        out = backend.attend(call)
        if out is None:
            pytest.skip(f"{method} runs dense at chunk {chunk_index}")
        outputs.append(out)
    assert not torch.equal(
        outputs[0], outputs[1]
    ), f"{method} reused stale current-chunk keys"


@requires_cuda
def test_lightforcing_pooled_history_resets_on_new_video():
    """A new video must not reuse the previous video's pooled history keys.

    The pooled-history cache is keyed by the KV layout signature, which says
    nothing about which video the keys came from: a new request whose first
    sparse chunk matches the previous video's last cached signature would
    silently select blocks against stale keys. A chunk-counter regression is
    the new-video signal (same rule as OSA's calibration reset).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(
        "lightforcing", {"num_output_frames": 81, "sparsity": 0.8}
    )
    for chunk_index in range(0, 3):
        backend.begin_forward(_geometry(chunk_index))
        backend.attend(_self_forcing_call(device, chunk_index=chunk_index))
    assert backend._pooled_history._entries, "cache never populated"

    backend.begin_forward(_geometry(0))  # chunk counter regressed: new video
    assert not backend._pooled_history._entries


@requires_cuda
def test_tempcache_merge_is_exact_for_identical_keys():
    """FAST-AR's Lemma 5.1: merging identical keys must change nothing.

    With the representative key, the group's mean value and a log(group size)
    logit bias, attention over the compacted cache has to equal attention over
    the original. If it does not, the bias or the value averaging is wrong, and
    the error would show up only as a slowly degrading video.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, frames, positions, head_dim = 2, 6, 5, 16
    repeated = torch.randn(heads, 1, positions, head_dim, device=device)
    keys = torch.cat(
        [
            repeated.expand(heads, 4, positions, head_dim),
            torch.randn(heads, 2, positions, head_dim, device=device),
        ],
        dim=1,
    ).contiguous()
    values = torch.randn(heads, frames, positions, head_dim, device=device)

    keep, group_size, merged = temporal_merge(keys, values, threshold=0.9)
    assert keep[0, :, 0].tolist() == [False, False, False, True, True, True]
    assert group_size[0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 1.0]

    query = torch.randn(heads, 3, head_dim, device=device)
    flat_keys = keys.reshape(heads, -1, head_dim)
    scale = head_dim**-0.5
    exact = torch.softmax(
        query @ flat_keys.transpose(1, 2) * scale, -1
    ) @ values.reshape(heads, -1, head_dim)
    scores = (
        query @ flat_keys.transpose(1, 2) * scale
        + group_size.reshape(heads, 1, -1).log()
    )
    scores = scores.masked_fill(~keep.reshape(heads, 1, -1), float("-inf"))
    compacted = torch.softmax(scores, -1) @ merged.reshape(heads, -1, head_dim)
    torch.testing.assert_close(compacted, exact, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_osa_fully_sparse_has_no_whole_frames_and_exact_density():
    """OSA 2-D with keep_whole_frames=False: no frame is kept whole — sink and
    own chunk get the frozen tile pattern like every other frame — and the
    achieved per-call density equals the knob (no floor)."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        BlockSparsePlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    query_tile = 128
    density = 0.3
    backend = build_sparse_attention_backend(
        "osa",
        {
            "density": density,
            "spatial_tile": 64,
            "calibration_query_stride": 4,
            "query_tile": query_tile,
            "keep_own_chunk_full": False,
            "keep_sink_full": False,
            "keep_recent_frames_full": False,
        },
    )

    backend.begin_forward(_geometry(0))
    assert backend.attend(_self_forcing_call(device, chunk_index=0)) is None

    # Even chunk 1 — where the whole-frame geometry used to force density
    # far above the knob — now runs at the knob.
    for chunk_index in (1, 9):
        call = _self_forcing_call(device, chunk_index=chunk_index)
        backend.begin_forward(_geometry(chunk_index))
        out = backend.attend(call)
        assert out is not None and out.shape == call.query.shape

    plan, _flat = backend._plans._entries[0][1]
    assert isinstance(plan, BlockSparsePlan)
    heads = 4
    kv_len = 10 * FRAMES_PER_BLOCK * FRAME_SEQLEN
    num_frames = kv_len // FRAME_SEQLEN
    q_tiles = -(-FRAME_SEQLEN // query_tile)
    allow = torch.zeros(heads, q_tiles, kv_len, dtype=torch.bool, device=device)
    for h in range(heads):
        for t in range(q_tiles):
            for s0 in plan.starts[h, t].tolist():
                allow[h, t, s0 : s0 + plan.key_tile] = True
    frames = allow.view(heads, q_tiles, num_frames, FRAME_SEQLEN)
    # No frame is whole — not the sink, not the own chunk ...
    assert not frames.all(dim=-1).any()
    # ... every frame gets the same per-query-tile pattern ...
    for frame in range(1, num_frames):
        assert torch.equal(frames[:, :, frame], frames[:, :, 0])
    # ... and the density tracks the knob without a floor.
    assert abs(plan.density - density) < 0.05


@requires_cuda
def test_osa_window_query_plans_per_chunk_and_executes():
    """A rolling-window call (multi-chunk query, Rolling Forcing) gets one plan
    per query chunk: each chunk keeps its *own* whole frames — its chunk, the
    sink frame and its recent frame — and the block-sparse execution over the
    shared K/V view matches a masked reference chunk by chunk."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        BlockSparsePlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, head_dim = 4, 64
    query_tile = 128
    chunk_tokens = FRAMES_PER_BLOCK * FRAME_SEQLEN
    window_chunks = 3
    backend = build_sparse_attention_backend(
        "osa",
        {
            "density": 0.6,
            "num_recent_frames": 1,
            "sink_latent_frames": 1,
            "calibration_query_stride": 4,
            "query_tile": query_tile,
        },
    )

    # Calibration: the last ramp-up window starts at chunk 0 and covers the
    # whole (window-sized) query; its keys are the window's own tokens.
    ramp_query = torch.randn(
        1,
        window_chunks * chunk_tokens,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    ramp_call = SparseAttentionCall(
        layer_index=0,
        query=ramp_query,
        key=torch.randn_like(ramp_query),
        value=torch.randn_like(ramp_query),
        key_segments=((0, window_chunks * chunk_tokens),),
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
    )
    backend.begin_forward(_geometry(0))
    assert backend.attend(ramp_call) is None  # ramp-up runs dense, measuring

    # Steady state: window of chunks [5, 8), keys = sink chunk 0 (re-roped
    # anchor) + working chunk 4 + the window's own fresh keys.
    window_start_chunk = 5
    key_segments = (
        (0, chunk_tokens),
        (4 * chunk_tokens, chunk_tokens),
        (window_start_chunk * chunk_tokens, window_chunks * chunk_tokens),
    )
    kv_len = sum(length for _, length in key_segments)
    query = torch.randn(
        1,
        window_chunks * chunk_tokens,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(1, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    call = SparseAttentionCall(
        layer_index=0,
        query=query,
        key=key,
        value=torch.randn_like(key),
        key_segments=key_segments,
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
    )
    backend.begin_forward(_geometry(window_start_chunk))
    out = backend.attend(call)
    assert out is not None and out.shape == query.shape

    plan, _flat = backend._plans._entries[0][1]
    assert isinstance(plan, tuple) and len(plan) == window_chunks
    assert all(isinstance(entry, BlockSparsePlan) for entry in plan)

    global_frame_ids = np.concatenate(
        [
            np.arange(start // FRAME_SEQLEN, (start + length) // FRAME_SEQLEN)
            for start, length in key_segments
        ]
    )
    num_frames = kv_len // FRAME_SEQLEN
    q_tiles = -(-FRAME_SEQLEN // query_tile)
    for offset, chunk_plan in enumerate(plan):
        allow = torch.zeros(heads, q_tiles, kv_len, dtype=torch.bool, device=device)
        for h in range(heads):
            for t in range(q_tiles):
                for s0 in chunk_plan.starts[h, t].tolist():
                    allow[h, t, s0 : s0 + chunk_plan.key_tile] = True
        frames = allow.view(heads, q_tiles, num_frames, FRAME_SEQLEN)
        own_first = (window_start_chunk + offset) * FRAMES_PER_BLOCK
        whole = (
            (global_frame_ids == 0)  # sink frame
            | (global_frame_ids == own_first - 1)  # recent frame
            | (
                (global_frame_ids >= own_first)
                & (global_frame_ids < own_first + FRAMES_PER_BLOCK)
            )
        )
        body = (FRAME_SEQLEN // 64) * 64
        assert frames[:, :, torch.from_numpy(whole).to(device), :body].all()
        # Every non-whole frame repeats one per-(head, query tile) pattern.
        others = np.flatnonzero(~whole)
        for frame in others[1:]:
            assert torch.equal(
                frames[:, :, int(frame)], frames[:, :, int(others[0])]
            )
        assert 0.3 < allow.float().mean().item() < 0.7

        # The per-chunk execution equals masked attention over the shared
        # view; the mask varies per query tile.
        rows = slice(offset * chunk_tokens, (offset + 1) * chunk_tokens)
        row_tile = (
            torch.arange(chunk_tokens, device=device) % FRAME_SEQLEN
        ) // query_tile
        scores = (
            torch.einsum("qhd,khd->hqk", query[0, rows].float(), key[0].float())
            * call.softmax_scale
        )
        scores = scores.masked_fill(~allow[:, row_tile, :], float("-inf"))
        reference = torch.softmax(scores, dim=-1) @ call.value[0].float().permute(
            1, 0, 2
        )
        torch.testing.assert_close(
            out[0, rows].float(),
            reference.permute(1, 0, 2),
            atol=3e-2,
            rtol=3e-2,
        )



def _gapped_call(device, *, heads=4, head_dim=64):
    """A LongLive-2/LingBot-shaped call: sink + gap + rolling window.

    Two key segments — video frames 0-1 pinned as a sink, then frames 10-17
    with frames 15-17 being the query's own chunk (chunk index 5).
    """
    sink_frames, window_frames = 2, 8
    kv_len = (sink_frames + window_frames) * FRAME_SEQLEN
    chunk_tokens = FRAMES_PER_BLOCK * FRAME_SEQLEN
    query = torch.randn(
        1, chunk_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(1, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    return SparseAttentionCall(
        layer_index=0,
        query=query,
        key=key,
        value=torch.randn_like(key),
        key_segments=(
            (0, sink_frames * FRAME_SEQLEN),
            (10 * FRAME_SEQLEN, window_frames * FRAME_SEQLEN),
        ),
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
    )


def _window_call(device, *, heads=4, head_dim=64, reroped_sink=True):
    """A Rolling-Forcing-shaped steady window: 3-chunk query, re-roped sink.

    Key segments: the 1-frame sink block (global frame 0, re-roped to sit just
    before the working cache), 4 working frames, then the 9-frame window that
    is also the query. The query's chunk index is the *oldest* window chunk.
    """
    query_frames = 3 * FRAMES_PER_BLOCK
    working_frames = 4
    sink_frames = 1
    first_window_frame = 12
    kv_len = (sink_frames + working_frames + query_frames) * FRAME_SEQLEN
    query = torch.randn(
        1,
        query_frames * FRAME_SEQLEN,
        heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(1, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    working_start = (first_window_frame - working_frames) * FRAME_SEQLEN
    segments = (
        (0, sink_frames * FRAME_SEQLEN),
        (working_start, working_frames * FRAME_SEQLEN),
        (first_window_frame * FRAME_SEQLEN, query_frames * FRAME_SEQLEN),
    )
    anchor_frame = first_window_frame - working_frames - sink_frames
    rope_starts = (
        (anchor_frame * FRAME_SEQLEN, working_start, segments[2][0])
        if reroped_sink
        else None
    )
    return SparseAttentionCall(
        layer_index=0,
        query=query,
        key=key,
        value=torch.randn_like(key),
        key_segments=segments,
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
        key_segment_rope_starts=rope_starts,
    )


def _window_geometry() -> ChunkGeometry:
    # query_token_start = oldest window chunk (frame 12 -> chunk 4).
    return ChunkGeometry(
        frame_seqlen=FRAME_SEQLEN,
        frames_per_block=FRAMES_PER_BLOCK,
        query_token_start=12 * FRAME_SEQLEN,
        grid_height=15,
        grid_width=26,
    )


_ALL_METHOD_CONFIGS = [
    ("osa", {"density": 0.4, "num_recent_frames": 1, "sink_latent_frames": 1}),
    ("xattention", {"threshold": 0.6}),
    ("svg1", {"num_sampled_blocks": 2, "band_frames": 0.5, "dense_sink_frames": 2}),
    ("svg2", {"cluster_size": 32, "kmeans_iters": 2, "top_p": 0.5}),
    ("radial", {}),
    ("sta", {"kernel_t": 3}),
    ("lightforcing", {}),
]


@requires_cuda
@pytest.mark.parametrize("method,config", _ALL_METHOD_CONFIGS)
@pytest.mark.parametrize("shape", ["gapped", "window"])
def test_method_is_exact_on_noncontiguous_layouts(method, config, shape):
    """Layouts B/C (sink + gap) and A (multi-chunk window query).

    Every method must either decline or compute exactly what its plan says on
    the layouts the capped-window models (LongLive-2, LingBot) and Rolling
    Forcing actually produce. This is the same kernel-exactness contract as
    ``test_method_output_matches_its_own_plan``, on the layouts that test does
    not cover.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(method, config)

    if shape == "gapped":
        call = _gapped_call(device)
        geometry = _geometry(5)
    else:
        call = _window_call(device)
        geometry = _window_geometry()
    # Walk earlier chunks so anything that calibrates (OSA) has done so.
    for earlier in range(geometry.query_chunk_index):
        backend.begin_forward(_geometry(earlier))
        backend.attend(_self_forcing_call(device, chunk_index=earlier))

    backend.begin_forward(geometry)
    out = backend.attend(call)
    if out is None:
        pytest.skip(f"{method} declines the {shape} layout")
    if method == "osa":
        # OSA executes through its own gather + FA3 path, not prepare();
        # its exactness on these layouts is asserted in
        # test_osa_window_query_plans_per_chunk_and_executes. Reaching a
        # non-None output through attend() is the contract checked here.
        assert out.shape == call.query.shape
        return

    layout = visible_layout(
        call.key_segments,
        geometry=geometry,
        query_tokens=call.query.shape[1],
        rope_starts=call.key_segment_rope_starts,
    )
    execution = backend.prepare(call, layout)
    assert execution is not None
    plan = execution.plan
    key_mask = plan_key_mask(plan, kv_len=execution.key.shape[1])
    q_len = execution.query.shape[1]
    rows = (torch.arange(q_len, device=device) // plan.block_m).clamp(
        max=key_mask.shape[1] - 1
    )
    scores = (
        torch.einsum(
            "qhd,khd->hqk", execution.query[0].float(), execution.key[0].float()
        )
        * call.softmax_scale
    )
    if plan.logit_bias is not None:
        scores = scores + plan.logit_bias[:, None, :]
    if plan.shared_q_block:
        rows = torch.zeros_like(rows)
    scores = scores.masked_fill(~key_mask[:, rows, :], float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ execution.value[0].float().permute(
        1, 0, 2
    )
    sparse_out = sparse_attention(
        query=execution.query,
        key=execution.key,
        value=execution.value,
        plan=plan,
        softmax_scale=call.softmax_scale,
    )
    torch.testing.assert_close(
        sparse_out[0].float(), reference.permute(1, 0, 2), atol=3e-2, rtol=3e-2
    )


def test_visible_layout_positional_ids_track_rope_starts():
    geometry = _window_geometry()
    device = torch.device("cpu")
    call = _window_call(device) if torch.cuda.is_available() else None
    segments = (
        (0, 1 * FRAME_SEQLEN),
        (8 * FRAME_SEQLEN, 4 * FRAME_SEQLEN),
        (12 * FRAME_SEQLEN, 9 * FRAME_SEQLEN),
    )
    rope_starts = (7 * FRAME_SEQLEN, 8 * FRAME_SEQLEN, 12 * FRAME_SEQLEN)
    layout = visible_layout(
        segments,
        geometry=geometry,
        query_tokens=9 * FRAME_SEQLEN,
        rope_starts=rope_starts,
    )
    assert layout is not None
    assert layout.global_frame_ids[0] == 0
    assert layout.positional_frame_ids[0] == 7
    np.testing.assert_array_equal(
        layout.positional_frame_ids[1:], layout.global_frame_ids[1:]
    )
    # Without rope starts, positional == global.
    plain = visible_layout(segments, geometry=geometry, query_tokens=9 * FRAME_SEQLEN)
    np.testing.assert_array_equal(plain.positional_frame_ids, plain.global_frame_ids)


@requires_cuda
def test_radial_reroped_sink_uses_positional_distance():
    """The re-roped sink must get the band of its *positional* neighbourhood.

    With ``dense_sink_frames=0`` the sink frame is treated by distance alone:
    re-roped to sit right before the working cache it is a near frame (wide
    band), while at its global position it would be decimated to nothing for
    most query frames. The masks must therefore differ, and the re-roped one
    must keep at least as much of the sink column as the global one.
    """
    device = torch.device("cuda")
    geometry = _window_geometry()
    config = RadialConfig(dense_sink_frames=0)
    masks = {}
    for reroped in (False, True):
        call = _window_call(device, reroped_sink=reroped)
        layout = visible_layout(
            call.key_segments,
            geometry=geometry,
            query_tokens=call.query.shape[1],
            rope_starts=call.key_segment_rope_starts,
        )
        masks[reroped] = build_radial_block_mask(
            layout=layout,
            q_len=call.query.shape[1],
            kv_len=call.key.shape[1],
            config=config,
            device=device,
        )
    sink_blocks = FRAME_SEQLEN // 128 + 1
    kept_reroped = masks[True][:, :sink_blocks].sum().item()
    kept_global = masks[False][:, :sink_blocks].sum().item()
    assert kept_reroped > kept_global


@requires_cuda
def test_svg1_dense_sink_frames_protects_that_many_frames():
    """`dense_sink_frames=n` keeps the first n view frames in the spatial mask."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.svg import (
        Svg1Config,
        build_svg1_segment_masks,
    )

    device = torch.device("cuda")
    call = _gapped_call(device)
    geometry = _geometry(5)
    layout = visible_layout(
        call.key_segments, geometry=geometry, query_tokens=call.query.shape[1]
    )
    q_len = call.query.shape[1]
    kv_len = call.key.shape[1]
    for sink_frames in (1, 2):
        config = Svg1Config(band_frames=0.5, dense_sink_frames=sink_frames)
        spatial, _, _ = build_svg1_segment_masks(
            layout=layout, q_len=q_len, kv_len=kv_len, config=config, device=device
        )
        tiles_per_frame = -(-FRAME_SEQLEN // config.key_tile)
        sink_cols = spatial[:, : sink_frames * tiles_per_frame]
        assert bool(sink_cols.all()), f"sink frames not dense at n={sink_frames}"
        beyond = spatial[0, sink_frames * tiles_per_frame : 3 * tiles_per_frame]
        assert not bool(beyond.all()), "everything past the sink is dense too"


def test_flops_matched_schedule_front_loads_and_matches_mean():
    """The schedule decays as 1/sqrt(kv) and its kv-weighted mean is the knob."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.osa import (
        flops_matched_densities,
    )

    mean = 0.2
    schedule = flops_matched_densities(
        num_frames=30,
        frames_per_block=3,
        window_frames=-1,
        mean_density=mean,
        floor_density=0.02,
    )
    assert schedule[0] == 1.0  # chunk 0 is the dense calibration chunk
    tail = schedule[1:]
    assert all(a >= b for a, b in zip(tail, tail[1:], strict=False))
    assert tail[0] > mean > tail[-1]
    kv = list(range(6, 31, 3))
    weighted = sum(d * k for d, k in zip(tail, kv, strict=True)) / sum(kv)
    assert abs(weighted - mean) < 1e-6


@requires_cuda
def test_osa_demand_weighted_allocation_follows_group_demand():
    """Demand weighting gives own-chunk frames more tiles than old history,
    keeps the per-(head, query tile) trip count uniform, and — with exemptions
    uncharged — enabling the sink anchor does not shrink the pattern budget."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        BlockSparsePlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    base = {
        "density": 0.2,
        "spatial_tile": 64,
        "calibration_query_stride": 4,
        "query_tile": 128,
        "keep_own_chunk_full": False,
        "keep_sink_full": False,
        "keep_recent_frames_full": False,
        "demand_weighted": True,
        "chunk_schedule": "flops_matched",
        "schedule_num_frames": 30,
    }

    def plan_at(config: dict, chunk_index: int) -> BlockSparsePlan:
        backend = build_sparse_attention_backend("osa", config)
        backend.begin_forward(_geometry(0))
        assert backend.attend(_self_forcing_call(device, chunk_index=0)) is None
        backend.begin_forward(_geometry(chunk_index))
        out = backend.attend(_self_forcing_call(device, chunk_index=chunk_index))
        assert out is not None
        plan, _flat = backend._plans._entries[0][1]
        assert isinstance(plan, BlockSparsePlan)
        return plan

    plan = plan_at(base, 9)
    num_frames = 30
    frame_of = (plan.starts[0, 0] // FRAME_SEQLEN).tolist()
    per_frame = [frame_of.count(f) for f in range(num_frames)]
    own = per_frame[-FRAMES_PER_BLOCK:]
    old_hist = per_frame[1 : num_frames // 2]
    assert min(own) > max(old_hist)  # own chunk out-weighs old history
    recent = per_frame[num_frames - FRAMES_PER_BLOCK - 1]
    assert recent > max(old_hist)  # the temporal anchor out-weighs old history
    # Uniform trip count is structural: every (head, q_tile) shares n_blocks.
    assert plan.starts.shape[2] == len(frame_of)

    # Free exemptions: adding the sink anchor must not reduce any patterned
    # frame's allocation (the osa2s starvation failure mode).
    charged = plan_at(
        {**base, "keep_sink_full": True, "sink_latent_frames": 1}, 9
    )
    frame_of_charged = (charged.starts[0, 0] // FRAME_SEQLEN).tolist()
    free = plan_at(
        {
            **base,
            "keep_sink_full": True,
            "sink_latent_frames": 1,
            "charge_exempt_frames": False,
        },
        9,
    )
    frame_of_free = (free.starts[0, 0] // FRAME_SEQLEN).tolist()
    for frame in range(1, num_frames):
        assert frame_of_free.count(frame) >= frame_of_charged.count(frame)
    assert free.density > charged.density  # the sink is extra, not carved out


@requires_cuda
def test_osa_flat_row_dense_fallback_is_exact_and_fades_with_budget():
    """Flat rows fall back to exact dense attention over the full KV; at a
    generous budget every row clears the threshold and no row falls back."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    base = {
        "density": 0.15,
        "spatial_tile": 64,
        "calibration_query_stride": 4,
        "query_tile": 128,
        "keep_own_chunk_full": False,
        "keep_sink_full": False,
        "keep_recent_frames_full": False,
        "demand_weighted": True,
        "flat_row_dense": True,
        "flat_row_retention": 0.9,  # aggressive so some rows trip on noise data
        "flat_row_max_fraction": 0.5,
    }
    backend = build_sparse_attention_backend("osa", base)
    backend.begin_forward(_geometry(0))
    assert backend.attend(_self_forcing_call(device, chunk_index=0)) is None
    call = _self_forcing_call(device, chunk_index=6)
    backend.begin_forward(_geometry(6))
    out = backend.attend(call)
    assert out is not None
    _plan, flat_index = backend._plans._entries[0][1]
    assert flat_index is not None and flat_index.numel() > 0
    flat_index = flat_index.to(device)
    reference = torch.nn.functional.scaled_dot_product_attention(
        call.query[:, flat_index].transpose(1, 2),
        call.key.transpose(1, 2),
        call.value.transpose(1, 2),
        scale=call.softmax_scale,
    ).transpose(1, 2)
    torch.testing.assert_close(
        out[:, flat_index], reference.to(out.dtype), atol=3e-2, rtol=3e-2
    )

    # A generous budget clears the threshold: no fallback rows.
    generous = build_sparse_attention_backend(
        "osa", {**base, "density": 0.85, "flat_row_retention": 0.05}
    )
    generous.begin_forward(_geometry(0))
    assert generous.attend(_self_forcing_call(device, chunk_index=0)) is None
    generous.begin_forward(_geometry(6))
    assert generous.attend(_self_forcing_call(device, chunk_index=6)) is not None
    _plan, flat_index = generous._plans._entries[0][1]
    assert flat_index is None or flat_index.numel() == 0


def test_osa_replan_rejects_pattern_only_knobs():
    """demand_weighted / flat_row_dense describe the frozen replicated pattern
    and cannot be combined with per-chunk measured re-planning."""
    for extra in ({"demand_weighted": True}, {"flat_row_dense": True}):
        with pytest.raises(ValueError, match="replan_each_chunk"):
            build_sparse_attention_backend(
                "osa", {"density": 0.3, "replan_each_chunk": True, **extra}
            )


@requires_cuda
def test_osa_replan_each_chunk_is_exact_and_beats_frozen_recall():
    """replan_each_chunk plans from each chunk's own measured mass: chunk 0
    stores no calibration, whole frames still follow the keep switches, the
    kernel output equals masked SDPA under the plan, and — on the same
    tensors — the measured plan captures at least as much attention mass as
    the frozen replicated pattern at the same whole-frame geometry."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        BlockSparsePlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, query_tile = 4, 128
    chunk_index = 9
    base = {
        "density": 0.4,
        "spatial_tile": 64,
        "calibration_query_stride": 4,
        "query_tile": query_tile,
    }
    call_c0 = _self_forcing_call(device, chunk_index=0, heads=heads)
    call = _self_forcing_call(device, chunk_index=chunk_index, heads=heads)

    def plan_and_out(config: dict):
        backend = build_sparse_attention_backend("osa", config)
        backend.begin_forward(_geometry(0))
        assert backend.attend(call_c0) is None
        backend.begin_forward(_geometry(chunk_index))
        out = backend.attend(call)
        assert out is not None
        plan, _flat = backend._plans._entries[0][1]
        assert isinstance(plan, BlockSparsePlan)
        return backend, plan, out

    backend, plan, out = plan_and_out({**base, "replan_each_chunk": True})
    assert not backend._section_mass and not backend._section_order

    kv_len = call.key.shape[1]
    num_frames = kv_len // FRAME_SEQLEN
    q_tiles = -(-FRAME_SEQLEN // query_tile)
    body = (FRAME_SEQLEN // 64) * 64

    def allow_mask(p: BlockSparsePlan) -> torch.Tensor:
        allow = torch.zeros(heads, q_tiles, kv_len, dtype=torch.bool, device=device)
        for h in range(heads):
            for t in range(q_tiles):
                for s0 in p.starts[h, t].tolist():
                    allow[h, t, s0 : s0 + p.key_tile] = True
        return allow

    allow = allow_mask(plan)
    frames = allow.view(heads, q_tiles, num_frames, FRAME_SEQLEN)
    # Whole frames per the default keep switches: sink 0, the two recent
    # frames, and the own chunk.
    whole = [0, num_frames - 5, num_frames - 4, num_frames - 3, num_frames - 2,
             num_frames - 1]
    assert frames[:, :, whole, :body].all()

    # The execution equals masked SDPA under the plan's own mask.
    row_tile = (
        torch.arange(call.query.shape[1], device=device) % FRAME_SEQLEN
    ) // query_tile
    scores = (
        torch.einsum("qhd,khd->hqk", call.query[0].float(), call.key[0].float())
        * call.softmax_scale
    )
    masked = scores.masked_fill(~allow[:, row_tile, :], float("-inf"))
    reference = torch.softmax(masked, dim=-1) @ call.value[0].float().permute(1, 0, 2)
    torch.testing.assert_close(
        out[0].float(), reference.permute(1, 0, 2), atol=3e-2, rtol=3e-2
    )

    # Same tensors, frozen replicated pattern at the same geometry: the
    # measured plan must capture at least as much true attention mass.
    _backend, frozen_plan, _out = plan_and_out(base)
    probs = torch.softmax(scores, dim=-1)
    measured_mass = probs.masked_fill(~allow[:, row_tile, :], 0.0).sum()
    frozen_mass = probs.masked_fill(
        ~allow_mask(frozen_plan)[:, row_tile, :], 0.0
    ).sum()
    assert measured_mass >= frozen_mass - 1e-4


@requires_cuda
def test_osa_replan_declines_window_queries():
    """A rolling-window call (multi-chunk query) falls back to dense under
    replan_each_chunk instead of building a mis-scoped plan."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, head_dim = 4, 64
    chunk_tokens = FRAMES_PER_BLOCK * FRAME_SEQLEN
    backend = build_sparse_attention_backend(
        "osa", {"density": 0.4, "replan_each_chunk": True}
    )
    key_segments = (
        (0, chunk_tokens),
        (4 * chunk_tokens, chunk_tokens),
        (5 * chunk_tokens, 3 * chunk_tokens),
    )
    kv_len = sum(length for _, length in key_segments)
    query = torch.randn(
        1, 3 * chunk_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key = torch.randn(1, kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
    call = SparseAttentionCall(
        layer_index=0,
        query=query,
        key=key,
        value=torch.randn_like(key),
        key_segments=key_segments,
        head_start=0,
        num_local_heads=heads,
        softmax_scale=head_dim**-0.5,
    )
    backend.begin_forward(_geometry(5))
    assert backend.attend(call) is None


# --------------------------------------------------------------------------
# MSA (mixed sparse attention)
# --------------------------------------------------------------------------


def _msa_taxonomy(tmp_path, families):
    """A taxonomy file assigning ``families[h]`` to head h of layer 0."""
    import json as _json

    heads = {}
    for head, (family, params) in enumerate(families):
        heads[f"L00_h{head}"] = {"family": family, **params}
    path = tmp_path / "taxonomy.json"
    path.write_text(_json.dumps({"heads": heads}))
    return str(path)


def _msa_backend(tmp_path, **config):
    families = [
        ("local", {"r": 1}),
        ("shortwin", {"m": 3}),
        ("diffuse", {}),
        ("content", {}),
    ]
    return build_sparse_attention_backend(
        "msa",
        {
            "taxonomy_path": _msa_taxonomy(tmp_path, families),
            "content_density": 0.3,
            "block_q": 64,
            "block_k": 64,
            **config,
        },
    )


def _msa_expected_mask(backend, call, layout):
    """The key mask MSA's own descriptors describe, [heads, q_blocks, kv_len]."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (
        frame_aligned_block_bounds,
    )
    from sglang.multimodal_gen.runtime.layers.attention.sparse.msa import (
        diffuse_frame_step,
        local_ranges,
    )

    config = backend._config
    device = call.query.device
    q_len = call.query.shape[1]
    kv_len = call.key.shape[1]
    frame_seqlen = layout.frame_seqlen
    num_frames = layout.num_frames
    q_blocks = -(-q_len // config.block_q)
    mask = torch.zeros(4, q_blocks, kv_len, dtype=torch.bool, device=device)
    for q_block in range(q_blocks):
        for lo, hi in local_ranges(
            q_block,
            radius=1,
            block_m=config.block_q,
            q_len=q_len,
            frame_seqlen=frame_seqlen,
            grid_height=15,
            grid_width=26,
        ):
            for frame in range(num_frames):
                mask[0, q_block, frame * frame_seqlen + lo : frame * frame_seqlen + hi] = True
    mask[1, :, (num_frames - 3) * frame_seqlen :] = True
    step = diffuse_frame_step(config.diffuse_density)
    tail = min(layout.query_frames, num_frames)
    kept_frames = list(range(0, num_frames - tail, step)) + list(
        range(num_frames - tail, num_frames)
    )
    for frame in kept_frames:
        mask[2, :, frame * frame_seqlen : (frame + 1) * frame_seqlen] = True
    lut, _, kv_blocks = backend._content_lut(call, layout, [3])
    content_mask = torch.zeros(
        lut.shape[0], lut.shape[1], kv_blocks, dtype=torch.bool, device=device
    )
    content_mask.scatter_(-1, lut, True)
    block_lo, block_hi = frame_aligned_block_bounds(
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        block=config.block_k,
        device=device,
    )
    for q_block in range(q_blocks):
        for block, kept in enumerate(content_mask[0, q_block].tolist()):
            if kept:
                mask[3, q_block, block_lo[block] : block_hi[block]] = True
    return mask


@requires_cuda
def test_msa_matches_its_own_descriptor_mask(tmp_path):
    """Both MSA kernels compute exactly the attention their descriptors say."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = _msa_backend(tmp_path)
    call = _self_forcing_call(device, chunk_index=4)
    backend.begin_forward(_geometry(4))
    layout = visible_layout(
        call.key_segments, geometry=_geometry(4), query_tokens=call.query.shape[1]
    )
    out = backend.attend(call)
    assert out is not None
    key_mask = _msa_expected_mask(backend, call, layout)
    reference = masked_reference(
        call.query,
        call.key,
        call.value,
        key_mask,
        block_m=backend._config.block_q,
        softmax_scale=call.softmax_scale,
    )
    torch.testing.assert_close(
        out.float(), reference.float(), atol=3e-2, rtol=3e-2
    )
    density = key_mask.float().mean().item()
    assert density < 0.7, f"msa kept too much ({density:.3f})"


@requires_cuda
def test_msa_declines_chunk0_but_sparsifies_cache_updates(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = _msa_backend(tmp_path)
    call = _self_forcing_call(device, chunk_index=0)
    backend.begin_forward(_geometry(0))
    assert backend.attend(call) is None
    call = _self_forcing_call(device, chunk_index=4)
    backend.begin_forward(_geometry(4))
    # Cache refreshes run sparse too (LightForcing parity).
    with backend.cache_update_scope():
        assert backend.attend(call) is not None
    assert backend.attend(call) is not None


@requires_cuda
def test_msa_static_descriptors_are_cached_per_layout(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = _msa_backend(tmp_path)
    call = _self_forcing_call(device, chunk_index=4)
    backend.begin_forward(_geometry(4))
    layout = visible_layout(
        call.key_segments, geometry=_geometry(4), query_tokens=call.query.shape[1]
    )
    backend.attend(call)
    hit, plan = backend._static_cache.get(
        0, (layout.num_frames, layout.frame_seqlen, call.query.shape[1])
    )
    assert hit and plan.head_ids.shape[0] == 3
    # The static side of the density is exact and strictly below dense.
    assert 0.0 < plan.fraction_sum < 3.0


def test_msa_local_ranges_straddle_and_merge():
    from sglang.multimodal_gen.runtime.layers.attention.sparse.msa import (
        local_ranges,
    )

    # Query block 6 of block_m=64 at frame_seqlen 390 covers tokens 384..448,
    # straddling the frame boundary: two windows, one at the frame's bottom
    # rows and one at its top.
    ranges = local_ranges(
        6,
        radius=1,
        block_m=64,
        q_len=1170,
        frame_seqlen=390,
        grid_height=15,
        grid_width=26,
    )
    assert len(ranges) == 2
    for lo, hi in ranges:
        assert 0 <= lo < hi <= 390
    assert ranges[0][0] == 0  # the top window starts at row 0
    assert ranges[1][1] == 390  # the bottom window ends at the last row


def test_msa_diffuse_frame_step_matches_density():
    from sglang.multimodal_gen.runtime.layers.attention.sparse.msa import (
        diffuse_frame_step,
    )

    assert diffuse_frame_step(0.10) == 10
    assert diffuse_frame_step(0.5) == 2
    assert diffuse_frame_step(1.0) == 1


@requires_cuda
def test_msa_content_schedule_thins_late_chunks(tmp_path):
    """flops_matched: later chunks keep fewer content blocks, mean preserved."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.osa import (
        flops_matched_densities,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = _msa_backend(
        tmp_path,
        content_schedule="flops_matched",
        schedule_num_frames=21,
        schedule_floor_density=0.05,
    )
    topks = {}
    for chunk_index in (1, 6):
        call = _self_forcing_call(device, chunk_index=chunk_index)
        backend.begin_forward(_geometry(chunk_index))
        layout = visible_layout(
            call.key_segments,
            geometry=_geometry(chunk_index),
            query_tokens=call.query.shape[1],
        )
        _, topk, kv_blocks = backend._content_lut(call, layout, [3])
        topks[chunk_index] = (topk, kv_blocks)
    # Early chunk denser than the knob, late chunk sparser.
    assert topks[1][0] / topks[1][1] > 0.3
    assert topks[6][0] / topks[6][1] < 0.3
    # The schedule's kv-weighted mean equals the knob.
    schedule = flops_matched_densities(
        num_frames=21,
        frames_per_block=3,
        window_frames=-1,
        mean_density=0.3,
        floor_density=0.05,
    )
    kv = list(range(6, 22, 3))
    mean = sum(d * k for d, k in zip(schedule[1:], kv)) / sum(kv)
    assert abs(mean - 0.3) < 1e-6


@requires_cuda
def test_msa_step_density_scale_shrinks_late_steps(tmp_path):
    """Per-step prefix plans: later denoise calls read fewer keys; the cache
    refresh reuses the fullest plan; all plans slice one ranked scoring."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = _msa_backend(tmp_path, step_density_scale=(1.0, 0.5))
    call = _self_forcing_call(device, chunk_index=4)
    backend.begin_forward(_geometry(4))

    def kept_tokens():
        assert backend.attend(call) is not None
        plans, topks, _, _ = backend._content_cache._entries[0][1]
        return topks

    topks = kept_tokens()
    assert len(topks) == 2 and topks[1] == max(1, round(topks[0] * 0.5))
    plans, topks, _, calls = backend._content_cache._entries[0][1]
    kept0 = int(plans[0].kept_tokens().sum())
    kept1 = int(plans[1].kept_tokens().sum())
    assert kept1 < kept0
    # Second call (step 1) must use the smaller plan; cache refresh the full.
    backend.attend(call)
    with backend.cache_update_scope():
        assert backend.attend(call) is not None


@requires_cuda
def test_msa_frame_granular_content_matches_reference(tmp_path):
    """Frame-granular content: whole-frame ranges, own frames always kept."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = _msa_backend(tmp_path, content_granularity="frame")
    call = _self_forcing_call(device, chunk_index=5)
    backend.begin_forward(_geometry(5))
    layout = visible_layout(
        call.key_segments, geometry=_geometry(5), query_tokens=call.query.shape[1]
    )
    out = backend.attend(call)
    assert out is not None
    plans, topks, kv_blocks, _ = backend._content_cache._entries[0][1]
    assert kv_blocks == layout.num_frames  # frame-granular accounting
    lut = backend._content_lut(call, layout, [3])[0]
    # own frames (the newest 3) rank first via the +inf force
    own = set(range(layout.num_frames - 3, layout.num_frames))
    assert set(lut[0, 0, :3].tolist()) == own
    # the plan's kept keys are whole frames: every range spans multiples of T
    plan = plans[0]
    starts = plan.range_starts[0, 0, : plan.range_counts[0, 0]]
    ends = plan.range_ends[0, 0, : plan.range_counts[0, 0]]
    assert all(int(s) % FRAME_SEQLEN == 0 for s in starts)
    assert all(int(e) % FRAME_SEQLEN == 0 for e in ends)
    # exactness vs masked SDPA on the plan's own mask
    from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
        plan_key_mask,
    )
    key_mask = plan_key_mask(plan, kv_len=call.key.shape[1])
    # content head 3 only: compare its rows
    reference = masked_reference(
        call.query[:, :, 3:4],
        call.key[:, :, 3:4],
        call.value[:, :, 3:4],
        key_mask,
        block_m=backend._config.block_q,
        softmax_scale=call.softmax_scale,
    )
    torch.testing.assert_close(
        out[:, :, 3:4].float(), reference.float(), atol=3e-2, rtol=3e-2
    )
