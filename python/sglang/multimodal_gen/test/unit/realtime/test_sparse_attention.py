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
        grid_height=10,
        grid_width=10,
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
def test_osa_freezes_the_last_chunk0_step_and_replicates():
    """OSA: chunk 0's *last* denoising step is the oracle, the frozen per-head
    tile set repeats in every non-full frame, the achieved density tracks the
    requested one, and the gather + FA3 execution matches a masked
    reference."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.replicate_kernel import (
        ReplicateGatherPlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(
        "osa",
        {
            "density": 0.4,
            "num_recent_frames": 1,
            "sink_latent_frames": 1,
            "spatial_tile": 64,
            "calibration_query_stride": 4,
        },
    )

    backend.begin_forward(_geometry(0))
    for _ in range(3):  # denoising steps of chunk 0 run dense while measuring
        assert backend.attend(_self_forcing_call(device, chunk_index=0)) is None
    frozen = backend._spatial_mass[0].clone()
    with backend.cache_update_scope():  # the KV refresh must not overwrite
        assert backend.attend(_self_forcing_call(device, chunk_index=0)) is None
    assert torch.equal(backend._spatial_mass[0], frozen)

    chunk_index = 9
    call = _self_forcing_call(device, chunk_index=chunk_index)
    backend.begin_forward(_geometry(chunk_index))
    out = backend.attend(call)
    assert out is not None and out.shape == call.query.shape

    plan = backend._plans._entries[0][1]
    assert isinstance(plan, ReplicateGatherPlan)
    kv_len = call.key.shape[1]
    heads = call.query.shape[2]
    allow = torch.zeros(heads, kv_len, dtype=torch.bool, device=device)
    allow.scatter_(1, plan.indices, True)
    num_frames = kv_len // FRAME_SEQLEN
    frames = allow.view(heads, num_frames, FRAME_SEQLEN)
    # own chunk, sink frame and the recent frame are whole ...
    assert frames[:, -FRAMES_PER_BLOCK:].all() and frames[:, 0].all()
    assert frames[:, -FRAMES_PER_BLOCK - 1].all()
    # ... and every other frame repeats one per-head tile pattern.
    for frame in range(2, num_frames - FRAMES_PER_BLOCK - 1):
        assert torch.equal(frames[:, frame], frames[:, 1])
    density = allow.float().mean().item()
    assert abs(density - 0.4) < 0.1
    assert abs(plan.density - density) < 1e-6

    # The gather + FA3 execution computes exactly the masked attention.
    scores = (
        torch.einsum("qhd,khd->hqk", call.query[0].float(), call.key[0].float())
        * call.softmax_scale
    )
    scores = scores.masked_fill(~allow[:, None, :], float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ call.value[0].float().permute(1, 0, 2)
    torch.testing.assert_close(
        out[0].float(), reference.permute(1, 0, 2), atol=3e-2, rtol=3e-2
    )


@requires_cuda
def test_osa_window_query_plans_per_chunk_and_executes():
    """A rolling-window call (multi-chunk query, Rolling Forcing) gets one plan
    per query chunk: each chunk keeps its *own* whole frames — its chunk, the
    sink frame and its recent frame — and the gather execution over the shared
    K/V view matches a masked reference chunk by chunk."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.osa import (
        WindowGatherPlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, head_dim = 4, 64
    chunk_tokens = FRAMES_PER_BLOCK * FRAME_SEQLEN
    window_chunks = 3
    backend = build_sparse_attention_backend(
        "osa",
        {
            "density": 0.6,
            "num_recent_frames": 1,
            "sink_latent_frames": 1,
            "calibration_query_stride": 4,
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

    plan = backend._plans._entries[0][1]
    assert isinstance(plan, WindowGatherPlan)
    assert len(plan.chunk_plans) == window_chunks

    global_frame_ids = np.concatenate(
        [
            np.arange(start // FRAME_SEQLEN, (start + length) // FRAME_SEQLEN)
            for start, length in key_segments
        ]
    )
    num_frames = kv_len // FRAME_SEQLEN
    for offset, chunk_plan in enumerate(plan.chunk_plans):
        allow = torch.zeros(heads, kv_len, dtype=torch.bool, device=device)
        allow.scatter_(1, chunk_plan.indices, True)
        frames = allow.view(heads, num_frames, FRAME_SEQLEN)
        own_first = (window_start_chunk + offset) * FRAMES_PER_BLOCK
        whole = (
            (global_frame_ids == 0)  # sink frame
            | (global_frame_ids == own_first - 1)  # recent frame
            | (
                (global_frame_ids >= own_first)
                & (global_frame_ids < own_first + FRAMES_PER_BLOCK)
            )
        )
        assert frames[:, torch.from_numpy(whole).to(device)].all()
        # Every non-whole frame repeats one per-head tile pattern.
        others = np.flatnonzero(~whole)
        for frame in others[1:]:
            assert torch.equal(frames[:, int(frame)], frames[:, int(others[0])])
        assert 0.3 < allow.float().mean().item() < 0.6

        # The per-chunk execution equals masked attention over the shared view.
        rows = slice(offset * chunk_tokens, (offset + 1) * chunk_tokens)
        scores = (
            torch.einsum("qhd,khd->hqk", query[0, rows].float(), key[0].float())
            * call.softmax_scale
        )
        scores = scores.masked_fill(~allow[:, None, :], float("-inf"))
        reference = torch.softmax(scores, dim=-1) @ call.value[0].float().permute(
            1, 0, 2
        )
        torch.testing.assert_close(
            out[0, rows].float(),
            reference.permute(1, 0, 2),
            atol=3e-2,
            rtol=3e-2,
        )
