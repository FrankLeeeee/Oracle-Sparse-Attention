# SPDX-License-Identifier: Apache-2.0
"""Correctness of the sparse-attention package.

Two things have to be true for any of these methods to be usable, and they are
tested separately:

1. the shared kernel computes *exactly* the attention its plan describes —
   checked against masked SDPA in float32 for every method's real plan, so a
   permutation, a merged key range or a logit bias that is wrong shows up here
   rather than as a subtly worse video;
2. each method's selection logic is the pattern it claims — OSA's per-head
   policy is chunk-relative and monotone, X-Attention's estimator matches a
   brute-force antidiagonal sum, and so on.
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
from sglang.multimodal_gen.runtime.layers.attention.sparse.fastar import (
    temporal_merge,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    frame_mask_to_ranges,
    visible_layout,
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
from sglang.multimodal_gen.runtime.layers.attention.sparse.osa import (
    PATTERN_BAND,
    PATTERN_BLOCK,
    PATTERN_DT_COMB,
    PATTERN_V_COMB,
    PatternHeadPolicies,
    choose_head_policies,
    choose_pattern_head_policies,
    dt_policy_qblock_mask,
    fold_mass_into_bins,
    policy_frame_mask,
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
    block_mask = (
        torch.rand(heads, 4, 9, device=device, generator=rng) < 0.4
    )
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
        batch, kv_len + 2 * FRAME_SEQLEN, heads, head_dim,
        device=device, dtype=torch.bfloat16,
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
        query, key, value, plan_key_mask(plan, kv_len=kv_len),
        block_m=128, softmax_scale=scale,
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
        expected = sorted({token % 10 for token in range(int(lo[index]), int(hi[index]))})
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
    key_frame = (
        torch.arange(mask.shape[1]) * block // FRAME_SEQLEN
    )
    query_frame = layout.num_frames - layout.query_frames
    widths = {}
    for distance in (1, 2, 4):
        columns = key_frame == (query_frame - distance)
        widths[distance] = mask[:, columns].float().mean().item()
    assert widths[1] > widths[2] > widths[4]


# --------------------------------------------------------------------------
# OSA
# --------------------------------------------------------------------------


def test_choose_head_policies_picks_the_cheapest_sufficient_policy():
    # head 0 lives on its own chunk plus the sink; head 1 on the two most
    # recent chunks; head 2 spreads mass over everything it can see.
    own = np.array([0.5, 0.5, 0.3])
    sink = np.array([0.45, 0.02, 0.1])
    offsets = np.array(
        [[0.02, 0.02, 0.01], [0.30, 0.15, 0.03], [0.25, 0.20, 0.15]]
    )
    policies = choose_head_policies(
        own_mass=own,
        sink_mass=sink,
        offset_mass=offsets,
        retention=0.9,
        frames_per_block=FRAMES_PER_BLOCK,
        sink_frames=FRAMES_PER_BLOCK,
    )
    assert policies.keep_sink.tolist() == [True, False, False]
    assert policies.num_recent.tolist() == [0, 2, 0]
    assert policies.dense.tolist() == [False, False, True]
    assert policies.observed_retention[0] == pytest.approx(0.95)


def test_choose_head_policies_marks_saturated_heads_dense():
    """Needing every observed past chunk means the horizon was not observed."""
    policies = choose_head_policies(
        own_mass=np.array([0.4]),
        sink_mass=np.array([0.0]),
        offset_mass=np.array([[0.3, 0.3]]),
        retention=0.9,
        frames_per_block=FRAMES_PER_BLOCK,
        sink_frames=FRAMES_PER_BLOCK,
    )
    assert policies.dense.tolist() == [True]


def test_policy_is_chunk_relative_and_monotone():
    """The kept chunk set shifts by exactly one chunk per generated chunk."""
    policies = choose_head_policies(
        own_mass=np.array([0.6, 0.6]),
        sink_mass=np.array([0.35, 0.0]),
        offset_mass=np.array([[0.02, 0.03], [0.35, 0.05]]),
        retention=0.9,
        frames_per_block=FRAMES_PER_BLOCK,
        sink_frames=FRAMES_PER_BLOCK,
    )
    assert policies.keep_sink.tolist() == [True, False]
    assert policies.num_recent.tolist() == [0, 1]

    def kept_chunks(chunk_index):
        layout = _layout(chunk_index)
        keep = policy_frame_mask(
            policies, layout=layout, sink_frames=FRAMES_PER_BLOCK
        )
        chunk_of_frame = layout.global_frame_ids // FRAMES_PER_BLOCK
        return [set(np.unique(chunk_of_frame[row])) for row in keep]

    # Monotonicity: head 0 reads the sink at every later chunk, head 1 reads
    # the one most recent chunk at every later chunk — the same relative set,
    # shifted forward.
    for chunk_index in range(3, 12):
        sink_head, recent_head = kept_chunks(chunk_index)
        assert sink_head == {0, chunk_index}
        assert recent_head == {chunk_index - 1, chunk_index}


@requires_cuda
def test_osa_recalibration_refreshes_policies_on_schedule():
    """`recalibrate_every` re-observes and swaps policies; 0 keeps them frozen.

    The refresh must happen on the scheduled chunk's first denoising step,
    update the bookkeeping, and keep producing valid sparse output; without
    the knob the reference-chunk policies stay frozen forever.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    def run(config):
        backend = build_sparse_attention_backend("osa", config)
        seen = {}
        for chunk_index in range(2, 8):
            backend.begin_forward(_geometry(chunk_index))
            out = backend.attend(_self_forcing_call(device, chunk_index=chunk_index))
            if 0 in backend.policies:
                seen[chunk_index] = backend.policies[0]
            if chunk_index > 2:
                assert out is not None, f"expected sparse output at chunk {chunk_index}"
        return backend, seen

    frozen_backend, frozen = run({"reference_chunk": 2, "retention": 0.5})
    assert all(p is frozen[3] for c, p in frozen.items() if c >= 3)
    assert frozen_backend._calibrated_at[0] == 2

    recal_backend, recal = run(
        {"reference_chunk": 2, "retention": 0.5, "recalibrate_every": 2}
    )
    # Refreshed at chunks 4 and 6: new policy objects, bookkeeping advanced.
    assert recal[4] is not recal[3]
    assert recal[6] is not recal[5]
    assert recal[5] is recal[4]
    assert recal_backend._calibrated_at[0] == 6


def test_frame_granular_policy_is_never_costlier_and_stays_monotone():
    """Frame granularity refines chunk granularity without changing the shape.

    Every chunk policy is expressible in frame units (k chunks = 3k frames), so
    at equal retention and sink size the cheapest frame policy keeps at most as
    many frames. And the kept set must still be the same chunk-relative set at
    every later chunk, shifted forward.
    """
    from sglang.multimodal_gen.runtime.layers.attention.sparse.osa import (
        fold_mass_into_bins,
        fold_mass_into_frame_bins,
        frame_ages,
    )

    rng = np.random.default_rng(0)
    layout = _layout(4)
    heads = 6
    frame_mass = rng.dirichlet(np.ones(layout.num_frames) * 0.4, size=heads)
    sink_frames = 1

    own_c, sink_c, chunk_bins = fold_mass_into_bins(
        frame_mass, layout=layout, sink_frames=sink_frames
    )
    own_f, sink_f, frame_bins = fold_mass_into_frame_bins(
        frame_mass, layout=layout, sink_frames=sink_frames
    )
    np.testing.assert_allclose(own_c, own_f)
    np.testing.assert_allclose(sink_c, sink_f)
    # Frame bins are an exact refinement: summing ages within a chunk offset
    # reproduces the chunk bin.
    np.testing.assert_allclose(
        frame_bins.reshape(heads, -1, FRAMES_PER_BLOCK).sum(-1), chunk_bins
    )

    retention = 0.7
    chunk_policies = choose_head_policies(
        own_mass=own_c, sink_mass=sink_c, offset_mass=chunk_bins,
        retention=retention, frames_per_block=FRAMES_PER_BLOCK,
        sink_frames=sink_frames,
    )
    frame_policies = choose_head_policies(
        own_mass=own_f, sink_mass=sink_f, offset_mass=frame_bins,
        retention=retention, frames_per_block=1, sink_frames=sink_frames,
    )

    def cost(policies, unit):
        return (
            policies.num_recent.astype(int) * unit
            + policies.keep_sink.astype(int) * sink_frames
            + policies.dense.astype(int) * 10_000
        )

    assert (
        cost(frame_policies, 1) <= cost(chunk_policies, FRAMES_PER_BLOCK)
    ).all()

    def kept_ages(chunk_index):
        layout_k = _layout(chunk_index)
        keep = policy_frame_mask(
            policies=frame_policies, layout=layout_k,
            sink_frames=sink_frames, frame_granular=True,
        )
        ages = frame_ages(layout_k)
        return [
            set(ages[row & (ages > 0)].tolist()) for row in keep
        ]

    reference_ages = kept_ages(5)
    for chunk_index in range(6, 10):
        # Chunk-relative stationarity: the same past *ages* stay kept (up to
        # frames that leave the visible window), only the sink is age-absolute.
        for head in range(heads):
            if frame_policies.dense[head] or frame_policies.keep_sink[head]:
                continue
            assert kept_ages(chunk_index)[head] == reference_ages[head]


def _planted_dt_mass(query_frame_ids, key_frame_ids, per_head):
    """[heads, nqf, K] rows summing to 1 from {dt: mass} / {("edge", i): mass}."""
    heads = len(per_head)
    mass = np.zeros((heads, query_frame_ids.size, key_frame_ids.size))
    for head, spec in enumerate(per_head):
        for qi, qf in enumerate(query_frame_ids):
            for where, value in spec.items():
                if isinstance(where, tuple):  # ("edge", view_index)
                    mass[head, qi, where[1]] += value
                else:  # dt = qf - kf
                    kf = qf - where
                    hits = np.flatnonzero(key_frame_ids == kf)
                    if hits.size:
                        mass[head, qi, hits[0]] += value
            remainder = 1.0 - mass[head, qi].sum()
            mass[head, qi] += remainder / key_frame_ids.size
    return mass


def _fit_patterns(mass, layout, *, retention=0.9, max_edge_frames=3):
    return choose_pattern_head_policies(
        query_frame_mass=mass,
        query_frame_ids=layout.global_frame_ids[layout.own_frames],
        key_frame_ids=layout.global_frame_ids,
        retention=retention,
        frames_per_block=FRAMES_PER_BLOCK,
        max_future=FRAMES_PER_BLOCK - 1,
        max_edge_frames=max_edge_frames,
    )


def test_choose_pattern_head_policies_recovers_planted_bands():
    # Chunk 5 of a growing window: query frames 15..17, key frames 0..17.
    layout = _layout(5)
    query_frame_ids = layout.global_frame_ids[layout.own_frames]
    key_frame_ids = layout.global_frame_ids
    # head 0 is diagonal over its own and previous frame; head 1 reads its own
    # frame and the next one (bidirectional inside the chunk; the newest frame
    # has no next, so its mass stays on dt 0); head 2 reads its own frame plus
    # the oldest visible frame.
    mass = _planted_dt_mass(
        query_frame_ids,
        key_frame_ids,
        [
            {0: 0.70, 1: 0.25},
            {0: 0.60, -1: 0.35},
            {0: 0.60, ("edge", 0): 0.35},
        ],
    )
    mass[1, -1] = 0.0
    mass[1, -1, np.flatnonzero(key_frame_ids == query_frame_ids[-1])[0]] = 0.95
    mass[1, -1] += 0.05 / key_frame_ids.size
    mass /= mass.sum(axis=-1, keepdims=True)
    policies = _fit_patterns(mass, layout)
    assert policies.dense.tolist() == [False, False, False]
    assert policies.pattern.tolist() == [PATTERN_BAND] * 3
    assert policies.params[:, 0].tolist() == [1, 0, 0]  # num_past
    assert policies.params[:, 1].tolist() == [0, 1, 0]  # num_future
    assert policies.edge_frames.tolist() == [0, 0, 1]
    assert (policies.observed_retention >= 0.9).all()


def test_choose_pattern_head_policies_recovers_periodic_patterns():
    layout = _layout(5)
    query_frame_ids = layout.global_frame_ids[layout.own_frames]
    key_frame_ids = layout.global_frame_ids
    # head 0: periodic diagonals — mass at dt 0, 3, 6 (one chunk apart);
    # head 1: chunk-aligned block — uniform over its own and previous chunk,
    #   which no cheap band can cover but two chunks can;
    # head 2: vertical stripes — every third *absolute* frame (chunk starts),
    #   regardless of the query frame.
    mass = _planted_dt_mass(
        query_frame_ids,
        key_frame_ids,
        [
            {0: 0.40, 3: 0.30, 6: 0.25},
            {},
            {},
        ],
    )
    query_chunk = query_frame_ids[0] // FRAMES_PER_BLOCK
    own_or_previous = (
        key_frame_ids // FRAMES_PER_BLOCK >= query_chunk - 1
    )
    mass[1] = np.where(own_or_previous[None, :], 1.0, 0.02)
    stripes = key_frame_ids % FRAMES_PER_BLOCK == 0
    mass[2] = 0.002
    mass[2, :, stripes] += 0.13
    for qi, qf in enumerate(query_frame_ids):
        mass[2, qi, np.flatnonzero(key_frame_ids == qf)[0]] += 0.20
    mass /= mass.sum(axis=-1, keepdims=True)
    policies = _fit_patterns(mass, layout, retention=0.95, max_edge_frames=0)
    assert policies.dense.tolist() == [False, False, False]
    assert policies.pattern.tolist() == [
        PATTERN_DT_COMB,
        PATTERN_BLOCK,
        PATTERN_V_COMB,
    ]
    period, start, width, depth = policies.params[0]
    assert (period, start, width) == (3, 0, 1) and depth >= 3
    assert policies.params[1, 0] == 1  # own + one recent chunk
    v_period, v_phase, v_width, _ = policies.params[2]
    assert (v_period, v_phase, v_width) == (3, 0, 1)


def test_choose_pattern_head_policies_marks_saturated_heads_dense():
    """Needing the deepest visible dt means the horizon was not observed."""
    layout = _layout(5)
    query_frame_ids = layout.global_frame_ids[layout.own_frames]
    key_frame_ids = layout.global_frame_ids
    horizon = int(query_frame_ids.min() - key_frame_ids.min())
    mass = _planted_dt_mass(
        query_frame_ids, key_frame_ids, [{0: 0.50, horizon: 0.45}]
    )
    policies = _fit_patterns(mass, layout, max_edge_frames=0)
    assert policies.dense.tolist() == [True]


def test_dt_policy_mask_is_query_frame_relative_and_monotone():
    """Each query frame keeps its own band, at every later chunk."""
    policies = PatternHeadPolicies(
        pattern=np.array([PATTERN_BAND, PATTERN_BAND], dtype=np.int8),
        params=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32),
        edge_frames=np.array([0, 2], dtype=np.int32),
        dense=np.array([False, False]),
        observed_retention=np.ones(2, dtype=np.float32),
    )
    for chunk_index in range(5, 9):
        layout = _layout(chunk_index)
        # block_m == frame_seqlen so query block b is exactly query frame b
        keep = dt_policy_qblock_mask(
            policies, layout=layout, block_m=FRAME_SEQLEN
        )
        query_frame_ids = layout.global_frame_ids[layout.own_frames]
        visible = set(layout.global_frame_ids.tolist())
        for block, qf in enumerate(query_frame_ids):
            band_head = set(layout.global_frame_ids[keep[0, block]].tolist())
            assert band_head == {qf - 1, qf} & visible
            edge_head = set(layout.global_frame_ids[keep[1, block]].tolist())
            oldest = set(layout.global_frame_ids[:2].tolist())
            assert edge_head == ({qf, qf + 1} & visible) | oldest


def test_dt_comb_mask_shifts_with_the_query_frame():
    """A periodic-diagonal head keeps dt 0, P, 2P... from *each* query frame."""
    period, depth = 3, 3
    policies = PatternHeadPolicies(
        pattern=np.array([PATTERN_DT_COMB], dtype=np.int8),
        params=np.array([[period, 0, 1, depth]], dtype=np.int32),
        edge_frames=np.array([0], dtype=np.int32),
        dense=np.array([False]),
        observed_retention=np.ones(1, dtype=np.float32),
    )
    for chunk_index in (6, 8):
        layout = _layout(chunk_index)
        keep = dt_policy_qblock_mask(policies, layout=layout, block_m=FRAME_SEQLEN)
        query_frame_ids = layout.global_frame_ids[layout.own_frames]
        visible = set(layout.global_frame_ids.tolist())
        for block, qf in enumerate(query_frame_ids):
            expected = {qf - tooth * period for tooth in range(depth)} & visible
            assert set(layout.global_frame_ids[keep[0, block]].tolist()) == expected


def test_dt_qblock_mask_unions_straddling_frames():
    """A query block spanning two frames keeps both frames' bands."""
    policies = PatternHeadPolicies(
        pattern=np.array([PATTERN_BAND], dtype=np.int8),
        params=np.array([[0, 0, 0, 0]], dtype=np.int32),
        edge_frames=np.array([0], dtype=np.int32),
        dense=np.array([False]),
        observed_retention=np.ones(1, dtype=np.float32),
    )
    layout = _layout(4)
    block_m = 256  # FRAME_SEQLEN=390, so block 1 covers tokens 256..511: two frames
    keep = dt_policy_qblock_mask(policies, layout=layout, block_m=block_m)
    query_frame_ids = layout.global_frame_ids[layout.own_frames]
    kept_frames = set(layout.global_frame_ids[keep[0, 1]].tolist())
    assert kept_frames == {query_frame_ids[0], query_frame_ids[1]}
    # a block wholly inside one frame keeps only that frame
    kept_first = set(layout.global_frame_ids[keep[0, 0]].tolist())
    assert kept_first == {query_frame_ids[0]}


@requires_cuda
def test_osa_dt_refreshes_once_at_the_first_full_window():
    """A partial-window calibration is re-observed once eviction begins.

    The reference chunk's window still starts at frame 0, so its policies
    cannot bound the steady-state horizon; the first chunk whose view has
    evicted frame 0 triggers exactly one recalibration, after which the
    policies are frozen for good.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(
        "osa", {"reference_chunk": 2, "retention": 0.5, "granularity": "dt"}
    )

    def sliding_call(chunk_index, window_chunks):
        chunk_tokens = FRAMES_PER_BLOCK * FRAME_SEQLEN
        first = chunk_index + 1 - window_chunks
        call = _self_forcing_call(device, chunk_index=chunk_index)
        return msgspec.structs.replace(
            call,
            key=call.key[:, -window_chunks * chunk_tokens :],
            value=call.value[:, -window_chunks * chunk_tokens :],
            key_segments=((first * chunk_tokens, window_chunks * chunk_tokens),),
        )

    for chunk_index in range(0, 4):  # growing window through the reference
        backend.begin_forward(_geometry(chunk_index))
        backend.attend(_self_forcing_call(device, chunk_index=chunk_index))
    calibrated = backend.policies[0]
    assert backend._calibration_window_start[0] == 0

    backend.begin_forward(_geometry(5))
    backend.attend(sliding_call(5, window_chunks=4))  # frame 0 evicted
    refreshed = backend.policies[0]
    assert refreshed is not calibrated
    assert backend._calibration_window_start[0] > 0

    backend.begin_forward(_geometry(6))
    backend.attend(sliding_call(6, window_chunks=4))
    assert backend.policies[0] is refreshed  # refresh happens only once


def test_fold_mass_into_bins_counts_the_sink_once():
    layout = _layout(3)
    frame_mass = np.full((1, layout.num_frames), 1.0 / layout.num_frames)
    own, sink, offsets = fold_mass_into_bins(
        frame_mass, layout=layout, sink_frames=FRAMES_PER_BLOCK
    )
    # chunks 0..3 visible: chunk 0 is the sink, so offsets -1 and -2 hold
    # chunks 2 and 1 and offset -3 (chunk 0) is empty.
    assert own == pytest.approx(0.25)
    assert sink == pytest.approx(0.25)
    assert offsets[0].tolist() == pytest.approx([0.25, 0.25, 0.0])
    assert own + sink + offsets.sum() == pytest.approx(1.0)


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
        ("osa", {"reference_chunk": 2, "retention": 0.8}),
        ("osa", {"reference_chunk": 2, "retention": 0.8, "granularity": "frame",
                 "sink_latent_frames": 1}),
        ("osa", {"reference_chunk": 2, "retention": 0.8, "granularity": "dt"}),
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
    about the selection policy itself.
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
        ("osa", {"reference_chunk": 2, "retention": 0.5}),
        ("osa", {"reference_chunk": 2, "retention": 0.5, "granularity": "frame",
                 "sink_latent_frames": 1}),
        ("osa", {"reference_chunk": 2, "retention": 0.5, "granularity": "dt"}),
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
    assert not torch.equal(outputs[0], outputs[1]), (
        f"{method} reused stale current-chunk keys"
    )


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
        [repeated.expand(heads, 4, positions, head_dim),
         torch.randn(heads, 2, positions, head_dim, device=device)],
        dim=1,
    ).contiguous()
    values = torch.randn(heads, frames, positions, head_dim, device=device)

    keep, group_size, merged = temporal_merge(keys, values, threshold=0.9)
    assert keep[0, :, 0].tolist() == [False, False, False, True, True, True]
    assert group_size[0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 1.0, 1.0]

    query = torch.randn(heads, 3, head_dim, device=device)
    flat_keys = keys.reshape(heads, -1, head_dim)
    scale = head_dim**-0.5
    exact = (
        torch.softmax(query @ flat_keys.transpose(1, 2) * scale, -1)
        @ values.reshape(heads, -1, head_dim)
    )
    scores = query @ flat_keys.transpose(1, 2) * scale + group_size.reshape(
        heads, 1, -1
    ).log()
    scores = scores.masked_fill(~keep.reshape(heads, 1, -1), float("-inf"))
    compacted = torch.softmax(scores, -1) @ merged.reshape(heads, -1, head_dim)
    torch.testing.assert_close(compacted, exact, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_osa_runs_dense_until_the_reference_chunk_then_freezes_its_policy():
    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend("osa", {"reference_chunk": 2})

    for chunk_index in range(0, 3):
        call = _self_forcing_call(device, chunk_index=chunk_index)
        backend.begin_forward(_geometry(chunk_index))
        assert backend.attend(call) is None  # dense up to and including chunk 2
    assert 0 in backend.policies

    frozen = backend.policies[0]
    call = _self_forcing_call(device, chunk_index=5)
    backend.begin_forward(_geometry(5))
    backend.attend(call)
    assert backend.policies[0] is frozen  # never recalibrated


@requires_cuda
def test_osa_replicate_freezes_the_last_chunk0_step_and_replicates():
    """The replicate policy: chunk 0's *last* denoising step is the oracle,
    the frozen per-head tile set repeats in every non-full frame, the achieved
    density tracks the requested one, and the gather + FA3 execution matches a
    masked reference."""
    from sglang.multimodal_gen.runtime.layers.attention.sparse.replicate_kernel import (
        ReplicateGatherPlan,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    backend = build_sparse_attention_backend(
        "osa",
        {
            "granularity": "replicate",
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
    reference = torch.softmax(scores, dim=-1) @ call.value[0].float().permute(
        1, 0, 2
    )
    torch.testing.assert_close(
        out[0].float(), reference.permute(1, 0, 2), atol=3e-2, rtol=3e-2
    )
