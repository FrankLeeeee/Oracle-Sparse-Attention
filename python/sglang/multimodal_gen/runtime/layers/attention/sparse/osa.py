# SPDX-License-Identifier: Apache-2.0
"""Oracle Sparse Attention (OSA) for block-causal video DiTs.

The design rests on one empirical property of the Self-Forcing family
(measured in ``notes/experiments/self-forcing.md``): a head's *temporal*
attention footprint is static in **chunk-relative** coordinates. If head ``i``
reads the sink chunk while generating chunk ``j``, it reads the sink chunk for
every chunk ``k > j``; if it reads the two most recent chunks at ``j``, it
reads the two most recent chunks at ``k``. Nothing about the footprint depends
on ``j`` beyond the sliding window that contains it.

That makes the pattern *observable once and reusable forever*, which is what
"oracle" means here — not an unattainable post-hoc top-k, but a real oracle
consulted once during generation:

1. Chunks ``0 .. R`` run dense. While generating the **reference chunk** ``R``
   (2 or 3; the first chunk with a representative history), the first denoising
   step recomputes ``softmax(q k^T)`` on a strided subset of that chunk's
   queries and reduces it to attention mass per chunk-relative bin, per
   ``(layer, head)``: the sink chunk, offsets ``-1, -2, ...``, and the head's
   own chunk.
2. Each ``(layer, head)`` then picks the cheapest policy that retains a target
   fraction of its mass: a boolean ``keep_sink`` and an integer
   ``num_recent`` — the number of most-recent past chunks it keeps. Its own
   chunk is always kept (attention is bidirectional inside a chunk). A head
   that needs *every* observed past chunk is marked dense, because the
   reference chunk cannot tell us where its true horizon ends.
3. Chunks ``> R`` apply the frozen policy. Kept latent frames are contiguous
   runs, so each head's key set is two or three token ranges and the shared
   block-sparse kernel walks them at nearly dense throughput.

Because the policy is frozen at ``R`` and expressed in chunk-relative terms,
the monotonicity the design asks for holds by construction rather than by
check: the kept set at chunk ``k`` is the same relative set as at chunk ``k-1``,
shifted forward by one chunk.

**Query-frame-relative (dt) granularity.** The 2026-08-16 stationarity
measurement (qk dumps at chunks 3..13 of a Self-Forcing run) sharpened the
design property: most heads' footprints are stationary in *query-frame*
coordinates, not merely chunk coordinates. Aligning the three query frames of a
chunk at identical ``dt = query_frame - key_frame`` correlates their profiles
at a median r of 0.9; aligning them chunk-relative reaches only 0.36-0.74. A
head that attends diagonally does so *per frame*, so a chunk-uniform mask pays
for the union of its three query frames' bands. ``granularity="dt"`` therefore
classifies each head into a small frame-level **pattern library** — diagonal
band, chunk-aligned block, periodic diagonals (``dt_comb``), absolute-periodic
vertical stripes (``v_comb``) — fits that family's parameters (band reach,
comb period/phase/width/depth, block depth), composes it with an
``edge_frames`` tail of oldest *visible* frames (the sink until eviction
begins, dt-stationary afterwards because the window slides), and freezes the
result. Selection is by cost: the cheapest family instance whose retained mass
reaches the target. Measured shares on Self-Forcing at retention 0.9: band
90%, block 6%, combs ~3% — the periodic families are rare on a 21-frame
window but their saving scales with window length. Every pattern compiles to
per-query-block token ranges for the shared kernel, whose throughput is
pattern-agnostic, so no specialized kernel is needed; plans stay memoized per
``(chunk, layer)``. Measured at steady state and retention 0.9 this costs a
density of ~0.50 against ~0.57 for the chunk-granular policy — and 0.57 is
exactly the shared kernel's break-even.

The same measurement quantified the reference-chunk limitation: policies frozen
on a partial window under-retain once the window outgrows the calibration
horizon (median 0.90 -> 0.75 by chunk 13 when frozen at chunk 3), while
policies frozen at the first full-window chunk hold 0.90 through the run.
``refresh_at_full_window`` re-observes once, on the first chunk whose window
has started evicting, and then freezes for good.
"""

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    LayoutCache,
    SparseAttentionBackend,
    SparseAttentionCall,
    SparseAttentionExecution,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    VisibleLayout,
    frame_mask_to_ranges,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    DEFAULT_BLOCK_M,
    plan_from_segment_mask,
    plan_from_shared_ranges,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class OsaConfig(msgspec.Struct, frozen=True):
    """Knobs of Oracle Sparse Attention.

    ``reference_chunk`` is the chunk whose attention is observed; everything up
    to and including it runs dense. ``retention`` is the fraction of each
    head's attention mass its policy must keep. ``sink_chunks`` defines how much
    of the video's start counts as the sink.
    """

    reference_chunk: int = 3
    retention: float = 0.9
    sink_chunks: int = 1
    # Unit of the recent window: "chunk" keeps whole 3-frame chunks (the
    # original design); "frame" keeps a contiguous window of latent frames,
    # cutting the selection quantum from frames_per_block/window to 1/window.
    # The chunk-relative stationarity argument is unchanged — a frame offset
    # is just a finer chunk-relative coordinate. "dt" goes one step further:
    # the policy is a per-head diagonal band around *each query frame*
    # (measured stationary in dt = query_frame - key_frame across both the
    # frames of a chunk and the chunks of a run) plus an optional window-edge
    # tail, so each query block keeps at most two key ranges instead of the
    # chunk-wide union.
    granularity: str = "chunk"
    # Sink size in latent frames; None falls back to sink_chunks whole chunks.
    # Frame granularity usually wants 1 (the classic first-frame sink).
    sink_latent_frames: int | None = None
    # Re-observe and refresh the frozen policies every N chunks after the
    # reference (0 = calibrate once, the original design). Chunk-relative
    # stationarity holds for *which* offsets a head prefers but not for the
    # *fraction* of mass near the chunk once the visible history outgrows the
    # calibration horizon (unbounded-KV runs): frozen policies then
    # under-retain and the output degrades. Caveat, measured on the 20 s
    # full-context runs: a refresh observes the sparse run's *own*
    # activations, so once quality slips the refreshed policies adapt to the
    # degradation (feedback) — recalibration at retention 0.3 made the output
    # *worse*. For unbounded-KV runs prefer a later reference chunk plus a
    # retention margin (reference_chunk 8, retention 0.7 recovered clean
    # 20 s video at ~0.20 sparse-call density); use recalibration only on top
    # of a configuration that already holds quality.
    recalibrate_every: int = 0
    # Re-observe once, on the first chunk whose visible window has started
    # evicting (the first *full* window), when the reference chunk itself saw a
    # partial one. Measured on the 2026-08-16 qk dumps: policies frozen on a
    # partial window under-retain once history outgrows the calibration
    # horizon (0.90 -> ~0.75 median retention), while policies frozen at the
    # first full-window chunk hold their target for the rest of the run. The
    # refresh observes the sparse run's own step-1 activations (the feedback
    # caveat on `recalibrate_every` applies), but a single refresh at a
    # near-target retention is measurably benign, unlike periodic
    # recalibration on a degraded run.
    refresh_at_full_window: bool = True
    # Query sub-sampling of the one calibration pass. The per-head frame
    # profile is an average over thousands of queries, so a stride of 8 costs
    # 1/8 of the recompute and moves the bins by well under a percent.
    calibration_query_stride: int = 8
    calibration_query_tile: int = 64


class HeadPolicies(msgspec.Struct, frozen=True):
    """One layer's frozen per-head policy (local heads under TP)."""

    keep_sink: np.ndarray  # bool [heads]
    num_recent: np.ndarray  # int32 [heads]
    dense: np.ndarray  # bool [heads]
    observed_retention: np.ndarray  # float32 [heads]


def choose_head_policies(
    *,
    own_mass: np.ndarray,  # [heads]
    sink_mass: np.ndarray,  # [heads]
    offset_mass: np.ndarray,  # [heads, num_past_chunks]; column j == offset -(j+1)
    retention: float,
    frames_per_block: int,
    sink_frames: int,
) -> HeadPolicies:
    """Cheapest ``(keep_sink, num_recent)`` per head that retains ``retention``.

    ``offset_mass`` excludes whatever the sink bin already accounts for, so the
    bins are disjoint and ``own + sink + offsets`` sums to one. Cost is counted
    in latent frames, the unit the kernel actually reads.

    A head whose minimal ``num_recent`` is the whole observed history is marked
    dense: the reference chunk saw no further past, so freezing the largest
    observable window would be an extrapolation the observation does not
    support.
    """
    num_heads, num_offsets = offset_mass.shape
    # Retained mass as a function of num_recent: own + prefix of the offsets.
    prefix = np.concatenate(
        [np.zeros((num_heads, 1), dtype=np.float64), np.cumsum(offset_mass, axis=1)],
        axis=1,
    )  # [heads, num_offsets + 1]
    base = own_mass[:, None] + prefix
    with_sink = base + sink_mass[:, None]

    recent_frames = np.arange(num_offsets + 1) * frames_per_block
    cost_no_sink = recent_frames[None, :].repeat(num_heads, axis=0).astype(np.float64)
    cost_with_sink = cost_no_sink + sink_frames

    keep_sink = np.zeros(num_heads, dtype=bool)
    num_recent = np.zeros(num_heads, dtype=np.int32)
    dense = np.zeros(num_heads, dtype=bool)
    achieved = np.zeros(num_heads, dtype=np.float32)

    for head in range(num_heads):
        best: tuple[float, bool, int, float] | None = None
        for use_sink in (False, True):
            retained = with_sink[head] if use_sink else base[head]
            costs = cost_with_sink[head] if use_sink else cost_no_sink[head]
            feasible = np.flatnonzero(retained >= retention)
            if feasible.size == 0:
                continue
            index = int(feasible[np.argmin(costs[feasible])])
            candidate = (float(costs[index]), use_sink, index, float(retained[index]))
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            dense[head] = True
            achieved[head] = 1.0
            continue
        _, use_sink, chosen_recent, retained_mass = best
        # Needing every observed past chunk means the horizon is not bounded by
        # what we saw; keep the head dense instead of guessing.
        if chosen_recent >= num_offsets and num_offsets > 0:
            dense[head] = True
            achieved[head] = 1.0
            continue
        keep_sink[head] = use_sink
        num_recent[head] = chosen_recent
        achieved[head] = retained_mass

    return HeadPolicies(
        keep_sink=keep_sink,
        num_recent=num_recent,
        dense=dense,
        observed_retention=achieved,
    )


def measure_query_frame_mass(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    layout: VisibleLayout,
    softmax_scale: float,
    query_stride: int,
    query_tile: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean attention mass per visible latent frame, per head and *query frame*.

    Recomputes the probabilities the fused kernel never materializes, for a
    strided subset of the chunk's queries, and reduces each query's row to one
    number per latent frame — keeping the query-frame axis, which is what the
    dt-granular policy is calibrated on. Returns
    ``(mass [heads, query_frames, frames], counts [query_frames])`` where each
    ``mass[h, qf]`` row sums to one.
    """
    batch, q_len, num_heads, _ = query.shape
    num_frames = layout.num_frames
    frame_seqlen = layout.frame_seqlen
    num_query_frames = layout.query_frames
    sampled = query[:1, ::query_stride]  # rank-0 batch element is enough
    keys = key[:1]
    sample_frames = (
        torch.arange(0, q_len, query_stride, device=query.device) // frame_seqlen
    )

    mass = torch.zeros(
        num_heads,
        num_query_frames,
        num_frames,
        dtype=torch.float32,
        device=query.device,
    )
    counts = torch.zeros(num_query_frames, dtype=torch.float32, device=query.device)
    num_sampled = sampled.shape[1]
    for tile_start in range(0, num_sampled, query_tile):
        tile = sampled[:, tile_start : tile_start + query_tile]
        scores = torch.einsum("bqhd,bkhd->hqk", tile.float(), keys.float())
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        per_frame = probs.view(num_heads, -1, num_frames, frame_seqlen).sum(-1)
        tile_frames = sample_frames[tile_start : tile_start + per_frame.shape[1]]
        mass.index_add_(1, tile_frames, per_frame)
        counts.index_add_(0, tile_frames, torch.ones_like(tile_frames, dtype=counts.dtype))
    del batch
    mass /= counts.clamp(min=1.0)[None, :, None]
    return (
        mass.cpu().numpy().astype(np.float64),
        counts.cpu().numpy().astype(np.float64),
    )


def measure_chunk_relative_mass(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    layout: VisibleLayout,
    softmax_scale: float,
    query_stride: int,
    query_tile: int,
) -> np.ndarray:
    """Mean attention mass per visible latent frame, per head: ``[heads, frames]``.

    The query-frame-pooled view of :func:`measure_query_frame_mass`, which the
    chunk- and frame-granular policies are calibrated on.
    """
    mass, counts = measure_query_frame_mass(
        query=query,
        key=key,
        layout=layout,
        softmax_scale=softmax_scale,
        query_stride=query_stride,
        query_tile=query_tile,
    )
    weights = counts / max(counts.sum(), 1.0)
    return (mass * weights[None, :, None]).sum(axis=1)


def fold_mass_into_bins(
    frame_mass: np.ndarray,
    *,
    layout: VisibleLayout,
    sink_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split per-frame mass into ``(own, sink, offsets)`` bins.

    The sink bin wins ties: frames that are both "the start of the video" and
    "chunk offset ``-3``" are counted once, in the sink bin, so a head that
    keeps the sink is not also charged for an offset.
    """
    sink = layout.sink_frames(sink_frames)
    own = layout.own_frames
    own_mass = frame_mass[:, own].sum(axis=1)
    sink_mass = frame_mass[:, sink & ~own].sum(axis=1)

    num_past = layout.num_past_chunks
    num_heads = frame_mass.shape[0]
    offset_mass = np.zeros((num_heads, num_past), dtype=np.float64)
    for j in range(num_past):
        frames = layout.frames_of_offset(-(j + 1))
        offset_mass[:, j] = frame_mass[:, frames & ~sink].sum(axis=1)
    return own_mass, sink_mass, offset_mass


def frame_ages(layout: VisibleLayout) -> np.ndarray:
    """Age of each visible frame in latent frames; own chunk is age <= 0."""
    own_start = int(layout.global_frame_ids[layout.own_frames].min())
    return own_start - layout.global_frame_ids


def fold_mass_into_frame_bins(
    frame_mass: np.ndarray,
    *,
    layout: VisibleLayout,
    sink_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split per-frame mass into ``(own, sink, ages)`` bins, one column per frame.

    Column ``j`` is the mass on the past frame of age ``j + 1`` (the frame
    immediately before the chunk first). Same disjointness convention as
    :func:`fold_mass_into_bins`: the sink bin wins the overlap.
    """
    sink = layout.sink_frames(sink_frames)
    own = layout.own_frames
    own_mass = frame_mass[:, own].sum(axis=1)
    sink_mass = frame_mass[:, sink & ~own].sum(axis=1)

    ages = frame_ages(layout)
    num_past = int(ages.max(initial=0))
    num_heads = frame_mass.shape[0]
    age_mass = np.zeros((num_heads, num_past), dtype=np.float64)
    for j in range(num_past):
        frames = ages == j + 1
        age_mass[:, j] = frame_mass[:, frames & ~sink].sum(axis=1)
    return own_mass, sink_mass, age_mass


def policy_frame_mask(
    policies: HeadPolicies,
    *,
    layout: VisibleLayout,
    sink_frames: int,
    frame_granular: bool = False,
) -> np.ndarray:
    """``[heads, frames]`` keep mask of a frozen policy on this visible window.

    ``num_recent`` counts chunks by default and latent frames when the policy
    was chosen at frame granularity.
    """
    own = layout.own_frames
    sink = layout.sink_frames(sink_frames)
    if frame_granular:
        age = frame_ages(layout)
        past = age > 0
    else:
        # -chunk_offsets is the age in chunks; own chunk is age 0.
        age = -layout.chunk_offsets
        past = layout.chunk_offsets < 0
    recent = past[None, :] & (age[None, :] <= policies.num_recent[:, None])
    keep = own[None, :] | recent | (policies.keep_sink[:, None] & sink[None, :])
    keep |= policies.dense[:, None]
    return keep


# Pattern families of the frame-level policy. Every family is stationary in
# the coordinates it is stated in, so its parameters freeze: BAND and DT_COMB
# in dt = query_frame - key_frame, V_COMB in absolute frame ids modulo its
# period, BLOCK in chunk offsets. Measured shares at retention 0.9 on the
# 2026-08-16 Self-Forcing dumps: band 90%, block 6%, dt_comb 3%, v_comb 1% —
# the exotic families are rare on a 21-frame window but scale with window
# length (a period-P comb saves ~(1 - 1/P) of whatever the window adds).
PATTERN_BAND = 0  # params (num_past, num_future, 0, 0)
PATTERN_BLOCK = 1  # params (num_recent_chunks, 0, 0, 0)
PATTERN_DT_COMB = 2  # params (period, start_dt, width, depth)
PATTERN_V_COMB = 3  # params (period, phase, width, 0)

PATTERN_NAMES = ("band", "block", "dt_comb", "v_comb")


class PatternHeadPolicies(msgspec.Struct, frozen=True):
    """One layer's frozen per-head frame-level pattern (local heads under TP).

    ``pattern[h]`` names the family, ``params[h]`` its fitted parameters (see
    the ``PATTERN_*`` comments), ``edge_frames[h]`` a tail of oldest *visible*
    frames that composes with every family — the sink until eviction begins,
    and dt-stationary afterwards because the window slides with the query.
    """

    pattern: np.ndarray  # int8 [heads]
    params: np.ndarray  # int32 [heads, 4]
    edge_frames: np.ndarray  # int32 [heads]
    dense: np.ndarray  # bool [heads]
    observed_retention: np.ndarray  # float32 [heads]


def _pattern_keep(
    pattern: int,
    params: tuple[int, int, int, int],
    *,
    query_frame_ids: np.ndarray,
    key_frame_ids: np.ndarray,
    frames_per_block: int,
) -> np.ndarray:
    """``[query_frames, frames]`` keep mask of one family instance (no edge)."""
    key_row = key_frame_ids[None, :]
    query_column = query_frame_ids[:, None]
    if pattern == PATTERN_BAND:
        num_past, num_future = params[0], params[1]
        return (key_row >= query_column - num_past) & (
            key_row <= query_column + num_future
        )
    if pattern == PATTERN_BLOCK:
        num_recent = params[0]
        chunk_offset = key_row // frames_per_block - query_column // frames_per_block
        return (chunk_offset >= -num_recent) & (chunk_offset <= 0)
    if pattern == PATTERN_DT_COMB:
        period, start, width, depth = params
        dt = query_column - key_row
        keep = np.zeros(dt.shape, dtype=bool)
        for tooth in range(depth):
            first = start + tooth * period
            keep |= (dt >= first) & (dt < first + width)
        # the own frame is always kept — a comb whose phase misses dt 0 would
        # otherwise leave a query block with no self-attention
        return keep | (dt == 0)
    if pattern == PATTERN_V_COMB:
        period, phase, width = params[0], params[1], params[2]
        stripes = (key_row - phase) % period < width
        return stripes | (query_column == key_row)
    raise ValueError(f"unknown pattern {pattern}")


def pattern_keep_per_query_frame(
    policies: PatternHeadPolicies,
    *,
    query_frame_ids: np.ndarray,
    key_frame_ids: np.ndarray,
    frames_per_block: int,
) -> np.ndarray:
    """``[heads, query_frames, frames]`` keep mask of a frozen pattern policy."""
    num_heads = policies.pattern.shape[0]
    keep = np.zeros(
        (num_heads, query_frame_ids.size, key_frame_ids.size), dtype=bool
    )
    view_order = np.arange(key_frame_ids.size)[None, :]
    for head in range(num_heads):
        if policies.dense[head]:
            keep[head] = True
            continue
        keep[head] = _pattern_keep(
            int(policies.pattern[head]),
            tuple(policies.params[head]),
            query_frame_ids=query_frame_ids,
            key_frame_ids=key_frame_ids,
            frames_per_block=frames_per_block,
        ) | (view_order < policies.edge_frames[head])
    return keep


def enumerate_pattern_candidates(
    *,
    query_frame_ids: np.ndarray,
    key_frame_ids: np.ndarray,
    frames_per_block: int,
    max_future: int,
):
    """Yield ``(pattern, params)`` for every fittable family instance.

    The saturation rules mirror the chunk-granular chooser: an instance whose
    deepest reach equals the deepest past the observation could see is not
    yielded, because freezing it would extrapolate an unobserved horizon.
    V_COMB is exempt — absolute periodicity *is* its extrapolation rule — but
    must have shown at least two stripes to count as observed.
    """
    horizon = int(query_frame_ids.min() - key_frame_ids.min())
    num_frames = key_frame_ids.size
    num_past_chunks = int(
        query_frame_ids.min() // frames_per_block
        - key_frame_ids.min() // frames_per_block
    )
    for num_past in range(max(1, horizon)):
        for num_future in range(max_future + 1):
            yield PATTERN_BAND, (num_past, num_future, 0, 0)
    for num_recent in range(max(1, num_past_chunks)):
        yield PATTERN_BLOCK, (num_recent, 0, 0, 0)
    for period in (2, 3, 4, 5, 6):
        for width in (1, 2):
            if width >= period:
                continue
            for start in range(min(period, frames_per_block)):
                for depth in (2, 3, 4, 5, 6):
                    if start + (depth - 1) * period + width - 1 >= horizon:
                        continue  # the comb's deepest tooth saturates
                    yield PATTERN_DT_COMB, (period, start, width, depth)
            if num_frames >= 2 * period:  # at least two stripes observed
                for phase in range(period):
                    yield PATTERN_V_COMB, (period, phase, width, 0)


def choose_pattern_head_policies(
    *,
    query_frame_mass: np.ndarray,  # [heads, query_frames, frames], rows sum to 1
    query_frame_ids: np.ndarray,  # int64 [query_frames], global latent frames
    key_frame_ids: np.ndarray,  # int64 [frames], global, in key order
    retention: float,
    frames_per_block: int,
    max_future: int,
    max_edge_frames: int,
) -> PatternHeadPolicies:
    """Cheapest ``(pattern, params, edge_frames)`` per head at ``retention``.

    Model selection by cost: every family instance from
    :func:`enumerate_pattern_candidates`, composed with every edge-tail size,
    competes on the mean number of kept frames per query frame. Heads no
    instance can satisfy are dense.
    """
    num_heads = query_frame_mass.shape[0]
    view_order = np.arange(key_frame_ids.size)[None, :]

    best_cost = np.full(num_heads, np.inf)
    best_pattern = np.zeros(num_heads, dtype=np.int8)
    best_params = np.zeros((num_heads, 4), dtype=np.int32)
    best_edge = np.zeros(num_heads, dtype=np.int32)
    best_retained = np.zeros(num_heads, dtype=np.float32)

    for pattern, params in enumerate_pattern_candidates(
        query_frame_ids=query_frame_ids,
        key_frame_ids=key_frame_ids,
        frames_per_block=frames_per_block,
        max_future=max_future,
    ):
        base = _pattern_keep(
            pattern,
            params,
            query_frame_ids=query_frame_ids,
            key_frame_ids=key_frame_ids,
            frames_per_block=frames_per_block,
        )
        for edge in range(max_edge_frames + 1):
            keep = base | (view_order < edge)
            cost = float(keep.sum(axis=1).mean())
            if not np.any(cost < best_cost):
                continue
            retained = (query_frame_mass * keep[None]).sum(axis=2).mean(axis=1)
            better = (retained >= retention) & (cost < best_cost)
            best_cost[better] = cost
            best_pattern[better] = pattern
            best_params[better] = params
            best_edge[better] = edge
            best_retained[better] = retained[better]

    dense = np.isinf(best_cost)
    best_retained[dense] = 1.0
    return PatternHeadPolicies(
        pattern=np.where(dense, 0, best_pattern).astype(np.int8),
        params=np.where(dense[:, None], 0, best_params).astype(np.int32),
        edge_frames=np.where(dense, 0, best_edge).astype(np.int32),
        dense=dense,
        observed_retention=best_retained,
    )


def dt_policy_qblock_mask(
    policies: PatternHeadPolicies,
    *,
    layout: VisibleLayout,
    block_m: int,
) -> np.ndarray | None:
    """``[heads, q_blocks, frames]`` keep mask, or ``None`` if unmappable.

    A query block that straddles a frame boundary keeps the union of the two
    frames' patterns. ``None`` means the query is not exactly the own chunk's
    frames, which this policy has no coordinates for.
    """
    query_frame_ids = layout.global_frame_ids[layout.own_frames]
    if query_frame_ids.size != layout.query_frames:
        return None
    per_frame = pattern_keep_per_query_frame(
        policies,
        query_frame_ids=query_frame_ids,
        key_frame_ids=layout.global_frame_ids,
        frames_per_block=layout.frames_per_block,
    )
    q_len = layout.query_frames * layout.frame_seqlen
    num_blocks = (q_len + block_m - 1) // block_m
    keep = np.zeros(
        (per_frame.shape[0], num_blocks, layout.num_frames), dtype=bool
    )
    for block in range(num_blocks):
        first = (block * block_m) // layout.frame_seqlen
        last = (min((block + 1) * block_m, q_len) - 1) // layout.frame_seqlen
        keep[:, block] = per_frame[:, first : last + 1].any(axis=1)
    return keep


class OracleSparseAttention(SparseAttentionBackend):
    """OSA: observe the reference chunk once, then run the frozen per-head plan."""

    name = "osa"

    def __init__(self, config: OsaConfig) -> None:
        super().__init__()
        if config.granularity not in ("chunk", "frame", "dt"):
            raise ValueError(
                "granularity must be 'chunk', 'frame' or 'dt', got "
                f"{config.granularity!r}"
            )
        if config.recalibrate_every < 0:
            raise ValueError("recalibrate_every must be non-negative")
        self._config = config
        self._frame_granular = config.granularity == "frame"
        self._dt_granular = config.granularity == "dt"
        self._policies: dict[int, HeadPolicies | PatternHeadPolicies] = {}
        # Chunk each layer's policies were last observed at.
        self._calibrated_at: dict[int, int] = {}
        # Oldest visible global frame at each layer's calibration; 0 means the
        # window had not started evicting yet, i.e. the observation may not
        # have seen the full steady-state horizon.
        self._calibration_window_start: dict[int, int] = {}
        self._plans = LayoutCache()
        self._last_chunk_index = -1
        self._logged_summary = False

    def _on_begin_forward(self, geometry: ChunkGeometry) -> None:
        chunk_index = geometry.query_chunk_index
        if chunk_index < self._last_chunk_index:
            # A new video restarts the chunk counter; the previous video's
            # calibration says nothing about this one.
            self.reset()
        self._last_chunk_index = chunk_index

    def reset(self) -> None:
        self._policies.clear()
        self._calibrated_at.clear()
        self._calibration_window_start.clear()
        self._plans.clear()
        self._last_chunk_index = -1
        self._logged_summary = False

    @property
    def policies(self) -> dict[int, HeadPolicies | PatternHeadPolicies]:
        return self._policies

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        chunk_index = layout.query_chunk_index
        reference = self._config.reference_chunk
        if chunk_index < reference:
            return None
        if chunk_index == reference:
            if call.layer_index not in self._policies:
                self._calibrate(call, layout)
            return None

        policies = self._policies.get(call.layer_index)
        if policies is None:
            self.warn_dense_once(
                f"layer {call.layer_index} was never calibrated "
                f"(reference chunk {reference} not seen)"
            )
            return None
        every = self._config.recalibrate_every
        scheduled = every > 0 and chunk_index >= (
            self._calibrated_at.get(call.layer_index, reference) + every
        )
        # One-shot refresh at the first full window: the reference chunk saw a
        # window that had not started evicting, so its policies may not bound
        # the steady-state horizon; the first evicting chunk's window is the
        # deepest this model will ever show.
        window_refresh = (
            self._config.refresh_at_full_window
            and self._calibration_window_start.get(call.layer_index, 1) == 0
            and int(layout.global_frame_ids.min()) > 0
        )
        if scheduled or window_refresh:
            # Refresh on this chunk's first denoising step (same observation
            # point as the reference-chunk calibration) and apply immediately.
            self._calibrate(call, layout)
            policies = self._policies[call.layer_index]
        plan = self._plan(call, layout, policies)
        if plan is None:
            return None
        return SparseAttentionExecution(
            plan=plan, query=call.query, key=call.key, value=call.value
        )

    def _sink_frames(self) -> int:
        if self._config.sink_latent_frames is not None:
            return self._config.sink_latent_frames
        return self._config.sink_chunks * self._geometry.frames_per_block

    def _calibrate(self, call: SparseAttentionCall, layout: VisibleLayout) -> None:
        config = self._config
        if self._dt_granular:
            self._calibrate_dt(call, layout)
            return
        frame_mass = measure_chunk_relative_mass(
            query=call.query,
            key=call.key,
            layout=layout,
            softmax_scale=call.softmax_scale,
            query_stride=config.calibration_query_stride,
            query_tile=config.calibration_query_tile,
        )
        if self._frame_granular:
            own_mass, sink_mass, offset_mass = fold_mass_into_frame_bins(
                frame_mass, layout=layout, sink_frames=self._sink_frames()
            )
            frames_per_unit = 1
        else:
            own_mass, sink_mass, offset_mass = fold_mass_into_bins(
                frame_mass, layout=layout, sink_frames=self._sink_frames()
            )
            frames_per_unit = layout.frames_per_block
        self._policies[call.layer_index] = choose_head_policies(
            own_mass=own_mass,
            sink_mass=sink_mass,
            offset_mass=offset_mass,
            retention=config.retention,
            frames_per_block=frames_per_unit,
            sink_frames=self._sink_frames(),
        )
        self._record_calibration(call.layer_index, layout)

    def _calibrate_dt(self, call: SparseAttentionCall, layout: VisibleLayout) -> None:
        config = self._config
        query_frame_ids = layout.global_frame_ids[layout.own_frames]
        if query_frame_ids.size != layout.query_frames:
            self.warn_dense_once(
                "dt calibration needs the query to be exactly the own chunk's "
                f"frames; layer {call.layer_index} sees {layout.query_frames} "
                f"query frames but {query_frame_ids.size} own-chunk frames"
            )
            return
        mass, _ = measure_query_frame_mass(
            query=call.query,
            key=call.key,
            layout=layout,
            softmax_scale=call.softmax_scale,
            query_stride=config.calibration_query_stride,
            query_tile=config.calibration_query_tile,
        )
        self._policies[call.layer_index] = choose_pattern_head_policies(
            query_frame_mass=mass,
            query_frame_ids=query_frame_ids,
            key_frame_ids=layout.global_frame_ids,
            retention=config.retention,
            frames_per_block=layout.frames_per_block,
            max_future=layout.frames_per_block - 1,
            max_edge_frames=self._sink_frames(),
        )
        self._record_calibration(call.layer_index, layout)

    def _record_calibration(self, layer_index: int, layout: VisibleLayout) -> None:
        self._calibrated_at[layer_index] = layout.query_chunk_index
        self._calibration_window_start[layer_index] = int(
            layout.global_frame_ids.min()
        )

    def _plan(
        self,
        call: SparseAttentionCall,
        layout: VisibleLayout,
        policies: HeadPolicies | PatternHeadPolicies,
    ):
        signature = (
            call.key_segments,
            layout.query_frames,
            call.head_start,
            call.num_local_heads,
        )
        hit, cached = self._plans.get(call.layer_index, signature)
        if hit:
            return cached
        if self._dt_granular:
            keep = dt_policy_qblock_mask(
                policies, layout=layout, block_m=DEFAULT_BLOCK_M
            )
            if keep is None:
                self.warn_dense_once(
                    "the query is not exactly the own chunk's frames; the dt "
                    "policy has no coordinates for it"
                )
        else:
            keep = policy_frame_mask(
                policies,
                layout=layout,
                sink_frames=self._sink_frames(),
                frame_granular=self._frame_granular,
            )
        if keep is not None and keep.shape[0] != call.num_local_heads:
            self.warn_dense_once(
                f"calibrated {keep.shape[0]} heads but layer "
                f"{call.layer_index} has {call.num_local_heads}"
            )
            keep = None
        plan = None
        if keep is not None and not keep.all():
            if self._dt_granular:
                plan = self._dt_plan_from_mask(keep, layout, call.query.device)
            else:
                plan = plan_from_shared_ranges(
                    frame_mask_to_ranges(keep, frame_seqlen=layout.frame_seqlen),
                    block_m=DEFAULT_BLOCK_M,
                    device=call.query.device,
                )
        self._plans.put(call.layer_index, signature, plan)
        self._log_summary(layout)
        return plan

    @staticmethod
    def _dt_plan_from_mask(
        keep: np.ndarray, layout: VisibleLayout, device: torch.device
    ):
        """Per-query-block plan from a ``[heads, q_blocks, frames]`` keep mask.

        Latent frames are the segments; runs of kept frames merge into single
        token ranges, so a head's band plus its edge tail is at most two.
        """
        frame_ids = torch.arange(layout.num_frames, dtype=torch.int32)
        return plan_from_segment_mask(
            torch.from_numpy(keep).to(device, non_blocking=True),
            segment_starts=(frame_ids * layout.frame_seqlen).to(device),
            segment_ends=((frame_ids + 1) * layout.frame_seqlen).to(device),
            block_m=DEFAULT_BLOCK_M,
        )

    def _log_summary(self, layout: VisibleLayout) -> None:
        if self._logged_summary or not self._policies:
            return
        self._logged_summary = True
        if self._dt_granular:
            dense = np.concatenate([p.dense for p in self._policies.values()])
            pattern = np.concatenate([p.pattern for p in self._policies.values()])
            edge = np.concatenate([p.edge_frames for p in self._policies.values()])
            live = ~dense
            shares = " ".join(
                f"{name} {100.0 * ((pattern == kind) & live).mean():.0f}%"
                for kind, name in enumerate(PATTERN_NAMES)
            )
            logger.info(
                "OSA (dt) calibrated %d layers at chunk %d: %.0f%% of heads "
                "dense, patterns %s, edge mean %.2f frames; visible window %d "
                "frames",
                len(self._policies),
                self._config.reference_chunk,
                100.0 * dense.mean(),
                shares,
                float(edge[live].mean()) if live.any() else 0.0,
                layout.num_frames,
            )
            return
        dense = np.concatenate([p.dense for p in self._policies.values()])
        keep_sink = np.concatenate([p.keep_sink for p in self._policies.values()])
        num_recent = np.concatenate([p.num_recent for p in self._policies.values()])
        logger.info(
            "OSA calibrated %d layers at chunk %d: %.0f%% of heads dense, "
            "%.0f%% keep the sink, recent window mean %.2f (max %d) "
            + ("frames" if self._frame_granular else "chunks")
            + "; visible window %d frames",
            len(self._policies),
            self._config.reference_chunk,
            100.0 * dense.mean(),
            100.0 * keep_sink[~dense].mean() if (~dense).any() else 0.0,
            float(num_recent[~dense].mean()) if (~dense).any() else 0.0,
            int(num_recent.max(initial=0)),
            layout.num_frames,
        )
