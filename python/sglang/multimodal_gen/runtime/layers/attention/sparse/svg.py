# SPDX-License-Identifier: Apache-2.0
"""Sparse VideoGen baselines — SVG1 (spatial/temporal heads) and SVG2 (semantic).

Reproduces https://github.com/svg-project/Sparse-VideoGen for the block-causal
setting. Both were designed for bidirectional diffusion over a whole clip; here
the query side is only the chunk being generated and the key side is the
visible KV view, which is exactly the adaptation the user asked for.

**SVG1** rests on the observation that a video-DiT head is one of two kinds. A
*spatial* head reads a temporal neighbourhood of whole latent frames; a
*temporal* head reads the same spatial position in every frame. Neither mask is
chosen offline: at every denoising step a strided sample of the chunk's queries
is attended exactly, both masks are scored against that exact output, and each
head takes the mask with the lower error.

**SVG2** drops the two hand-designed patterns for a semantic one. Keys are
clustered (k-means over the key vectors), the sequence is permuted so each
cluster is contiguous, and every query block keeps the clusters whose
centroid-estimated mass reaches ``top_p``. Here the clustering runs over the
*history* keys only — they are the part that is stable across the denoising
steps of a chunk, so the clustering is paid once per ``(chunk, layer)`` — while
the chunk's own keys stay dense and in place.
"""

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    LayoutCache,
    SparseAttentionBackend,
    SparseAttentionCall,
    SparseAttentionExecution,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.blocks import block_bounds
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    VisibleLayout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_segment_mask,
    sparse_attention,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (
    frame_aligned_block_bounds,
)


class Svg1Config(msgspec.Struct, frozen=True):
    block: int = 128
    # Upstream's `block_thres`, as a multiple of the frame length: both masks are
    # token-distance bands of half-width `band_frames * frame_seqlen`, the
    # spatial one in natural (frame-major) order and the temporal one in
    # spatial-major order. Upstream hard-codes its *profiling* masks at 2
    # frames (`svg/models/wan/utils.py`) but executes a band derived from its
    # `sparsity` knob via `sparsity_to_width`; here one width drives both
    # scoring and execution, and it is the knob to turn for a sparsity sweep.
    band_frames: float = 2.0
    # Upstream's `pixel_attn_mask[:, :frame_size] = 1` — the leading frame(s)
    # of the view are a sink for *both* candidate masks. Upstream protects one
    # frame; models distilled onto a multi-frame sink block (Rolling Forcing 3,
    # LongLive-2 8, LingBot 9) set this to their sink size, since the sink
    # block is exactly the leading frames of their visible view.
    dense_sink_frames: int = 1
    # Query blocks attended exactly to score the two candidate masks. Upstream
    # samples 32 individual rows; whole blocks keep the profiling pass on the
    # block-sparse fast path, and two of them is the nearest equivalent cost.
    num_sampled_blocks: int = 2
    # Key-segment granularity of the executed masks, in tokens. The temporal
    # mask keeps a narrow spatial-cell interval of every visible frame, so its
    # per-frame ranges quantize on this grid; 32 keeps the quantization loss
    # a few percent while adjacent kept segments still merge into one kernel
    # range each.
    key_tile: int = 32


class Svg2Config(msgspec.Struct, frozen=True):
    block: int = 128
    # Average keys per semantic cluster; the cluster count follows from the
    # history length. Upstream's Wan config works out to ~33 tokens/cluster on
    # a fixed 33k-token clip; a constant tokens-per-cluster would make the
    # assignment matmul quadratic in KV length on these unbounded-KV runs, so
    # the default trades cluster resolution for a bounded planning cost.
    cluster_size: int = 256
    # Cold-start iterations for a layer's first clustering, and the warm
    # iterations used when the previous chunk's centroids seed the next one
    # (upstream's `kmeans_step` warm-starts from cached centroids the same
    # way; the history only grows by one chunk between refits). 0 warm
    # iterations disables the warm start.
    kmeans_iters: int = 4
    kmeans_warm_iters: int = 2
    top_p: float = 0.9


def svg1_token_masks(
    *,
    num_frames: int,
    frame_seqlen: int,
    band_frames: float,
    dense_sink_frames: int,
    block: int = 128,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(spatial, temporal)`` token-level masks, upstream's operations in order.

    Mirrors ``svg/models/wan/utils.py::get_attention_mask`` step for step, because
    the order of its steps is load-bearing:

    1. a **block-quantized** band — ``|i // block - j // block| < band_tokens //
       block`` — not a token-distance band, so its edges sit on block boundaries;
    2. the first ``frame_seqlen`` *columns* forced on (upstream's sink);
    3. for the temporal mask only, a permutation
       ``(frame_seqlen, num_frames, frame_seqlen, num_frames) -> (1, 0, 3, 2)``.

    Because the sink is applied *before* the permutation, the temporal mask's
    always-on columns are not the first frame: they are the columns whose
    spatial-major index is below ``frame_seqlen``, i.e. the lowest
    ``frame_seqlen / num_frames`` spatial cells of *every* frame. That is
    upstream's behaviour, quirk included.

    Materializes ``total x total``, so it is for tests and for defining the
    block-level builder — at Wan's geometry it would be a gigabyte.
    """
    total = num_frames * frame_seqlen
    index = torch.arange(total, device=device)
    band_blocks = int(band_frames * frame_seqlen) // block
    banded = ((index[:, None] // block) - (index[None, :] // block)).abs() < band_blocks
    banded[:, : dense_sink_frames * frame_seqlen] = True

    permuted = (
        banded.reshape(frame_seqlen, num_frames, frame_seqlen, num_frames)
        .permute(1, 0, 3, 2)
        .reshape(total, total)
    )
    return banded, permuted


def _spatial_major_index(
    tokens: torch.Tensor, *, frame_seqlen: int, num_frames: int
) -> torch.Tensor:
    """The pre-permutation index of a natural token index: ``cell * num_frames + frame``."""
    return (tokens % frame_seqlen) * num_frames + tokens // frame_seqlen


def chunk_spatial_major_permutation(
    *, q_len: int, frame_seqlen: int, device: torch.device
) -> torch.Tensor:
    """Spatial-major order of the chunk's own queries: ``[q_len]`` long.

    Sorts the chunk's tokens by (spatial cell, frame), which is upstream's
    temporal-head placement restricted to the query side. A 128-query block
    then covers ~``128 / query_frames`` consecutive spatial cells instead of
    128, which is what keeps the per-block union of the spatial-major band
    narrow.
    """
    tokens = torch.arange(q_len, device=device)
    query_frames = q_len // frame_seqlen
    spatial_major = (tokens % frame_seqlen) * query_frames + tokens // frame_seqlen
    return torch.argsort(spatial_major)


def build_svg1_segment_masks(
    *,
    layout: VisibleLayout,
    q_len: int,
    kv_len: int,
    config: Svg1Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(spatial, temporal, query_permutation)`` for one visible layout.

    Both masks are ``[q_blocks, segments]`` over frame-aligned ``key_tile``
    segments (:func:`frame_aligned_block_bounds`). The token-level masks are
    exactly :func:`svg1_token_masks`'s; what this builder chooses is the
    *executed* reduction, and two choices keep the temporal mask's executed
    density near its token density instead of collapsing to ~2x it (the fate
    of a global 128-block any-overlap reduction, 0.42 vs 0.22 at Self-Forcing
    geometry):

    * temporal rows are taken in **spatial-major query order** — upstream's
      head placement applied to the query side only, keys stay in place; the
      returned permutation applies to temporal-classified heads' queries and
      the executed rows of ``temporal[b]`` are ``query_permutation[b*block :
      (b+1)*block]``;
    * the key axis is quantized on per-frame ``key_tile`` segments, so the
      per-frame spatial-cell interval the band solves to is kept at token
      rather than 128-block resolution.
    """
    block = config.block
    tile = config.key_tile
    frame_seqlen = layout.frame_seqlen
    num_frames = layout.num_frames
    band_blocks = int(config.band_frames * frame_seqlen) // block
    offset = kv_len - q_len  # the chunk's queries are the last keys of the view
    sink_tokens = min(config.dense_sink_frames, num_frames) * frame_seqlen
    sink = sink_tokens > 0

    seg_lo, seg_hi = frame_aligned_block_bounds(
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        block=tile,
        device=device,
    )
    frames = torch.arange(num_frames, device=device)
    tiles_per_frame = -(-frame_seqlen // tile)
    within_lo = seg_lo[:tiles_per_frame]  # spatial-cell tile bounds in one frame
    within_hi = seg_hi[:tiles_per_frame].clamp(max=frame_seqlen)

    def overlapping(a: int, b: int) -> torch.Tensor:
        """Segments overlapping view-token interval ``[a, b)``: ``[segments]``."""
        return (seg_hi > a) & (seg_lo < b)

    permutation = chunk_spatial_major_permutation(
        q_len=q_len, frame_seqlen=frame_seqlen, device=device
    )
    q_lo, q_hi = block_bounds(q_len, block, device=device)
    spatial_rows = []
    temporal_rows = []
    for lo, hi in zip(q_lo.tolist(), q_hi.tolist(), strict=True):
        # Natural-order band, block-quantized on the global grid exactly as
        # upstream states it; as a token set it is one interval per query block.
        rlo, rhi = lo + offset, hi + offset
        row_block_min, row_block_max = rlo // block, (rhi - 1) // block
        spatial = overlapping(
            max(0, (row_block_min - band_blocks + 1) * block),
            min(kv_len, (row_block_max + band_blocks) * block),
        )
        if sink:
            spatial = spatial | overlapping(0, sink_tokens)
        spatial_rows.append(spatial | overlapping(rlo, rhi))

        # Spatial-major band over sm = cell * F + f, for the *permuted* rows
        # of this block. Row-to-row sm steps are < block, so the union over
        # the block's rows fills the sm-block interval; the kept token set per
        # frame f is then the spatial-cell interval that sm in [A, B) solves
        # to, exact on the tile grid.
        rows = permutation[lo:hi] + offset
        sm = (rows % frame_seqlen) * num_frames + rows // frame_seqlen
        sm_block_min = int(sm.min()) // block
        sm_block_max = int(sm.max()) // block
        band_lo = max(0, (sm_block_min - band_blocks + 1) * block)
        band_hi = min(num_frames * frame_seqlen, (sm_block_max + band_blocks) * block)
        cell_lo = ((band_lo - frames + num_frames - 1) // num_frames).clamp(min=0)
        cell_hi = ((band_hi - frames + num_frames - 1) // num_frames).clamp(
            min=0, max=frame_seqlen
        )
        kept = (within_hi[None, :] > cell_lo[:, None]) & (
            within_lo[None, :] < cell_hi[:, None]
        )
        if sink:
            # Upstream applies the sink before the permutation, so the
            # temporal sink is sm < sink_tokens: the lowest spatial cells of
            # every frame, not the leading frames themselves.
            sink_hi = ((sink_tokens - frames + num_frames - 1) // num_frames).clamp(
                min=0
            )
            kept |= within_lo[None, :] < sink_hi[:, None]
        temporal = kept.reshape(-1)
        if band_blocks < 1:
            # Degenerate band: guarantee each permuted row still sees itself.
            temporal = temporal | overlapping(int(rows.min()), int(rows.max()) + 1)
        temporal_rows.append(temporal)

    return torch.stack(spatial_rows), torch.stack(temporal_rows), permutation


def choose_mask_per_head(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,
    value: torch.Tensor,
    candidates: list,  # (plan, sampled_rows) per candidate
    softmax_scale: float,
) -> torch.Tensor:
    """Index of the candidate mask with the lowest error, per head: ``[heads]``.

    Upstream's ``sample_mse``: exact attention on a small sample of the queries,
    each candidate mask scored against it by mean squared error, lowest wins.
    Each candidate's ``sampled_rows`` are in its own executed query order
    (natural for the spatial mask, spatial-major for the temporal one), so the
    plan's sampled query blocks line up with the rows they were built for.

    Two deviations from ``svg/models/wan/attention.py``, both for cost. Upstream
    samples 32 individual random rows and evaluates the candidates by masking a
    materialized ``[heads, sampled, kv_len]`` score matrix; we sample whole query
    *blocks* and evaluate through the same block-sparse kernel the method will
    actually use, because a masked SDPA on that intermediate costs more than the
    attention it is choosing. The candidate plans depend only on the visible
    layout, so the caller builds them once per chunk rather than once per step.
    """
    errors = []
    for plan, sampled_rows in candidates:
        sampled_query = query[:1, sampled_rows]
        exact = torch.nn.functional.scaled_dot_product_attention(
            sampled_query.transpose(1, 2),
            key[:1].transpose(1, 2),
            value[:1].transpose(1, 2),
            scale=softmax_scale,
        ).transpose(1, 2)
        out = sparse_attention(
            query=sampled_query,
            key=key[:1],
            value=value[:1],
            plan=plan,
            softmax_scale=softmax_scale,
        )
        errors.append(((out - exact).float() ** 2).mean(dim=(0, 1, 3)))
    return torch.stack(errors).argmin(dim=0)


def select_clusters_by_top_p(logits: torch.Tensor, *, top_p: float) -> torch.Tensor:
    """Upstream ``identify_dynamic_map``: top-p over size-weighted cluster mass.

    ``logits`` must already include ``log(cluster size)`` — softmax of that is
    upstream's ``weighted_softmax(scores, sizes)``, which weights each cluster's
    exponential by how many keys it holds, so a large diffuse cluster can outrank
    a small sharp one.

    Boundary convention, copied deliberately: upstream computes
    ``remove = cumsum > p`` and then shifts it right one place with
    ``remove[..., 0] = False``, so the cluster that *crosses* ``p`` is kept and
    the strongest cluster is always kept even at ``p = 0``. The kept set is
    therefore the descending prefix whose **exclusive** cumulative mass is
    ``<= p``.
    """
    probabilities = torch.softmax(logits, dim=-1)
    ordered, order = torch.sort(probabilities, dim=-1, descending=True)
    inclusive = ordered.cumsum(dim=-1)
    remove_ordered = inclusive > top_p
    remove_ordered = torch.cat(
        [torch.zeros_like(remove_ordered[..., :1]), remove_ordered[..., :-1]], dim=-1
    )
    return torch.zeros_like(remove_ordered).scatter(-1, order, ~remove_ordered)


class Svg1Attention(SparseAttentionBackend):
    name = "svg1"

    def __init__(self, config: Svg1Config) -> None:
        super().__init__()
        self._config = config
        self._masks = LayoutCache()

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        config = self._config
        q_len = call.query.shape[1]
        kv_len = call.key.shape[1]
        if kv_len <= q_len:
            return None
        device = call.query.device

        # Both candidate masks, the sampled rows and the plans that profile
        # them depend only on the *shape* of the visible layout — bands and
        # sink are view-token constructions — so the whole lot is built once
        # and shared by every layer, denoising step, and (on Rolling Forcing)
        # every steady window, whose absolute segments change while the shape
        # repeats. Only the per-head choice is remade per call, which is the
        # part SVG defines as online. The slot is the frame count so the
        # alternating denoise/updating layouts of one window don't evict each
        # other.
        signature = (layout.num_frames, q_len, call.num_local_heads)
        hit, cached = self._masks.get(layout.num_frames, signature)
        if not hit:
            cached = self._build_candidates(layout, q_len, kv_len, call, device)
            self._masks.put(layout.num_frames, signature, cached)
        if cached is None:
            return None
        spatial, temporal, permutation, seg_lo, seg_hi, candidates = cached

        choice = choose_mask_per_head(
            query=call.query,
            key=call.key,
            value=call.value,
            candidates=candidates,
            softmax_scale=call.softmax_scale,
        )
        temporal_heads = choice == 1
        keep = torch.where(temporal_heads[:, None, None], temporal[None], spatial[None])
        plan = plan_from_segment_mask(
            keep, segment_starts=seg_lo, segment_ends=seg_hi, block_m=config.block
        )
        if not bool(temporal_heads.any()):
            return SparseAttentionExecution(
                plan=plan, query=call.query, key=call.key, value=call.value
            )
        # Temporal heads' queries execute in spatial-major order (their mask
        # rows were built for it); keys stay in place, and the base class
        # scatters the output back through query_permutation.
        natural = torch.arange(q_len, device=device)
        index = torch.where(
            temporal_heads[None, :], permutation[:, None], natural[:, None]
        )  # [q_len, heads]
        gathered = call.query.gather(1, index[None, :, :, None].expand_as(call.query))
        return SparseAttentionExecution(
            plan=plan,
            query=gathered,
            key=call.key,
            value=call.value,
            query_permutation=index[None].expand(call.query.shape[0], -1, -1),
        )

    def _build_candidates(self, layout, q_len, kv_len, call, device):
        config = self._config
        spatial, temporal, permutation = build_svg1_segment_masks(
            layout=layout, q_len=q_len, kv_len=kv_len, config=config, device=device
        )
        if bool(spatial.all()) and bool(temporal.all()):
            return None
        seg_lo, seg_hi = frame_aligned_block_bounds(
            num_frames=layout.num_frames,
            frame_seqlen=layout.frame_seqlen,
            block=config.key_tile,
            device=device,
        )
        num_q_blocks = spatial.shape[0]
        sampled_blocks = torch.linspace(
            0,
            num_q_blocks - 1,
            min(config.num_sampled_blocks, num_q_blocks),
            device=device,
        ).long()
        natural_rows = (
            (
                sampled_blocks[:, None] * config.block
                + torch.arange(config.block, device=device)
            )
            .flatten()
            .clamp(max=q_len - 1)
        )
        candidates = [
            (
                plan_from_segment_mask(
                    candidate[sampled_blocks][None].expand(
                        call.num_local_heads, -1, -1
                    ),
                    segment_starts=seg_lo,
                    segment_ends=seg_hi,
                    block_m=config.block,
                ),
                rows,
            )
            for candidate, rows in (
                (spatial, natural_rows),
                (temporal, permutation[natural_rows]),
            )
        ]
        return spatial, temporal, permutation, seg_lo, seg_hi, candidates


class KeyClustering(msgspec.Struct, frozen=True):
    """A per-head k-means partition of the history keys, in permuted order."""

    permutation: torch.Tensor  # [heads, history_len] long, into history keys
    centroids: torch.Tensor  # [heads, clusters, head_dim] float
    segment_starts: torch.Tensor  # [heads, 1, clusters + 1] int32 (last = own chunk)
    segment_ends: torch.Tensor  # [heads, 1, clusters + 1] int32
    log_size: torch.Tensor  # [heads, clusters] float32


def cluster_keys(
    keys: torch.Tensor,  # [heads, history_len, head_dim]
    *,
    num_clusters: int,
    iters: int,
    own_chunk_len: int,
    initial_centroids: torch.Tensor | None = None,
) -> KeyClustering:
    """k-means over the history keys, packed into contiguous per-head segments.

    ``initial_centroids`` (the previous chunk's fit) warm-starts the iteration:
    the history only grows by one chunk between refits, so a couple of warm
    iterations recover what a full cold start would. Extra clusters demanded by
    the longer history are seeded from the keys as in a cold start.
    """
    num_heads, history_len, head_dim = keys.shape
    seeds = torch.linspace(0, history_len - 1, num_clusters, device=keys.device).long()
    if initial_centroids is None:
        centroids = keys[:, seeds].clone()
    else:
        carried = initial_centroids[:, :num_clusters]
        centroids = torch.cat([carried, keys[:, seeds[carried.shape[1] :]]], dim=1)
    labels = torch.zeros(num_heads, history_len, dtype=torch.long, device=keys.device)
    for _ in range(iters):
        similarity = keys @ centroids.transpose(1, 2)
        labels = (
            2 * similarity.float() - (centroids.float() ** 2).sum(-1)[:, None, :]
        ).argmax(-1)
        # Centroid update as a one-hot matmul rather than scatter_add. With
        # ~28k keys falling into ~110 clusters, scatter_add serializes on
        # atomics into the same 110 rows; the matmul does the same arithmetic on
        # tensor cores with no contention, and dominated the clustering cost.
        membership = torch.zeros(
            num_heads, num_clusters, history_len, device=keys.device, dtype=keys.dtype
        )
        membership.scatter_(1, labels[:, None, :], 1.0)
        counts = membership.sum(-1)
        sums = membership @ keys
        occupied = counts > 0
        centroids = torch.where(
            occupied[..., None], sums / counts.clamp(min=1)[..., None], centroids
        )

    permutation = labels.argsort(dim=1, stable=True)
    counts = torch.zeros(
        num_heads, num_clusters, device=keys.device, dtype=torch.long
    ).scatter_add_(1, labels, torch.ones_like(labels))
    ends = counts.cumsum(dim=1)
    starts = ends - counts
    # One extra segment pins the chunk's own keys, which are never clustered.
    own = torch.full((num_heads, 1), history_len, device=keys.device, dtype=torch.long)
    return KeyClustering(
        permutation=permutation,
        centroids=centroids,
        segment_starts=torch.cat([starts, own], dim=1)[:, None].to(torch.int32),
        segment_ends=torch.cat([ends, own + own_chunk_len], dim=1)[:, None].to(
            torch.int32
        ),
        log_size=counts.clamp(min=1).float().log(),
    )


class Svg2Attention(SparseAttentionBackend):
    name = "svg2"

    def __init__(self, config: Svg2Config) -> None:
        super().__init__()
        self._config = config
        self._clusters: dict[int, tuple[tuple, KeyClustering]] = {}
        self._last_chunk_index = -1

    def _on_begin_forward(self, geometry: ChunkGeometry) -> None:
        chunk_index = geometry.query_chunk_index
        if chunk_index < self._last_chunk_index:
            # A new video restarts the chunk counter; its keys have nothing to
            # do with the cached clusterings, and the warm start must not seed
            # from the previous video's centroids either.
            self._clusters.clear()
        self._last_chunk_index = chunk_index

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        config = self._config
        q_len = call.query.shape[1]
        kv_len = call.key.shape[1]
        history_len = kv_len - q_len
        num_clusters = history_len // config.cluster_size
        if history_len <= 0 or num_clusters < 2:
            return None

        clustering = self._clustering(call, history_len, num_clusters)
        keys, values = self._permute_kv(call, clustering, history_len)
        query, query_permutation = self._permute_queries(call, clustering)
        keep = self._select_clusters(
            query=query, clustering=clustering, softmax_scale=call.softmax_scale
        )
        if bool(keep.all()):
            return None
        plan = plan_from_segment_mask(
            keep,
            segment_starts=clustering.segment_starts,
            segment_ends=clustering.segment_ends,
            block_m=config.block,
        )
        return SparseAttentionExecution(
            plan=plan,
            query=query,
            key=keys,
            value=values,
            query_permutation=query_permutation,
        )

    def _clustering(
        self, call: SparseAttentionCall, history_len: int, num_clusters: int
    ) -> KeyClustering:
        signature = (call.key_segments, history_len, num_clusters)
        cached = self._clusters.get(call.layer_index)
        if cached is not None and cached[0] == signature:
            return cached[1]
        history = call.key[0, :history_len].permute(1, 0, 2)
        warm = self._config.kmeans_warm_iters > 0 and cached is not None
        clustering = cluster_keys(
            history,
            num_clusters=num_clusters,
            iters=self._config.kmeans_warm_iters if warm else self._config.kmeans_iters,
            own_chunk_len=call.key.shape[1] - history_len,
            initial_centroids=cached[1].centroids if warm else None,
        )
        self._clusters[call.layer_index] = (signature, clustering)
        return clustering

    def _permute_kv(
        self,
        call: SparseAttentionCall,
        clustering: KeyClustering,
        history_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [heads, history] -> [batch, history, heads, head_dim] gather index
        index = clustering.permutation.T[None, :, :, None].expand(
            call.key.shape[0], history_len, -1, call.head_dim
        )
        keys = torch.cat(
            [call.key[:, :history_len].gather(1, index), call.key[:, history_len:]],
            dim=1,
        )
        values = torch.cat(
            [call.value[:, :history_len].gather(1, index), call.value[:, history_len:]],
            dim=1,
        )
        return keys, values

    def _permute_queries(
        self, call: SparseAttentionCall, clustering: KeyClustering
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = call.query[0].permute(1, 0, 2)  # [heads, q_len, head_dim]
        similarity = queries @ clustering.centroids.transpose(1, 2)
        labels = (
            2 * similarity.float()
            - (clustering.centroids.float() ** 2).sum(-1)[:, None, :]
        ).argmax(-1)
        permutation = labels.argsort(dim=1, stable=True).T  # [q_len, heads]
        index = permutation[None, :, :, None].expand_as(call.query)
        return call.query.gather(1, index), permutation[None].expand(
            call.query.shape[0], -1, -1
        )

    def _select_clusters(
        self,
        *,
        query: torch.Tensor,
        clustering: KeyClustering,
        softmax_scale: float,
    ) -> torch.Tensor:
        config = self._config
        q_len = query.shape[1]
        num_blocks = -(-q_len // config.block)
        pad = num_blocks * config.block - q_len
        padded = torch.nn.functional.pad(query[0], (0, 0, 0, 0, 0, pad))
        block_sum = padded.view(num_blocks, config.block, -1, query.shape[-1]).sum(1)
        # Divide by the true row count: a zero-padded partial last block would
        # otherwise scale its dot-logits down while log(cluster size) stays
        # put, reordering that block's cluster scores.
        rows = torch.full(
            (num_blocks, 1, 1), config.block, device=query.device, dtype=block_sum.dtype
        )
        if pad:
            rows[-1] = config.block - pad
        block_mean = (block_sum / rows).permute(1, 0, 2)  # [heads, q_blocks, head_dim]

        logits = (
            block_mean @ clustering.centroids.transpose(1, 2)
        ).float() * softmax_scale + clustering.log_size[:, None, :]
        keep = select_clusters_by_top_p(logits, top_p=config.top_p)
        # The chunk's own keys are the extra trailing segment and always stay.
        own = torch.ones_like(keep[..., :1])
        return torch.cat([keep, own], dim=-1)
