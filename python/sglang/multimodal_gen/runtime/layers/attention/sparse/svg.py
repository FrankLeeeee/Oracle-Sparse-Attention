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
from sglang.multimodal_gen.runtime.layers.attention.sparse.blocks import (
    block_bounds,
    own_block_mask,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import VisibleLayout
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_block_mask,
    plan_from_segment_mask,
    sparse_attention,
)


class Svg1Config(msgspec.Struct, frozen=True):
    block: int = 128
    # Upstream's `block_thres`, as a multiple of the frame length: both masks are
    # token-distance bands of half-width `band_frames * frame_seqlen`, the
    # spatial one in natural (frame-major) order and the temporal one in
    # spatial-major order. `svg/models/wan/utils.py` hard-codes
    # `block_thres = frame_size * 2`, so 2.0 is upstream's setting and the knob
    # to turn for a sparsity sweep.
    band_frames: float = 2.0
    # Upstream's `pixel_attn_mask[:, :frame_size] = 1` — the first frame of the
    # video is a sink for *both* candidate masks.
    dense_sink_frames: int = 1
    # Query blocks attended exactly to score the two candidate masks. Upstream
    # samples 32 individual rows; whole blocks keep the profiling pass on the
    # block-sparse fast path, and two of them is the nearest equivalent cost.
    num_sampled_blocks: int = 2


class Svg2Config(msgspec.Struct, frozen=True):
    block: int = 128
    # Average keys per semantic cluster; the cluster count follows from the
    # history length.
    cluster_size: int = 256
    kmeans_iters: int = 4
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
    banded[:, :frame_seqlen] |= dense_sink_frames > 0

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


def build_svg1_masks(
    *,
    layout: VisibleLayout,
    q_len: int,
    kv_len: int,
    config: Svg1Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(spatial, temporal)`` ``[q_blocks, key_blocks]`` masks for one layout.

    The same two masks as :func:`svg1_token_masks`, reduced to blocks by "keep a
    block pair if any of its token pairs is kept" — the convention upstream's own
    block-level execution path uses. Built one query block at a time so the
    intermediate is ``[block, kv_len]`` rather than ``[kv_len, kv_len]``, which is
    what makes it affordable at Wan's geometry.
    """
    block = config.block
    frame_seqlen = layout.frame_seqlen
    num_frames = layout.num_frames
    band_blocks = int(config.band_frames * frame_seqlen) // block
    offset = kv_len - q_len  # the chunk's queries are the last keys of the view

    keys = torch.arange(kv_len, device=device)
    key_block = keys // block
    key_spatial_major = _spatial_major_index(
        keys, frame_seqlen=frame_seqlen, num_frames=num_frames
    )
    sink_columns = keys < frame_seqlen if config.dense_sink_frames > 0 else None
    sink_columns_permuted = (
        key_spatial_major < frame_seqlen if config.dense_sink_frames > 0 else None
    )

    q_lo, q_hi = block_bounds(q_len, block, device=device)
    num_key_blocks = -(-kv_len // block)
    spatial_columns = []
    temporal_columns = []
    for lo, hi in zip(q_lo.tolist(), q_hi.tolist(), strict=True):
        rows = torch.arange(lo + offset, hi + offset, device=device)
        natural = ((rows[:, None] // block) - key_block[None, :]).abs() < band_blocks
        if sink_columns is not None:
            natural |= sink_columns[None, :]
        spatial_columns.append(natural.any(dim=0))

        rows_spatial_major = _spatial_major_index(
            rows, frame_seqlen=frame_seqlen, num_frames=num_frames
        )
        permuted = (
            (rows_spatial_major[:, None] // block)
            - (key_spatial_major[None, :] // block)
        ).abs() < band_blocks
        if sink_columns_permuted is not None:
            permuted |= sink_columns_permuted[None, :]
        temporal_columns.append(permuted.any(dim=0))

    def _to_blocks(columns: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(columns)  # [q_blocks, kv_len]
        padded = torch.nn.functional.pad(
            stacked, (0, num_key_blocks * block - kv_len)
        )
        return padded.view(len(columns), num_key_blocks, block).amax(dim=-1)

    own = own_block_mask(
        q_lo=q_lo,
        q_hi=q_hi,
        k_lo=torch.arange(num_key_blocks, device=device) * block,
        k_hi=((torch.arange(num_key_blocks, device=device) + 1) * block).clamp(
            max=kv_len
        ),
        query_offset_in_view=offset,
    )
    return _to_blocks(spatial_columns) | own, _to_blocks(temporal_columns) | own


def choose_mask_per_head(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_plans: list,  # one SparseAttentionPlan per candidate, sampled rows
    sampled_rows: torch.Tensor,  # [sampled] long
    softmax_scale: float,
) -> torch.Tensor:
    """Index of the candidate mask with the lowest error, per head: ``[heads]``.

    Upstream's ``sample_mse``: exact attention on a small sample of the queries,
    each candidate mask scored against it by mean squared error, lowest wins.

    Two deviations from ``svg/models/wan/attention.py``, both for cost. Upstream
    samples 32 individual random rows and evaluates the candidates by masking a
    materialized ``[heads, sampled, kv_len]`` score matrix; we sample whole query
    *blocks* and evaluate through the same block-sparse kernel the method will
    actually use, because a masked SDPA on that intermediate costs more than the
    attention it is choosing. The candidate plans depend only on the visible
    layout, so the caller builds them once per chunk rather than once per step.
    """
    sampled_query = query[:1, sampled_rows]
    exact = torch.nn.functional.scaled_dot_product_attention(
        sampled_query.transpose(1, 2),
        key[:1].transpose(1, 2),
        value[:1].transpose(1, 2),
        scale=softmax_scale,
    ).transpose(1, 2)

    errors = []
    for plan in candidate_plans:
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

        # Both candidate masks, the sampled rows and the plans that profile them
        # depend only on the visible layout, so the whole lot is built once per
        # chunk and shared by every layer and every denoising step. Only the
        # per-head choice is remade per call, which is the part SVG defines as
        # online.
        signature = (call.key_segments, q_len, call.num_local_heads)
        hit, cached = self._masks.get(0, signature)
        if not hit:
            cached = self._build_candidates(layout, q_len, kv_len, call, device)
            self._masks.put(0, signature, cached)
        if cached is None:
            return None
        spatial, temporal, sampled_rows, candidate_plans = cached

        choice = choose_mask_per_head(
            query=call.query,
            key=call.key,
            value=call.value,
            candidate_plans=candidate_plans,
            sampled_rows=sampled_rows,
            softmax_scale=call.softmax_scale,
        )
        keep = torch.where(
            (choice == 0)[:, None, None], spatial[None], temporal[None]
        )
        plan = plan_from_block_mask(
            keep, block_n=config.block, kv_len=kv_len, block_m=config.block
        )
        return SparseAttentionExecution(
            plan=plan, query=call.query, key=call.key, value=call.value
        )

    def _build_candidates(self, layout, q_len, kv_len, call, device):
        config = self._config
        spatial, temporal = build_svg1_masks(
            layout=layout, q_len=q_len, kv_len=kv_len, config=config, device=device
        )
        if bool(spatial.all()) and bool(temporal.all()):
            return None
        num_q_blocks = spatial.shape[0]
        sampled_blocks = torch.linspace(
            0, num_q_blocks - 1, min(config.num_sampled_blocks, num_q_blocks),
            device=device,
        ).long()
        sampled_rows = (
            sampled_blocks[:, None] * config.block
            + torch.arange(config.block, device=device)
        ).flatten().clamp(max=q_len - 1)
        candidate_plans = [
            plan_from_block_mask(
                candidate[sampled_blocks][None].expand(call.num_local_heads, -1, -1),
                block_n=config.block,
                kv_len=kv_len,
                block_m=config.block,
            )
            for candidate in (spatial, temporal)
        ]
        return spatial, temporal, sampled_rows, candidate_plans


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
) -> KeyClustering:
    """k-means over the history keys, packed into contiguous per-head segments."""
    num_heads, history_len, head_dim = keys.shape
    seeds = torch.linspace(0, history_len - 1, num_clusters, device=keys.device).long()
    centroids = keys[:, seeds].clone()
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
        segment_ends=torch.cat(
            [ends, own + own_chunk_len], dim=1
        )[:, None].to(torch.int32),
        log_size=counts.clamp(min=1).float().log(),
    )


class Svg2Attention(SparseAttentionBackend):
    name = "svg2"

    def __init__(self, config: Svg2Config) -> None:
        super().__init__()
        self._config = config
        self._clusters: dict[int, tuple[tuple, KeyClustering]] = {}

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
        clustering = cluster_keys(
            history,
            num_clusters=num_clusters,
            iters=self._config.kmeans_iters,
            own_chunk_len=call.key.shape[1] - history_len,
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
            2 * similarity.float() - (clustering.centroids.float() ** 2).sum(-1)[:, None, :]
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
        block_mean = padded.view(num_blocks, config.block, -1, query.shape[-1]).mean(1)
        block_mean = block_mean.permute(1, 0, 2)  # [heads, q_blocks, head_dim]

        logits = (
            block_mean @ clustering.centroids.transpose(1, 2)
        ).float() * softmax_scale + clustering.log_size[:, None, :]
        keep = select_clusters_by_top_p(logits, top_p=config.top_p)
        # The chunk's own keys are the extra trailing segment and always stay.
        own = torch.ones_like(keep[..., :1])
        return torch.cat([keep, own], dim=-1)
