# SPDX-License-Identifier: Apache-2.0
"""Per-chunk self-attention map probe for block-causal video DiTs.

Answers "while generating chunk *c*, how much attention mass does each layer
spend on every earlier chunk?" for the Self-Forcing family (Causal Forcing,
Rolling Forcing), where the KV cache makes the attention matrix invisible to
ordinary tooling: the fused attention kernels never materialize the
probabilities, and the visible key range is a rolling view of the cache rather
than the plain token sequence.

Enabled by setting ``SGLANG_DIFFUSION_ATTENTION_MAP_DIR``; disabled (zero cost
beyond one ``None`` check per attention call) otherwise. When on, each
attention layer recomputes ``softmax(q k^T)`` for a strided subset of the
current queries and reduces it to *attention mass per latent frame, per head*
(mean over the sampled queries of the chunk; each head's row sums to 1), then
buffers the result. :meth:`ChunkAttentionRecorder.flush` writes one
``chunk_<c>.npz`` per generated chunk — arrays shaped
``[steps, layers, heads, frames]`` — plus a ``meta.json``. Render with
``python -m sglang.multimodal_gen.tools.plot_chunk_attention_maps``.

The head and frame axes are kept rather than collapsed because sparse-attention
patterns are per-head properties at sub-chunk granularity; reduce them at
analysis time for the coarse chunk-level view.

The probe is meant for single-GPU debugging runs: it only records on world rank
0, so under tensor parallelism the reported mass covers rank 0's heads only.
"""

import json
import pathlib
import time
from collections.abc import Sequence
from contextlib import contextmanager

import msgspec
import numpy as np
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Pass kinds. Block-causal pipelines run two flavors of DiT forward per chunk:
# the denoising steps, and a final clean-latent forward that refreshes the KV
# cache. They are recorded separately so plots can ignore the latter.
DENOISE_PASS = "denoise"
CACHE_UPDATE_PASS = "cache_update"

_DEFAULT_QUERY_STRIDE = 8
_QUERY_TILE = 64

# Spatial displacement is recorded per temporal offset, because "attends to my
# own neighbourhood in this frame" and "tracks the same pixel across frames" are
# different patterns that a sparse kernel would exploit differently.
TEMPORAL_BUCKETS = ("dt=0", "dt=1", "dt=2", "dt>=3", "dt<0")
_DEFAULT_SPATIAL_QUERY_STRIDE = 32
# Chunk 0-1 attend to a cache of only a few frames, so their dt distribution is
# not representative of steady state; skip them by default.
_DEFAULT_SPATIAL_MIN_CHUNK = 2


class ChunkAttentionEvent(msgspec.Struct, frozen=True):
    """Attention mass of one (pass, chunk, layer, step), per head and frame."""

    pass_kind: str
    query_chunk: int
    layer_index: int
    step_index: int
    scores: torch.Tensor  # [heads, num_frames_so_far], each head's row sums to 1


class ForwardScope(msgspec.Struct, frozen=True):
    """Geometry shared by every attention layer of one DiT forward."""

    frame_seqlen: int
    num_frames_per_block: int
    query_token_start: int
    pass_kind: str
    grid_height: int = 0
    grid_width: int = 0

    @property
    def chunk_tokens(self) -> int:
        return self.frame_seqlen * self.num_frames_per_block


def segment_positions(
    segments: Sequence[tuple[int, int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Global token index of every key, given its ``(token_start, length)`` segments.

    The visible key range of a causal DiT is generally not contiguous in global
    token space (Rolling Forcing prepends a re-roped attention sink and appends
    the in-flight window), so callers describe it as a list of contiguous global
    token ranges, in the order they appear along the key axis.
    """
    return torch.cat(
        [
            torch.arange(start, start + length, device=device)
            for start, length in segments
        ]
    )


def segment_frame_ids(
    segments: Sequence[tuple[int, int]],
    *,
    frame_seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    """Latent-frame index of every key (see :func:`segment_positions`)."""
    return segment_positions(segments, device=device) // frame_seqlen


def temporal_bucket_ids(frame_delta: torch.Tensor) -> torch.Tensor:
    """Bucket ``query_frame - key_frame`` into the classes of `TEMPORAL_BUCKETS`."""
    bucketed = frame_delta.clamp(min=0, max=3)
    return torch.where(frame_delta < 0, torch.full_like(frame_delta, 4), bucketed)


def attention_mass_by_frame(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_chunk_ids: torch.Tensor,
    key_frame_ids: torch.Tensor,
    num_chunks: int,
    num_frames: int,
    token_mass: torch.Tensor | None = None,
    query_tile: int = _QUERY_TILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention mass from each query chunk to each key frame, per head.

    ``query``/``key`` are ``[seq, heads, head_dim]`` (single batch element,
    post-RoPE). Returns ``(mass, counts)``: ``mass[i, h, f]`` is the summed
    softmax probability that head ``h`` of the queries in chunk ``i`` puts on
    latent frame ``f``, and ``counts[i]`` the number of such queries, so
    ``mass[i] / counts[i]`` is a per-head distribution over frames.

    The head axis is kept because sparse-attention patterns are usually a
    per-head property; reduce it afterwards for a head-averaged view.

    ``token_mass`` ``[chunks, heads, key_seq]``, when given, is additionally
    accumulated with the same probabilities left at *key-token* resolution
    (no reduction over the key axis at all). It shares this function's single
    softmax pass rather than recomputing it.

    Computed in query tiles because the full probability matrix
    (heads x queries x cache) does not fit in memory for long videos.
    """
    num_heads = query.shape[1]
    scale = query.shape[-1] ** -0.5
    mass = torch.zeros(
        num_chunks, num_heads, num_frames, dtype=torch.float32, device=query.device
    )
    counts = torch.zeros(num_chunks, dtype=torch.float32, device=query.device)
    key_by_head = key.permute(1, 2, 0)  # [heads, head_dim, key_seq]

    for start in range(0, query.shape[0], query_tile):
        query_tile_by_head = query[start : start + query_tile].permute(1, 0, 2)
        probs = torch.softmax(
            torch.bmm(query_tile_by_head, key_by_head).float() * scale, dim=-1
        )  # [heads, tile, key_seq]
        per_frame = torch.zeros(
            probs.shape[0],
            probs.shape[1],
            num_frames,
            dtype=torch.float32,
            device=query.device,
        )
        per_frame.index_add_(2, key_frame_ids, probs)
        tile_ids = query_chunk_ids[start : start + query_tile]
        mass.index_add_(0, tile_ids, per_frame.transpose(0, 1))  # [tile, heads, frames]
        counts.index_add_(0, tile_ids, torch.ones_like(tile_ids, dtype=torch.float32))
        if token_mass is not None:
            token_mass.index_add_(0, tile_ids, probs.transpose(0, 1))

    return mass, counts


def _concentration_ranks(frame_seqlen: int, *, device: torch.device) -> torch.Tensor:
    """Top-k cut-offs (0-based indices) for the per-query concentration curve."""
    ranks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if k < frame_seqlen]
    ranks.append(frame_seqlen)
    return torch.tensor([k - 1 for k in ranks], device=device)


def _grow_token_buffer(buffer: torch.Tensor, *sizes: int) -> torch.Tensor:
    """Grow ``buffer`` so every axis is at least the requested size."""
    assert len(sizes) == buffer.ndim, (
        f"expected {buffer.ndim} sizes for a {buffer.ndim}-d buffer, got {len(sizes)}"
    )
    target = tuple(max(have, want) for have, want in zip(buffer.shape, sizes))
    if target == tuple(buffer.shape):
        return buffer
    grown = torch.zeros(target, dtype=buffer.dtype, device=buffer.device)
    grown[tuple(slice(0, n) for n in buffer.shape)] = buffer
    return grown


def _grow_to_layer(accumulator: torch.Tensor, layer_index: int) -> torch.Tensor:
    """Extend the layer axis so ``layer_index`` is addressable."""
    if layer_index < accumulator.shape[0]:
        return accumulator
    grown = torch.zeros(
        layer_index + 1,
        *accumulator.shape[1:],
        dtype=accumulator.dtype,
        device=accumulator.device,
    )
    grown[: accumulator.shape[0]] = accumulator
    return grown


def spatial_displacement_mass(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    frame_seqlen: int,
    grid_height: int,
    grid_width: int,
    accumulator: torch.Tensor,
    absolute: torch.Tensor | None = None,
    concentration: torch.Tensor | None = None,
    concentration_ranks: torch.Tensor | None = None,
    query_tile: int = _QUERY_TILE,
) -> int:
    """Accumulate attention mass by *spatial displacement*, per head and dt bucket.

    ``accumulator`` is ``[heads, buckets, 2 * grid_height - 1, 2 * grid_width - 1]``
    and is added to in place; the returned int is how many queries contributed.
    Bin ``(dy + grid_height - 1, dx + grid_width - 1)`` holds the mass a query at
    latent position ``(y, x)`` sends to keys at ``(y + dy, x + dx)``, so a
    spatially local head concentrates on the centre and a head that tracks the
    same position across time concentrates on the exact centre of bucket ``dt>0``.

    Two optional companions disambiguate what a *diffuse* displacement map means.
    ``absolute`` ``[heads, buckets, grid_height, grid_width]`` accumulates the
    same mass in absolute frame coordinates, so a head that always looks at one
    region of the frame (rather than near its own query) is still visibly
    concentrated. ``concentration`` ``[heads, buckets, len(ranks)]`` accumulates,
    per query, the mass held by its top-``k`` cells for each k in
    ``concentration_ranks`` — attention can be highly sparse per query while
    being unstructured in displacement space, and only that case forces a
    content-dependent (rather than static) sparse pattern.

    Queries are processed one *frame* at a time so that the key-side dt bucket is
    constant within a tile; the per-key scatter is therefore over
    ``buckets x frame_seqlen`` rather than over every (query, key) pair, which is
    what keeps this affordable.
    """
    num_heads = query.shape[1]
    scale = query.shape[-1] ** -0.5
    buckets, height_bins, width_bins = accumulator.shape[1:]
    key_by_head = key.permute(1, 2, 0)  # [heads, head_dim, key_seq]

    key_frames = key_positions // frame_seqlen
    key_spatial = key_positions % frame_seqlen
    cell_y = torch.arange(frame_seqlen, device=query.device) // grid_width
    cell_x = torch.arange(frame_seqlen, device=query.device) % grid_width
    flat_accumulator = accumulator.reshape(num_heads * buckets, -1)

    query_frames = query_positions // frame_seqlen
    recorded = 0
    for frame in query_frames.unique().tolist():
        in_frame = (query_frames == frame).nonzero().flatten()
        key_bin = (
            temporal_bucket_ids(frame - key_frames) * frame_seqlen + key_spatial
        )
        for start in range(0, in_frame.numel(), query_tile):
            rows = in_frame[start : start + query_tile]
            tile = rows.numel()
            probs = torch.softmax(
                torch.bmm(query[rows].permute(1, 0, 2), key_by_head).float() * scale,
                dim=-1,
            )  # [heads, tile, key_seq]
            per_cell = torch.zeros(
                num_heads,
                tile,
                buckets * frame_seqlen,
                dtype=torch.float32,
                device=query.device,
            )
            per_cell.index_add_(2, key_bin, probs)

            spatial = query_positions[rows] % frame_seqlen
            offset_y = cell_y[None, :] - (spatial // grid_width)[:, None] + grid_height - 1
            offset_x = cell_x[None, :] - (spatial % grid_width)[:, None] + grid_width - 1
            bins = (offset_y * width_bins + offset_x).reshape(-1)  # [tile * frame_seqlen]

            by_bucket = per_cell.view(num_heads, tile, buckets, frame_seqlen)
            source = by_bucket.permute(0, 2, 1, 3).reshape(
                num_heads * buckets, tile * frame_seqlen
            )
            flat_accumulator.index_add_(1, bins, source)

            if absolute is not None:
                absolute.view(num_heads, buckets, frame_seqlen).add_(
                    by_bucket.sum(dim=1)
                )
            if concentration is not None:
                assert concentration_ranks is not None
                ranked = torch.sort(by_bucket, dim=-1, descending=True).values
                concentration.add_(
                    ranked.cumsum(dim=-1)[..., concentration_ranks].sum(dim=1)
                )
            recorded += tile
    return recorded


def dense_chunk_scores(
    events: Sequence[ChunkAttentionEvent],
    *,
    num_frames: int,
) -> np.ndarray:
    """Stack one chunk's events into ``[num_steps, num_layers, num_heads, num_frames]``.

    Entries with no event stay ``nan`` (e.g. a chunk that reached fewer
    denoising steps than another).
    """
    num_steps = max(event.step_index for event in events) + 1
    num_layers = max(event.layer_index for event in events) + 1
    num_heads = events[0].scores.shape[0]
    dense = np.full(
        (num_steps, num_layers, num_heads, num_frames), np.nan, dtype=np.float32
    )
    for event in events:
        scores = event.scores.numpy()
        dense[event.step_index, event.layer_index, :, : scores.shape[1]] = scores
    return dense


class ChunkAttentionRecorder:
    """Buffers per-chunk attention mass across a generation, then writes it out.

    The model calls :meth:`begin_forward` once per DiT forward (the key layout
    is shared by all its layers) and :meth:`record` once per attention layer;
    the denoising stage calls :meth:`flush` when the video is done.
    """

    def __init__(
        self,
        *,
        output_dir: str,
        query_stride: int = _DEFAULT_QUERY_STRIDE,
        query_tile: int = _QUERY_TILE,
        spatial: bool = False,
        spatial_query_stride: int = _DEFAULT_SPATIAL_QUERY_STRIDE,
        spatial_min_chunk: int = _DEFAULT_SPATIAL_MIN_CHUNK,
        token_scores: bool = False,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.query_stride = max(1, query_stride)
        self.query_tile = query_tile
        self.spatial = spatial
        self.spatial_query_stride = max(1, spatial_query_stride)
        self.spatial_min_chunk = spatial_min_chunk
        self.token_scores = token_scores
        self.current_pass_kind = DENOISE_PASS
        self._scope: ForwardScope | None = None
        self._events: list[ChunkAttentionEvent] = []
        self._step_counter: dict[tuple[str, int, int], int] = {}
        # [layers, heads, buckets, 2*grid_h-1, 2*grid_w-1], built on first record
        self._displacement: torch.Tensor | None = None
        self._absolute: torch.Tensor | None = None
        self._concentration: torch.Tensor | None = None
        self._displacement_queries: dict[int, int] = {}
        # [chunks, layers, heads, global_tokens] + [chunks, layers] query counts
        # and a [chunks, global_tokens] "was ever in the cache" mask, so that
        # never-visible tokens can be told from evicted ones (exact 0) later.
        self._token_mass: torch.Tensor | None = None
        self._token_counts: torch.Tensor | None = None
        self._token_visible: torch.Tensor | None = None

    @contextmanager
    def pass_kind_scope(self, pass_kind: str):
        """Tag the forwards run inside the block (e.g. KV cache refreshes)."""
        previous = self.current_pass_kind
        self.current_pass_kind = pass_kind
        try:
            yield
        finally:
            self.current_pass_kind = previous

    def begin_forward(
        self,
        *,
        frame_seqlen: int,
        num_frames_per_block: int,
        query_token_start: int,
        pass_kind: str | None = None,
        grid_height: int = 0,
        grid_width: int = 0,
    ) -> None:
        self._scope = ForwardScope(
            frame_seqlen=frame_seqlen,
            num_frames_per_block=num_frames_per_block,
            query_token_start=query_token_start,
            pass_kind=pass_kind or self.current_pass_kind,
            grid_height=grid_height,
            grid_width=grid_width,
        )

    def end_forward(self) -> None:
        self._scope = None

    @torch.no_grad()
    def record(
        self,
        *,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        key_segments: Sequence[tuple[int, int]],
    ) -> None:
        """Record one attention layer.

        ``query``/``key`` are the post-RoPE ``[batch, seq, heads, head_dim]``
        tensors handed to the attention kernel; ``key_segments`` describes where
        the visible keys live in global token space (see
        :func:`segment_frame_ids`). Only batch element 0 is recorded.
        """
        scope = self._scope
        if scope is None:
            return
        sampled_query = query[0, :: self.query_stride]
        visible_key = key[0]
        query_positions = scope.query_token_start + torch.arange(
            0, query.shape[1], self.query_stride, device=query.device
        )
        query_chunk_ids = query_positions // scope.chunk_tokens
        key_frame_ids = segment_frame_ids(
            key_segments,
            frame_seqlen=scope.frame_seqlen,
            device=query.device,
        )
        assert key_frame_ids.shape[0] == visible_key.shape[0], (
            f"attention probe key segments cover {key_frame_ids.shape[0]} tokens "
            f"but the attention call sees {visible_key.shape[0]}"
        )
        num_chunks = int(query_chunk_ids.max().item()) + 1
        num_frames = int(key_frame_ids.max().item()) + 1
        want_tokens = self.token_scores and scope.pass_kind == DENOISE_PASS
        token_mass = (
            torch.zeros(
                num_chunks,
                sampled_query.shape[1],
                visible_key.shape[0],
                dtype=torch.float32,
                device=query.device,
            )
            if want_tokens
            else None
        )
        mass, counts = attention_mass_by_frame(
            query=sampled_query,
            key=visible_key,
            query_chunk_ids=query_chunk_ids,
            key_frame_ids=key_frame_ids,
            num_chunks=num_chunks,
            num_frames=num_frames,
            token_mass=token_mass,
            query_tile=self.query_tile,
        )
        if token_mass is not None:
            self._accumulate_token_scores(
                layer_index=layer_index,
                token_mass=token_mass,
                counts=counts,
                key_positions=segment_positions(key_segments, device=query.device),
                scope=scope,
            )
        self._append_events(
            scope.pass_kind,
            layer_index=layer_index,
            mass=mass,
            counts=counts,
        )
        if self.spatial and scope.pass_kind == DENOISE_PASS:
            self._record_displacement(
                layer_index=layer_index,
                query=query[0],
                key=visible_key,
                key_positions=segment_positions(key_segments, device=query.device),
                scope=scope,
            )

    @torch.no_grad()
    def _accumulate_token_scores(
        self,
        *,
        layer_index: int,
        token_mass: torch.Tensor,
        counts: torch.Tensor,
        key_positions: torch.Tensor,
        scope: ForwardScope,
    ) -> None:
        """Scatter one layer's per-key-token mass into global token space."""
        num_chunks, num_heads, _ = token_mass.shape
        # the cache never reaches past the end of the newest query chunk
        num_tokens = max(
            int(key_positions.max().item()) + 1, num_chunks * scope.chunk_tokens
        )
        if self._token_mass is None:
            self._token_mass = torch.zeros(
                num_chunks, layer_index + 1, num_heads, num_tokens,
                dtype=torch.float32, device=token_mass.device,
            )
            self._token_counts = torch.zeros(
                num_chunks, layer_index + 1,
                dtype=torch.float32, device=token_mass.device,
            )
            self._token_visible = torch.zeros(
                num_chunks, num_tokens, dtype=torch.bool, device=token_mass.device
            )
        self._token_mass = _grow_token_buffer(
            self._token_mass, num_chunks, layer_index + 1, num_heads, num_tokens
        )
        self._token_counts = _grow_token_buffer(
            self._token_counts, num_chunks, layer_index + 1
        )
        self._token_visible = _grow_token_buffer(
            self._token_visible, num_chunks, num_tokens
        )

        for chunk in counts.nonzero().flatten().tolist():
            self._token_mass[chunk, layer_index].index_add_(
                1, key_positions, token_mass[chunk]
            )
            self._token_counts[chunk, layer_index] += counts[chunk]
            self._token_visible[chunk, key_positions] = True

    @torch.no_grad()
    def _record_displacement(
        self,
        *,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        key_positions: torch.Tensor,
        scope: ForwardScope,
    ) -> None:
        assert scope.grid_height and scope.grid_width, (
            "spatial recording needs the latent grid; pass grid_height/grid_width "
            "to begin_forward()"
        )
        positions = scope.query_token_start + torch.arange(
            0, query.shape[0], self.spatial_query_stride, device=query.device
        )
        keep = (positions // scope.chunk_tokens) >= self.spatial_min_chunk
        if not bool(keep.any()):
            return
        positions = positions[keep]
        sampled = query[:: self.spatial_query_stride][keep]

        heads = query.shape[1]
        buckets = len(TEMPORAL_BUCKETS)
        if self._displacement is None:
            self._concentration_ranks = _concentration_ranks(
                scope.frame_seqlen, device=query.device
            )
            self._displacement = torch.zeros(
                1, heads, buckets,
                2 * scope.grid_height - 1, 2 * scope.grid_width - 1,
                dtype=torch.float32, device=query.device,
            )
            self._absolute = torch.zeros(
                1, heads, buckets, scope.grid_height, scope.grid_width,
                dtype=torch.float32, device=query.device,
            )
            self._concentration = torch.zeros(
                1, heads, buckets, self._concentration_ranks.numel(),
                dtype=torch.float32, device=query.device,
            )
        self._displacement = _grow_to_layer(self._displacement, layer_index)
        self._absolute = _grow_to_layer(self._absolute, layer_index)
        self._concentration = _grow_to_layer(self._concentration, layer_index)

        recorded = spatial_displacement_mass(
            query=sampled,
            key=key,
            query_positions=positions,
            key_positions=key_positions,
            frame_seqlen=scope.frame_seqlen,
            grid_height=scope.grid_height,
            grid_width=scope.grid_width,
            accumulator=self._displacement[layer_index],
            absolute=self._absolute[layer_index],
            concentration=self._concentration[layer_index],
            concentration_ranks=self._concentration_ranks,
            query_tile=self.query_tile,
        )
        self._displacement_queries[layer_index] = (
            self._displacement_queries.get(layer_index, 0) + recorded
        )

    def _append_events(
        self,
        pass_kind: str,
        *,
        layer_index: int,
        mass: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        for query_chunk in counts.nonzero().flatten().tolist():
            step_key = (pass_kind, query_chunk, layer_index)
            step_index = self._step_counter.get(step_key, 0)
            self._step_counter[step_key] = step_index + 1
            self._events.append(
                ChunkAttentionEvent(
                    pass_kind=pass_kind,
                    query_chunk=query_chunk,
                    layer_index=layer_index,
                    step_index=step_index,
                    scores=(mass[query_chunk] / counts[query_chunk]).cpu(),
                )
            )

    def flush(self, *, model_tag: str, meta: dict | None = None) -> str | None:
        """Write one ``chunk_<c>.npz`` per chunk plus ``meta.json``; reset state."""
        events = self._events
        displacement = self._displacement
        absolute = self._absolute
        concentration = self._concentration
        displacement_queries = self._displacement_queries
        token_mass = self._token_mass
        token_counts = self._token_counts
        token_visible = self._token_visible
        self._token_mass = None
        self._token_counts = None
        self._token_visible = None
        self._events = []
        self._step_counter = {}
        self._scope = None
        self._displacement = None
        self._absolute = None
        self._concentration = None
        self._displacement_queries = {}
        if not events:
            return None

        run_dir = self.output_dir / f"{model_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        num_frames = max(event.scores.shape[1] for event in events)
        num_chunks = max(event.query_chunk for event in events) + 1
        by_chunk: dict[int, list[ChunkAttentionEvent]] = {}
        for event in events:
            by_chunk.setdefault(event.query_chunk, []).append(event)

        num_layers = max(event.layer_index for event in events) + 1
        for query_chunk, chunk_events in sorted(by_chunk.items()):
            arrays = {}
            for pass_kind in (DENOISE_PASS, CACHE_UPDATE_PASS):
                pass_events = [e for e in chunk_events if e.pass_kind == pass_kind]
                if pass_events:
                    arrays[pass_kind] = dense_chunk_scores(
                        pass_events, num_frames=num_frames
                    )
            np.savez_compressed(run_dir / f"chunk_{query_chunk:03d}.npz", **arrays)

        spatial_meta: dict = {}
        if displacement is not None:
            # normalise to a distribution: every query row sums to 1 over all
            # keys, so dividing by the query count makes the five dt buckets of
            # one (layer, head) sum to 1 together
            per_layer = torch.tensor(
                [max(displacement_queries.get(i, 0), 1) for i in range(num_layers)],
                dtype=torch.float32,
                device=displacement.device,
            )
            scale = per_layer[:, None, None, None, None]
            np.savez_compressed(
                run_dir / "spatial_displacement.npz",
                displacement=(displacement / scale).cpu().numpy().astype(np.float32),
                absolute=(absolute / scale).cpu().numpy().astype(np.float32),
                concentration=(concentration / per_layer[:, None, None, None])
                .cpu()
                .numpy()
                .astype(np.float32),
                concentration_ranks=(self._concentration_ranks + 1).cpu().numpy(),
            )
            spatial_meta = {
                "spatial_layout": "[layers, heads, dt_buckets, dy, dx]",
                "absolute_layout": "[layers, heads, dt_buckets, y, x]",
                "concentration_layout": "[layers, heads, dt_buckets, top_k]",
                "temporal_buckets": list(TEMPORAL_BUCKETS),
                "spatial_query_stride": self.spatial_query_stride,
                "spatial_min_chunk": self.spatial_min_chunk,
                "spatial_queries_per_layer": displacement_queries.get(0, 0),
            }

        if token_mass is not None:
            # each (layer, head) row becomes a distribution over key tokens:
            # mean over the chunk's sampled queries and over denoising steps
            scores = token_mass / token_counts.clamp(min=1)[:, :, None, None]
            scores[~token_visible[:, None, None, :].expand_as(scores)] = torch.nan
            np.savez_compressed(
                run_dir / "token_scores.npz",
                token_scores=scores.cpu().numpy().astype(np.float32),
            )
            spatial_meta["token_scores_layout"] = "[chunks, layers, heads, tokens]"

        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "model_tag": model_tag,
                    "layout": "[steps, layers, heads, frames]",
                    "num_chunks": num_chunks,
                    "num_layers": num_layers,
                    # the key axis of the dumps; callers pass pixel `num_frames`
                    "num_latent_frames": num_frames,
                    "num_heads": int(events[0].scores.shape[0]),
                    "query_stride": self.query_stride,
                    **spatial_meta,
                    **(meta or {}),
                },
                indent=2,
            )
        )
        logger.info("Wrote per-chunk attention maps to %s", run_dir)
        return str(run_dir)


_warned_unsupported: set[str] = set()


def warn_unsupported_once(reason: str) -> None:
    """Warn (once per reason) that a layout the probe cannot map was skipped."""
    if reason in _warned_unsupported:
        return
    _warned_unsupported.add(reason)
    logger.warning("Attention-map probe skipping attention calls: %s", reason)


_recorder: ChunkAttentionRecorder | None = None
_recorder_resolved = False


def get_attention_map_recorder() -> ChunkAttentionRecorder | None:
    """The process-wide recorder, or ``None`` when the probe is disabled."""
    global _recorder, _recorder_resolved
    if _recorder_resolved:
        return _recorder
    _recorder_resolved = True
    output_dir = envs.SGLANG_DIFFUSION_ATTENTION_MAP_DIR
    if output_dir is None:
        return None

    from sglang.multimodal_gen.runtime.distributed import get_world_rank

    if get_world_rank() != 0:
        return None
    _recorder = ChunkAttentionRecorder(
        output_dir=output_dir,
        query_stride=envs.SGLANG_DIFFUSION_ATTENTION_MAP_QUERY_STRIDE,
        spatial=envs.SGLANG_DIFFUSION_ATTENTION_MAP_SPATIAL,
        spatial_query_stride=envs.SGLANG_DIFFUSION_ATTENTION_MAP_SPATIAL_QUERY_STRIDE,
        token_scores=envs.SGLANG_DIFFUSION_ATTENTION_MAP_TOKEN_SCORES,
    )
    logger.info(
        "Chunk attention-map probe enabled (dir=%s, query_stride=%d, spatial=%s)",
        output_dir,
        _recorder.query_stride,
        _recorder.spatial,
    )
    return _recorder
