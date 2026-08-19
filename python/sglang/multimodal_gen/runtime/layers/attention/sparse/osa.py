# SPDX-License-Identifier: Apache-2.0
"""Oracle Sparse Attention (OSA) for block-causal video DiTs.

The design rests on one empirical property of the Self-Forcing family
(measured on the qk dumps in ``notes/exploration/qk-attention-maps.md``): a
head's attention map is a *replication of one frame-to-frame pattern*. The
within-frame spatial structure of the mass a head sends to a key frame is the
same for every (query frame, key frame) pair, at every chunk and — once the
denoising trajectory has settled — at every denoising step.

That makes the pattern *observable once and reusable forever*, which is what
"oracle" means here — not an unattainable post-hoc top-k, but a real oracle
consulted once during generation:

1. Chunk 0 runs dense. Each of its denoising steps recomputes
   ``softmax(q k^T)`` on a strided subset of queries and folds the key axis
   modulo the frame, giving per-head mass over spatial key *tiles*; every step
   overwrites the last, so the surviving measurement is the final step's,
   after the per-head pattern has settled.
2. Each ``(layer, head)`` freezes its tiles ordered by descending mass. On
   every later chunk, the query's own chunk, the sink and the
   ``num_recent_frames`` most recent frames are kept whole, and the remaining
   token budget buys each head its top-mass tiles — the *same* tile set in
   every other visible frame. ``density`` (fraction of visible keys read) is
   met exactly by construction.
3. Execution gathers each head's kept K/V rows into a compact contiguous
   buffer and runs the ordinary FA3 varlen kernel over it (see
   ``replicate_kernel.py``) — ~2x the shared range-walking kernel on these
   scattered-tile plans.
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
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.replicate_kernel import (
    ReplicateGatherPlan,
    build_gather_plan,
    replicate_gather_attention,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class OsaConfig(msgspec.Struct, frozen=True):
    """Knobs of Oracle Sparse Attention.

    Exactly one of ``density`` / ``sparsity`` must be set
    (``sparsity = 1 - density``); it is the fraction of visible keys read,
    met exactly by construction. ``sink_chunks`` / ``sink_latent_frames``
    define how much of the video's start counts as the sink.
    """

    # Fraction of visible keys read: the query's own chunk,
    # `sink_latent_frames` and `num_recent_frames` are always kept whole, and
    # the remaining token budget buys each head its top-mass tiles, the same
    # tile set in every other frame.
    density: float | None = None
    sparsity: float | None = None
    sink_chunks: int = 1
    # Sink size in latent frames; None falls back to sink_chunks whole chunks.
    # 1 is the classic first-frame sink.
    sink_latent_frames: int | None = None
    # Most-recent past latent frames kept whole (the diagonal band every
    # measured head shows), counted against the density budget.
    num_recent_frames: int = 2
    # Run the clean-latent KV-refresh pass dense. Its hidden states feed the
    # K/V every later chunk reads, so sparsification errors there compound.
    # Off by default to keep the read-density comparable with the baselines
    # (which sparsify that pass too); turn on as a quality lever.
    dense_cache_update: bool = False
    # Spatial selection quantum in key tokens; 64 matches the kernel's key
    # tile, so a kept tile is never partially wasted.
    spatial_tile: int = 64
    # Query sub-sampling of the calibration passes. The per-head tile profile
    # is an average over thousands of queries, so a stride of 8 costs 1/8 of
    # the recompute and moves the profile by well under a percent.
    calibration_query_stride: int = 8
    calibration_query_tile: int = 64


@torch.no_grad()
def measure_spatial_tile_mass(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    layout: VisibleLayout,
    softmax_scale: float,
    query_stride: int,
    query_tile: int,
    spatial_tile: int,
) -> torch.Tensor:
    """Mean attention mass per within-frame spatial tile: ``[heads, tiles]``.

    Folds the key axis modulo the frame — summing a head's probabilities over
    all visible frames and all sampled queries — so the result is the
    frame-to-frame pattern the policy freezes. Rows sum to ~1.
    Stays on device; the caller decides when (if ever) to sync.
    """
    frame_seqlen = layout.frame_seqlen
    num_frames = layout.num_frames
    num_tiles = (frame_seqlen + spatial_tile - 1) // spatial_tile
    sampled = query[:1, ::query_stride]
    keys = key[:1]
    num_heads = query.shape[2]
    spatial_mass = torch.zeros(
        num_heads, frame_seqlen, dtype=torch.float32, device=query.device
    )
    num_sampled = sampled.shape[1]
    for tile_start in range(0, num_sampled, query_tile):
        tile = sampled[:, tile_start : tile_start + query_tile]
        scores = torch.einsum("bqhd,bkhd->hqk", tile.float(), keys.float())
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        folded = probs.view(num_heads, -1, num_frames, frame_seqlen).sum((1, 2))
        spatial_mass += folded
    spatial_mass /= max(num_sampled, 1)
    padded = torch.zeros(
        num_heads, num_tiles * spatial_tile, dtype=torch.float32, device=query.device
    )
    padded[:, :frame_seqlen] = spatial_mass
    return padded.view(num_heads, num_tiles, spatial_tile).sum(-1)


def frame_ages(layout: VisibleLayout) -> np.ndarray:
    """Age of each visible frame in latent frames; own chunk is age <= 0."""
    own_start = int(layout.global_frame_ids[layout.own_frames].min())
    return own_start - layout.global_frame_ids


class OracleSparseAttention(SparseAttentionBackend):
    """OSA: measure the per-head tile pattern on chunk 0, replicate it forever."""

    name = "osa"

    def __init__(self, config: OsaConfig) -> None:
        super().__init__()
        if config.density is not None and config.sparsity is not None:
            raise ValueError("set either density or sparsity, not both")
        density = (
            config.density
            if config.density is not None
            else (1.0 - config.sparsity if config.sparsity is not None else None)
        )
        if density is None or not 0.0 < density <= 1.0:
            raise ValueError(
                "osa needs density in (0, 1] "
                "(or the equivalent sparsity in [0, 1))"
            )
        self._density = density
        self._config = config
        # layer -> [heads, tiles] spatial mass, overwritten at every denoise
        # step of chunk 0 so the surviving measurement is the last step's.
        self._spatial_mass: dict[int, torch.Tensor] = {}
        # layer -> [heads, tiles] tile indices by descending mass, frozen at
        # the first post-calibration chunk.
        self._tile_order: dict[int, torch.Tensor] = {}
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
        self._spatial_mass.clear()
        self._tile_order.clear()
        self._plans.clear()
        self._last_chunk_index = -1
        self._logged_summary = False

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        raise RuntimeError(
            "OSA executes through attend() (gather + FA3), not through the "
            "shared-kernel prepare() path"
        )

    def attend(self, call: SparseAttentionCall) -> torch.Tensor | None:
        """Gather + FA3 varlen execution (2x the shared range kernel on OSA's
        scattered-tile plans)."""
        layout = self._layout(call)
        if layout is None:
            return None
        plan = self._prepare_plan(call, layout)
        if plan is None:
            self._record_density(None, kv_len=call.key.shape[1])
            return None
        self._record_density(None, kv_len=call.key.shape[1], fraction=plan.density)
        return replicate_gather_attention(
            query=call.query,
            key=call.key,
            value=call.value,
            plan=plan,
            softmax_scale=call.softmax_scale,
        )

    def _prepare_plan(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> ReplicateGatherPlan | None:
        if self._config.dense_cache_update and self.in_cache_update:
            return None
        if layout.query_chunk_index == 0:
            # Chunk 0 runs dense; every denoising step overwrites the
            # measurement so the frozen pattern is the last step's, which is
            # when the per-head pattern has settled. The clean-latent
            # KV-refresh pass runs after the last step and must not overwrite.
            if not self.in_cache_update:
                self._spatial_mass[call.layer_index] = measure_spatial_tile_mass(
                    query=call.query,
                    key=call.key,
                    layout=layout,
                    softmax_scale=call.softmax_scale,
                    query_stride=self._config.calibration_query_stride,
                    query_tile=self._config.calibration_query_tile,
                    spatial_tile=self._config.spatial_tile,
                )
            return None
        order = self._tile_order.get(call.layer_index)
        if order is None:
            mass = self._spatial_mass.get(call.layer_index)
            if mass is None:
                self.warn_dense_once(
                    f"layer {call.layer_index} was never calibrated "
                    "(chunk 0 not seen)"
                )
                return None
            # The frame's short tail tile (when frame_seqlen % spatial_tile
            # != 0) is excluded from selection so every head keeps exactly the
            # same token count — the gather execution needs a uniform varlen
            # batch. Whole frames still include the tail.
            full_tiles = layout.frame_seqlen // self._config.spatial_tile
            order = torch.argsort(
                mass[:, :full_tiles], dim=1, descending=True
            )
            self._tile_order[call.layer_index] = order
            self._spatial_mass.pop(call.layer_index, None)
        return self._plan(call, layout, order)

    def _plan(
        self,
        call: SparseAttentionCall,
        layout: VisibleLayout,
        order: torch.Tensor,  # [heads, full_tiles] tile ids by descending mass
    ) -> ReplicateGatherPlan | None:
        signature = (
            call.key_segments,
            layout.query_frames,
            call.head_start,
            call.num_local_heads,
        )
        hit, cached = self._plans.get(call.layer_index, signature)
        if hit:
            return cached
        num_heads, num_tiles = order.shape
        if num_heads != call.num_local_heads:
            self.warn_dense_once(
                f"calibrated {num_heads} heads but layer "
                f"{call.layer_index} has {call.num_local_heads}"
            )
            return None
        frame_seqlen = layout.frame_seqlen
        tile_size = self._config.spatial_tile
        num_frames = layout.num_frames
        device = order.device

        # Frames kept whole: the query's own chunk, the sink, and the most
        # recent past frames. Everything else gets the replicated tile set.
        ages = frame_ages(layout)
        full = (
            layout.own_frames
            | layout.sink_frames(self._sink_frames())
            | ((ages > 0) & (ages <= self._config.num_recent_frames))
        )
        num_full = int(full.sum())
        num_other = num_frames - num_full
        budget = self._density * num_frames * frame_seqlen
        remaining = budget - num_full * frame_seqlen
        if num_other == 0 or remaining >= num_other * frame_seqlen:
            plan = None  # everything is kept — dense is strictly better
        else:
            tiles_kept = min(
                max(0, int(round(remaining / (num_other * tile_size)))),
                num_tiles,
            )
            full_frames = torch.from_numpy(full).to(device)
            frame_starts = (
                torch.arange(num_frames, dtype=torch.int64, device=device)
                * frame_seqlen
            )
            # Whole frames: every token, identical for all heads.
            full_tokens = (
                frame_starts[full_frames][:, None]
                + torch.arange(frame_seqlen, dtype=torch.int64, device=device)
            ).reshape(-1)
            # Replicated tiles: each head's top tiles in every other frame.
            kept_offsets = (
                order[:, :tiles_kept].to(torch.int64)[:, :, None] * tile_size
                + torch.arange(tile_size, dtype=torch.int64, device=device)
            ).reshape(num_heads, -1)  # [heads, tiles_kept * tile]
            other_starts = frame_starts[~full_frames]
            replicated = (
                other_starts[None, :, None] + kept_offsets[:, None, :]
            ).reshape(num_heads, -1)
            indices = torch.cat(
                [
                    full_tokens[None, :].expand(num_heads, -1),
                    replicated,
                ],
                dim=1,
            )
            indices, _ = torch.sort(indices, dim=1)
            plan = build_gather_plan(
                indices=indices,
                q_len=call.query.shape[1],
                kv_len=layout.kv_len,
            )
        self._plans.put(call.layer_index, signature, plan)
        self._log_summary(layout)
        return plan

    def _log_summary(self, layout: VisibleLayout) -> None:
        # Freezing is lazy per layer during the first post-calibration chunk;
        # wait one more chunk so the count covers every layer.
        if self._logged_summary or layout.query_chunk_index < 2:
            return
        self._logged_summary = True
        logger.info(
            "OSA frozen from chunk 0's last denoising step: "
            "%d layers, target density %.2f, %d recent + %d sink frames kept "
            "whole; visible window %d frames",
            len(self._tile_order),
            self._density,
            self._config.num_recent_frames,
            self._sink_frames(),
            layout.num_frames,
        )

    def _sink_frames(self) -> int:
        if self._config.sink_latent_frames is not None:
            return self._config.sink_latent_frames
        return self._config.sink_chunks * self._geometry.frames_per_block
