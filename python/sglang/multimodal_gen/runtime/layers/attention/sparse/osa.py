# SPDX-License-Identifier: Apache-2.0
"""Oracle Sparse Attention (OSA) for block-causal video DiTs.

The design rests on one empirical property of the Self-Forcing family
(measured on the qk dumps in ``notes/exploration/qk-attention-maps.md``): a
head's attention map is a *replication of one frame-to-frame pattern*. The 2-D
section map — attention mass per (query tile, key tile) within a frame pair —
is the same for every (query frame, key frame) pair, at every chunk and, once
the denoising trajectory has settled, at every denoising step. The map is
dominated by a within-frame diagonal band (query tile ``q`` attends to key
tiles near ``q``); 61–76% of its energy lies outside the best rank-1
approximation, so no query-independent ("vertical slash") selection can
represent it — an earlier 1-D variant that froze the map's column marginal
captured only ~0.25 of the history-frame mass where the 2-D pattern captures
~0.68 at the same budget, and was removed.

The pattern is *observable once and reusable forever*, which is what "oracle"
means here — not an unattainable post-hoc top-k, but a real oracle consulted
once during generation:

1. Chunk 0 runs dense. Each of its denoising steps recomputes
   ``softmax(q k^T)`` on a strided subset of queries, keeping the query-tile
   axis and folding the key axis modulo the frame; every step overwrites the
   last, so the surviving measurement is the final step's, after the pattern
   has settled.
2. Each ``(layer, head, query tile)`` freezes its key tiles ordered by
   descending mass. On every later chunk the token budget buys each query
   tile its top-mass key tiles — the *same* set in every replicated frame.
   The ``keep_own_chunk_full`` / ``keep_sink_full`` /
   ``keep_recent_frames_full`` switches choose which frames are exempt from
   the pattern and kept whole instead. ``density`` (fraction of visible keys read) is met exactly
   by construction.
3. Execution runs through the uniform block-sparse Triton kernel
   (``block_kernel.py``): every (head, query tile) keeps the same number of
   key tiles, so the trip count is grid-uniform and the K/V loads pipeline.
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
from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
    BlockSparsePlan,
    block_sparse_attention,
    build_block_plan,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    VisibleLayout,
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

    # Fraction of visible keys read. Frames selected by the keep_*_full
    # switches are kept whole; the remaining budget buys each (head, query
    # tile) its top-mass key tiles, the same set in every other frame.
    density: float | None = None
    sparsity: float | None = None
    sink_chunks: int = 1
    # Sink size in latent frames; None falls back to sink_chunks whole chunks.
    # 1 is the classic first-frame sink.
    sink_latent_frames: int | None = None
    # How many most-recent past latent frames keep_recent_frames_full keeps
    # whole, counted against the density budget.
    num_recent_frames: int = 2
    # Which frames are exempt from the tile pattern and kept whole ("full").
    # The three exemptions are independent; their union is the kept-whole set,
    # and its share of the visible window is the density floor. All three on
    # is the classic geometry (fastest at matched density, highest PSNR);
    # sink + recent alone are the two anchor frames whose dense keeps prevent
    # the chunk-periodic camera oscillation of the fully sparse setting; all
    # three off has no floor but shakes and corrupts below ~0.2.
    keep_own_chunk_full: bool = True
    keep_sink_full: bool = True
    keep_recent_frames_full: bool = True
    # Run the first this-many denoising forwards of every chunk dense (the
    # early steps set the chunk's structure; proposal 3.6's hybrid-dense
    # lever). 0 disables.
    dense_first_steps: int = 0
    # Run the clean-latent KV-refresh pass dense. Its hidden states feed the
    # K/V every later chunk reads, so sparsification errors there compound.
    # Off by default to keep the read-density comparable with the baselines
    # (which sparsify that pass too); turn on as a quality lever.
    dense_cache_update: bool = False
    # --- Chunk-aware density schedule ---------------------------------------
    # "constant" reproduces the historical behavior: every chunk gets the same
    # per-call budget. "flops_matched" front-loads: chunk i's density decays
    # toward ``schedule_floor_density`` as ``1/sqrt(kv frames)``, with the
    # slope solved so the kv-weighted mean density over the whole video equals
    # ``density`` (LightForcing's schedule applied to OSA's budget). Motivated
    # by the 2026-08-26 5 s probes: attention demand is front-loaded (the own
    # chunk alone holds 50-70% of the mass and early windows are small), so a
    # constant per-call budget starves exactly where the mass concentrates.
    chunk_schedule: str = "constant"
    # Latent frames of the whole video; the flops_matched beta solve needs
    # every chunk's kv length up front. The campaign runner injects it.
    schedule_num_frames: int | None = None
    # KV window cap in latent frames (-1 = full context).
    schedule_window_frames: int = -1
    # The density the schedule decays toward on late chunks.
    schedule_floor_density: float = 0.02
    # --- Demand-weighted allocation -----------------------------------------
    # Split each chunk's pattern budget over the patterned frames proportional
    # to measured per-frame attention demand instead of uniformly. Weights are
    # relative per-frame demand fitted from the 2026-08-26 group probes
    # (Self-Forcing 720p, stable across knobs and chunks): an own-chunk frame
    # carries ~8x the mass of an old history frame, the recent frame ~4x, the
    # sink ~2.5x, and history decays roughly as 1/age.
    demand_weighted: bool = False
    demand_w_own: float = 8.0
    demand_w_recent: float = 4.0
    demand_w_sink: float = 2.5
    # History frame of age a weighs clip(demand_w_hist_scale / a, 1, w_recent).
    demand_w_hist_scale: float = 8.0
    # When False, keep_*_full frames are read whole but NOT charged against
    # the density budget, so enabling an anchor can never starve the patterned
    # frames below their floor (the osa2s/osa2a 5 s failure mode). True keeps
    # the historical accounting where exemptions come out of the same budget.
    charge_exempt_frames: bool = True
    # Spatial selection quantum in key tokens; 64 matches the kernel's key
    # tile, so a kept tile is never partially wasted.
    spatial_tile: int = 64
    # Query tiling quantum of the frozen pattern; also the kernel's BLOCK_M.
    query_tile: int = 128
    # Query sub-sampling of the calibration passes. The per-tile profile is an
    # average over many queries, so a stride of 8 costs 1/8 of the recompute
    # and moves the profile by well under a percent.
    calibration_query_stride: int = 8


@torch.no_grad()
def measure_section_tile_mass(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    layout: VisibleLayout,
    softmax_scale: float,
    query_stride: int,
    query_tile: int,
    spatial_tile: int,
) -> torch.Tensor:
    """Mean attention mass per (query tile, key tile): ``[heads, q_tiles, k_tiles]``.

    The key axis is folded modulo the frame; the query axis keeps its
    within-frame tile identity. Query frames fold together — the
    frame-to-frame pattern is per *section*, and all sections of a head share
    it.
    """
    frame_seqlen = layout.frame_seqlen
    num_frames = layout.num_frames
    num_heads = query.shape[2]
    k_tiles = (frame_seqlen + spatial_tile - 1) // spatial_tile
    q_tiles = (frame_seqlen + query_tile - 1) // query_tile
    device = query.device

    positions = torch.arange(0, query.shape[1], query_stride, device=device)
    sampled = query[:1, ::query_stride]
    bucket = (positions % frame_seqlen) // query_tile  # [nq]
    keys = key[:1]
    mass = torch.zeros(num_heads, q_tiles, k_tiles, dtype=torch.float32, device=device)
    counts = torch.zeros(q_tiles, dtype=torch.float32, device=device)
    counts.index_add_(0, bucket, torch.ones_like(bucket, dtype=torch.float32))
    block = 256
    for start in range(0, sampled.shape[1], block):
        tile = sampled[:, start : start + block]
        scores = torch.einsum("bqhd,bkhd->hqk", tile.float(), keys.float())
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        folded = probs.view(num_heads, -1, num_frames, frame_seqlen).sum(2)
        padded = torch.zeros(
            num_heads, folded.shape[1], k_tiles * spatial_tile, device=device
        )
        padded[:, :, :frame_seqlen] = folded
        per_tile = padded.view(num_heads, -1, k_tiles, spatial_tile).sum(-1)
        mass.index_add_(1, bucket[start : start + block], per_tile)
    mass /= counts.clamp(min=1.0)[None, :, None]
    return mass


def flops_matched_densities(
    *,
    num_frames: int,
    frames_per_block: int,
    window_frames: int,
    mean_density: float,
    floor_density: float,
) -> list[float]:
    """Per-chunk read densities, front-loaded as ``floor + beta / sqrt(kv)``.

    Entry 0 is 1.0 (chunk 0 is the dense calibration chunk); entry ``i > 0``
    is chunk i's density, with beta solved so the kv-length-weighted mean
    density over the video equals ``mean_density`` — the same normalization as
    LightForcing's ``calculate_chunk_sparsities``, in density form.
    """
    counts = list(range(2 * frames_per_block, num_frames + 1, frames_per_block))
    kv = [c if window_frames < 0 else min(c, window_frames) for c in counts]
    alphas = [1.0 / (c**0.5) for c in counts]
    target = sum(mean_density * k for k in kv)
    base = sum(floor_density * k for k in kv)
    weighted = sum(a * k for a, k in zip(alphas, kv, strict=True))
    if weighted == 0:
        return [1.0] + [mean_density] * len(counts)
    beta = (target - base) / weighted
    return [1.0] + [
        min(1.0, max(floor_density, floor_density + a * beta)) for a in alphas
    ]


def frame_ages(layout: VisibleLayout, *, query_chunk_offset: int = 0) -> np.ndarray:
    """Age of each visible frame in latent frames; the query chunk is age <= 0.

    ``query_chunk_offset`` selects which chunk of a multi-chunk query (a
    rolling-forcing window) the ages are relative to; 0 is the first.
    """
    own = layout.frames_of_offset(query_chunk_offset)
    own_start = int(layout.global_frame_ids[own].min())
    return own_start - layout.global_frame_ids


class OracleSparseAttention(SparseAttentionBackend):
    """OSA: measure the per-(head, query tile) pattern on chunk 0, replicate it."""

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
                "osa needs density in (0, 1] " "(or the equivalent sparsity in [0, 1))"
            )
        if config.chunk_schedule not in ("constant", "flops_matched"):
            raise ValueError(
                f"unknown chunk_schedule {config.chunk_schedule!r} "
                "(use 'constant' or 'flops_matched')"
            )
        if config.chunk_schedule == "flops_matched" and not config.schedule_num_frames:
            raise ValueError(
                "chunk_schedule='flops_matched' needs schedule_num_frames "
                "(the video's latent frame count)"
            )
        self._density = density
        self._config = config
        # chunk index -> per-call density, solved once (flops_matched only).
        self._schedule: list[float] | None = None
        # layer -> [heads, q_tiles, k_tiles] section mass, overwritten at
        # every denoise step of chunk 0 so the surviving measurement is the
        # last step's.
        self._section_mass: dict[int, torch.Tensor] = {}
        # layer -> [heads, q_tiles, k_tiles] key tiles by descending mass,
        # frozen at the first post-calibration chunk.
        self._section_order: dict[int, torch.Tensor] = {}
        self._plans = LayoutCache()
        self._last_chunk_index = -1
        self._forwards_this_chunk = 0
        self._logged_summary = False

    def _on_begin_forward(self, geometry: ChunkGeometry) -> None:
        chunk_index = geometry.query_chunk_index
        if chunk_index < self._last_chunk_index:
            # A new video restarts the chunk counter; the previous video's
            # calibration says nothing about this one.
            self.reset()
        if chunk_index != self._last_chunk_index:
            self._forwards_this_chunk = 0
        else:
            self._forwards_this_chunk += 1
        self._last_chunk_index = chunk_index

    def reset(self) -> None:
        self._section_mass.clear()
        self._section_order.clear()
        self._plans.clear()
        self._last_chunk_index = -1
        self._forwards_this_chunk = 0
        self._logged_summary = False

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        raise RuntimeError(
            "OSA executes through attend() (the uniform block-sparse kernel), "
            "not through the shared-kernel prepare() path"
        )

    def attend(self, call: SparseAttentionCall) -> torch.Tensor | None:
        layout = self._layout(call)
        if layout is None:
            return None
        plan = self._prepare_plan(call, layout)
        if plan is None:
            self._record_density(None, kv_len=call.key.shape[1])
            return None
        if isinstance(plan, BlockSparsePlan):
            self._record_density(None, kv_len=call.key.shape[1], fraction=plan.density)
            return block_sparse_attention(
                query=call.query,
                key=call.key,
                value=call.value,
                plan=plan,
                softmax_scale=call.softmax_scale,
            )
        # A rolling window: one plan per query chunk over the shared KV view.
        # A None entry means that chunk keeps everything, which the block
        # kernel cannot express — fall back to dense for the whole call
        # rather than splicing partial outputs.
        if any(chunk_plan is None for chunk_plan in plan):
            self._record_density(None, kv_len=call.key.shape[1])
            return None
        tokens = layout.frames_per_block * layout.frame_seqlen
        outputs = [
            block_sparse_attention(
                query=call.query[:, index * tokens : (index + 1) * tokens],
                key=call.key,
                value=call.value,
                plan=chunk_plan,
                softmax_scale=call.softmax_scale,
            )
            for index, chunk_plan in enumerate(plan)
        ]
        self._record_density(
            None,
            kv_len=call.key.shape[1],
            fraction=sum(chunk_plan.density for chunk_plan in plan) / len(plan),
        )
        return torch.cat(outputs, dim=1)

    def _prepare_plan(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> BlockSparsePlan | tuple[BlockSparsePlan | None, ...] | None:
        if self._config.dense_cache_update and self.in_cache_update:
            return None
        if (
            self._config.dense_first_steps > 0
            and not self.in_cache_update
            and self._forwards_this_chunk < self._config.dense_first_steps
        ):
            return None
        if layout.query_chunk_index == 0:
            # Chunk 0 runs dense; every denoising step overwrites the
            # measurement so the frozen pattern is the last step's, which is
            # when the pattern has settled. The clean-latent KV-refresh pass
            # runs after the last step and must not overwrite.
            if not self.in_cache_update:
                self._section_mass[call.layer_index] = measure_section_tile_mass(
                    query=call.query,
                    key=call.key,
                    layout=layout,
                    softmax_scale=call.softmax_scale,
                    query_stride=self._config.calibration_query_stride,
                    query_tile=self._config.query_tile,
                    spatial_tile=self._config.spatial_tile,
                )
            return None
        order = self._section_order.get(call.layer_index)
        if order is None:
            mass = self._section_mass.get(call.layer_index)
            if mass is None:
                self.warn_dense_once(
                    f"layer {call.layer_index} was never calibrated "
                    "(chunk 0 not seen)"
                )
                return None
            # The frame's short tail tile (when frame_seqlen % spatial_tile
            # != 0) is excluded from selection so every (head, query tile)
            # keeps the same token count — the kernel's uniform trip count.
            # Whole frames still include the tail.
            full_tiles = layout.frame_seqlen // self._config.spatial_tile
            order = torch.argsort(mass[:, :, :full_tiles], dim=2, descending=True).to(
                torch.int32
            )
            self._section_order[call.layer_index] = order
            self._section_mass.pop(call.layer_index, None)
        num_query_chunks, ragged = divmod(layout.query_frames, layout.frames_per_block)
        if ragged:
            self.warn_dense_once(
                f"query spans {layout.query_frames} frames, not a whole number "
                f"of {layout.frames_per_block}-frame chunks"
            )
            return None
        signature = (
            call.key_segments,
            layout.query_frames,
            call.head_start,
            call.num_local_heads,
        )
        hit, cached = self._plans.get(call.layer_index, signature)
        if hit:
            return cached
        if order.shape[0] != call.num_local_heads:
            self.warn_dense_once(
                f"calibrated {order.shape[0]} heads but layer "
                f"{call.layer_index} has {call.num_local_heads}"
            )
            return None
        if num_query_chunks == 1:
            plan = self._chunk_plan(layout, order, query_chunk_offset=0)
        else:
            plans = tuple(
                self._chunk_plan(layout, order, query_chunk_offset=offset)
                for offset in range(num_query_chunks)
            )
            plan = None if all(entry is None for entry in plans) else plans
        self._plans.put(call.layer_index, signature, plan)
        self._log_summary(layout)
        return plan

    def _whole_frame_mask(
        self, layout: VisibleLayout, own: np.ndarray, ages: np.ndarray
    ) -> np.ndarray:
        full = np.zeros(layout.num_frames, dtype=bool)
        if self._config.keep_own_chunk_full:
            full |= own
        if self._config.keep_sink_full:
            full |= layout.sink_frames(self._sink_frames())
        if self._config.keep_recent_frames_full:
            full |= (ages > 0) & (ages <= self._config.num_recent_frames)
        return full

    def _chunk_plan(
        self,
        layout: VisibleLayout,
        order: torch.Tensor,  # [heads, q_tiles, full_tiles] by descending mass
        *,
        query_chunk_offset: int,
    ) -> BlockSparsePlan | None:
        """One query chunk's plan, or ``None`` if it keeps everything."""
        num_heads, q_tiles, num_tiles = order.shape
        frame_seqlen = layout.frame_seqlen
        tile_size = self._config.spatial_tile
        num_frames = layout.num_frames
        device = order.device

        own = layout.frames_of_offset(query_chunk_offset)
        if not own.any():
            # The query chunk's own keys are always in the view for the models
            # this supports; an unmapped chunk keeps everything to stay safe.
            return None
        ages = frame_ages(layout, query_chunk_offset=query_chunk_offset)
        full = self._whole_frame_mask(layout, own, ages)
        num_full = int(full.sum())
        num_other = num_frames - num_full
        budget = self._call_density(layout) * num_frames * frame_seqlen
        if self._config.charge_exempt_frames:
            budget -= num_full * frame_seqlen
        if num_other == 0 or budget >= num_other * frame_seqlen:
            return None  # everything is kept
        frame_ids = torch.arange(num_frames, dtype=torch.int32, device=device)
        full_frames = torch.from_numpy(full).to(device)
        hist_offsets = frame_ids[~full_frames] * frame_seqlen
        if self._config.demand_weighted:
            weights = self._demand_weights(layout, own, ages, full)
            # At least one tile per frame — a zero-tile plan would leave that
            # frame attending to nothing.
            counts = np.clip(
                np.round(budget * weights / weights.sum() / tile_size),
                1,
                num_tiles,
            ).astype(np.int64)
            return build_block_plan(
                tiles=order,
                hist_offsets=hist_offsets,
                hist_tile_counts=torch.from_numpy(counts).to(device),
                whole_offsets=frame_ids[full_frames] * frame_seqlen,
                frame_seqlen=frame_seqlen,
                query_tile=self._config.query_tile,
                key_tile=tile_size,
                kv_len=layout.kv_len,
            )
        # Uniform allocation: the same tile count for every patterned frame.
        tiles_kept = min(
            max(1, int(round(budget / (num_other * tile_size)))),
            num_tiles,
        )
        return build_block_plan(
            tiles=order[:, :, :tiles_kept],
            hist_offsets=hist_offsets,
            whole_offsets=frame_ids[full_frames] * frame_seqlen,
            frame_seqlen=frame_seqlen,
            query_tile=self._config.query_tile,
            key_tile=tile_size,
            kv_len=layout.kv_len,
        )

    def _call_density(self, layout: VisibleLayout) -> float:
        """This chunk's per-call density: the knob, or its scheduled value."""
        if self._config.chunk_schedule == "constant":
            return self._density
        if self._schedule is None:
            self._schedule = flops_matched_densities(
                num_frames=self._config.schedule_num_frames,
                frames_per_block=layout.frames_per_block,
                window_frames=self._config.schedule_window_frames,
                mean_density=self._density,
                floor_density=self._config.schedule_floor_density,
            )
        index = min(max(int(layout.query_chunk_index), 0), len(self._schedule) - 1)
        return self._schedule[index]

    def _demand_weights(
        self,
        layout: VisibleLayout,
        own: np.ndarray,
        ages: np.ndarray,
        full: np.ndarray,
    ) -> np.ndarray:
        """Relative per-frame demand of the patterned (non-full) frames."""
        config = self._config
        weights = np.clip(
            config.demand_w_hist_scale / np.maximum(ages.astype(np.float64), 1.0),
            1.0,
            config.demand_w_recent,
        )
        weights[own] = config.demand_w_own
        recent = (ages > 0) & (ages <= config.num_recent_frames)
        weights[recent] = config.demand_w_recent
        weights[layout.sink_frames(self._sink_frames()) & ~own & ~recent] = (
            config.demand_w_sink
        )
        return weights[~full]

    def _log_summary(self, layout: VisibleLayout) -> None:
        # Freezing is lazy per layer during the first post-calibration chunk;
        # wait one more chunk so the count covers every layer.
        if self._logged_summary or layout.query_chunk_index < 2:
            return
        self._logged_summary = True
        logger.info(
            "OSA frozen from chunk 0's last denoising step: "
            "%d layers, target density %.2f, kept full: own=%s sink=%s(%d) "
            "recent=%s(%d); visible window %d frames",
            len(self._section_order),
            self._density,
            self._config.keep_own_chunk_full,
            self._config.keep_sink_full,
            self._sink_frames(),
            self._config.keep_recent_frames_full,
            self._config.num_recent_frames,
            layout.num_frames,
        )

    def _sink_frames(self) -> int:
        if self._config.sink_latent_frames is not None:
            return self._config.sink_latent_frames
        return self._config.sink_chunks * self._geometry.frames_per_block
