# SPDX-License-Identifier: Apache-2.0
"""Light Forcing — chunk-aware growth + hierarchical block selection.

Reproduces https://github.com/chengtao-lv/LightForcing (files
``wan/modules/sparse_attention.py`` and ``wan/modules/kernel.py``) for the
block-causal setting. Parity against verbatim transcriptions of those is
asserted in ``test/unit/realtime/test_lightforcing_parity.py``.

Light Forcing is two mechanisms on top of a Self-Forcing-style causal Wan:

**Chunk-aware growth** notices that early chunks have little history to drop
and later chunks a lot, so one global sparsity knob is unfair to both. The
per-chunk sparsity starts from ``sparsity_base`` and is lowered by an amount
proportional to ``1 / sqrt(kv frames)``, with the proportionality constant
solved so the *total* FLOPs over the whole video match the target ``sparsity``
(:func:`calculate_chunk_sparsities`). Chunk 0 is always dense.

**Hierarchical selection** scores mean-pooled query blocks against mean-pooled
key blocks and keeps the top ``(1 - sparsity)`` fraction of key blocks per
query block. While the history is short (``<= keep_frames`` past frames) the
top-k runs over every block. Once it is longer, a first stage picks
``keep_frames - keep_sink - keep_near`` whole *frames* per query block by
frame-level pooled scores; the block-level top-k then runs with everything
outside {those frames, the first ``keep_sink`` frames of the view, the
``keep_near`` frames nearest the chunk, the chunk itself} scored ``-inf``.

Two deliberate adaptations to this runtime, both invisible at upstream's own
geometry (``frame_seqlen % block == 0``; theirs is 1536):

* Key blocks are **frame-aligned**: each latent frame is covered by
  ``ceil(frame_seqlen / block_k)`` blocks, the last one short. Upstream's
  fixed-size grid would straddle frames at Wan 480p's 1560-token frames and
  its frame-level stage (``frame_blk = frame_seq // BLKK``) is undefined
  there. The kernel takes token ranges, so ragged blocks cost nothing.
* The chunk index is **clamped** into the sparsity schedule; upstream indexes
  past the end of ``sparsity_list`` and crashes if the video is longer than
  the schedule it was computed for.
"""

import math

import msgspec
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
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_segment_mask,
)


class LightForcingConfig(msgspec.Struct, frozen=True):
    # Upstream's BLKQ/BLKK, defaulting to its SM90/SM100 sizes — they also fit
    # this runtime's kernel, whose key tiles are 128 wide, so 64-token blocks
    # would waste half of every tile. block_q is the kernel's query tile, so it
    # must be a power of two.
    block_q: int = 128
    block_k: int = 128
    # Target average sparsity over the whole video and the value the schedule
    # decays from; upstream ships 0.88/0.98 (short model) and 0.85/0.95 (long).
    sparsity: float = 0.88
    sparsity_base: float = 0.98
    # Frame budget of the hierarchical stage: keep_sink view-leading frames +
    # keep_near chunk-adjacent frames + (keep_frames - keep_sink - keep_near)
    # frames chosen per query block by frame-level scores.
    keep_frames: int = 6
    keep_sink: int = 1
    keep_near: int = 2
    # The schedule is solved over the whole video, so it needs the intended
    # length (in latent frames) and the KV window cap (-1 = uncapped) up
    # front — upstream computes it in inference.py from the CLI arguments.
    num_output_frames: int = 21
    local_attn_size: int = -1


def calculate_chunk_sparsities(
    *,
    num_output_frames: int,
    frames_per_block: int,
    local_attn_size: int,
    sparsity: float,
    sparsity_base: float,
) -> list[float]:
    """Per-chunk sparsity, upstream's ``calculate_chunk_sparsities`` verbatim.

    Entry ``i`` is chunk ``i``'s sparsity; entry 0 is always ``0.0`` (dense
    first chunk). Chunk ``i > 0`` sees ``(i + 1) * frames_per_block`` frames of
    KV (capped by ``local_attn_size``); its sparsity is
    ``sparsity_base - beta / sqrt(frame_count)`` with ``beta`` solved so the
    kv-length-weighted mean sparsity equals the target ``sparsity``.
    """
    chunk_frame_counts = range(
        2 * frames_per_block, num_output_frames + 1, frames_per_block
    )
    kv_lengths = [
        frame_count if local_attn_size == -1 else min(frame_count, local_attn_size)
        for frame_count in chunk_frame_counts
    ]
    alphas = [1 / math.sqrt(frame_count) for frame_count in chunk_frame_counts]

    target_flops = sum((1 - sparsity) * kv_length for kv_length in kv_lengths)
    base_flops = sum((1 - sparsity_base) * kv_length for kv_length in kv_lengths)
    alpha_weighted_flops = sum(
        alpha * kv_length for alpha, kv_length in zip(alphas, kv_lengths, strict=True)
    )
    if alpha_weighted_flops == 0:
        return [0.0] + [sparsity_base] * len(alphas)

    beta = (target_flops - base_flops) / alpha_weighted_flops
    return [0.0] + [sparsity_base - alpha * beta for alpha in alphas]


def frame_aligned_block_bounds(
    *, num_frames: int, frame_seqlen: int, block: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-open token bounds of the frame-aligned key blocks, ``[num_blocks]``.

    Each frame is covered independently by ``ceil(frame_seqlen / block)``
    blocks, so no block straddles a frame boundary and every block's frame is
    ``block_index // blocks_per_frame``.
    """
    blocks_per_frame = -(-frame_seqlen // block)
    within = (torch.arange(blocks_per_frame, device=device) * block).clamp(
        max=frame_seqlen
    )
    frame_starts = torch.arange(num_frames, device=device) * frame_seqlen
    lo = (frame_starts[:, None] + within[None, :]).flatten()
    hi = torch.cat([lo[1:], lo.new_tensor([num_frames * frame_seqlen])]).clamp(
        max=num_frames * frame_seqlen
    )
    return lo, hi


def mean_pool_blocks(
    x: torch.Tensor, *, block: int, group: int | None = None
) -> torch.Tensor:
    """Block means of ``x [length, heads, dim]`` → ``[num_blocks, heads, dim]``.

    Sums accumulate in float32 and the result is stored back in ``x.dtype``,
    matching upstream's ``mean_pool_blhd`` Triton kernel. With ``group`` set,
    ``length`` is treated as ``group``-sized rows pooled independently — the
    frame-aligned key pooling — so the short last block of each row divides by
    its true token count, exactly like upstream's masked tail block.
    """
    length, heads, dim = x.shape
    group = length if group is None else group
    rows = length // group
    blocks_per_row = -(-group // block)
    pad = blocks_per_row * block - group
    padded = torch.nn.functional.pad(
        x.view(rows, group, heads, dim), (0, 0, 0, 0, 0, pad)
    )
    # Accumulate in float32 without materializing a float32 copy of the input.
    sums = padded.view(rows, blocks_per_row, block, heads, dim).sum(
        dim=2, dtype=torch.float32
    )
    counts = torch.full((blocks_per_row,), block, device=x.device, dtype=torch.float32)
    counts[-1] = group - (blocks_per_row - 1) * block
    pooled = sums / counts[None, :, None, None]
    return pooled.view(rows * blocks_per_row, heads, dim).to(x.dtype)


def select_middle_frames(
    *,
    pooled_query: torch.Tensor,  # [heads, q_blocks, dim]
    pooled_frames: torch.Tensor,  # [heads, num_frames, dim]
    past_frames: int,
    keep_frames: int,
    keep_sink: int,
    keep_near: int,
) -> torch.Tensor:
    """Stage 1: per query block, the kept middle frames — ``[heads, q_blocks, kept]``.

    Upstream's ``_select_2stage_middle_frames``: frame-level pooled scores over
    the frames between the sink and the near window, top
    ``keep_frames - keep_sink - keep_near`` per query block. Returned indices
    are absolute view-local frame ids (upstream adds ``keep_offset=keep_sink``
    inside its scoring kernel).
    """
    if keep_sink < 0 or keep_near < 0:
        raise ValueError("keep_sink and keep_near must be non-negative.")
    if keep_sink + keep_near > keep_frames:
        raise ValueError("keep_sink + keep_near must be <= keep_frames.")
    keep_middle = keep_frames - keep_sink - keep_near
    heads, q_blocks, _ = pooled_query.shape
    if keep_middle == 0:
        return torch.empty(
            (heads, q_blocks, 0), device=pooled_query.device, dtype=torch.int64
        )
    middle = pooled_frames[:, keep_sink : past_frames - keep_near]
    frame_score = pooled_query @ middle.transpose(-1, -2)
    kept = torch.topk(frame_score, keep_middle, dim=-1, sorted=False).indices
    return kept + keep_sink


def lightforcing_block_mask(
    *,
    pooled_query: torch.Tensor,  # [heads, q_blocks, dim]
    pooled_key: torch.Tensor,  # [heads, kv_blocks, dim]
    blocks_per_frame: int,
    past_frames: int,
    keep_frames: int,
    keep_sink: int,
    keep_near: int,
    topk: int,
) -> torch.Tensor:
    """The kept key blocks per query block — bool ``[heads, q_blocks, kv_blocks]``.

    One-stage (``past_frames <= keep_frames``): plain top-k of the pooled
    scores over every key block. Two-stage: eligibility is the union of the
    stage-1 frames, the sink, the near window and the chunk's own frames;
    ineligible blocks score ``-inf`` before the same top-k. Upstream keeps
    whatever top-k returns, ``-inf`` entries included, and so does this.
    """
    heads, q_blocks, _ = pooled_query.shape
    kv_blocks = pooled_key.shape[1]
    num_frames = kv_blocks // blocks_per_frame
    scores = pooled_query @ pooled_key.transpose(-1, -2)

    if past_frames > keep_frames:
        pooled_frames = pooled_key.view(
            heads, num_frames, blocks_per_frame, -1
        ).mean(dim=2)
        kept_middle = select_middle_frames(
            pooled_query=pooled_query,
            pooled_frames=pooled_frames,
            past_frames=past_frames,
            keep_frames=keep_frames,
            keep_sink=keep_sink,
            keep_near=keep_near,
        )
        # Ineligible frames get -inf added rather than a block-level bool mask
        # materialized and masked_fill'd: frame-aligned blocks make the frame
        # axis a free reshape of the block axis, and the plan is rebuilt every
        # step, so launch count is what the planning cost is made of.
        bias = scores.new_full((heads, q_blocks, num_frames), float("-inf"))
        bias.scatter_(-1, kept_middle, 0.0)
        bias[..., :keep_sink] = 0.0
        bias[..., past_frames - keep_near : past_frames] = 0.0
        bias[..., past_frames:] = 0.0
        scores = (
            scores.view(heads, q_blocks, num_frames, blocks_per_frame)
            + bias[..., None]
        ).view(heads, q_blocks, kv_blocks)

    lut = torch.topk(scores, topk, dim=-1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, lut, True)
    return mask


class LightForcingAttention(SparseAttentionBackend):
    name = "lightforcing"

    def __init__(self, config: LightForcingConfig) -> None:
        super().__init__()
        block_q = config.block_q
        if block_q < 16 or block_q & (block_q - 1) != 0:
            raise ValueError(f"block_q must be a power of two >= 16, got {block_q}")
        if config.keep_sink < 0 or config.keep_near < 0:
            raise ValueError("keep_sink and keep_near must be non-negative.")
        if config.keep_sink + config.keep_near > config.keep_frames:
            raise ValueError("keep_sink + keep_near must be <= keep_frames.")
        self._config = config
        # Keyed by frames_per_block, which is all the schedule takes from the
        # geometry; everything else is frozen config.
        self._schedules: dict[int, list[float]] = {}
        # Pooled *history* key blocks per layer. The history keys are stable
        # across the denoising steps of a chunk — only the chunk's own keys are
        # rewritten every step — so their pooling is paid once per
        # ``(chunk, layer)``. Own-chunk keys are pooled fresh on every call
        # (memoizing them is the FAST-AR class of bug).
        self._pooled_history = LayoutCache()
        self._last_chunk_index = -1

    def _on_begin_forward(self, geometry: ChunkGeometry) -> None:
        chunk_index = geometry.query_chunk_index
        if chunk_index < self._last_chunk_index:
            # A new video restarts the chunk counter. Its first sparse chunk
            # can carry the same layout signature as the previous video's last
            # cached one, which would silently reuse the previous video's
            # pooled history keys for block selection.
            self._pooled_history.clear()
        self._last_chunk_index = chunk_index

    def _chunk_sparsity(self, layout: VisibleLayout) -> float:
        config = self._config
        schedule = self._schedules.get(layout.frames_per_block)
        if schedule is None:
            schedule = calculate_chunk_sparsities(
                num_output_frames=config.num_output_frames,
                frames_per_block=layout.frames_per_block,
                local_attn_size=config.local_attn_size,
                sparsity=config.sparsity,
                sparsity_base=config.sparsity_base,
            )
            self._schedules[layout.frames_per_block] = schedule
        return schedule[min(layout.query_chunk_index, len(schedule) - 1)]

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        config = self._config
        q_len = call.query.shape[1]
        kv_len = call.key.shape[1]
        if kv_len <= q_len:
            return None
        sparsity = self._chunk_sparsity(layout)
        if sparsity <= 0.0:
            return None

        frame_seqlen = layout.frame_seqlen
        blocks_per_frame = -(-frame_seqlen // config.block_k)
        kv_blocks = layout.num_frames * blocks_per_frame
        topk = min(kv_blocks, int((1.0 - sparsity) * kv_blocks))
        if topk <= 0 or topk >= kv_blocks:
            return None

        # The plan has no batch axis, so the selection is scored on the
        # batch-mean activations; causal video runs batch 1 in practice.
        query = call.query.mean(dim=0) if call.query.shape[0] > 1 else call.query[0]
        key = call.key.mean(dim=0) if call.key.shape[0] > 1 else call.key[0]
        pooled_query = mean_pool_blocks(query, block=config.block_q)

        history_len = (layout.num_frames - layout.query_frames) * frame_seqlen
        signature = (call.key_segments, kv_len, frame_seqlen)
        hit, pooled_history = self._pooled_history.get(call.layer_index, signature)
        if not hit:
            pooled_history = mean_pool_blocks(
                key[:history_len], block=config.block_k, group=frame_seqlen
            )
            self._pooled_history.put(call.layer_index, signature, pooled_history)
        pooled_own = mean_pool_blocks(
            key[history_len:], block=config.block_k, group=frame_seqlen
        )
        pooled_key = torch.cat([pooled_history, pooled_own], dim=0)

        mask = lightforcing_block_mask(
            pooled_query=pooled_query.permute(1, 0, 2),
            pooled_key=pooled_key.permute(1, 0, 2),
            blocks_per_frame=blocks_per_frame,
            past_frames=layout.num_frames - layout.query_frames,
            keep_frames=config.keep_frames,
            keep_sink=config.keep_sink,
            keep_near=config.keep_near,
            topk=topk,
        )
        block_lo, block_hi = frame_aligned_block_bounds(
            num_frames=layout.num_frames,
            frame_seqlen=frame_seqlen,
            block=config.block_k,
            device=call.query.device,
        )
        plan = plan_from_segment_mask(
            mask,
            segment_starts=block_lo,
            segment_ends=block_hi,
            block_m=config.block_q,
        )
        return SparseAttentionExecution(
            plan=plan, query=call.query, key=call.key, value=call.value
        )
