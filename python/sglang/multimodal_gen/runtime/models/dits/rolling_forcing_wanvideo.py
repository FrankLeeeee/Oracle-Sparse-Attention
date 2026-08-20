# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/TencentARC/RollingForcing
"""Rolling Forcing causal Wan DiT.

Same architecture and weights layout as :class:`CausalWanTransformer3DModel`
(Wan2.1-T2V-1.3B), but with the Rolling Forcing streaming attention semantics
from ``wan/modules/causal_model.py`` upstream:

- One denoising forward covers a rolling *window* of blocks at staggered noise
  levels; only the window's **first** block is written into the persistent KV
  cache. The remaining window tokens attend to each other via the fresh keys.
- The first-ever block is the **attention sink**: stored un-RoPE'd, never
  evicted, and re-roped on the fly to a *relative* position just before the
  working cache (long-horizon memory without unbounded RoPE positions).
- After each window pass a second forward (``updating_cache=True``) re-runs
  the just-finished block at the context timestep to overwrite its cache slots
  with clean features.

The cache write layout is identical for all transformer layers of one forward,
so the model computes it once (:func:`compute_rolling_cache_layout`) and
stashes it on each layer's :class:`RollingForcingSelfAttentionKVCache`.
"""

from contextlib import nullcontext

import msgspec
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    RollingForcingSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
    CausalWanTransformerBlock,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
    CACHE_UPDATE_PASS,
    get_attention_map_recorder,
)
from sglang.multimodal_gen.runtime.utils.chunk_timing_probe import (
    timing_pass_kind_scope,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RollingCacheLayout(msgspec.Struct, frozen=True):
    """Cache write/read plan for one Rolling Forcing DiT forward.

    All indices are token offsets into the per-layer KV cache buffer, except
    ``anchor_start_frame`` (latent-frame index used to re-rope the sink block;
    ``-1`` when the sink is not re-roped this forward).
    """

    updating_cache: bool
    num_new_tokens: int
    num_evicted_tokens: int
    num_rolled_tokens: int
    local_start_index: int
    local_end_index: int
    global_end_after: int
    local_end_after: int
    working_start: int
    working_end: int
    anchor_start_frame: int


def compute_rolling_cache_layout(
    *,
    global_end_index: int,
    local_end_index: int,
    cache_size: int,
    current_start: int,
    block_tokens: int,
    sink_tokens: int,
    max_attention_tokens: int,
    q_tokens: int,
    frame_seqlen: int,
    num_frames_per_block: int,
    updating_cache: bool,
) -> RollingCacheLayout:
    """Mirror of the upstream ``CausalWanSelfAttention`` cache index math."""
    cache_end = current_start + block_tokens
    num_new_tokens = cache_end - global_end_index
    assert num_new_tokens >= 0, (
        f"rolling cache went backwards: cache_end={cache_end}, "
        f"global_end_index={global_end_index}"
    )

    if num_new_tokens > 0 and num_new_tokens + local_end_index > cache_size:
        num_evicted_tokens = num_new_tokens + local_end_index - cache_size
        num_rolled_tokens = local_end_index - num_evicted_tokens - sink_tokens
        new_local_end = local_end_index + num_new_tokens - num_evicted_tokens
    else:
        num_evicted_tokens = 0
        num_rolled_tokens = 0
        new_local_end = local_end_index + num_new_tokens
    local_start = new_local_end - block_tokens

    working_start = 0
    working_end = 0
    anchor_start_frame = -1
    if local_start > 0:
        if updating_cache:
            working_end = new_local_end
            working_start = max(0, new_local_end - max_attention_tokens)
            if working_start == 0:
                # The sink is part of the working range; re-rope it at its
                # absolute position (frame 0).
                anchor_start_frame = 0
        else:
            working_cache_max = max_attention_tokens - q_tokens - block_tokens
            working_end = local_start
            working_start = max(block_tokens, local_start - working_cache_max)
            working_frames = (working_end - working_start) // frame_seqlen
            current_start_frame = current_start // frame_seqlen
            # Re-rope the sink to sit immediately before the working cache.
            anchor_start_frame = (
                current_start_frame - working_frames - num_frames_per_block
            )
            assert anchor_start_frame >= 0

    return RollingCacheLayout(
        updating_cache=updating_cache,
        num_new_tokens=num_new_tokens,
        num_evicted_tokens=num_evicted_tokens,
        num_rolled_tokens=num_rolled_tokens,
        local_start_index=local_start,
        local_end_index=new_local_end,
        global_end_after=cache_end if num_new_tokens > 0 else global_end_index,
        local_end_after=new_local_end if num_new_tokens > 0 else local_end_index,
        working_start=working_start,
        working_end=working_end,
        anchor_start_frame=anchor_start_frame,
    )


def _cache_range_segments(
    local_start: int,
    local_end: int,
    *,
    sink_tokens: int,
    window_start: int,
) -> tuple[tuple[int, int], ...]:
    """Map a cache-buffer range to global ``(token_start, length)`` ranges.

    The sink slots ``[0, sink_tokens)`` always hold the first block of the video;
    everything after them rolls, so slot ``i`` holds global token
    ``window_start + i``.
    """
    segments = []
    if local_start < sink_tokens:
        segments.append((local_start, min(local_end, sink_tokens) - local_start))
    if local_end > sink_tokens:
        start = max(local_start, sink_tokens)
        segments.append((window_start + start, local_end - start))
    return tuple(segments)


def visible_key_segments(
    layout: RollingCacheLayout,
    *,
    sink_tokens: int,
    block_tokens: int,
    current_start: int,
    num_query_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Global ``(token_start, length)`` ranges of the keys one attention call sees.

    Mirrors the key assembly in :meth:`RollingForcingWanSelfAttention.forward`
    (only the per-chunk attention-map probe consumes this).
    """
    if layout.local_start_index == 0:
        # Ramp-up: everything visible is the window's own freshly computed keys.
        return ((current_start, num_query_tokens),)

    window_start = layout.global_end_after - layout.local_end_after
    working = _cache_range_segments(
        layout.working_start,
        layout.working_end,
        sink_tokens=sink_tokens,
        window_start=window_start,
    )
    if layout.updating_cache:
        return working
    # Denoising pass: re-roped sink, working cache, then the window's own keys.
    return ((0, block_tokens),) + working + ((current_start, num_query_tokens),)


class RollingForcingWanSelfAttention(nn.Module):
    """Self-attention with the Rolling Forcing cache protocol.

    Reads the per-forward :class:`RollingCacheLayout` (and anchor RoPE table)
    off the layer's cache object; see the module docstring for the protocol.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        head_dim: int | None = None,
        head_start: int = 0,
    ) -> None:
        if head_dim is None:
            assert dim % num_heads == 0
            head_dim = dim // num_heads
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Set by the transformer after the block list is built; only read by the
        # per-chunk attention-map probe.
        self.layer_index = -1
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FA,
                AttentionBackendEnum.AITER,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    @staticmethod
    def _write_first_block(
        kv_cache: RollingForcingSelfAttentionKVCache,
        layout: RollingCacheLayout,
        *,
        key: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        block_tokens = kv_cache.rolling_block_tokens
        sink_tokens = kv_cache.sink_tokens

        if layout.num_evicted_tokens > 0 and layout.num_rolled_tokens > 0:
            src = slice(
                sink_tokens + layout.num_evicted_tokens,
                sink_tokens + layout.num_evicted_tokens + layout.num_rolled_tokens,
            )
            dst = slice(sink_tokens, sink_tokens + layout.num_rolled_tokens)
            kv_cache.k[:, dst] = kv_cache.k[:, src].clone()
            kv_cache.v[:, dst] = kv_cache.v[:, src].clone()

        write = slice(layout.local_start_index, layout.local_end_index)
        if layout.local_start_index == 0:
            # The very first block is the attention sink: keep it un-RoPE'd so
            # it can be re-roped to a relative position later.
            kv_cache.k[:, write] = key[:, :block_tokens]
        else:
            kv_cache.k[:, write] = roped_key[:, :block_tokens]
        kv_cache.v[:, write] = value[:, :block_tokens]

        kv_cache.write_indices(
            global_end_index=layout.global_end_after,
            local_end_index=layout.local_end_after,
        )

    def _rerope_anchor(
        self,
        kv_cache: RollingForcingSelfAttentionKVCache,
        value: torch.Tensor,
    ) -> torch.Tensor:
        anchor_freqs = kv_cache.rolling_anchor_freqs
        assert anchor_freqs is not None
        cos, sin = anchor_freqs
        block_tokens = kv_cache.rolling_block_tokens
        return _apply_rotary_emb(
            kv_cache.k[:, :block_tokens], cos, sin, is_neox_style=False
        ).type_as(value)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask,
        kv_cache: RollingForcingSelfAttentionKVCache | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        cos, sin = freqs_cis
        roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
        roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)

        if kv_cache is None:
            return self.attn(roped_query, roped_key, v)

        layout = kv_cache.rolling_layout
        assert isinstance(layout, RollingCacheLayout), (
            "RollingForcingWanSelfAttention requires the model to precompute "
            "the rolling cache layout before the block loop"
        )
        self._write_first_block(kv_cache, layout, key=k, roped_key=roped_key, value=v)

        block_tokens = kv_cache.rolling_block_tokens
        recorder = get_attention_map_recorder()
        key_segments = (
            visible_key_segments(
                layout,
                sink_tokens=kv_cache.sink_tokens,
                block_tokens=block_tokens,
                current_start=current_start,
                num_query_tokens=q.shape[1],
            )
            if recorder is not None
            else ()
        )

        def record(key: torch.Tensor) -> None:
            if recorder is not None:
                recorder.record(
                    layer_index=self.layer_index,
                    query=roped_query,
                    key=key,
                    key_segments=key_segments,
                )

        if layout.local_start_index == 0:
            # Ramp-up windows: everything visible is inside the current window.
            record(roped_key)
            return self.attn(roped_query, roped_key, v)

        working = slice(layout.working_start, layout.working_end)
        if layout.updating_cache:
            working_k = kv_cache.k[:, working]
            working_v = kv_cache.v[:, working]
            if layout.anchor_start_frame >= 0:
                working_k = working_k.clone()
                working_k[:, :block_tokens] = self._rerope_anchor(kv_cache, v)
            record(working_k)
            return self.attn(roped_query, working_k, working_v)

        anchor_k = self._rerope_anchor(kv_cache, v)
        anchor_v = kv_cache.v[:, :block_tokens]
        input_k = torch.cat([anchor_k, kv_cache.k[:, working], roped_key], dim=1)
        input_v = torch.cat([anchor_v, kv_cache.v[:, working], v], dim=1)
        record(input_k)
        return self.attn(roped_query, input_k, input_v)


class RollingForcingWanTransformerBlock(CausalWanTransformerBlock):
    _self_attn_cls = RollingForcingWanSelfAttention


class RollingForcingWanTransformer3DModel(CausalWanTransformer3DModel):
    _block_cls = RollingForcingWanTransformerBlock

    def _anchor_freqs_cis(
        self,
        *,
        anchor_start_frame: int,
        post_patch_height: int,
        post_patch_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        cos, sin = get_rotary_pos_embed(
            (
                self.config.arch_config.num_frames_per_block,
                post_patch_height,
                post_patch_width,
            ),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
            rope_theta=10000,
            start_frame=anchor_start_frame,
        )
        return cos.to(device).float(), sin.to(device).float()

    def _prepare_rolling_forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: list[RollingForcingSelfAttentionKVCache],
        current_start: int,
        updating_cache: bool,
    ) -> None:
        arch_config = self.config.arch_config
        _, _, num_frames, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size[1]
        post_patch_width = width // self.patch_size[2]
        frame_seqlen = post_patch_height * post_patch_width
        block_tokens = arch_config.num_frames_per_block * frame_seqlen

        template_cache = kv_cache[0]
        global_end_index, local_end_index = template_cache.read_indices()
        layout = compute_rolling_cache_layout(
            global_end_index=global_end_index,
            local_end_index=local_end_index,
            cache_size=template_cache.cache_size,
            current_start=current_start,
            block_tokens=block_tokens,
            sink_tokens=template_cache.sink_tokens,
            max_attention_tokens=arch_config.max_attention_num_frames * frame_seqlen,
            q_tokens=(num_frames // self.patch_size[0]) * frame_seqlen,
            frame_seqlen=frame_seqlen,
            num_frames_per_block=arch_config.num_frames_per_block,
            updating_cache=updating_cache,
        )

        anchor_freqs = None
        if layout.anchor_start_frame >= 0:
            anchor_freqs = self._anchor_freqs_cis(
                anchor_start_frame=layout.anchor_start_frame,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
                device=hidden_states.device,
            )

        for layer_cache in kv_cache:
            layer_cache.rolling_layout = layout
            layer_cache.rolling_block_tokens = block_tokens
            layer_cache.rolling_anchor_freqs = anchor_freqs

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: list[RollingForcingSelfAttentionKVCache] | None = None,
        crossattn_cache=None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        updating_cache: bool = False,
    ) -> torch.Tensor:
        if kv_cache is not None:
            self._prepare_rolling_forward(
                hidden_states,
                kv_cache,
                current_start,
                updating_cache,
            )
        recorder = get_attention_map_recorder()
        pass_scope = (
            recorder.pass_kind_scope(CACHE_UPDATE_PASS)
            if recorder is not None and updating_cache
            else nullcontext()
        )
        timing_scope = (
            timing_pass_kind_scope(CACHE_UPDATE_PASS)
            if updating_cache
            else nullcontext()
        )
        with pass_scope, timing_scope:
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_hidden_states_image=encoder_hidden_states_image,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                start_frame=start_frame,
            )


EntryClass = RollingForcingWanTransformer3DModel
