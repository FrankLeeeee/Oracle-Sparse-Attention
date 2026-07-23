# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shengshu-ai/minWM (HY15/hyvideo/models/transformers)
"""Causal (autoregressive) HunyuanVideo 1.5 DiT, as trained by minWM.

Architecture is the stock HunyuanVideo-1.5 8B dual-stream MMDiT (54 double
blocks, no single blocks, patch size 1).  minWM's causal rollout splits every
double block into two halves:

* ``txt`` half — run once per request over the combined condition tokens
  (SigLIP vision tokens + ByT5 glyph tokens + Qwen2.5-VL text tokens, each
  offset by ``cond_type_embedding``).  The per-layer post-qk-norm K/V of the
  evolving text stream are cached; this maps onto the framework's
  ``CrossAttentionKVCache``.
* ``img`` half — run per chunk of latent frames.  Attention keys/values are
  the concatenation of the cached text K/V and the vision KV cache (all clean
  chunks so far plus the current chunk); causality comes purely from cache
  construction, no mask.  This maps onto ``CausalSelfAttentionKVCache``.

Parameter names match the checkpoint (original tencent naming, e.g.
``double_blocks.N.img_attn_q``), so no weight-name mapping is required.
"""

import math
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.hunyuanvideo15 import (
    CausalHunyuanVideo15Config,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    visible_key_segments,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
    get_attention_map_recorder,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_ATTENTION_BACKENDS = (
    AttentionBackendEnum.FA,
    AttentionBackendEnum.TORCH_SDPA,
)


# ---------------------------------------------------------------------------
# Rotary position embedding (minWM ``posemb_layers`` math, interleaved style)
# ---------------------------------------------------------------------------


def _get_1d_rotary_freqs(dim: int, pos: torch.Tensor, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    return torch.outer(pos, freqs)  # [S, dim/2]


def get_nd_rotary_pos_embed(
    rope_dim_list: list[int],
    rope_sizes: tuple[int, int, int],
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape [T*H*W, head_dim], interleave-duplicated pairs."""
    axes = [torch.arange(s, dtype=torch.float32) for s in rope_sizes]
    grid = torch.meshgrid(*axes, indexing="ij")
    freqs = torch.cat(
        [
            _get_1d_rotary_freqs(rope_dim_list[i], grid[i].reshape(-1), theta)
            for i in range(3)
        ],
        dim=1,
    )  # [S, head_dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=1)
    sin = freqs.sin().repeat_interleave(2, dim=1)
    return cos, sin


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)


def apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q/k: [B, S, H, D]; cos/sin: [S, D]."""
    cos = cos.view(1, -1, 1, cos.shape[-1])
    sin = sin.view(1, -1, 1, sin.shape[-1])
    q_out = (q.float() * cos + _rotate_half_interleaved(q.float()) * sin).type_as(q)
    k_out = (k.float() * cos + _rotate_half_interleaved(k.float()) * sin).type_as(k)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Small modules (parameter names match the checkpoint)
# ---------------------------------------------------------------------------


def _timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """``time_in`` / ``txt_in.t_embedder``: sinusoidal -> MLP (names mlp.0/mlp.2)."""

    def __init__(self, hidden_size: int, freq_embed_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = freq_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = _timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.type_as(self.mlp[0].weight))


class TextProjection(nn.Module):
    """``txt_in.c_embedder`` (names linear_1/linear_2, SiLU between)."""

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(caption)))


class MLP(nn.Module):
    """Transformer MLP (names fc1/fc2, gelu-tanh)."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ModulateDiT(nn.Module):
    """``img_mod`` / ``txt_mod``: SiLU -> Linear(hidden, factor*hidden)."""

    def __init__(self, hidden_size: int, factor: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


class PatchEmbed(nn.Module):
    """``img_in``: Conv3d over 65 channels (32 latent + 32 cond + 1 mask)."""

    def __init__(
        self, patch_size: list[int], in_chans: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=tuple(patch_size),
            stride=tuple(patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # BCTHW -> B(THW)C


class ByT5Mapper(nn.Module):
    """``byt5_in``: LayerNorm -> fc1 -> GELU -> fc2 -> GELU -> fc3."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, out_dim1: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        return self.fc3(x)


class VisionProjection(nn.Module):
    """``vision_in``: LN -> Linear -> GELU -> Linear -> LN (Sequential names)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(vision_embeds)


class FinalLayer(nn.Module):
    """``final_layer``: adaLN (Sequential index 1 = Linear) + projection."""

    def __init__(
        self, hidden_size: int, patch_size: list[int], out_channels: int
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, math.prod(patch_size) * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # vec: per-token [B, S, C]
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads_num: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads_num = heads_num
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = _RefinerMLP(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        qkv = self.self_attn_qkv(self.norm1(x))
        b, s, _ = qkv.shape
        q, k, v = qkv.view(b, s, 3, self.heads_num, -1).unbind(2)
        # Dense-masked SDPA: the framework attention backends don't thread
        # pairwise masks, and this refiner is only two small layers.
        attn = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
        )
        attn = attn.transpose(1, 2).flatten(2)
        x = x + self.self_attn_proj(attn) * gate_msa.unsqueeze(1)
        x = x + self.mlp(self.norm2(x)) * gate_mlp.unsqueeze(1)
        return x


class _RefinerMLP(nn.Module):
    """Refiner MLP uses SiLU (names fc1/fc2)."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class IndividualTokenRefiner(nn.Module):
    def __init__(
        self, hidden_size: int, heads_num: int, depth: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(hidden_size, heads_num, mlp_ratio)
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_mask = None
        if mask is not None:
            mask = mask.clone().bool()
            mask[:, 0] = True  # prevent NaN rows
            # [B, L] -> [B, 1, L, L] pairwise validity
            m1 = mask.view(mask.shape[0], 1, 1, mask.shape[1])
            attn_mask = m1 & m1.transpose(2, 3)
        for block in self.blocks:
            x = block(x, c, attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """``txt_in``: Qwen2.5-VL text projection (LI-DiT token refiner, depth 2)."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = TextProjection(in_channels, hidden_size)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size, heads_num, depth, mlp_ratio
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        timestep_aware = self.t_embedder(t)
        if mask is None:
            context_aware = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)
            context_aware = (
                (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            ).to(x.dtype)
        c = timestep_aware + self.c_embedder(context_aware)
        x = self.input_embedder(x)
        return self.individual_token_refiner(x, c, mask)


# ---------------------------------------------------------------------------
# Double-stream block
# ---------------------------------------------------------------------------


class CausalMMDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        layer_index: int,
    ) -> None:
        super().__init__()
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        self.layer_index = layer_index
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(hidden_size, factor=6)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.img_attn_q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.img_attn_k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim)

        self.txt_mod = ModulateDiT(hidden_size, factor=6)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.txt_attn_q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.txt_attn_k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim)

        self.attn = LocalAttention(
            num_heads=heads_num,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=set(_ATTENTION_BACKENDS),
        )

    def _qkv(
        self,
        x: torch.Tensor,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        q = to_q(x).view(b, s, self.heads_num, self.head_dim)
        k = to_k(x).view(b, s, self.heads_num, self.head_dim)
        v = to_v(x).view(b, s, self.heads_num, self.head_dim)
        q = q_norm(q).to(v)
        k = k_norm(k).to(v)
        return q, k, v

    def forward_txt(
        self,
        txt: torch.Tensor,
        vec_txt: torch.Tensor,
        crossattn_cache: CrossAttentionKVCache | None,
    ) -> torch.Tensor:
        """txt half; caches this layer's text K/V for the vision passes."""
        (
            shift1,
            scale1,
            gate1,
            shift2,
            scale2,
            gate2,
        ) = self.txt_mod(vec_txt).chunk(6, dim=-1)
        txt_modulated = self.txt_norm1(txt) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        q, k, v = self._qkv(
            txt_modulated,
            self.txt_attn_q,
            self.txt_attn_k,
            self.txt_attn_v,
            self.txt_attn_q_norm,
            self.txt_attn_k_norm,
        )
        if crossattn_cache is not None:
            crossattn_cache.store(k, v)
        attn = self.attn(q, k, v).flatten(2)
        txt = txt + self.txt_attn_proj(attn) * gate1.unsqueeze(1)
        txt_modulated2 = (
            self.txt_norm2(txt) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        )
        txt = txt + self.txt_mlp(txt_modulated2) * gate2.unsqueeze(1)
        return txt

    def forward_vision(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: CausalSelfAttentionKVCache,
        crossattn_cache: CrossAttentionKVCache,
        current_start: int,
    ) -> torch.Tensor:
        """img half; attends over [text K/V, clean-chunk KV cache, current chunk]."""
        (
            shift1,
            scale1,
            gate1,
            shift2,
            scale2,
            gate2,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        img_modulated = self.img_norm1(img) * (1 + scale1) + shift1
        q, k, v = self._qkv(
            img_modulated,
            self.img_attn_q,
            self.img_attn_k,
            self.img_attn_v,
            self.img_attn_q_norm,
            self.img_attn_k_norm,
        )
        cos, sin = freqs_cis
        q, k = apply_rotary_emb_qk(q, k, cos, sin)
        k = k.type_as(v)
        q = q.type_as(v)

        cache_view = kv_cache.update_and_get_attention_kv(
            key=k,
            value=v,
            current_chunk_start=current_start,
            cache_head_start=0,
            debug_name="CausalHunyuanVideo15 KV cache",
        )
        recorder = get_attention_map_recorder()
        if recorder is not None:
            # Latent-token map only: the joint attention also sees the text
            # KV, but the probe's chunk/token axes are latent positions, so
            # record queries against the visible vision keys.
            recorder.record(
                layer_index=self.layer_index,
                query=q,
                key=cache_view.k,
                key_segments=visible_key_segments(kv_cache, cache_view),
            )
        full_k = torch.cat([crossattn_cache.k, cache_view.k], dim=1)
        full_v = torch.cat([crossattn_cache.v, cache_view.v], dim=1)
        attn = self.attn(q, full_k, full_v).flatten(2)

        img = img + self.img_attn_proj(attn) * gate1
        img_modulated2 = self.img_norm2(img) * (1 + scale2) + shift2
        img = img + self.img_mlp(img_modulated2) * gate2
        return img


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class CausalHunyuanVideo15Transformer3DModel(BaseDiT):
    _fsdp_shard_conditions = CausalHunyuanVideo15Config()._fsdp_shard_conditions
    _compile_conditions = CausalHunyuanVideo15Config()._compile_conditions
    _supported_attention_backends = (
        CausalHunyuanVideo15Config()._supported_attention_backends
    )
    param_names_mapping = CausalHunyuanVideo15Config().param_names_mapping
    reverse_param_names_mapping = (
        CausalHunyuanVideo15Config().reverse_param_names_mapping
    )
    lora_param_names_mapping: dict = {}

    def __init__(
        self,
        config: CausalHunyuanVideo15Config,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.heads_num
        self.attention_head_dim = arch.hidden_size // arch.heads_num
        self.num_channels_latents = arch.out_channels
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.patch_size = list(arch.patch_size)
        self.rope_dim_list = list(arch.rope_dim_list)
        self.rope_theta = float(arch.rope_theta)
        self.local_attn_size = arch.local_attn_size
        # Consumed by CausalDMDDenoisingStage; HY1.5 has no independent first
        # frame — every chunk has the same size.
        self.independent_first_frame = False

        if arch.concat_condition:
            # 32 noisy latent + 32 cond latent + 1 mask channels
            in_chans = arch.in_channels * 2 + 1
        else:
            in_chans = arch.in_channels

        self.img_in = PatchEmbed(self.patch_size, in_chans, self.hidden_size)
        self.txt_in = SingleTokenRefiner(
            arch.text_states_dim, self.hidden_size, arch.heads_num, depth=2
        )
        self.byt5_in = ByT5Mapper(
            in_dim=arch.byt5_states_dim,
            hidden_dim=2048,
            out_dim=2048,
            out_dim1=self.hidden_size,
        )
        self.vision_in = VisionProjection(arch.vision_states_dim, self.hidden_size)
        self.time_in = TimestepEmbedder(self.hidden_size)
        self.cond_type_embedding = nn.Embedding(3, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                CausalMMDoubleStreamBlock(
                    self.hidden_size,
                    arch.heads_num,
                    arch.mlp_width_ratio,
                    layer_index=i,
                )
                for i in range(arch.mm_double_blocks_depth)
            ]
        )
        self.final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )

        self._freqs_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.__post_init__()

    # ------------------------------------------------------------------
    # Condition-token assembly (mirrors minWM ``get_text_and_mask``)
    # ------------------------------------------------------------------

    def _cond_type_embed(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        type_ids = torch.full(
            tokens.shape[:2], type_id, device=tokens.device, dtype=torch.long
        )
        return tokens + self.cond_type_embedding(type_ids)

    def _assemble_condition_tokens(
        self,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        byt5_states: torch.Tensor,
        byt5_mask: torch.Tensor,
        image_embeds: torch.Tensor | None,
        timestep_txt: torch.Tensor,
    ) -> torch.Tensor:
        """Project, tag and concatenate the valid condition tokens.

        Returns [1, n_valid, hidden] with token order
        [vision, byt5_valid, qwen_valid] — matching minWM's double reordering
        followed by pre-masking (invalid tokens are dropped entirely, which is
        exactly what ``txt[text_mask.bool()]`` does upstream).
        """
        if text_states.shape[0] != 1:
            raise ValueError(
                "causal HunyuanVideo 1.5 text prefill supports batch size 1"
            )

        txt = self.txt_in(text_states, timestep_txt, text_mask)
        txt = self._cond_type_embed(txt, 0)

        byt5_txt = self._cond_type_embed(self.byt5_in(byt5_states), 1)

        parts = []
        if image_embeds is not None:
            vision_tokens = self._cond_type_embed(self.vision_in(image_embeds), 2)
            parts.append(vision_tokens[0])
        parts.append(byt5_txt[0][byt5_mask[0].bool()])
        parts.append(txt[0][text_mask[0].bool()])
        return torch.cat(parts, dim=0).unsqueeze(0)

    def _prefill_text_kv(
        self,
        crossattn_cache: list[CrossAttentionKVCache],
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        byt5_states: torch.Tensor,
        byt5_mask: torch.Tensor,
        image_embeds: torch.Tensor | None,
    ) -> None:
        """Run the txt half of every block once, caching per-layer K/V."""
        timestep_txt = torch.zeros(
            text_states.shape[0], device=text_states.device, dtype=torch.float32
        )
        txt = self._assemble_condition_tokens(
            text_states=text_states,
            text_mask=text_mask,
            byt5_states=byt5_states,
            byt5_mask=byt5_mask,
            image_embeds=image_embeds,
            timestep_txt=timestep_txt,
        )
        vec_txt = self.time_in(timestep_txt)
        for block, cache in zip(self.double_blocks, crossattn_cache):
            txt = block.forward_txt(txt, vec_txt, cache)

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------

    def _get_freqs_cis(
        self,
        start_frame: int,
        num_frames: int,
        th: int,
        tw: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_frames = start_frame + num_frames
        key = (total_frames, th, tw)
        if key not in self._freqs_cache:
            cos, sin = get_nd_rotary_pos_embed(
                self.rope_dim_list, (total_frames, th, tw), self.rope_theta
            )
            self._freqs_cache[key] = (cos, sin)
        cos, sin = self._freqs_cache[key]
        per_frame = th * tw
        sl = slice(start_frame * per_frame, total_frames * per_frame)
        return cos[sl].to(device), sin[sl].to(device)

    # ------------------------------------------------------------------
    # Forward (vision pass over one chunk)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        *,
        encoder_attention_mask: list[torch.Tensor] | None = None,
        image_embeds: torch.Tensor | None = None,
        cond_latents: torch.Tensor | None = None,
        kv_cache: list[CausalSelfAttentionKVCache] | None = None,
        crossattn_cache: list[CrossAttentionKVCache] | None = None,
        current_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        assert kv_cache is not None and crossattn_cache is not None, (
            "causal HunyuanVideo 1.5 only supports KV-cached rollout"
        )
        b, c, t, h, w = hidden_states.shape
        pt, ph, pw = self.patch_size
        tt, th, tw = t // pt, h // ph, w // pw

        # Text prefill once per request (cache reset clears is_init).
        if not crossattn_cache[0].is_init:
            text_states, byt5_states = encoder_hidden_states
            text_mask, byt5_mask = encoder_attention_mask
            self._prefill_text_kv(
                crossattn_cache,
                text_states=text_states.to(hidden_states.dtype),
                text_mask=text_mask,
                byt5_states=byt5_states.to(hidden_states.dtype),
                byt5_mask=byt5_mask,
                image_embeds=(
                    None
                    if image_embeds is None
                    else image_embeds.to(hidden_states.dtype)
                ),
            )

        # i2v channel-concat conditioning, sliced to this chunk.
        assert cond_latents is not None
        cond_chunk = cond_latents[:, :, start_frame : start_frame + t].to(
            hidden_states.dtype
        )
        x = torch.cat([hidden_states, cond_chunk], dim=1)

        img = self.img_in(x)

        # Per-frame timestep -> per-token modulation vector [B, S, C].
        t_frames = timestep.reshape(b, -1).float()
        if t_frames.shape[1] == 1:
            t_frames = t_frames.expand(b, tt)
        vec = self.time_in(t_frames.reshape(-1)).view(b, tt, -1)
        vec = vec.repeat_interleave(th * tw, dim=1).type_as(img)

        cos, sin = self._get_freqs_cis(start_frame, tt, th, tw, img.device)

        recorder = get_attention_map_recorder()
        if recorder is not None:
            # The visible key layout is shared by every layer of this forward.
            recorder.begin_forward(
                frame_seqlen=th * tw,
                num_frames_per_block=tt,
                query_token_start=current_start,
                grid_height=th,
                grid_width=tw,
            )
        try:
            for block, layer_kv, layer_cross in zip(
                self.double_blocks, kv_cache, crossattn_cache
            ):
                img = block.forward_vision(
                    img,
                    vec,
                    (cos, sin),
                    layer_kv,
                    layer_cross,
                    current_start,
                )
        finally:
            if recorder is not None:
                recorder.end_forward()

        img = self.final_layer(img, vec)
        return self._unpatchify(img, tt, th, tw)

    def _unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        pt, ph, pw = self.patch_size
        x = x.reshape(x.shape[0], t, h, w, c, pt, ph, pw)
        x = torch.einsum("nthwcopq->nctohpwq", x)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)


EntryClass = CausalHunyuanVideo15Transformer3DModel
