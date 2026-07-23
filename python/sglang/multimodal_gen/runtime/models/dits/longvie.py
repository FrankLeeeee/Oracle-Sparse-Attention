# SPDX-License-Identifier: Apache-2.0
"""LongVie 2 — Wan2.1-I2V-14B with a dual-control (depth + track) side network.

Adapted from https://github.com/Vchitect/LongVie
(``diffsynth/models/wan_video_dit_dual_control.py`` for the module layout and
``diffsynth/pipelines/wan_video_new_longvie.py::model_fn_wan_video`` for the
fusion, which upstream lives in the pipeline rather than the model).

The side network is two half-width towers — one for the dense signal (depth),
one for the sparse signal (point tracks) — that run alongside the first
``control_layers`` main blocks. After each of those blocks their outputs are
summed, projected back to the model width, and added into the main stream::

    dense  = in_proj_dense(patchify(dense_latents))     # 5120 -> 2560
    sparse = in_proj_sparse(patchify(sparse_latents))
    for i, block in enumerate(blocks):
        x = block(x, context, t_mod, freqs)
        if i < control_layers:
            dense  = blocks_dense[i](dense,  control_context, control_t_mod, freqs)
            sparse = blocks_sparse[i](sparse, control_context, control_t_mod, freqs)
            x = x + combine_linears[i](dense + sparse)

Both control latents are 36-channel like the I2V input (16 VAE channels of the
control video, plus a 4-channel first-frame mask and the 16-channel encoded
control image), so they go through the *same* patch embedding as the video.
"""

from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.longvie import LongVie2VideoConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanTransformer3DModel,
    WanTransformerBlock,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LongVie2ControlBranch(nn.Module):
    """The dual-stream control tower and its fusion projections."""

    def __init__(
        self,
        *,
        dim: int,
        control_dim: int,
        control_num_heads: int,
        control_ffn_dim: int,
        control_layers: int,
        qk_norm: str,
        cross_attn_norm: bool,
        eps: float,
        supported_attention_backends: set,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.control_layers = control_layers

        def make_blocks(stream: str) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    WanTransformerBlock(
                        control_dim,
                        control_ffn_dim,
                        control_num_heads,
                        qk_norm,
                        cross_attn_norm,
                        eps,
                        # the control cross-attention carries the I2V image
                        # branch too, at the side network's own width
                        control_dim,
                        supported_attention_backends,
                        prefix=f"control.blocks_{stream}.{i}",
                        quant_config=quant_config,
                    )
                    for i in range(control_layers)
                ]
            )

        self.blocks_dense = make_blocks("dense")
        self.blocks_sparse = make_blocks("sparse")

        # project the patch-embedded control latents down into the side network
        self.in_proj_dense = nn.Linear(dim, control_dim)
        self.in_proj_sparse = nn.Linear(dim, control_dim)
        # text context and timestep modulation, at side-network width
        self.text_proj = nn.Linear(dim, control_dim)
        self.time_proj = nn.Linear(dim, control_dim)
        # zero-initialised at training time so the branch starts as a no-op;
        # the released checkpoint's values are trained
        self.combine_linears = nn.ModuleList(
            [nn.Linear(control_dim, dim) for _ in range(control_layers)]
        )

    def project_inputs(
        self,
        *,
        dense_tokens: torch.Tensor,
        sparse_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.in_proj_dense(dense_tokens),
            self.in_proj_sparse(sparse_tokens),
            self.text_proj(encoder_hidden_states),
            self.time_proj(timestep_proj),
        )


class LongVie2Transformer3DModel(WanTransformer3DModel):
    """Wan I2V transformer whose first blocks are steered by the control tower."""

    # the base class binds these to `WanVideoConfig()` at class-definition time,
    # so they must be re-bound here or the control branch's rewrite rules never
    # reach the weight loader
    param_names_mapping = LongVie2VideoConfig().param_names_mapping
    reverse_param_names_mapping = LongVie2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = LongVie2VideoConfig().lora_param_names_mapping

    def __init__(
        self,
        config: LongVie2VideoConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        inner_dim = config.num_attention_heads * config.attention_head_dim
        arch = config.arch_config
        self.control_layers = int(hf_config.get("control_layers", arch.control_layers))
        self.control = LongVie2ControlBranch(
            dim=inner_dim,
            control_dim=int(hf_config.get("control_dim", arch.control_dim)),
            control_num_heads=int(
                hf_config.get("control_num_heads", arch.control_num_heads)
            ),
            control_ffn_dim=int(hf_config.get("control_ffn_dim", arch.control_ffn_dim)),
            control_layers=self.control_layers,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            supported_attention_backends=self._supported_attention_backends,
            quant_config=quant_config,
        )
        self.layer_names = ["blocks", "control.blocks_dense", "control.blocks_sparse"]

    def _patchify_control(
        self, latents: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """``[B, 36, F, H, W]`` control latents -> ``[B, tokens, dim]``."""
        tokens = self.patch_embedding(latents.to(dtype))
        return tokens.flatten(2).transpose(1, 2).contiguous()

    def _control_latents(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-forward control signals, attached to the batch by the stage."""
        forward_batch = get_forward_context().forward_batch
        if forward_batch is None:
            return None
        dense = forward_batch.longvie_dense_latents
        sparse = forward_batch.longvie_sparse_latents
        if dense is None or sparse is None:
            return None
        if forward_batch.enable_sequence_shard and self.sp_size > 1:
            # the control streams are not sharded, so the tail-slice fusion
            # below would steer the wrong tokens on every rank but the last
            raise NotImplementedError(
                "LongVie 2 control fusion does not support sequence sharding yet"
            )
        return dense, sparse

    def _history_latents(self) -> torch.Tensor | None:
        forward_batch = get_forward_context().forward_batch
        if forward_batch is None:
            return None
        return forward_batch.longvie_history_latents

    def _prepend_history_tokens(
        self,
        hidden_states: torch.Tensor,
        history_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, int, tuple[torch.Tensor, torch.Tensor]]:
        """Prepend patchified history tokens and extend RoPE over them.

        Upstream (``model_fn_wan_video``) concatenates ``[history, current]``
        and computes ``freqs_history`` over the combined frame count — history
        sits at temporal positions ``[0, f_h)`` and the current clip is shifted
        to start at ``f_h``. The control streams keep the unshifted ``freqs``.
        """
        history_tokens = self._patchify_control(
            history_latents, dtype=hidden_states.dtype
        )
        p_t, p_h, p_w = self.patch_size
        f_h = history_latents.shape[2] // p_t
        grid_h = history_latents.shape[3] // p_h
        grid_w = history_latents.shape[4] // p_w
        current_frames = hidden_states.shape[1] // (grid_h * grid_w)
        freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
            (f_h + current_frames, grid_h, grid_w),
            shard_dim=0,
            start_frame=0,
            device=hidden_states.device,
        )
        hidden_states = torch.cat([history_tokens, hidden_states], dim=1)
        return (
            hidden_states,
            history_tokens.shape[1],
            (freqs_cos.float(), freqs_sin.float()),
        )

    def _run_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        control = self._control_latents()
        if control is None:
            # no control signal supplied -> plain Wan I2V behaviour
            return super()._run_transformer_blocks(
                hidden_states, encoder_hidden_states, timestep_proj, freqs_cis
            )

        dense_latents, sparse_latents = control
        dtype = hidden_states.dtype
        dense, sparse, control_context, control_t_mod = self.control.project_inputs(
            dense_tokens=self._patchify_control(dense_latents, dtype=dtype),
            sparse_tokens=self._patchify_control(sparse_latents, dtype=dtype),
            encoder_hidden_states=encoder_hidden_states,
            timestep_proj=timestep_proj,
        )
        if dense.shape[1] != hidden_states.shape[1]:
            raise ValueError(
                "LongVie control tokens must match the generated tokens: "
                f"control={dense.shape[1]} main={hidden_states.shape[1]}"
            )

        # Clip-by-clip AR: history tokens sit in front of the main stream and
        # shift its RoPE; the control streams cover only the generated tail.
        history_latents = self._history_latents()
        history_len = 0
        main_freqs_cis = freqs_cis
        if history_latents is not None:
            hidden_states, history_len, main_freqs_cis = self._prepend_history_tokens(
                hidden_states, history_latents
            )

        for index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, main_freqs_cis
            )
            if index >= self.control_layers:
                continue
            dense = self.control.blocks_dense[index](
                dense, control_context, control_t_mod, freqs_cis
            )
            sparse = self.control.blocks_sparse[index](
                sparse, control_context, control_t_mod, freqs_cis
            )
            fused = self.control.combine_linears[index](dense + sparse)
            # history tokens sit at the front and must not be steered
            hidden_states[:, -fused.shape[1] :] = (
                hidden_states[:, -fused.shape[1] :] + fused
            )

        if history_len:
            hidden_states = hidden_states[:, history_len:]
        return hidden_states


EntryClass = LongVie2Transformer3DModel
