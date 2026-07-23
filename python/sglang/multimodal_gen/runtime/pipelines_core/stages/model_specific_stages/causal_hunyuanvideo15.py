# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shengshu-ai/minWM (HY15/hy15_inference.py)
"""Causal denoising stage for the minWM HunyuanVideo 1.5 TI2V model.

Differences from the Wan-family ``CausalDMDDenoisingStage``:

* Chunk denoising uses deterministic Euler flow-matching steps
  (``x += v * dsigma``) instead of the DMD x0-predict + re-noise recipe.
* Timesteps come from ``FlowMatchDiscreteScheduler.set_timesteps`` (shift-5
  SD3 warp), not from a configured ``dmd_denoising_steps`` list.
* TI2V conditioning is a channel concat of the first-frame VAE latent and a
  frame mask (65 input channels), passed to the DiT as ``cond_latents``; there
  is no cache warm-up from image latents.
* The text KV "cross-attention" cache is filled by the DiT's internal text
  prefill on the first chunk forward.
"""

import PIL.Image
import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CausalHunyuanVideo15ImagePreprocessStage(PipelineStage):
    """Squash-resize the condition image to the requested output size.

    minWM resizes the input image to exactly (height, width) with bilinear
    interpolation (``transforms.Resize``) — no aspect-preserving crop, and the
    output size is never derived from the image.  The shared TI2V branch of
    ``InputValidationStage`` is skipped via ``skip_input_image_preprocess``.
    """

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return batch
        images = batch.condition_image
        if not isinstance(images, list):
            images = [images]
        batch.condition_image = [
            img.resize((batch.width, batch.height), PIL.Image.Resampling.BILINEAR)
            for img in images
        ]
        return batch

# SigLIP semantic tokens + ByT5 max length + Qwen2.5-VL max length; the
# cross-attention cache stores the combined valid condition tokens, and
# ``CrossAttentionKVCache.store`` replaces the buffers, so this only sizes the
# initial allocation.
_NOMINAL_MAX_TEXT_LEN = 729 + 256 + 1000


class CausalHunyuanVideo15DenoisingStage(CausalDMDDenoisingStage):
    """KV-cached chunkwise Euler denoising for causal HunyuanVideo 1.5."""

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        scheduler.set_timesteps(batch.num_inference_steps, device=device)
        logger.info("Using timesteps: %s", scheduler.timesteps)
        return scheduler.timesteps

    def _get_max_text_len(self, server_args: ServerArgs) -> int:
        return _NOMINAL_MAX_TEXT_LEN

    def _prepare_causal_dmd_image_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        """Build the TI2V channel-concat conditioning and SigLIP embeds."""
        assert batch.latents is not None
        b, c, t, h, w = batch.latents.shape
        device = batch.latents.device

        cond_latents = torch.zeros(
            (b, c, t, h, w), device=device, dtype=target_dtype
        )
        mask = torch.zeros((b, 1, t, h, w), device=device, dtype=target_dtype)
        if batch.image_latent is not None:
            cond_latents[:, :, :1] = batch.image_latent.to(
                device=device, dtype=target_dtype
            )
            mask[:, :, 0] = 1.0

        image_embeds = None
        if batch.image_embeds:
            image_embeds = batch.image_embeds[0].to(
                device=device, dtype=target_dtype
            )

        return {
            "cond_latents": torch.cat([cond_latents, mask], dim=1),
            "image_embeds": image_embeds,
        }

    def _denoise_causal_dmd_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input,
        progress_bar=None,
    ) -> tuple[torch.Tensor, object | None]:
        """Euler flow-matching over one chunk (minWM ``euler`` solver)."""
        current_latents = chunk_latents
        for i, timestep in enumerate(timesteps):
            latent_model_input = prepare_model_input(current_latents).to(target_dtype)
            timestep_2d = self._expand_timestep(
                timestep, latent_model_input.shape[0], device
            ).unsqueeze(1)
            v_pred = self._forward_causal_transformer(
                batch,
                latent_model_input=latent_model_input,
                prompt_embeds=prompt_embeds,
                timestep=timestep_2d,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=pos_cond_kwargs,
                current_timestep=i,
                attn_metadata=None,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            current_latents = scheduler.step_at(v_pred, i, current_latents).to(
                chunk_latents.dtype
            )
            if progress_bar is not None:
                progress_bar.update()
        return current_latents, None

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self._component_residency_manager is not None:
            self._manage_dit_use_site(self.transformer, "transformer", batch)
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        latents = ctx.latents
        if latents.is_inference() and not torch.is_inference_mode_enabled():
            latents = latents.clone()
        t = ctx.num_frames

        if t % self.num_frames_per_block != 0:
            raise ValueError(
                f"latent frame count {t} must be divisible by the chunk size "
                f"{self.num_frames_per_block} for causal HunyuanVideo 1.5"
            )

        # minWM keeps every generated frame in the vision KV cache — no
        # sliding window — so make sure the cache covers the whole request.
        self.sliding_window_num_frames = max(self.sliding_window_num_frames, t)

        if self.causal_kv_cache is None:
            self._initialize_causal_caches(
                batch_size=ctx.batch_size,
                max_text_len=self._get_max_text_len(server_args),
                dtype=ctx.target_dtype,
                device=latents.device,
            )
        else:
            assert self.crossattn_cache is not None
            if self.causal_kv_cache[0].cache_size < t * self.num_token_per_frame:
                self._initialize_causal_caches(
                    batch_size=ctx.batch_size,
                    max_text_len=self._get_max_text_len(server_args),
                    dtype=ctx.target_dtype,
                    device=latents.device,
                )
            else:
                self._reset_causal_caches(
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                )

        num_blocks = t // self.num_frames_per_block
        block_sizes = [self.num_frames_per_block] * num_blocks
        start_index = 0

        def prepare_model_input(current_latents):
            return current_latents

        def prepare_context_input(current_latents):
            return current_latents

        with self.progress_bar(
            total=len(block_sizes) * len(ctx.timesteps), batch=batch
        ) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[
                    :, :, start_index : start_index + current_num_frames
                ]
                current_latents = self._denoise_and_update_causal_block(
                    batch,
                    server_args,
                    chunk_latents=current_latents,
                    scheduler=ctx.scheduler,
                    timesteps=ctx.timesteps,
                    prompt_embeds=ctx.prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_tokens=start_index * self.num_token_per_frame,
                    start_frame=start_index,
                    image_kwargs=ctx.image_kwargs,
                    pos_cond_kwargs=ctx.pos_cond_kwargs,
                    target_dtype=ctx.target_dtype,
                    autocast_enabled=ctx.autocast_enabled,
                    device=ctx.device,
                    attn_raw_latent_shape=(current_num_frames, ctx.height, ctx.width),
                    prepare_model_input=prepare_model_input,
                    prepare_context_input=prepare_context_input,
                    progress_bar=progress_bar,
                )
                latents[:, :, start_index : start_index + current_num_frames] = (
                    current_latents
                )
                start_index += current_num_frames

        self._flush_attention_maps(batch)
        batch.latents = latents
        return batch
