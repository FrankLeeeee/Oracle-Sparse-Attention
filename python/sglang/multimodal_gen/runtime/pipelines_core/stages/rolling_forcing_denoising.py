# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/TencentARC/RollingForcing
"""Rolling Forcing denoising stage.

Port of ``pipeline/rolling_forcing_inference.py`` (``inference_rolling_forcing``)
onto the causal DMD stage infrastructure. Instead of fully denoising one block
before the next (Self-Forcing style), a rolling window of
``len(dmd_denoising_steps)`` blocks is denoised jointly, each block at a
different (staggered) noise level:

- window position 0 (oldest block) is one step from clean, the last position
  (newest block) is pure noise;
- one DiT forward denoises every block in the window by one step, then the
  window slides by one block;
- a block leaving the window is final; its x0 prediction is re-run at the
  context timestep with ``updating_cache=True`` to write clean features into
  the persistent KV cache (only the window's first block is ever cached);
- between window passes each unfinished block is stochastically re-noised to
  its next scheduled timestep (``scheduler.add_noise`` with fresh noise).
"""

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    RollingForcingSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    pred_noise_to_pred_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
    CausalDMDForwardContext,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def build_rolling_window_bounds(
    *, num_blocks: int, window_length_blocks: int
) -> list[tuple[int, int]]:
    """Inclusive (start_block, end_block) per window pass.

    The window ramps up at the start of the video (grows from 1 block) and
    drains at the end, giving ``num_blocks + window_length_blocks - 1`` passes.
    """
    bounds = []
    for window_index in range(num_blocks + window_length_blocks - 1):
        start_block = max(0, window_index - window_length_blocks + 1)
        end_block = min(num_blocks - 1, window_index)
        bounds.append((start_block, end_block))
    return bounds


def build_staggered_timesteps(
    timesteps: torch.Tensor,
    *,
    batch_size: int,
    num_frames_per_block: int,
) -> torch.Tensor:
    """Per-frame timesteps for a full window: oldest block cleanest.

    ``timesteps`` is the (warped) denoising step list, noisiest first. Window
    position 0 holds the oldest block (last denoising step), the final
    position the newest (pure noise), mirroring upstream ``shared_timestep``.
    """
    num_steps = timesteps.shape[0]
    shared = torch.ones(
        [batch_size, num_steps * num_frames_per_block],
        device=timesteps.device,
        dtype=torch.float32,
    )
    for index, timestep in enumerate(reversed(timesteps)):
        block = slice(index * num_frames_per_block, (index + 1) * num_frames_per_block)
        shared[:, block] *= timestep
    return shared


class RollingForcingDenoisingStage(CausalDMDDenoisingStage):
    """Rolling-window joint denoising with attention-sink KV cache."""

    def _allocate_causal_kv_cache(
        self,
        *,
        batch_size: int,
        kv_cache_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dtype: torch.dtype,
        device,
        use_int_indices: bool = False,
        sink_tokens: int = 0,
        global_sink_tokens: int = 0,
        attention_window_size: int | None = None,
        allow_growth: bool = False,
    ) -> list[RollingForcingSelfAttentionKVCache]:
        def zeros() -> torch.Tensor:
            return torch.zeros(
                [batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                dtype=dtype,
                device=device,
            )

        return [
            RollingForcingSelfAttentionKVCache(
                k=zeros(),
                v=zeros(),
                global_end_index=torch.zeros(1, dtype=torch.long, device=device),
                local_end_index=torch.zeros(1, dtype=torch.long, device=device),
                global_end_index_int=0,
                local_end_index_int=0,
                cache_size=kv_cache_size,
                sink_tokens=sink_tokens,
            )
            for _ in range(self.num_transformer_blocks)
        ]

    def _forward_rolling_transformer(
        self,
        batch: Req,
        ctx: CausalDMDForwardContext,
        *,
        latents_bcthw: torch.Tensor,
        timestep: torch.Tensor,
        current_start_frame: int,
        updating_cache: bool,
    ) -> torch.Tensor:
        with (
            torch.autocast(
                device_type=ctx.device.type,
                dtype=ctx.target_dtype,
                enabled=ctx.autocast_enabled,
            ),
            set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=batch,
            ),
        ):
            return self.transformer(
                latents_bcthw.to(ctx.target_dtype),
                ctx.prompt_embeds,
                timestep,
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.num_token_per_frame,
                start_frame=current_start_frame,
                updating_cache=updating_cache,
                **ctx.image_kwargs,
                **ctx.pos_cond_kwargs,
            )

    def _renoise_unfinished_blocks(
        self,
        batch: Req,
        ctx: CausalDMDForwardContext,
        *,
        x0_btchw: torch.Tensor,
        noisy_cache: torch.Tensor,
        window_index: int,
        start_block: int,
        end_block: int,
    ) -> None:
        """Re-noise each unfinished window block to its next scheduled timestep.

        Block ``b`` has completed ``window_index - b`` denoising steps, so its
        next timestep is ``timesteps[window_index - b + 1]`` (upstream recovers
        the same index by matching the block's current timestep value).
        """
        blk = self.num_frames_per_block
        num_steps = ctx.timesteps.shape[0]
        batch_size = x0_btchw.shape[0]
        for block_idx in range(start_block, end_block + 1):
            step_index = window_index - block_idx
            if step_index == num_steps - 1:
                continue  # final step: block exits the window clean
            next_timestep = ctx.timesteps[step_index + 1]
            in_window = slice(
                (block_idx - start_block) * blk, (block_idx - start_block + 1) * blk
            )
            x0_block = x0_btchw[:, in_window]
            noise = torch.randn(
                x0_block.shape,
                dtype=x0_block.dtype,
                generator=self._single_generator(batch),
                device=x0_block.device,
            )
            renoised = ctx.scheduler.add_noise(
                x0_block.flatten(0, 1),
                noise.flatten(0, 1),
                next_timestep.expand(batch_size * blk),
            ).unflatten(0, x0_block.shape[:2])
            in_video = slice(block_idx * blk, (block_idx + 1) * blk)
            noisy_cache[:, :, in_video] = renoised.permute(0, 2, 1, 3, 4)

    def _run_rolling_window(
        self,
        batch: Req,
        ctx: CausalDMDForwardContext,
        *,
        window_index: int,
        start_block: int,
        end_block: int,
        shared_timestep: torch.Tensor,
        noise: torch.Tensor,
        noisy_cache: torch.Tensor,
        output: torch.Tensor,
        total_frames: int,
    ) -> None:
        blk = self.num_frames_per_block
        window_frames = shared_timestep.shape[1]
        current_start_frame = start_block * blk
        current_end_frame = (end_block + 1) * blk
        current_num_frames = current_end_frame - current_start_frame

        # Assemble the window input: previously re-noised blocks, plus fresh
        # noise for a newly entering block (full and ramp-up windows only).
        if current_num_frames == window_frames or current_start_frame == 0:
            noisy_input = torch.cat(
                [
                    noisy_cache[:, :, current_start_frame : current_end_frame - blk],
                    noise[:, :, current_end_frame - blk : current_end_frame],
                ],
                dim=2,
            )
        else:  # draining at the end of the video
            noisy_input = noisy_cache[:, :, current_start_frame:current_end_frame]

        if current_num_frames == window_frames:
            current_timestep = shared_timestep
        elif current_start_frame == 0:  # ramp-up: newest positions only
            current_timestep = shared_timestep[:, -current_num_frames:]
        elif current_end_frame == total_frames:  # draining: oldest positions
            current_timestep = shared_timestep[:, :current_num_frames]
        else:
            raise ValueError(
                "rolling window must be full, ramping up, or draining; got "
                f"start_frame={current_start_frame}, end_frame={current_end_frame}"
            )

        pred_noise = self._forward_rolling_transformer(
            batch,
            ctx,
            latents_bcthw=noisy_input,
            timestep=current_timestep,
            current_start_frame=current_start_frame,
            updating_cache=False,
        )
        pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
        x0_btchw = pred_noise_to_pred_video(
            pred_noise=pred_noise_btchw.flatten(0, 1),
            noise_input_latent=noisy_input.permute(0, 2, 1, 3, 4).flatten(0, 1),
            timestep=current_timestep,
            scheduler=ctx.scheduler,
        ).unflatten(0, pred_noise_btchw.shape[:2])

        output[:, :, current_start_frame:current_end_frame] = x0_btchw.permute(
            0, 2, 1, 3, 4
        )

        self._renoise_unfinished_blocks(
            batch,
            ctx,
            x0_btchw=x0_btchw,
            noisy_cache=noisy_cache,
            window_index=window_index,
            start_block=start_block,
            end_block=end_block,
        )

        # Write clean features of the window's first block into the KV cache
        # (upstream context_noise is 0: the cache-update pass runs at t=0).
        clean_first_block = x0_btchw[:, :blk].permute(0, 2, 1, 3, 4)
        context_timestep = torch.zeros(
            [clean_first_block.shape[0], blk],
            device=clean_first_block.device,
            dtype=torch.float32,
        )
        self._forward_rolling_transformer(
            batch,
            ctx,
            latents_bcthw=clean_first_block,
            timestep=context_timestep,
            current_start_frame=current_start_frame,
            updating_cache=True,
        )

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if self._component_residency_manager is not None:
            self._manage_dit_use_site(self.transformer, "transformer", batch)
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        latents = ctx.latents  # [B, C, T, H, W] pure noise from latent prep
        total_frames = ctx.num_frames
        blk = self.num_frames_per_block
        if total_frames % blk != 0:
            raise ValueError(
                f"num latent frames ({total_frames}) must be divisible by "
                f"num_frames_per_block ({blk}) for rolling forcing"
            )
        num_blocks = total_frames // blk
        window_bounds = build_rolling_window_bounds(
            num_blocks=num_blocks,
            window_length_blocks=ctx.timesteps.shape[0],
        )

        if self.causal_kv_cache is None:
            self._initialize_causal_caches(
                batch_size=ctx.batch_size,
                max_text_len=self._get_max_text_len(server_args),
                dtype=ctx.target_dtype,
                device=latents.device,
            )
        else:
            assert self.crossattn_cache is not None
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
            )

        shared_timestep = build_staggered_timesteps(
            ctx.timesteps,
            batch_size=ctx.batch_size,
            num_frames_per_block=blk,
        )
        noise = latents
        output = torch.zeros_like(latents)
        noisy_cache = torch.zeros_like(latents)

        with self.progress_bar(total=len(window_bounds), batch=batch) as progress_bar:
            for window_index, (start_block, end_block) in enumerate(window_bounds):
                self._run_rolling_window(
                    batch,
                    ctx,
                    window_index=window_index,
                    start_block=start_block,
                    end_block=end_block,
                    shared_timestep=shared_timestep,
                    noise=noise,
                    noisy_cache=noisy_cache,
                    output=output,
                    total_frames=total_frames,
                )
                if progress_bar is not None:
                    progress_bar.update()

        self._flush_attention_maps(batch)
        batch.latents = output
        return batch
