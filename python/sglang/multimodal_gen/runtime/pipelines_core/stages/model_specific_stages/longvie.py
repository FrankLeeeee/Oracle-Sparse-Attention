# SPDX-License-Identifier: Apache-2.0
"""Encode LongVie 2's dual-control signals into the latents the DiT consumes.

Adapted from ``WanVideoUnit_LongVieControlEmbedder`` in
https://github.com/Vchitect/LongVie (``diffsynth/pipelines/wan_video_new_longvie.py``).

Each control stream (dense = depth, sparse = point tracks) becomes a
36-channel latent laid out exactly like the Wan I2V input, so it can go through
the *same* patch embedding as the video::

    [ 16 ch: VAE(control video)          ]
    [  4 ch: first-frame mask            ]  <- the 20-channel image-condition
    [ 16 ch: VAE(control first frame)    ]     block, built the same way I2V does

The result is attached to the request and read back inside
:class:`LongVie2Transformer3DModel`, which fuses it into the first
``control_layers`` blocks.
"""

import pathlib

import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)
from sglang.multimodal_gen.runtime.utils.vision import load_video

logger = init_logger(__name__)

LONGVIE_TOTAL_NUM_FRAMES_KEY = "longvie_total_num_frames"
LONGVIE_CONTROL_PIXELS_KEY = "longvie_control_pixels"


def load_control_video(
    path: str, *, num_frames: int, height: int, width: int
) -> torch.Tensor:
    """Read a control video as ``[1, 3, num_frames, H, W]`` in ``[-1, 1]``.

    Frames are sampled from the start and the last frame is repeated if the
    clip is short, so a control signal never silently shifts the timeline.
    """
    frames = load_video(path)
    if len(frames) < num_frames:
        logger.warning(
            "control video %s has %d frames, need %d; repeating the last frame",
            path,
            len(frames),
            num_frames,
        )
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    frames = frames[:num_frames]
    resized = np.stack(
        [
            np.asarray(
                frame.convert("RGB").resize(
                    (width, height), PIL.Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            )
            for frame in frames
        ]
    )
    # [F, H, W, 3] -> [1, 3, F, H, W] in [-1, 1]
    video = torch.from_numpy(resized).permute(3, 0, 1, 2).unsqueeze(0)
    return video / 127.5 - 1.0


class LongVie2ControlEncodingStage(PipelineStage):
    """Turn the depth / track control videos into 36-channel latents.

    Mirrors :class:`ImageVAEEncodingStage`: the VAE runs at the configured VAE
    precision and its latents go through the same sample-mode / scale-shift
    normalization, so the control tokens live in the same space as the video
    tokens they steer.
    """

    def __init__(self, vae: ParallelTiledVAE, component_name: str = "vae") -> None:
        super().__init__()
        self.vae = vae
        self.component_name = component_name

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                self.component_name,
                target_dtype=vae_dtype,
            )
        ]

    def _encode(
        self, video: torch.Tensor, *, vae, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        """VAE-encode ``[1, 3, F, H, W]`` pixels into normalized 16-ch latents."""
        pipeline_config = server_args.pipeline_config
        vae_dtype = resolve_precision(
            server_args, self.component_name, precision_attr="vae_precision"
        )
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if not vae_autocast_enabled:
                video = video.to(vae_dtype)
            video = pipeline_config.preprocess_vae_encode(video, vae)
            with temporary_module_dtype(
                vae, vae_dtype, enabled=not vae_autocast_enabled
            ) as scoped_vae:
                latent_dist: DiagonalGaussianDistribution = scoped_vae.encode(video)
        if isinstance(latent_dist, AutoencoderKLOutput):
            latent_dist = latent_dist.latent_dist

        sample_mode = pipeline_config.vae_config.encode_sample_mode()
        latents = (
            latent_dist.mode()
            if sample_mode == "argmax"
            else latent_dist.sample(batch.generator)
        )
        latents = pipeline_config.postprocess_vae_encode(latents, vae)
        normalized = pipeline_config.normalize_vae_encode(latents, vae)
        if normalized is not None:
            return normalized

        scaling_factor, shift_factor = pipeline_config.get_decode_scale_and_shift(
            device=latents.device, dtype=latents.dtype, vae=vae
        )
        return ImageVAEEncodingStage.scale_and_shift_encode_latents(
            latents, scaling_factor, shift_factor
        )

    def _encode_stream(
        self, video: torch.Tensor, *, vae, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        """Build one stream's 36-channel control latent from ``[1,3,F,H,W]`` pixels."""
        video_latents = self._encode(
            video, vae=vae, batch=batch, server_args=server_args
        )

        # the control stream's own I2V-style condition: its first frame, padded
        # with black, encoded and prefixed with the first-frame mask
        first_frame_condition = torch.cat(
            [video[:, :, :1], torch.zeros_like(video[:, :, 1:])], dim=2
        )
        first_frame_latents = self._encode(
            first_frame_condition, vae=vae, batch=batch, server_args=server_args
        )
        # [1, 20, F, h, w] — the same mask + image-latent block the video gets
        conditioned = server_args.pipeline_config.postprocess_image_latent(
            first_frame_latents, batch
        )
        return torch.cat([video_latents, conditioned.to(video_latents.dtype)], dim=1)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        dense_path = batch.longvie_dense_video
        sparse_path = batch.longvie_sparse_video
        clip_num_frames = server_args.pipeline_config.longvie_clip_num_frames
        multi_clip = batch.num_frames > clip_num_frames
        if not dense_path or not sparse_path:
            if dense_path or sparse_path:
                raise ValueError(
                    "LongVie 2 needs both control streams; got dense="
                    f"{dense_path!r} sparse={sparse_path!r}"
                )
            if multi_clip:
                raise ValueError(
                    "LongVie 2 clip-by-clip generation (num_frames "
                    f"{batch.num_frames} > {clip_num_frames}) requires both "
                    "control videos"
                )
            logger.info("No LongVie control videos given; running as plain Wan I2V")
            return batch

        for path in (dense_path, sparse_path):
            if not path.startswith(("http://", "https://")) and not pathlib.Path(
                path
            ).is_file():
                raise FileNotFoundError(f"control video not found: {path}")

        if multi_clip:
            # Clip loop: every downstream stage works at clip shape; the loop
            # stage slices and encodes the control pixels per clip.
            total = batch.num_frames
            batch.extra[LONGVIE_TOTAL_NUM_FRAMES_KEY] = total
            batch.num_frames = clip_num_frames
            batch.extra[LONGVIE_CONTROL_PIXELS_KEY] = tuple(
                load_control_video(
                    path,
                    num_frames=total,
                    height=batch.height,
                    width=batch.width,
                )
                for path in (dense_path, sparse_path)
            )
            logger.info(
                "LongVie clip loop: %d frames as %d-frame clips",
                total,
                clip_num_frames,
            )
            return batch

        device = get_local_torch_device()
        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            assert vae is not None
            self.vae = vae
            common = dict(vae=vae, batch=batch, server_args=server_args)
            batch.longvie_dense_latents = self._encode_stream(
                load_control_video(
                    dense_path,
                    num_frames=batch.num_frames,
                    height=batch.height,
                    width=batch.width,
                ).to(device=device, dtype=torch.float32),
                **common,
            )
            batch.longvie_sparse_latents = self._encode_stream(
                load_control_video(
                    sparse_path,
                    num_frames=batch.num_frames,
                    height=batch.height,
                    width=batch.width,
                ).to(device=device, dtype=torch.float32),
                **common,
            )

        logger.info(
            "Encoded LongVie control latents: dense=%s sparse=%s",
            tuple(batch.longvie_dense_latents.shape),
            tuple(batch.longvie_sparse_latents.shape),
        )
        return batch


class LongVie2ClipLoopStage(PipelineStage):
    """Clip-by-clip autoregressive denoise + decode (upstream ``inference.py``).

    For requests longer than one clip, generates ``longvie_clip_num_frames``
    frames at a time. Between clips (all following upstream):

    * the last decoded frame becomes the next clip's I2V condition image
      (CLIP + VAE re-encoded through the standard stages),
    * the last ``longvie_history_frames`` pixel frames are VAE-encoded into
      36-channel history latents that the DiT prepends to its token stream,
    * the *same* initial noise is reused, except latent frame 0 is blended
      toward the history's last latent frame,
    * the control videos are sliced per clip and re-encoded.

    Single-clip requests fall through to the wrapped standard denoise + decode
    unchanged.
    """

    def __init__(
        self,
        *,
        denoising_stage,
        decoding_stage,
        timestep_stage,
        image_encoding_stage,
        image_vae_stage,
        control_stage: LongVie2ControlEncodingStage,
    ) -> None:
        super().__init__()
        self.denoising_stage = denoising_stage
        self.decoding_stage = decoding_stage
        self.timestep_stage = timestep_stage
        self.image_encoding_stage = image_encoding_stage
        self.image_vae_stage = image_vae_stage
        self.control_stage = control_stage

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None):
        # Union of the wrapped stages' uses so the residency manager onloads
        # everything the clip loop touches; the inner stages themselves are
        # not registered with the manager.
        stage_name = self._component_stage_name(stage_name)
        uses = []
        for inner in (
            self.denoising_stage,
            self.decoding_stage,
            self.image_encoding_stage,
            self.image_vae_stage,
            self.control_stage,
        ):
            uses.extend(inner.component_uses(server_args, stage_name))
        return uses

    @staticmethod
    def _frames_to_pil(frames: torch.Tensor, index: int) -> PIL.Image.Image:
        """One frame of ``[1, 3, F, H, W]`` in [0, 1] -> PIL image."""
        frame = frames[0, :, index].permute(1, 2, 0).float().cpu().numpy()
        return PIL.Image.fromarray((frame * 255.0).round().clip(0, 255).astype("uint8"))

    def _encode_history(
        self, history_px: torch.Tensor, *, batch: Req, server_args: ServerArgs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[1, 3, n, H, W]`` pixels in [-1, 1] -> 36-ch history latents.

        Upstream's chunked VAE encoder consumes only the first
        ``1 + 4 * ((n - 1) // 4)`` frames of the history window (5 for n=8) and
        silently drops the rest; encoding exactly that prefix through our VAE
        is computation-identical.
        """
        n = history_px.shape[2]
        used = 1 + 4 * ((n - 1) // 4)
        with self.control_stage.use_declared_component(
            component_name=self.control_stage.component_name,
            module=self.control_stage.vae,
        ) as vae:
            assert vae is not None
            latents = self.control_stage._encode(
                history_px[:, :, :used],
                vae=vae,
                batch=batch,
                server_args=server_args,
            )
        ones = torch.ones(
            (latents.shape[0], 20, *latents.shape[2:]),
            device=latents.device,
            dtype=latents.dtype,
        )
        # upstream: torch.cat([ones, history_latents], dim=1) — ones FIRST,
        # the reverse of the control-latent layout
        return torch.cat([ones, latents], dim=1), latents

    def _encode_clip_controls(
        self,
        dense_px: torch.Tensor,
        sparse_px: torch.Tensor,
        *,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        device = get_local_torch_device()
        with self.control_stage.use_declared_component(
            component_name=self.control_stage.component_name,
            module=self.control_stage.vae,
        ) as vae:
            assert vae is not None
            common = dict(vae=vae, batch=batch, server_args=server_args)
            batch.longvie_dense_latents = self.control_stage._encode_stream(
                dense_px.to(device=device, dtype=torch.float32), **common
            )
            batch.longvie_sparse_latents = self.control_stage._encode_stream(
                sparse_px.to(device=device, dtype=torch.float32), **common
            )

    @staticmethod
    def _slice_clip(pixels: torch.Tensor, start: int, clip_frames: int) -> torch.Tensor:
        """``[1, 3, total, H, W]`` -> a full-length clip slice, last-frame padded."""
        clip = pixels[:, :, start : start + clip_frames]
        if clip.shape[2] < clip_frames:
            pad = clip[:, :, -1:].expand(-1, -1, clip_frames - clip.shape[2], -1, -1)
            clip = torch.cat([clip, pad], dim=2)
        return clip

    def _propagate_residency_manager(self) -> None:
        """Hand the wrapper's residency manager to the wrapped stages.

        The inner stages are not registered with the executor, so they never
        get the manager through ``before_stage`` — without it components that
        the memory policy offloads (e.g. the DiT under ``dit_cpu_offload``)
        would never be onloaded at their use sites inside the loop.
        """
        for inner in (
            self.denoising_stage,
            self.decoding_stage,
            self.timestep_stage,
            self.image_encoding_stage,
            self.image_vae_stage,
            self.control_stage,
        ):
            inner.set_component_residency_manager(self._component_residency_manager)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs):
        self._propagate_residency_manager()
        total = batch.extra.get(LONGVIE_TOTAL_NUM_FRAMES_KEY)
        if total is None:
            # single clip: exactly the pre-existing denoise + decode path
            batch = self.denoising_stage.forward(batch, server_args)
            return self.decoding_stage.forward(batch, server_args)

        pipeline_config = server_args.pipeline_config
        clip_frames = batch.num_frames  # clamped by the control stage
        history_frames = pipeline_config.longvie_history_frames
        blend_sigma = pipeline_config.longvie_first_frame_blend_sigma
        dense_px, sparse_px = batch.extra.pop(LONGVIE_CONTROL_PIXELS_KEY)
        num_clips = -(-total // clip_frames)

        base_noise = batch.latents.clone()
        sigmas_snapshot = batch.sigmas
        all_frames: list[torch.Tensor] = []
        output_batch = None

        for clip_index in range(num_clips):
            start = clip_index * clip_frames
            logger.info(
                "LongVie clip %d/%d (frames %d-%d)",
                clip_index + 1,
                num_clips,
                start,
                min(start + clip_frames, total) - 1,
            )
            self._encode_clip_controls(
                self._slice_clip(dense_px, start, clip_frames),
                self._slice_clip(sparse_px, start, clip_frames),
                batch=batch,
                server_args=server_args,
            )

            if clip_index > 0:
                frames = all_frames[-1]
                # I2V condition: the previous clip's last frame
                batch.condition_image = self._frames_to_pil(frames, -1)
                batch.image_embeds = []
                self.image_encoding_stage.forward(batch, server_args)
                batch.image_latent = None
                self.image_vae_stage.forward(batch, server_args)

                history_px = (
                    frames[:, :, -history_frames:].to(get_local_torch_device()) * 2.0
                    - 1.0
                )
                history_latents, raw_history = self._encode_history(
                    history_px, batch=batch, server_args=server_args
                )
                batch.longvie_history_latents = history_latents

                # unified noise, frame 0 re-noised toward the history tail
                latents = base_noise.clone()
                latents[:, :, :1] = (1.0 - blend_sigma) * raw_history[:, :, -1:].to(
                    latents.dtype
                ) + blend_sigma * base_noise[:, :, :1]
                batch.latents = latents

                # rewind the scheduler for the new clip
                batch.timesteps = None
                batch.sigmas = sigmas_snapshot
                self.timestep_stage.forward(batch, server_args)

            batch = self.denoising_stage.forward(batch, server_args)
            output_batch = self.decoding_stage.forward(batch, server_args)
            all_frames.append(output_batch.output)

        batch.longvie_history_latents = None
        assert output_batch is not None
        output_batch.output = torch.cat(all_frames, dim=2)[:, :, :total]
        return output_batch
