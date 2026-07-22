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
        self, video_path: str, *, vae, batch: Req, server_args: ServerArgs
    ) -> torch.Tensor:
        """Build one stream's 36-channel control latent."""
        device = get_local_torch_device()
        video = load_control_video(
            video_path,
            num_frames=batch.num_frames,
            height=batch.height,
            width=batch.width,
        ).to(device=device, dtype=torch.float32)

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
        if not dense_path or not sparse_path:
            if dense_path or sparse_path:
                raise ValueError(
                    "LongVie 2 needs both control streams; got dense="
                    f"{dense_path!r} sparse={sparse_path!r}"
                )
            logger.info("No LongVie control videos given; running as plain Wan I2V")
            return batch

        for path in (dense_path, sparse_path):
            if not path.startswith(("http://", "https://")) and not pathlib.Path(
                path
            ).is_file():
                raise FileNotFoundError(f"control video not found: {path}")

        with self.use_declared_component(
            component_name=self.component_name, module=self.vae
        ) as vae:
            assert vae is not None
            self.vae = vae
            common = dict(vae=vae, batch=batch, server_args=server_args)
            batch.longvie_dense_latents = self._encode_stream(dense_path, **common)
            batch.longvie_sparse_latents = self._encode_stream(sparse_path, **common)

        logger.info(
            "Encoded LongVie control latents: dense=%s sparse=%s",
            tuple(batch.longvie_dense_latents.shape),
            tuple(batch.longvie_sparse_latents.shape),
        )
        return batch
