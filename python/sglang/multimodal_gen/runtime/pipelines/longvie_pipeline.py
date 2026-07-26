# SPDX-License-Identifier: Apache-2.0
"""LongVie 2 pipeline — controllable ultra-long I2V on a Wan2.1-I2V-14B base.

Adapted from https://github.com/Vchitect/LongVie.

The generation path is Wan I2V; what LongVie adds is a dual-control side
network inside the transformer (see ``runtime/models/dits/longvie.py``) driven
by depth and point-track control videos. When no control latents are attached
to the forward batch the transformer falls back to plain Wan I2V, so this
pipeline is a strict superset of :class:`WanImageToVideoPipeline`.

Requests longer than ``longvie_clip_num_frames`` run upstream's clip-by-clip
autoregressive loop: each clip is conditioned on the previous clip's last
frame (I2V) and its last 8 frames as history tokens inside the transformer,
with the initial noise shared across clips. The denoise/decode stages are
owned by :class:`LongVie2ClipLoopStage`, which for single-clip requests simply
runs them once.
"""

from sglang.multimodal_gen.runtime.pipelines.wan_i2v_pipeline import (
    WanImageToVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
    ImageEncodingStage,
    ImageVAEEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longvie import (
    LongVie2ClipLoopStage,
    LongVie2ControlEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LongVie2Pipeline(WanImageToVideoPipeline):
    pipeline_name = "LongVie2Pipeline"

    def create_pipeline_stages(self, server_args: ServerArgs):
        # input validation resolves height/width from the condition image, and
        # the control latents must be encoded at that resolution — so the
        # control stage goes between validation and the rest of the I2V stages.
        # For multi-clip requests it clamps num_frames to the clip size so
        # every downstream stage works at clip shape.
        self.add_stage(InputValidationStage())
        control_stage = LongVie2ControlEncodingStage(vae=self.get_module("vae"))
        self.add_stage(control_stage)
        self.add_standard_text_encoding_stage()
        image_encoding_stage = ImageEncodingStage(
            image_encoder=self.get_module("image_encoder"),
            image_processor=self.get_module("image_processor"),
        )
        self.add_stage(image_encoding_stage)
        image_vae_stage = ImageVAEEncodingStage(vae=self.get_module("vae"))
        self.add_stage(image_vae_stage)
        self.add_standard_latent_preparation_stage()
        timestep_stage = TimestepPreparationStage(
            scheduler=self.get_module("scheduler"),
            prepare_extra_set_timesteps_kwargs=[],
        )
        self.add_stage(timestep_stage)
        # denoise + decode are owned by the clip loop (it re-runs them per
        # clip), so they are not registered as pipeline stages themselves
        self.add_stage(
            LongVie2ClipLoopStage(
                denoising_stage=DenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                    pipeline=self,
                ),
                decoding_stage=DecodingStage(
                    vae=self.get_module("vae"),
                    pipeline=self,
                    component_name="vae",
                ),
                timestep_stage=timestep_stage,
                image_encoding_stage=image_encoding_stage,
                image_vae_stage=image_vae_stage,
                control_stage=control_stage,
            )
        )


EntryClass = LongVie2Pipeline
