# SPDX-License-Identifier: Apache-2.0
"""LongVie 2 pipeline — controllable ultra-long I2V on a Wan2.1-I2V-14B base.

Adapted from https://github.com/Vchitect/LongVie.

The generation path is Wan I2V; what LongVie adds is a dual-control side
network inside the transformer (see ``runtime/models/dits/longvie.py``) driven
by depth and point-track control videos. When no control latents are attached
to the forward batch the transformer falls back to plain Wan I2V, so this
pipeline is a strict superset of :class:`WanImageToVideoPipeline`.
"""

from sglang.multimodal_gen.runtime.pipelines.wan_i2v_pipeline import (
    WanImageToVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longvie import (
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
        # It runs once per request; the transformer reads its output back on
        # every denoising step.
        self.add_stage(InputValidationStage())
        self.add_stage(LongVie2ControlEncodingStage(vae=self.get_module("vae")))
        self.add_standard_ti2v_stages(include_input_validation=False)


EntryClass = LongVie2Pipeline
