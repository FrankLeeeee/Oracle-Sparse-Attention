# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shengshu-ai/minWM
"""Causal (minWM) HunyuanVideo 1.5 TI2V pipeline.

Chunk-by-chunk KV-cached autoregressive generation on the HunyuanVideo-1.5 8B
backbone: text/vision conditions are prefilled once into a per-layer text KV
cache, then each chunk of 4 latent frames is denoised with 4 CFG-free Euler
flow-matching steps and re-encoded cleanly into the vision KV cache.
"""

from sglang.multimodal_gen.configs.pipeline_configs.hunyuanvideo15 import (
    CausalHunyuanVideo15TI2VConfig,
)
from sglang.multimodal_gen.configs.sample.hunyuanvideo15 import (
    CausalHunyuanVideo15TI2VSamplingParams,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_discrete import (
    FlowMatchDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.causal_hunyuanvideo15 import (
    CausalHunyuanVideo15DenoisingStage,
    CausalHunyuanVideo15ImagePreprocessStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CausalHunyuanVideo15Pipeline(ComposedPipelineBase):
    pipeline_name = "CausalHunyuanVideo15Pipeline"
    pipeline_config_cls = CausalHunyuanVideo15TI2VConfig
    sampling_params_cls = CausalHunyuanVideo15TI2VSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "text_encoder_2",
        "tokenizer_2",
        "image_encoder",
        "feature_extractor",
        "vae",
        "transformer",
    ]

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        # minWM: FlowMatchDiscreteScheduler(shift=5.0, reverse=True,
        # solver="euler").
        self.modules["scheduler"] = FlowMatchDiscreteScheduler(
            num_train_timesteps=1000,
            shift=server_args.pipeline_config.flow_shift,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_stage(CausalHunyuanVideo15ImagePreprocessStage())
        self.add_stage(
            TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )
        self.add_stage(
            ImageEncodingStage(
                image_processor=self.get_module("feature_extractor"),
                image_encoder=self.get_module("image_encoder"),
            ),
        )
        self.add_stage(
            ImageVAEEncodingStage(
                vae=self.get_module("vae"),
            ),
        )
        self.add_standard_latent_preparation_stage()
        self.add_stage(
            CausalHunyuanVideo15DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = CausalHunyuanVideo15Pipeline
