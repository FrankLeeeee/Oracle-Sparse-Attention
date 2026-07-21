# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/TencentARC/RollingForcing
"""
Rolling Forcing pipeline (TencentARC/RollingForcing).

Wan2.1-T2V-1.3B distilled streaming generator: a rolling window of 5 blocks is
denoised jointly at staggered noise levels, with an attention-sink KV cache
for long-horizon consistency. Checkpoints are converted with
``sglang.multimodal_gen.tools.convert_forcing_to_diffusers``.
"""

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    RollingForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.configs.sample.wan import RollingForcingT2VSamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.rolling_forcing_denoising import (
    RollingForcingDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class WanRollingForcingPipeline(WanCausalDMDPipeline):
    pipeline_name = "WanRollingForcingPipeline"
    pipeline_config_cls = RollingForcingWanT2V480PConfig
    sampling_params_cls = RollingForcingT2VSamplingParams

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        # Match upstream FlowMatchScheduler(shift=5, sigma_min=0.0,
        # extra_one_step=True) with 1000 shift-warped training timesteps.
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        self.add_stage(
            RollingForcingDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = WanRollingForcingPipeline
