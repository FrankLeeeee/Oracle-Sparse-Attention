# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/thu-ml/Causal-Forcing
"""
Causal Forcing pipeline (thu-ml/Causal-Forcing).

A Self-Forcing-family causal DMD generator on Wan2.1-T2V-1.3B. Inference is
identical to the block-wise causal DMD recipe (KV-cached few-step denoising,
clean-latent cache refresh), so this pipeline only swaps in the shift-warped
Self-Forcing flow-match scheduler used by the upstream implementation. The
block size (chunk-wise: 3 latent frames, frame-wise: 1) is read from the
converted checkpoint's ``transformer/config.json``.

Checkpoints are converted with
``sglang.multimodal_gen.tools.convert_forcing_to_diffusers``.
"""

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    CausalForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.configs.sample.wan import CausalForcingT2VSamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class CausalForcingPipeline(WanCausalDMDPipeline):
    pipeline_name = "CausalForcingPipeline"
    pipeline_config_cls = CausalForcingWanT2V480PConfig
    sampling_params_cls = CausalForcingT2VSamplingParams

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        # Match upstream: FlowMatchScheduler(shift=5, sigma_min=0.0,
        # extra_one_step=True) with 1000 shift-warped training timesteps used
        # both for warping dmd_denoising_steps and for add_noise re-noising.
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )


EntryClass = CausalForcingPipeline
