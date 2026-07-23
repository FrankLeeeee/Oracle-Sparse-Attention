# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class CausalHunyuanVideo15TI2VSamplingParams(SamplingParams):
    """minWM causal HunyuanVideo 1.5 TI2V — 4-step distilled generator, no CFG.

    Defaults match ``HY15/scripts/inference/run_infer_causal.sh``: 480x832,
    77 output frames (20 latent frames = 5 chunks of 4), fps 16.
    """

    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    negative_prompt: str | None = None
    height: int = 480
    width: int = 832
    num_frames: int = 77
    fps: int = 16
