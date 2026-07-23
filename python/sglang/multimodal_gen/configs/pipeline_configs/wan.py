# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.models.dits.longvie import LongVie2VideoConfig
from sglang.multimodal_gen.configs.models.dits.rolling_forcing_wanvideo import (
    RollingForcingWanVideoConfig,
)
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPVisionConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class WanI2VCommonConfig(PipelineConfig):
    # for all wan i2v pipelines
    def adjust_num_frames(self, num_frames):
        vae_scale_factor_temporal = self.vae_config.arch_config.scale_factor_temporal
        if num_frames % vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
            )
            return num_frames
        return num_frames


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    task_type: ModelTaskType = ModelTaskType.T2V
    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)

    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
        )

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]


@dataclass
class TurboWanT2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [988, 932, 852, 608]
    )


@dataclass
class TurboWanT2V1_3B480PConfig(TurboWanT2V480PConfig):
    """Configuration for TurboWan T2V 1.3B DMD pipeline."""

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
            keep_resident_min_available_gb=60,
            keep_resident_components=(
                "text_encoder",
                "image_encoder",
                "vae",
            ),
        )


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WanI2V480PConfig(WanT2V480PConfig, WanI2VCommonConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    max_area: int = 480 * 832
    # WanConfig-specific parameters with defaults
    task_type: ModelTaskType = ModelTaskType.I2V
    # Precision for each component
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
        )


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    max_area: int = 720 * 1280
    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class TurboWanI2V720Config(WanI2V720PConfig):
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [996, 932, 852, 608]
    )
    boundary_ratio: float | None = 0.9

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
            keep_resident_min_available_gb=60,
            keep_resident_components=(
                "text_encoder",
                "image_encoder",
                "vae",
            ),
        )


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    flow_shift: float | None = 5.0
    task_type: ModelTaskType = ModelTaskType.TI2V
    expand_timesteps: bool = True
    # ti2v, 5B
    vae_stride = (4, 16, 16)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config):
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio
        self.dit_config.torch_compile_mode = "default"


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V720PConfig):
    flow_shift: float | None = 5.0
    boundary_ratio: float | None = 0.900

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.boundary_ratio = self.boundary_ratio


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V480PConfig):
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # 1.3B realtime-oriented causal generator: keep the DiT resident when
        # memory allows instead of the generic Wan layerwise-offload default.
        return ModelDeploymentConfig(
            keep_resident_components=("dit", "vae"),
        )


@dataclass
class CausalForcingWanT2V480PConfig(SelfForcingWanT2V480PConfig):
    """Causal Forcing (thu-ml/Causal-Forcing), chunk-wise and frame-wise.

    Inference-identical to Self-Forcing (block-wise causal DMD, 4 warped
    denoising steps, shift-5 flow matching, no CFG). The block size
    (``num_frames_per_block``: 3 chunk-wise, 1 frame-wise) comes from the
    converted checkpoint's ``transformer/config.json``.
    """


@dataclass
class RollingForcingWanT2V480PConfig(SelfForcingWanT2V480PConfig):
    """Rolling Forcing (TencentARC/RollingForcing).

    Joint denoising over a rolling window of 5 blocks at staggered noise
    levels, with an attention-sink KV cache for long-horizon streaming.
    """

    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 800, 600, 400, 200]
    )
    dit_config: DiTConfig = field(default_factory=RollingForcingWanVideoConfig)


@dataclass
class LongVie2Config(WanI2V480PConfig):
    """LongVie 2: Wan I2V 14B 480P plus the dual-control side network.

    Only the DiT differs from the base model — the VAE, encoders and scheduler
    are Wan's — so this inherits everything and swaps the transformer config.
    """

    dit_config: DiTConfig = field(default_factory=LongVie2VideoConfig)

    # --- clip-by-clip autoregressive generation (upstream inference.py) ---
    # Frames generated per clip; requests longer than this run the clip loop.
    longvie_clip_num_frames: int = 81
    # Pixel frames of the previous clip carried as history. Upstream takes the
    # last 8 but its VAE's chunked encoder only consumes the first
    # ``1 + 4 * ((n - 1) // 4)`` of them (5 for n=8) — the encode helper
    # reproduces that.
    longvie_history_frames: int = 8
    # Blend factor for re-noising latent frame 0 toward the history's last
    # latent frame on clips after the first (upstream hardcodes this).
    longvie_first_frame_blend_sigma: float = 0.925926
