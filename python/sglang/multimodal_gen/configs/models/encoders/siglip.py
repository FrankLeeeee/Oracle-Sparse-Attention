# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    ImageEncoderArchConfig,
    ImageEncoderConfig,
)


@dataclass
class SiglipVisionArchConfig(ImageEncoderArchConfig):
    # Defaults follow google/siglip-so400m-patch14-384.
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 384
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )


@dataclass
class SiglipVisionConfig(ImageEncoderConfig):
    arch_config: ImageEncoderArchConfig = field(default_factory=SiglipVisionArchConfig)

    num_hidden_layers_override: int | None = None
    require_post_norm: bool | None = None
    prefix: str = "siglip"
