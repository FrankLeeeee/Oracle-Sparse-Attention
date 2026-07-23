# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HunyuanVideo15VAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: tuple[int, ...] = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True
    scaling_factor: float = 1.03682
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4


@dataclass
class HunyuanVideo15VAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=HunyuanVideo15VAEArchConfig)

    # Diffusers' AutoencoderKLHunyuanVideo15 only tiles spatially; temporal
    # tiling would change the causal-conv temporal context, so keep it off.
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
