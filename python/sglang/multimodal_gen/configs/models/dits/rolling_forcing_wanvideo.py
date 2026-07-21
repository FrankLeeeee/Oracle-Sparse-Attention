# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/TencentARC/RollingForcing

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)


@dataclass
class RollingForcingWanVideoArchConfig(WanVideoArchConfig):
    """Rolling Forcing causal attention geometry (latent-frame units).

    Upstream inference uses a 24-frame KV cache ring buffer, caps the visible
    attention context at 21 frames, and pins the first 3-frame block as a
    never-evicted attention sink that is re-roped to a relative position just
    before the working cache.
    """

    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 24
    sink_size: int = 3
    # Maximum attention context (sink + working cache + current window).
    max_attention_num_frames: int = 21


@dataclass
class RollingForcingWanVideoConfig(WanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=RollingForcingWanVideoArchConfig)
