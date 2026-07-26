# SPDX-License-Identifier: Apache-2.0
"""DiT config for the causal (minWM) HunyuanVideo 1.5 transformer.

The checkpoint (MIN-Lab/minWM ``HY15/TI2V/*``) keeps the original tencent
HunyuanVideo-1.5 parameter naming (``double_blocks.N.img_attn_q`` ...), so the
model implementation mirrors those names and ``param_names_mapping`` stays
empty (identity).  Architecture fields below use the checkpoint's
``config.json`` field names so that ``update_model_arch`` overrides them
directly.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def _is_double_block(n: str, m) -> bool:
    return "double_blocks" in n and n.split(".")[-1].isdigit()


@dataclass
class CausalHunyuanVideo15ArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_double_block])

    # Checkpoint names == module names; no renaming needed.
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)

    # --- fields mirroring the checkpoint's transformer/config.json ---
    patch_size: list = field(default_factory=lambda: [1, 1, 1])
    in_channels: int = 32
    concat_condition: bool = True
    out_channels: int = 32
    hidden_size: int = 2048
    heads_num: int = 16
    mlp_width_ratio: float = 4.0
    mlp_act_type: str = "gelu_tanh"
    mm_double_blocks_depth: int = 54
    mm_single_blocks_depth: int = 0
    rope_dim_list: list = field(default_factory=lambda: [16, 56, 56])
    qkv_bias: bool = True
    qk_norm: bool = True
    qk_norm_type: str = "rms"
    guidance_embed: bool = False
    use_meanflow: bool = False
    text_projection: str = "single_refiner"
    use_attention_mask: bool = True
    text_states_dim: int = 3584
    text_states_dim_2: int | None = None
    text_pool_type: str | None = None
    rope_theta: int = 256
    attn_mode: str = "flash"
    attn_param: dict | None = None
    glyph_byT5_v2: bool = True
    vision_projection: str = "linear"
    vision_states_dim: int = 1152
    byt5_states_dim: int = 1472
    is_reshape_temporal_channels: bool = False
    use_cond_type_embedding: bool = True
    ideal_resolution: str | None = None
    ideal_task: str | None = None

    # --- causal rollout parameters (consumed by the causal denoising stage) ---
    # minWM denoises in chunks of 4 latent frames and keeps every generated
    # frame in the KV cache (no sliding window): 77 output frames -> 20 latent
    # frames.
    num_frames_per_block: int = 4
    sliding_window_num_frames: int = 20
    sink_size: int = 0
    local_attn_size: int = -1

    def __post_init__(self):
        super().__post_init__()
        self.num_attention_heads = self.heads_num
        self.attention_head_dim = self.hidden_size // self.heads_num
        self.num_layers = self.mm_double_blocks_depth
        self.num_channels_latents = self.out_channels


@dataclass
class CausalHunyuanVideo15Config(DiTConfig):
    arch_config: DiTArchConfig = field(
        default_factory=CausalHunyuanVideo15ArchConfig
    )

    prefix: str = "HunyuanVideo15"
