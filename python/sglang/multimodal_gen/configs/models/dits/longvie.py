# SPDX-License-Identifier: Apache-2.0
"""Config for LongVie 2 — Wan2.1-I2V-14B plus a dual-control side network."""

import re
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)

# The control side network is built from ordinary Wan DiT blocks, so its weights
# need exactly the Wan block rewrites — just re-anchored under `control.*`.
_CONTROL_BLOCK_PREFIX = r"^control\.blocks_(dense|sparse)\."


def _reanchor_block_rules(rules: dict[str, str]) -> dict[str, str]:
    """Re-target `^blocks.N....` rewrite rules at the control side network.

    The capture-group indices shift by one because the re-anchored pattern
    captures the dense/sparse stream name before the layer index.
    """
    reanchored: dict[str, str] = {}
    for pattern, replacement in rules.items():
        if not pattern.startswith(r"^blocks\.") or not replacement:
            continue
        new_pattern = _CONTROL_BLOCK_PREFIX + pattern[len(r"^blocks\.") :]
        # \1 -> \2, \2 -> \3, ... then re-insert the stream name as \1
        shifted = re.sub(r"\\(\d)", lambda m: f"\\{int(m.group(1)) + 1}", replacement)
        new_replacement = r"control.blocks_\1." + shifted[len("blocks.") :]
        reanchored[new_pattern] = new_replacement
    return reanchored


@dataclass
class LongVie2VideoArchConfig(WanVideoArchConfig):
    # side network geometry; overridden from the checkpoint's config.json
    control_layers: int = 12
    control_dim: int = 2560
    control_num_heads: int = 20
    control_ffn_dim: int = 6912

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.param_names_mapping = {
            **self.param_names_mapping,
            **_reanchor_block_rules(self.param_names_mapping),
        }
        self.reverse_param_names_mapping = {
            **self.reverse_param_names_mapping,
            **{
                _CONTROL_BLOCK_PREFIX
                + p[len(r"^blocks\.") :]: r"control.blocks_\1."
                + re.sub(r"\\(\d)", lambda m: f"\\{int(m.group(1)) + 1}", r)[
                    len("blocks.") :
                ]
                for p, r in self.reverse_param_names_mapping.items()
                if p.startswith(r"^blocks\.") and r
            },
        }


@dataclass
class LongVie2VideoConfig(WanVideoConfig):
    arch_config: LongVie2VideoArchConfig = field(
        default_factory=LongVie2VideoArchConfig
    )
    prefix: str = "LongVie2"
