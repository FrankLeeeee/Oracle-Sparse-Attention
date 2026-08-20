# SPDX-License-Identifier: Apache-2.0
"""Layer and head selections shared by the attention-map captures.

Layers are taken at fixed percentiles of the model's depth so the same
"shallow / quarter / middle / three-quarter / deep" positions are compared
across models of different depth. Heads are drawn per (model, layer) from a
seeded RNG: "four random heads" is only meaningful inside a layer, since head
3 of layer 0 and head 3 of layer 29 are unrelated features. Seeding on the
model and layer names keeps the same heads across chunks, steps, denoising
steps, resolutions and durations, so every figure of a model is comparable.
"""

import random

# (num_layers, num_heads) per model, as built by their configs.
GEOMETRY = {
    "self_forcing": (30, 12),
    "rolling_forcing": (30, 12),
    "longlive2": (30, 24),
    "lingbot_world_v2": (40, 40),
}
LAYER_PERCENTILES = (0, 25, 50, 75, 100)
HEADS_PER_LAYER = 4
HEAD_SEED = 42


def layer_ids(model: str, percentiles=LAYER_PERCENTILES) -> list[int]:
    num_layers = GEOMETRY[model][0]
    return sorted({round(p / 100 * (num_layers - 1)) for p in percentiles})


def head_ids(model: str, layer: int, count: int = HEADS_PER_LAYER) -> list[int]:
    num_heads = GEOMETRY[model][1]
    rng = random.Random(f"{HEAD_SEED}:{model}:{layer}")
    return sorted(rng.sample(range(num_heads), count))


def head_spec(model: str, layers: list[int]) -> str:
    """``SGLANG_DIFFUSION_ATTENTION_MAP_QK_HEADS`` value for one model."""
    return ";".join(
        f"{layer}:" + ",".join(str(h) for h in head_ids(model, layer))
        for layer in layers
    )
