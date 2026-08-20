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
    # Wan2.1-T2V-1.3B: the bidirectional model the causal ones are built from,
    # same transformer shape as Self-Forcing.
    "wan2_1_t2v_1_3b": (30, 12),
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


# Chunks a config actually generates, measured from the frame-similarity sweep
# rather than re-derived: LingBot's realtime sessions emit one more chunk than
# the frame arithmetic suggests, and getting this wrong silently shifts every
# percentile.
CHUNK_COUNTS = {
    ("self_forcing", 5): 7,
    ("self_forcing", 10): 14,
    ("self_forcing", 20): 27,
    ("rolling_forcing", 5): 7,
    ("rolling_forcing", 10): 14,
    ("rolling_forcing", 20): 27,
    ("longlive2", 5): 4,
    ("longlive2", 10): 8,
    ("longlive2", 20): 15,
    ("lingbot_world_v2", 5): 8,
    ("lingbot_world_v2", 10): 15,
    ("lingbot_world_v2", 20): 28,
}
CHUNK_PERCENTILES = (0, 20, 40, 60, 80, 100)
# The last denoising step of a chunk, per model. Rolling Forcing keys a dump by
# the window's oldest block, and that window runs a single forward in which
# that block is at its cleanest -- so its "last step" is index 0.
# Denoising steps a model runs, where that matters for step percentiles.
NUM_STEPS = {"wan2_1_t2v_1_3b": 50}
STEP_PERCENTILES = (0, 25, 50, 75, 100)

LAST_STEP = {
    "self_forcing": 3,
    "rolling_forcing": 0,
    "longlive2": 3,
    "lingbot_world_v2": 3,
}


def chunk_ids(model: str, duration: int, percentiles=CHUNK_PERCENTILES) -> list[int]:
    count = CHUNK_COUNTS[(model, duration)]
    return sorted({round(p / 100 * (count - 1)) for p in percentiles})


def step_ids(model: str, percentiles=STEP_PERCENTILES) -> list[int]:
    count = NUM_STEPS[model]
    return sorted({round(p / 100 * (count - 1)) for p in percentiles})


def token_stride(width: int, height: int, num_frames: int, target: int = 2048) -> int:
    """Subsampling stride that leaves roughly ``target`` positions per axis.

    A bidirectional model attends over the whole video at once, so the matrix
    is (total tokens)^2 -- 291k tokens at 720p/20s. A fixed stride would make
    the short configs coarse and the long ones unplottable.
    """
    tokens = ((num_frames - 1) // 4 + 1) * (width // 16) * (height // 16)
    stride = 1
    while tokens // stride > target:
        stride *= 2
    return stride
