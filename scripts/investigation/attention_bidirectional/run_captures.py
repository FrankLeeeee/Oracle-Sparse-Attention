# SPDX-License-Identifier: Apache-2.0
"""Task 4 — attention maps of the bidirectional model the causal ones came from.

Wan2.1-T2V-1.3B attends over the *whole* video at once: no chunks, no KV cache,
no causal mask. It is the reference the block-causal models are distilled from,
so its maps are the baseline the earlier sections' structure should be read
against.

Per config: five depth percentiles x four seeded-random heads per layer x the
0 / 25 / 50 / 75 / 100 % denoising steps of its 50-step schedule. The head
selection is fixed per (model, layer), so the same heads appear at every step.

Only the conditional branch of classifier-free guidance is recorded, so a step
index here is a denoising step rather than a forward.

    python run_captures.py [--gpus auto|4,7] [--durations 5,10,20]

Results land in results/investigation/attention_bidirectional/runs/
wan2_1_t2v_1_3b/<res>_<dur>s/{qk_chunk_000_step_*.npz, meta.json, video.mp4}.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import video_sweep  # noqa: E402
from geometry import head_spec, layer_ids, step_ids, token_stride  # noqa: E402

MODEL = "wan2_1_t2v_1_3b"


def capture_env(model: str, res: str, durations: list[int]) -> dict:
    layers = layer_ids(model)
    width, height = video_sweep.MODELS[model]["resolutions"][res]
    frames = video_sweep.MODELS[model]["frames"][durations[0]]
    stride = token_stride(width, height, frames)
    return {
        # A full-attention pass has no blocks, so the probe groups it as a
        # single chunk 0 whose queries are the entire video.
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS": "0",
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_STEPS": ",".join(
            str(step) for step in step_ids(model)
        ),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_LAYERS": ",".join(
            str(layer) for layer in layers
        ),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_HEADS": head_spec(model, layers),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_ONLY": "1",
        "SGLANG_DIFFUSION_ATTENTION_MAP_QUERY_STRIDE": str(stride),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_KEY_STRIDE": str(stride),
    }


if __name__ == "__main__":
    video_sweep.main(
        topic="attention_bidirectional",
        probe_env={"SGLANG_DIFFUSION_ATTENTION_MAP_DIR": None},
        description=__doc__,
        default_port_base=41000,
        extra_env=capture_env,
        default_models=[MODEL],
    )
