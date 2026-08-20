# SPDX-License-Identifier: Apache-2.0
"""Task 3.1 — chunk 0's attention maps at every denoising step.

Chunk 0 is the only chunk generated with no prior context: whatever attention
structure the model has must be built from scratch inside its own few frames.
Dumping every denoising step of that chunk shows the pattern forming.

Per config: 4 heads per layer at 5 depth percentiles, chunk 0 only, all of its
denoising steps. The probe runs in QK-only mode, so it skips the per-frame
attention-mass pass and dumps just the matrices being plotted.

    python run_captures.py [--models m1,m2] [--gpus auto|4,7]

Results land in results/investigation/attention_chunk0/runs/<model>/
<res>_<dur>s/{qk_chunk_000_step_*.npz, meta.json, video.mp4}.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import video_sweep  # noqa: E402
from geometry import head_spec, layer_ids  # noqa: E402

# Rolling Forcing's chunk 0 is denoised by five successive ramp-up windows, the
# other models take four steps; listing extra indices is harmless.
MAX_STEPS = 5
# Chunk 0's visible keys are just its own few frames, so a stride of 16 on both
# axes still leaves a few hundred rows and columns per map.
QUERY_STRIDE = 16
KEY_STRIDE = 16


def capture_env(model: str, res: str, duration: int) -> dict:
    layers = layer_ids(model)
    return {
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS": "0",
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_STEPS": ",".join(
            str(step) for step in range(MAX_STEPS)
        ),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_LAYERS": ",".join(
            str(layer) for layer in layers
        ),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_HEADS": head_spec(model, layers),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_ONLY": "1",
        "SGLANG_DIFFUSION_ATTENTION_MAP_QUERY_STRIDE": str(QUERY_STRIDE),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_KEY_STRIDE": str(KEY_STRIDE),
    }


if __name__ == "__main__":
    video_sweep.main(
        topic="attention_chunk0",
        probe_env={"SGLANG_DIFFUSION_ATTENTION_MAP_DIR": None},
        description=__doc__,
        default_port_base=39000,
        extra_env=capture_env,
    )
