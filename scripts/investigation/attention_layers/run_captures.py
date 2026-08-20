# SPDX-License-Identifier: Apache-2.0
"""Tasks 3.2 + 3.3 — attention maps across the video and across network depth.

3.2 asks for layer 0 at chunks spaced over the whole video; 3.3 asks for the
same at the 25 / 50 / 75 / 100 % depth percentiles. Those differ only in which
layers are dumped, and the head selection has to be identical for the two to be
comparable, so one capture covers both: five depth percentiles x six chunk
percentiles x the same four seeded-random heads per layer as task 3.1.

Only each chunk's *last* denoising step is dumped. Task 3.1 already covers how
a chunk changes across its steps, and the late chunks of a full-context model
carry hundreds of thousands of visible keys -- dumping every step of those
would be tens of GB for no new question answered.

    python run_captures.py [--models m1,m2] [--gpus auto|4,7]

Results land in results/investigation/attention_layers/runs/<model>/
<res>_<dur>s/{qk_chunk_*_step_*.npz, meta.json, video.mp4}.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import video_sweep  # noqa: E402
from geometry import LAST_STEP, chunk_ids, head_spec, layer_ids  # noqa: E402

# A late full-context chunk sees ~280k keys at 720p, so the key axis is
# subsampled harder than in task 3.1, where a chunk only saw its own frames.
QUERY_STRIDE = 16
KEY_STRIDE = 32


def capture_env(model: str, res: str, durations: list[int]) -> dict:
    layers = layer_ids(model)
    # A realtime model serves every duration from one process, so one chunk
    # list has to cover all of them. Requesting the union is safe: a session
    # simply never reaches the chunk ids belonging to a longer duration.
    chunks = sorted({c for d in durations for c in chunk_ids(model, d)})
    return {
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS": ",".join(
            str(chunk) for chunk in chunks
        ),
        "SGLANG_DIFFUSION_ATTENTION_MAP_QK_STEPS": str(LAST_STEP[model]),
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
        topic="attention_layers",
        probe_env={"SGLANG_DIFFUSION_ATTENTION_MAP_DIR": None},
        description=__doc__,
        default_port_base=40000,
        extra_env=capture_env,
    )
