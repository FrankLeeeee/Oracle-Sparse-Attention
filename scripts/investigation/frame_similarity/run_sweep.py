# SPDX-License-Identifier: Apache-2.0
"""Intra-chunk frame-similarity sweep over four block-causal models.

Models x resolutions {480p, 720p} x durations {5, 10, 20}s, one run per config
with the frame-similarity probe on
(``SGLANG_DIFFUSION_FRAME_SIMILARITY_DIR``): for every denoising step and every
layer boundary, the cosine similarity between each pair of latent frames of the
chunk. Unlike the chunk_runtime sweep this measures no wall time, so it can be
fanned out across several GPUs.

    python run_sweep.py [--models m1,m2] [--gpus auto|4,7] [--durations 5,10,20]

Results land in results/investigation/frame_similarity/runs/<model>/
<res>_<dur>s/{run.log, frame_similarity.npz, meta.json, video.mp4}.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import video_sweep  # noqa: E402

if __name__ == "__main__":
    video_sweep.main(
        topic="frame_similarity",
        probe_env={"SGLANG_DIFFUSION_FRAME_SIMILARITY_DIR": None},
        description=__doc__,
        default_port_base=38000,
    )
