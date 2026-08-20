# SPDX-License-Identifier: Apache-2.0
"""Per-chunk forward / attention wall-time sweep over four block-causal models.

Models x resolutions {480p, 720p} x durations {5, 10, 20}s, one clean timing
run per config with the chunk-timing probe on
(``SGLANG_DIFFUSION_CHUNK_TIMING_DIR``). The probe brackets every DiT forward
and every attention module with CUDA events and only resolves them at chunk
boundaries, so the run stays close to an unprobed one. Run it serially on one
idle GPU: concurrent generations inflate wall time badly.

    python run_sweep.py [--models m1,m2] [--gpus auto|4,7] [--durations 5,10,20]

Results land in results/investigation/chunk_runtime/runs/<model>/<res>_<dur>s/
{run.log, chunk_timing.json, video.mp4}.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import video_sweep  # noqa: E402

if __name__ == "__main__":
    video_sweep.main(
        topic="chunk_runtime",
        probe_env={"SGLANG_DIFFUSION_CHUNK_TIMING_DIR": None},
        description=__doc__,
        default_port_base=35000,
    )
