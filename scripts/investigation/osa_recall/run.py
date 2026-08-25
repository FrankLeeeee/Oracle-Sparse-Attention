# SPDX-License-Identifier: Apache-2.0
"""Generate one OSA video and record how well chunk 0's pattern still fits.

    python run.py [--gpu 5] [--density 0.3] [--seconds 10]

Runs full-context Self-Forcing 1.3B at 720p through `sglang generate` with
`--sparse-attention osa`, with hook/sitecustomize.py on PYTHONPATH so every
sparse attention call also reports the attention mass its kept set captures.
Output: results/investigation/osa_recall/<tag>/{run.log,recall.jsonl,*.mp4}.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/investigation/sparse_baselines"))
from common import MODELS, PROMPTS, SEED  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = results_dir("osa_recall")
# The three 1.3B Wan block-causal models of the sparse-baselines study, which
# share prompt, seed and 720p geometry but differ in how much history a chunk
# can see: Self-Forcing keeps the whole video, Causal Forcing a 21-frame
# window, Rolling Forcing a 21-frame window denoised as a 5-chunk rolling
# window (so one forward carries several query chunks).
MODEL_KEYS = ("self_forcing", "causal_forcing", "rolling_forcing")
PROMPT_KEY = "p0_tokyo"
# 16 fps Wan latents: pixel frames = 4 * latent - 3, and the chunk is 3 latent
# frames, so the frame count is chosen to land on a whole number of chunks.
FRAMES_PER_BLOCK = 3


def frame_count(seconds: float, fps: int) -> int:
    blocks = max(1, round(seconds * fps / 4 / FRAMES_PER_BLOCK))
    return 4 * (blocks * FRAMES_PER_BLOCK) - 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="self_forcing", choices=MODEL_KEYS)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.3)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--res", default="720p")
    parser.add_argument("--query-stride", type=int, default=32)
    parser.add_argument("--port-base", type=int, default=29700)
    parser.add_argument("--tile-layer", type=int, default=-1)
    parser.add_argument("--section-chunks", default="")
    parser.add_argument("--query-tiled", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--no-hook", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    spec = MODELS[args.model]
    width, height = spec["resolutions"][args.res]
    frames = frame_count(args.seconds, spec["fps"])
    tag = args.tag or f"{args.model}_d{args.density:g}_{frames}f"
    out_dir = ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = out_dir / "sections"
    sections.mkdir(exist_ok=True)
    recall_path = out_dir / "recall.jsonl"
    if recall_path.exists():
        recall_path.unlink()

    config = {
        "density": args.density,
        "sink_latent_frames": 1,
        "num_recent_frames": 1,
    }
    if args.query_tiled:
        config["query_tiled"] = True
    sparse_flags = (
        []
        if args.dense
        else ["--sparse-attention", "osa", "--sparse-attention-config", json.dumps(config)]
    )
    cmd = [
        "sglang",
        "generate",
        "--model-path",
        spec["path"],
        "--prompt",
        PROMPTS[PROMPT_KEY],
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(frames),
        "--seed",
        str(SEED),
        "--save-output",
        *sparse_flags,
        "--master-port",
        str(args.port_base),
        "--scheduler-port",
        str(args.port_base + 1),
        "--port",
        str(args.port_base + 2),
    ]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(HERE / "hook"), str(REPO / "python")]),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        OSA_RECALL_OUT="" if args.no_hook else str(recall_path),
        OSA_RECALL_QUERY_STRIDE=str(args.query_stride),
        OSA_RECALL_TILE_LAYER=str(args.tile_layer),
        OSA_RECALL_TILE_OUT=str(out_dir / "tile_profile.jsonl"),
        OSA_SECTION_LAYER=str(args.tile_layer),
        OSA_SECTION_CHUNKS=args.section_chunks,
        OSA_SECTION_DIR=str(sections) if args.section_chunks else "",
    )
    print(
        f"[run] {args.model}: {frames} frames "
        f"({frames / spec['fps']:.1f}s @ {args.res}), density {args.density}, "
        f"gpu {args.gpu}"
    )
    started = time.time()
    with open(out_dir / "run.log", "w") as handle:
        proc = subprocess.run(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=out_dir
        )
    elapsed = time.time() - started
    lines = (
        sum(1 for _ in open(recall_path))
        if not args.no_hook and recall_path.exists()
        else 0
    )
    print(f"[run] rc={proc.returncode} in {elapsed:.0f}s, {lines} measurements")
    if proc.returncode != 0:
        print((out_dir / "run.log").read_text()[-3000:])
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
