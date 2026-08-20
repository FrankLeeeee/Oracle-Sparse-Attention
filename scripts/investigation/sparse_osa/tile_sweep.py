# SPDX-License-Identifier: Apache-2.0
"""Sweep OSA's within-frame spatial tile size at the ~0.30 density tier.

The replicate policy meets its token budget exactly for any tile size
(``tiles_kept = round(budget / (num_other * tile_size))``), so holding the
calibrated 0.3-tier density knob fixed and varying ``spatial_tile`` isolates
the selection granularity: smaller tiles capture per-head mass more precisely
but fragment the gather; larger tiles approach whole-frame keeps.

One 720p / 20 s video per (prompt, tile) for tiles 16 / 32 / 128 / 256 on a
single GPU; tile 64 reuses the main-sweep (fox) and multi-prompt runs. Then
PSNR against each prompt's dense reference and one labeled frame sheet per
prompt (rows: dense + tile sizes).

    python tile_sweep.py [--gpu 7] [--analyze-only]
Videos -> runs_tiles/<prompt>_tile<n>/, sheets -> tile_sheet_<prompt>.png,
numbers -> results_tiles.json.
"""

import argparse
import glob
import json
import pathlib
import sys

import imageio
import numpy as np
from common import PROMPT as FOX_PROMPT
from common import (
    extract_frames,
    render_frame_sheet,
    run_generate,
)
from multi_prompt import PROMPTS as NEW_PROMPTS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("sparse_osa")
TILES = (16, 32, 64, 128, 256)
NEW_TILES = (16, 32, 128, 256)
PROMPTS = {"p0_fox": FOX_PROMPT, **NEW_PROMPTS}
BASE_CONFIG = json.loads((ROOT / "configs.json").read_text())["osa"]["0.3"]["config"]


def tile_run_dir(prompt_id: str, tile: int) -> pathlib.Path:
    if tile == 64:  # reuse the existing 0.3-tier runs
        if prompt_id == "p0_fox":
            return ROOT / "runs" / "osa_0.3"
        return ROOT / "runs_prompts" / f"{prompt_id}_osa"
    return ROOT / "runs_tiles" / f"{prompt_id}_tile{tile}"


def dense_run_dir(prompt_id: str) -> pathlib.Path:
    if prompt_id == "p0_fox":
        return ROOT / "runs" / "dense"
    return ROOT / "runs_prompts" / f"{prompt_id}_dense"


def load_video(run_dir: pathlib.Path) -> np.ndarray | None:
    paths = sorted(glob.glob(str(run_dir / "outputs" / "*.mp4"))) or sorted(
        glob.glob(str(run_dir / "*.mp4"))
    )
    if not paths:
        return None
    return np.stack([frame for frame in imageio.get_reader(paths[-1])])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255**2 / max(mse, 1e-9)))


def generate_all(gpu: int) -> None:
    results_path = ROOT / "results_tiles.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    for prompt_id, prompt in PROMPTS.items():
        for tile in NEW_TILES:
            run_id = f"{prompt_id}_tile{tile}"
            if results.get(run_id, {}).get("returncode") == 0:
                print(f"skip {run_id} (already done)", flush=True)
                continue
            result = run_generate(
                out_dir=ROOT / "runs_tiles" / run_id,
                log_name="timing.log",
                gpu=gpu,
                port_base=42000,
                width=1280,
                height=720,
                num_frames=321,
                method="osa",
                method_config=dict(BASE_CONFIG, spatial_tile=tile),
                save_output=True,
                timeout_s=2400,
                prompt=prompt,
            )
            results[run_id] = result
            results_path.write_text(json.dumps(results, indent=2))
            print(
                f"DONE {run_id} rc={result['returncode']} "
                f"denoise={result.get('denoise_s')} density={result.get('density')}",
                flush=True,
            )
    print("GENERATION DONE", flush=True)


def tile64_stats(prompt_id: str) -> dict:
    """Timing/density of the reused tile-64 run from the earlier sweeps."""
    if prompt_id == "p0_fox":
        merged = json.loads((ROOT / "results_merged.json").read_text())
        return merged.get("osa_0.3", {})
    prompts = json.loads((ROOT / "results_prompts.json").read_text())
    return prompts.get(f"{prompt_id}_osa", {})


def analyze() -> None:
    results_path = ROOT / "results_tiles.json"
    results: dict = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )
    fps = 16
    first = 5 * fps
    frame_indices = [second * fps for second in (1, 4, 7, 10, 13, 16, 19)]
    for prompt_id in PROMPTS:
        dense = load_video(dense_run_dir(prompt_id))
        if dense is None:
            print(f"missing dense reference for {prompt_id}", flush=True)
            continue
        rows = [("Dense", extract_frames(dense_run_dir(prompt_id), frame_indices))]
        for tile in TILES:
            run_dir = tile_run_dir(prompt_id, tile)
            video = load_video(run_dir)
            if video is None:
                print(f"missing video: {prompt_id} tile {tile}", flush=True)
                continue
            entry = results.setdefault(f"{prompt_id}_tile{tile}", {})
            if tile == 64:
                entry.update(tile64_stats(prompt_id))
            n = min(len(dense), len(video))
            entry["psnr_overall_db"] = round(psnr(dense[:n], video[:n]), 2)
            entry["psnr_first5s_db"] = round(psnr(dense[:first], video[:first]), 2)
            print(
                f"{prompt_id:10s} tile {tile:3d}: density={entry.get('density')} "
                f"denoise={entry.get('denoise_s')} "
                f"PSNR overall={entry['psnr_overall_db']} "
                f"first5s={entry['psnr_first5s_db']}",
                flush=True,
            )
            rows.append((f"tile {tile}", extract_frames(run_dir, frame_indices)))
        results_path.write_text(json.dumps(results, indent=2))
        out = ROOT / f"tile_sheet_{prompt_id}.png"
        render_frame_sheet(
            rows=[(label, frames) for label, frames in rows if frames is not None],
            frame_indices=frame_indices,
            fps=fps,
            out_path=out,
        )
        print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        generate_all(args.gpu)
    analyze()


if __name__ == "__main__":
    main()
