# SPDX-License-Identifier: Apache-2.0
"""Shared runner for the sparse-attention comparison on Self-Forcing.

Model: full-context Self-Forcing 1.3B (config-patched `-fullctx-null`), the
attention-bound configuration where sparsity matters most. All runs share one
prompt and seed; a run's figure of merit is the *achieved* read density the
backend reports (cumulative over the run, dense fallbacks counted as 1.0).
"""

import glob
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO  # noqa: E402,F401

MODEL = "/data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers-fullctx-null"
PROMPT = (
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and "
    "animated city signage. She wears a black leather jacket, a long red dress, "
    "and black boots, and carries a black purse. She wears sunglasses and red "
    "lipstick. She walks confidently and casually. The street is damp and "
    "reflective, creating a mirror effect of the colorful lights. Many "
    "pedestrians walk about."
)
SEED = 42

METHOD_LABELS = {
    "dense": "Dense",
    "osa": "OSA",
    "lightforcing": "LightForcing",
    "radial": "Radial",
    "svg1": "SVG1",
    "svg2": "SVG2",
    "xattention": "XAttention",
}
SHEET_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def extract_frames(
    run_dir: pathlib.Path, frame_indices: list[int]
) -> list[np.ndarray] | None:
    """Half-res frames at the given indices from the newest mp4 under run_dir."""
    paths = sorted(glob.glob(str(run_dir / "outputs" / "*.mp4"))) or sorted(
        glob.glob(str(run_dir / "*.mp4"))
    )
    if not paths:
        return None
    found: dict[int, np.ndarray] = {}
    last: np.ndarray | None = None
    for index, frame in enumerate(imageio.get_reader(paths[-1])):
        last = np.asarray(frame)[::2, ::2]
        if index in frame_indices:
            found[index] = last
        if index >= max(frame_indices):
            break
    if last is None:
        return None
    return [found.get(index, last) for index in frame_indices]


def render_frame_sheet(
    *,
    rows: list[tuple[str, list[np.ndarray]]],
    frame_indices: list[int],
    fps: int,
    out_path: pathlib.Path,
) -> None:
    """Tile sheet with method-name row labels and frame-number column labels."""
    from PIL import Image, ImageDraw, ImageFont

    tile_h, tile_w = rows[0][1][0].shape[:2]
    row_font = ImageFont.truetype(SHEET_FONT, 30)
    col_font = ImageFont.truetype(SHEET_FONT, 26)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    margin = int(max(probe.textlength(label, font=row_font) for label, _ in rows)) + 32
    header = 56
    sheet = Image.new(
        "RGB",
        (margin + tile_w * len(frame_indices), header + tile_h * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for col, frame_index in enumerate(frame_indices):
        text = f"frame {frame_index} ({frame_index / fps:.0f} s)"
        text_w = probe.textlength(text, font=col_font)
        draw.text(
            (margin + col * tile_w + (tile_w - text_w) / 2, 13),
            text,
            fill="black",
            font=col_font,
        )
    for row, (label, frames) in enumerate(rows):
        top = header + row * tile_h
        draw.text((16, top + tile_h / 2 - 18), label, fill="black", font=row_font)
        for col, frame in enumerate(frames):
            sheet.paste(Image.fromarray(frame), (margin + col * tile_w, top))
    sheet.save(out_path)


DENSITY_LINE = re.compile(
    r"attention density so far: ([0-9.]+) over (\d+) calls \((\d+) dense\)"
)
STAGE_LINE = re.compile(r"\[(\w+Stage)\] finished in ([0-9.]+) seconds")
E2E_LINE = re.compile(r"Pixel data generated successfully in ([0-9.]+) seconds")


def run_generate(
    *,
    out_dir: pathlib.Path,
    log_name: str,
    gpu: int,
    port_base: int,
    width: int,
    height: int,
    num_frames: int,
    method: str | None,
    method_config: dict | None,
    save_output: bool = False,
    extra_env: dict | None = None,
    timeout_s: int = 3600,
    prompt: str = PROMPT,
) -> dict:
    """One `sglang generate` run; returns parsed timings and density."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / log_name
    args = [
        "sglang",
        "generate",
        "--model-path",
        MODEL,
        "--prompt",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(num_frames),
        "--seed",
        str(SEED),
        "--master-port",
        str(port_base),
        "--scheduler-port",
        str(port_base + 1),
        "--port",
        str(port_base + 2),
    ]
    if method is not None:
        args += ["--sparse-attention", method]
        if method_config:
            args += ["--sparse-attention-config", json.dumps(method_config)]
    if save_output:
        args += ["--save-output"]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        SGLANG_DIFFUSION_STAGE_LOGGING="1",
        SGLANG_DIFFUSION_SYNC_STAGE_PROFILING="1",
    )
    env.update(extra_env or {})
    started = time.time()
    with open(log, "w") as handle:
        proc = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=out_dir,
            timeout=timeout_s,
        )
    result = parse_log(log)
    result["returncode"] = proc.returncode
    result["wall_s"] = round(time.time() - started, 1)
    return result


def parse_log(log: pathlib.Path) -> dict:
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    out: dict = {}
    densities = DENSITY_LINE.findall(text)
    if densities:
        density, calls, dense_calls = densities[-1]
        out.update(
            density=float(density),
            density_calls=int(calls),
            density_dense_calls=int(dense_calls),
        )
    for stage, seconds in STAGE_LINE.findall(text):
        key = "denoise_s" if stage.endswith("DenoisingStage") else None
        if stage.endswith("DecodingStage"):
            key = "vae_decode_s"
        if key:
            out[key] = out.get(key, 0.0) + float(seconds)
    e2e = E2E_LINE.findall(text)
    if e2e:
        out["e2e_s"] = float(e2e[-1])
    return out
