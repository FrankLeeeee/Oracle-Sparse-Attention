# SPDX-License-Identifier: Apache-2.0
"""Generate the same video with every sparse-attention method and compare.

Runs dense plus each ``--sparse-attention`` method on one prompt and seed,
one method per GPU, then reports denoise wall-clock, the density each method
settled on, and the pixel delta against the dense reference.

    python -m sglang.multimodal_gen.tools.compare_sparse_attention_videos \
        --model-path /path/to/SelfForcing-Wan2.1-T2V-1.3B-Diffusers \
        --num-frames 321 --output-dir samples/sparse_attention_20s

Pixel delta is mean absolute difference in 0-255 units over all frames; it is
a fidelity measure against dense, not a quality measure — a method can drift
and still look fine, which is what the videos are for.
"""

import argparse
import json
import pathlib
import re
import subprocess
import time

METHODS = ("dense", "osa", "xattention", "svg1", "svg2", "radial", "fastar")

_DENOISE_TIME = re.compile(r"\[CausalDMDDenoisingStage\] finished in ([0-9.]+) seconds")
_TOTAL_TIME = re.compile(
    # the number is wrapped in ANSI colour codes, which are full of digits
    r"Pixel data generated successfully in .*?([0-9]+\.[0-9]+).*?seconds"
)
_DENSITY = re.compile(r"attention density so far: ([0-9.]+) over (\d+) calls")
_OSA_SUMMARY = re.compile(r"OSA calibrated .*")


def run_one(
    *,
    method: str,
    gpu: int,
    slot: int,
    model_path: str,
    prompt: str,
    num_frames: int,
    seed: int,
    output_dir: pathlib.Path,
    config: str | None,
) -> subprocess.Popen:
    command = [
        "sglang",
        "generate",
        "--model-path",
        model_path,
        "--prompt",
        prompt,
        "--num-frames",
        str(num_frames),
        "--seed",
        str(seed),
        "--save-output",
        "--output-path",
        str(output_dir / method),
        "--output-file-name",
        f"{method}.mp4",
        # Concurrent runs each stand up their own local worker, so every port
        # has to be distinct or the second one dies on EADDRINUSE.
        "--master-port",
        str(30005 + 20 * slot),
        "--port",
        str(30100 + 20 * slot),
        "--scheduler-port",
        str(35880 + 20 * slot),
    ]
    if method != "dense":
        command += ["--sparse-attention", method]
        if config:
            command += ["--sparse-attention-config", config]
    log_path = output_dir / f"{method}.log"
    log_file = log_path.open("w")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={
            "PATH": __import__("os").environ["PATH"],
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": __import__("os").environ.get("PYTHONPATH", ""),
            "HOME": __import__("os").environ.get("HOME", "/root"),
        },
    )


def parse_log(path: pathlib.Path) -> dict:
    text = path.read_text(errors="replace")
    densities = _DENSITY.findall(text)
    denoise = _DENOISE_TIME.findall(text)
    total = _TOTAL_TIME.findall(text)
    summary = _OSA_SUMMARY.findall(text)
    return {
        "denoise_s": float(denoise[-1]) if denoise else None,
        "total_s": float(total[-1]) if total else None,
        "density": float(densities[-1][0]) if densities else None,
        "density_calls": int(densities[-1][1]) if densities else None,
        "osa_summary": summary[-1] if summary else None,
    }


def pixel_delta(reference: pathlib.Path, other: pathlib.Path) -> dict | None:
    import imageio.v3 as iio
    import numpy as np

    a = iio.imread(reference)
    b = iio.imread(other)
    frames = min(len(a), len(b))
    a = a[:frames].astype(np.float32)
    b = b[:frames].astype(np.float32)
    difference = np.abs(a - b)
    return {
        "frames": frames,
        "mean_abs": round(float(difference.mean()), 3),
        "p99_abs": round(float(np.percentile(difference, 99)), 1),
        "psnr": round(
            float(10 * np.log10(255.0**2 / max(((a - b) ** 2).mean(), 1e-9))), 2
        ),
    }


def find_video(directory: pathlib.Path) -> pathlib.Path | None:
    videos = sorted(directory.glob("**/*.mp4"))
    return videos[0] if videos else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompt",
        default="A red fox trotting across a snowy field, camera follows",
    )
    parser.add_argument("--num-frames", type=int, default=321)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 4, 5, 6, 7])
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--sparse-attention-config", default=None)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = list(args.methods)
    running: dict[str, subprocess.Popen] = {}
    free_slots = list(range(len(args.gpus)))
    slot_of: dict[str, int] = {}
    started = time.time()
    while pending or running:
        while pending and free_slots:
            method = pending.pop(0)
            slot = free_slots.pop(0)
            slot_of[method] = slot
            gpu = args.gpus[slot]
            print(f"[{time.time() - started:6.0f}s] launching {method} on GPU {gpu}")
            running[method] = run_one(
                method=method,
                gpu=gpu,
                slot=slot,
                model_path=args.model_path,
                prompt=args.prompt,
                num_frames=args.num_frames,
                seed=args.seed,
                output_dir=output_dir,
                config=args.sparse_attention_config,
            )
        for method, process in list(running.items()):
            if process.poll() is not None:
                print(
                    f"[{time.time() - started:6.0f}s] {method} exited "
                    f"with {process.returncode}"
                )
                del running[method]
                free_slots.append(slot_of[method])
        time.sleep(5)

    reference = find_video(output_dir / "dense")
    results = {}
    for method in args.methods:
        row = parse_log(output_dir / f"{method}.log")
        video = find_video(output_dir / method)
        row["video"] = str(video) if video else None
        if reference is not None and video is not None and method != "dense":
            row["vs_dense"] = pixel_delta(reference, video)
        results[method] = row
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(
        f"\n{'method':<12}{'denoise s':>11}{'total s':>9}{'density':>9}"
        f"{'speedup':>9}{'mean|d|':>9}{'PSNR':>8}"
    )
    print("-" * 67)
    baseline = results.get("dense", {}).get("denoise_s")
    for method, row in results.items():
        delta = row.get("vs_dense") or {}
        speedup = (
            baseline / row["denoise_s"] if baseline and row.get("denoise_s") else None
        )
        print(
            f"{method:<12}"
            f"{row['denoise_s'] if row['denoise_s'] else float('nan'):>11.2f}"
            f"{row['total_s'] if row['total_s'] else float('nan'):>9.2f}"
            f"{row['density'] if row['density'] else float('nan'):>9.3f}"
            f"{speedup if speedup else float('nan'):>9.2f}"
            f"{delta.get('mean_abs', float('nan')):>9.2f}"
            f"{delta.get('psnr', float('nan')):>8.2f}"
        )
    for method, row in results.items():
        if row.get("osa_summary"):
            print(f"\n{row['osa_summary']}")


if __name__ == "__main__":
    main()
