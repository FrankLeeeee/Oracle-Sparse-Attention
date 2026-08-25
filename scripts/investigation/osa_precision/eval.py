# SPDX-License-Identifier: Apache-2.0
"""One evaluation run of an OSA variant: generate, then score against dense.

    python eval.py --model self_forcing --tag baseline [--config '{"density":0.19,...}']

Metrics per run (720p / 20 s / p0_tokyo / seed 42):
  denoise_s        summed DenoisingStage wall time
  density          achieved cumulative read density
  psnr / psnr_f5   PSNR vs the model's dense run (overall / first 5 s)
  lag1 / flips     x-shift smoothness (phase correlation): autocorr at lag 1,
                   direction reversals among |dx|>2px moves (dense ~11)

Results append to results/investigation/osa_precision/results.json under
"<model>/<tag>"; the video and log stay in <model>/<tag>/.
"""

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/investigation/sparse_baselines"))
from common import MODELS, run_generate  # noqa: E402

ROOT = results_dir("osa_precision")
BASELINES_ROOT = REPO / "results/investigation/sparse_baselines"


def latest_mp4(run_dir: pathlib.Path) -> str:
    paths = sorted(glob.glob(str(run_dir / "outputs" / "*.mp4")))
    if not paths:
        raise RuntimeError(f"no video under {run_dir}")
    return paths[-1]


def read_frames(path: str, max_frames: int = 321):
    import imageio.v2 as imageio

    for i, frame in enumerate(imageio.get_reader(path)):
        if i >= max_frames:
            break
        yield i, np.asarray(frame)


def psnr_vs(path: str, dense_path: str, fps: int) -> tuple[float, float]:
    dense = {i: f.astype(np.float64) for i, f in read_frames(dense_path)}
    overall, first5 = [], []
    for i, frame in read_frames(path):
        if i not in dense:
            continue
        if i % 4:  # every 4th pixel frame is plenty for a mean
            continue
        mse = ((frame.astype(np.float64) - dense[i]) ** 2).mean()
        value = 10 * np.log10(255**2 / max(mse, 1e-9))
        overall.append(value)
        if i < 5 * fps:
            first5.append(value)
    return float(np.mean(overall)), float(np.mean(first5))


def xshift_stats(path: str) -> tuple[float, int]:
    prev, shifts = None, []
    for _, frame in read_frames(path):
        g = frame[::4, ::4].mean(-1)
        g = g - g.mean()
        if prev is not None:
            F = np.fft.rfft2(prev) * np.conj(np.fft.rfft2(g))
            corr = np.fft.irfft2(F / (np.abs(F) + 1e-9))
            peak = np.unravel_index(np.argmax(corr), corr.shape)
            dx = peak[1] if peak[1] <= corr.shape[1] // 2 else peak[1] - corr.shape[1]
            shifts.append(dx * 4)
        prev = g
    s = np.array(shifts, dtype=float)
    lag1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    big = s[np.abs(s) > 2]
    flips = int(np.sum(np.abs(np.diff(np.sign(big))) > 0))
    return lag1, flips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", default=None, help="full OSA config JSON; default = model's osa 0.3 tier")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=29600)
    parser.add_argument("--skip-run", action="store_true", help="only (re)score an existing run")
    args = parser.parse_args()

    if args.config:
        config = json.loads(args.config)
    else:
        tiers = json.loads((BASELINES_ROOT / "configs.json").read_text())
        config = tiers[args.model]["osa"]["0.3"]["config"]

    out_dir = ROOT / args.model / args.tag
    result: dict = {"config": config}
    if not args.skip_run:
        result.update(
            run_generate(
                model=args.model,
                out_dir=out_dir,
                gpu=args.gpu,
                port_base=args.port_base,
                duration=20,
                method="osa",
                method_config=config,
            )
        )

    fps = MODELS[args.model]["fps"]
    video = latest_mp4(out_dir)
    dense_video = latest_mp4(BASELINES_ROOT / args.model / "runs" / "dense")
    result["psnr"], result["psnr_f5"] = psnr_vs(video, dense_video, fps)
    result["lag1"], result["flips"] = xshift_stats(video)
    dense_lag1, dense_flips = xshift_stats(dense_video)
    result["dense_lag1"], result["dense_flips"] = dense_lag1, dense_flips

    results_path = ROOT / "results.json"
    table = json.loads(results_path.read_text()) if results_path.exists() else {}
    table.setdefault(args.model, {})[args.tag] = result
    results_path.write_text(json.dumps(table, indent=1))
    print(json.dumps({args.model: {args.tag: {k: v for k, v in result.items() if k != 'config'}}}, indent=1))


if __name__ == "__main__":
    main()
