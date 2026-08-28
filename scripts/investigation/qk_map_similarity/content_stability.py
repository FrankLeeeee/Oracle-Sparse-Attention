# SPDX-License-Identifier: Apache-2.0
"""Cross-prompt stability of the head taxonomy — evidence for calibrate-once.

    python content_stability.py [--runs p1,p2,p3,p4,p5] [--step 3] [--chunk 6]

Needs the full-chunk capture for every run (``run.py --spec all9 --chunks
0,...,6 --prompts ...``). For each (layer, head) pick and each prompt:

``own@10%``
    frozen chunk-0 top-10% per-query positions of the *same* prompt, replicated
    over all frames of the deployment chunk — the taxonomy's headline metric.
``cross@10%``
    the same, but with positions calibrated on *another* prompt's chunk 0
    (mean over the other prompts). own ~ cross means one calibration serves
    any content — the direct test of the calibrate-once claim.
``overlap``
    mean per-query Jaccard-style overlap of the top-10% index sets between
    prompt pairs (secondary: sets can differ where the map is smooth without
    costing mass).
``local_r1 / local_r9``
    geometric-window mass (Chebyshev radius 1 / 9), per prompt.
``frames_90``
    key frames needed for 90% of the mass, per prompt.

A metric whose per-prompt spread is small (and cross ~ own) is
content-independent, so the per-head family assignment holds across content.

Output: results/investigation/qk_map_similarity/deep_dive/content_stability.json
        + plots/content_stability.png
"""

import argparse
import json
import pathlib
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from deep_dive import (  # noqa: E402
    SPECS,
    frame_mass,
    full_softmax,
    load,
    local_window,
    q_frame,
    self_map,
)

ROOT = results_dir("qk_map_similarity")
TOP_FRACTION = 0.10


def to_device(data: dict, device: str) -> dict:
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()
    }


@torch.no_grad()
def chunk0_topk(run_dir, spec, step, device) -> list[torch.Tensor]:
    reference = to_device(load(run_dir, spec["layer"], spec["head"], 0, step), device)
    k = round(TOP_FRACTION * reference["frame_seqlen"])
    return [self_map(reference, i).topk(k, dim=1).indices for i in range(3)]


@torch.no_grad()
def replicated_mass(data: dict, topk: list[torch.Tensor], device) -> float:
    t = data["frame_seqlen"]
    num_frames = data["key"].shape[0] // t
    offsets = torch.arange(num_frames, device=device)[None, :, None] * t
    total = 0.0
    for i in range(3):
        probs = full_softmax(q_frame(data, i), data["key"])
        index = (topk[i][:, None, :] + offsets).reshape(t, -1)
        total += float(probs.gather(1, index).sum(1).mean())
    return total / 3


@torch.no_grad()
def topk_overlap(a: list[torch.Tensor], b: list[torch.Tensor], t: int) -> float:
    total = 0.0
    for i in range(3):
        mask_a = torch.zeros(t, t, dtype=torch.bool, device=a[i].device)
        mask_b = torch.zeros_like(mask_a)
        mask_a.scatter_(1, a[i], True)
        mask_b.scatter_(1, b[i], True)
        total += float((mask_a & mask_b).sum(1).float().mean() / a[i].shape[1])
    return total / 3


def key_of(spec: dict) -> str:
    return f"L{spec['layer']:02d}_h{spec['head']}"


def plot(results: dict, runs: list[str], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    labels = [key_of(spec).replace("_", "·") for spec in SPECS]
    x = range(len(SPECS))
    panels = (
        ("own@10%", "frozen@10% mass, chunk 6", axes[0]),
        ("local_r9", "local-window mass (r=9), chunk 6", axes[1]),
        ("frames_90", "frames for 90% mass (of 21)", axes[2]),
    )
    for name, title, ax in panels:
        for index, spec in enumerate(SPECS):
            values = [results[key_of(spec)][run][name] for run in runs]
            ax.plot([index] * len(values), values, "o", color="#4c72b0",
                    markersize=4, alpha=0.7)
            if name == "own@10%":
                cross = [results[key_of(spec)][run]["cross@10%"] for run in runs]
                ax.plot([index] * len(cross), cross, "x", color="#d62728",
                        markersize=5, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.grid(alpha=0.25)
        if name != "frames_90":
            ax.set_ylim(0, 1.02)
    axes[0].legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color="#4c72b0",
                       label="own calibration (5 prompts)"),
            plt.Line2D([], [], marker="x", ls="", color="#d62728",
                       label="cross-prompt calibration"),
        ],
        fontsize=7, loc="center left",
    )
    fig.suptitle(
        "Content independence: taxonomy metrics across the five prompts (step 3)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="p1,p2,p3,p4,p5")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=6)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    runs = args.runs.split(",")

    results: dict[str, dict] = {}
    for spec in SPECS:
        topk = {
            run: chunk0_topk(ROOT / "runs" / run, spec, args.step, args.device)
            for run in runs
        }
        per_run: dict[str, dict] = {}
        for run in runs:
            run_dir = ROOT / "runs" / run
            data = to_device(
                load(run_dir, spec["layer"], spec["head"], args.chunk, args.step),
                args.device,
            )
            others = [o for o in runs if o != run]
            window = local_window(
                run_dir, spec, args.step, args.device, chunk=args.chunk
            )
            per_run[run] = {
                "own@10%": round(replicated_mass(data, topk[run], args.device), 4),
                "cross@10%": round(
                    sum(replicated_mass(data, topk[o], args.device) for o in others)
                    / len(others),
                    4,
                ),
                "overlap": round(
                    sum(
                        topk_overlap(topk[run], topk[o], data["frame_seqlen"])
                        for o in others
                    )
                    / len(others),
                    4,
                ),
                "local_r1": window["1"],
                "local_r9": window["9"],
                "frames_90": frame_mass(
                    run_dir, spec, args.step, args.device, chunk=args.chunk
                )["frames_for_90pct"],
            }
        results[key_of(spec)] = per_run
        print(f"[stab] {key_of(spec)}: done", flush=True)

    out = ROOT / "deep_dive" / "content_stability.json"
    out.write_text(json.dumps({"chunk": args.chunk, "step": args.step,
                               "runs": runs, "picks": results}, indent=2))
    plot(results, runs, ROOT / "plots" / "content_stability.png")
    print(f"[stab] wrote {out}")


if __name__ == "__main__":
    main()
