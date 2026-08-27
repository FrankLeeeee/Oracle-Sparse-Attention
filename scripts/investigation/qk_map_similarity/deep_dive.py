# SPDX-License-Identifier: Apache-2.0
"""Sparse-opportunity deep dive on the dense Self-Forcing Q/K dumps.

    python deep_dive.py --run p1 [--step 3] [--device cuda]

Needs the full-chunk capture (``run.py --spec all9 --chunks 0,1,2,3,4,5,6``).
Five measurements, each answering one design question for sparse attention on
causal video DiTs:

``ref_matrix``
    mean cosine of chunk C's self maps vs chunk c's frame-pair maps, for every
    (C, c) — does calibrating later (or re-calibrating periodically) help,
    and how fast does a calibration go stale?
``mass_transfer``
    the user's oracle-recall experiment: from the chunk-0 self map
    ``A_{0,i,i}``, take each query token's top-p% key positions (frame-relative
    indices), replicate them over every visible frame while generating chunk
    c, and measure the softmax mass they capture. Mass near 1 means frozen
    per-query positions are enough for sparse attention. A same-chunk
    "refreshed" variant separates pattern staleness from pattern inadequacy.
``local_window``
    mass within Chebyshev radius r of the query's own (y, x) grid position,
    replicated over all frames — how far a purely geometric (zero-calibration)
    local pattern gets, per head.
``frame_mass``
    per-key-frame mass distribution at the last chunk — how many whole frames
    a frame-granular policy needs for 90% of the mass, per head.
``step_consistency``
    mean frame-pair-map cosine of denoising steps 0-2 vs step 3 — whether a
    per-chunk plan can be computed once and reused across steps.

Output: results/investigation/qk_map_similarity/deep_dive/<run>/*.json + figures.
"""

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from run import SPEC_SETS  # noqa: E402

ROOT = results_dir("qk_map_similarity")
ALL_CHUNKS = tuple(range(7))
SPECS = sorted(SPEC_SETS["all9"], key=lambda s: (s["layer"], s["head"]))
TOP_FRACTIONS = (0.05, 0.10, 0.20)
RADII = (0, 1, 2, 4, 9, 16)


def load(run_dir: pathlib.Path, layer: int, head: int, chunk: int, step: int) -> dict:
    data = np.load(run_dir / "qk" / f"qk_L{layer:02d}_c{chunk}_s{step}.npz")
    column = list(data["head_ids"]).index(head)
    return {
        "query": torch.from_numpy(data["query"][:, column]).float(),
        "key": torch.from_numpy(data["key"][:, column]).float(),
        "frame_seqlen": int(data["frame_seqlen"]),
        "grid": (int(data["grid_height"]), int(data["grid_width"])),
    }


def pair_softmax(q_frame: torch.Tensor, k_frame: torch.Tensor) -> torch.Tensor:
    scale = q_frame.shape[-1] ** -0.5
    return torch.softmax((q_frame @ k_frame.T) * scale, dim=-1)


def full_softmax(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    return torch.softmax((query @ key.T) * scale, dim=-1)


def q_frame(data: dict, i: int) -> torch.Tensor:
    t = data["frame_seqlen"]
    return data["query"][i * t : (i + 1) * t]


def k_frame(data: dict, j: int) -> torch.Tensor:
    t = data["frame_seqlen"]
    return data["key"][j * t : (j + 1) * t]


def self_map(data: dict, i: int) -> torch.Tensor:
    """``A_{i, self}``: the chunk's own frames are the newest 3 of the cache."""
    key_frames = data["key"].shape[0] // data["frame_seqlen"]
    return pair_softmax(q_frame(data, i), k_frame(data, key_frames - 3 + i))


@torch.no_grad()
def ref_matrix(run_dir, spec, step, device) -> list[list[float]]:
    """``M[C][c]``: mean cos of chunk C's self maps vs chunk c's pair maps."""
    per_chunk = {
        c: {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in load(run_dir, spec["layer"], spec["head"], c, step).items()
        }
        for c in ALL_CHUNKS
    }
    refs = {}  # i -> [num_refs, T*T] normalized
    for i in range(3):
        stack = torch.stack(
            [self_map(per_chunk[C], i).flatten() for C in ALL_CHUNKS]
        )
        refs[i] = stack / stack.norm(dim=1, keepdim=True)
    sums = torch.zeros(len(ALL_CHUNKS), len(ALL_CHUNKS), device=device)
    counts = torch.zeros(len(ALL_CHUNKS), device=device)
    for c in ALL_CHUNKS:
        data = per_chunk[c]
        key_frames = data["key"].shape[0] // data["frame_seqlen"]
        for i in range(3):
            for j in range(key_frames):
                flat = pair_softmax(q_frame(data, i), k_frame(data, j)).flatten()
                sums[:, c] += refs[i] @ (flat / flat.norm())
        counts[c] = 3 * key_frames
    return (sums / counts[None, :]).cpu().tolist()


@torch.no_grad()
def mass_transfer(run_dir, spec, step, device) -> dict:
    """Frozen chunk-0 top-p% per-query positions, replicated over all frames."""
    reference = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in load(run_dir, spec["layer"], spec["head"], 0, step).items()
    }
    t = reference["frame_seqlen"]
    frozen_topk = {}  # (i, fraction) -> [T, k] frame-relative indices
    for i in range(3):
        ref_map = self_map(reference, i)
        for fraction in TOP_FRACTIONS:
            k = round(fraction * t)
            frozen_topk[(i, fraction)] = ref_map.topk(k, dim=1).indices
    out = {f"frozen@{f:.2f}": [] for f in TOP_FRACTIONS}
    out["refreshed@0.10"] = []
    # Positions from the *previous* chunk's self map — what a scheme that plans
    # during chunk c-1's KV-cache-update forward would deploy on chunk c.
    out["prev@0.10"] = []
    previous_topk = None
    for c in ALL_CHUNKS:
        data = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in load(run_dir, spec["layer"], spec["head"], c, step).items()
        }
        num_frames = data["key"].shape[0] // t
        offsets = torch.arange(num_frames, device=device)[None, :, None] * t
        accum = {name: 0.0 for name in out}
        fresh_topk = []
        for i in range(3):
            probs = full_softmax(q_frame(data, i), data["key"])  # [T, kv]
            for fraction in TOP_FRACTIONS:
                index = (frozen_topk[(i, fraction)][:, None, :] + offsets).reshape(
                    t, -1
                )
                accum[f"frozen@{fraction:.2f}"] += float(
                    probs.gather(1, index).sum(1).mean()
                )
            fresh = self_map(data, i).topk(round(0.10 * t), dim=1).indices
            fresh_topk.append(fresh)
            index = (fresh[:, None, :] + offsets).reshape(t, -1)
            accum["refreshed@0.10"] += float(probs.gather(1, index).sum(1).mean())
            prev = fresh if previous_topk is None else previous_topk[i]
            index = (prev[:, None, :] + offsets).reshape(t, -1)
            accum["prev@0.10"] += float(probs.gather(1, index).sum(1).mean())
        previous_topk = fresh_topk
        for name, total in accum.items():
            out[name].append(round(total / 3, 4))
    return out


@torch.no_grad()
def local_window(run_dir, spec, step, device, chunk=6) -> dict:
    """Mass within Chebyshev radius r of the query's own grid position."""
    data = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in load(run_dir, spec["layer"], spec["head"], chunk, step).items()
    }
    t = data["frame_seqlen"]
    height, width = data["grid"]
    out = {r: 0.0 for r in RADII}
    for i in range(3):
        probs = full_softmax(q_frame(data, i), data["key"])
        summed = probs.view(t, -1, height, width).sum(1)  # [T, H, W]
        integral = torch.nn.functional.pad(
            summed.cumsum(1).cumsum(2), (1, 0, 1, 0)
        )  # [T, H+1, W+1]
        y = torch.arange(t, device=device) // width
        x = torch.arange(t, device=device) % width
        rows = torch.arange(t, device=device)
        for r in RADII:
            y0, y1 = (y - r).clamp(min=0), (y + r + 1).clamp(max=height)
            x0, x1 = (x - r).clamp(min=0), (x + r + 1).clamp(max=width)
            window = (
                integral[rows, y1, x1]
                - integral[rows, y0, x1]
                - integral[rows, y1, x0]
                + integral[rows, y0, x0]
            )
            out[r] += float(window.mean())
    return {str(r): round(v / 3, 4) for r, v in out.items()}


@torch.no_grad()
def frame_mass(run_dir, spec, step, device, chunk=6) -> dict:
    """Per-key-frame mass distribution and frames needed for 90%."""
    data = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in load(run_dir, spec["layer"], spec["head"], chunk, step).items()
    }
    t = data["frame_seqlen"]
    total = torch.zeros(data["key"].shape[0] // t, device=device)
    for i in range(3):
        probs = full_softmax(q_frame(data, i), data["key"])
        total += probs.view(t, -1, t).sum((0, 2))
    distribution = (total / (3 * t)).cpu()
    ranked = distribution.sort(descending=True).values
    frames_90 = int((ranked.cumsum(0) < 0.9).sum().item()) + 1
    return {
        "per_frame": [round(v, 4) for v in distribution.tolist()],
        "frames_for_90pct": frames_90,
    }


@torch.no_grad()
def step_consistency(run_dir, spec, device, chunk=6, last_step=3) -> dict:
    """Mean frame-pair-map cosine of each step vs the last step."""
    stable = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in load(run_dir, spec["layer"], spec["head"], chunk, last_step).items()
    }
    key_frames = stable["key"].shape[0] // stable["frame_seqlen"]
    out = {}
    for step in range(last_step):
        data = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in load(run_dir, spec["layer"], spec["head"], chunk, step).items()
        }
        total = 0.0
        for i in range(3):
            for j in range(key_frames):
                a = pair_softmax(q_frame(stable, i), k_frame(stable, j)).flatten()
                b = pair_softmax(q_frame(data, i), k_frame(data, j)).flatten()
                total += float((a / a.norm()) @ (b / b.norm()))
        out[f"s{step}_vs_s{last_step}"] = round(total / (3 * key_frames), 4)
    return out


def plot_ref_matrix(results: dict, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), dpi=150)
    for ax, spec in zip(axes.flat, SPECS):
        matrix = np.array(results[key_of(spec)])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(
                col, row, f"{value:.2f}", ha="center", va="center",
                fontsize=6.5, color="white" if value < 0.6 else "black",
            )
        ax.set_title(f"L{spec['layer']} · h{spec['head']}", fontsize=10)
        ax.set_xlabel("generation chunk c", fontsize=8)
        ax.set_ylabel("reference chunk C", fontsize=8)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.tick_params(labelsize=7)
    fig.colorbar(image, ax=axes, shrink=0.6, label="mean cosine")
    fig.suptitle(
        "Reference-chunk sweep: cos(self maps of C, pair maps of c), step 3",
        fontsize=12,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_mass_transfer(results: dict, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), dpi=150, sharey=True)
    for ax, spec in zip(axes.flat, SPECS):
        record = results[key_of(spec)]
        for fraction, color in zip(TOP_FRACTIONS, ("#9ecae1", "#4292c6", "#084594")):
            ax.plot(
                ALL_CHUNKS, record[f"frozen@{fraction:.2f}"],
                color=color, linewidth=1.6, label=f"frozen top {fraction:.0%}",
            )
        ax.plot(
            ALL_CHUNKS, record["refreshed@0.10"],
            color="#d62728", linewidth=1.6, linestyle="--", label="refreshed top 10%",
        )
        ax.plot(
            ALL_CHUNKS, record["prev@0.10"],
            color="#e6a23c", linewidth=1.6, linestyle=":", label="prev-chunk top 10%",
        )
        ax.set_title(f"L{spec['layer']} · h{spec['head']}", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("generation chunk c", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("captured mass", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Oracle mass recall: chunk-0 per-query top-p% frame positions "
        "replicated over all visible frames (step 3)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def plot_local_window(results: dict, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    cmap = plt.get_cmap("tab10")
    for index, spec in enumerate(SPECS):
        record = results[key_of(spec)]
        ax.plot(
            range(len(RADII)), [record[str(r)] for r in RADII],
            color=cmap(index % 10), marker="o", markersize=3, linewidth=1.5,
            label=f"L{spec['layer']}·h{spec['head']}",
        )
    densities = [min((2 * r + 1) ** 2, 3600) / 3600 for r in RADII]
    ax.set_xticks(range(len(RADII)))
    ax.set_xticklabels(
        [f"r={r}\n({d:.1%})" for r, d in zip(RADII, densities)], fontsize=8
    )
    ax.set_xlabel("Chebyshev radius around the query's own (y, x) — density")
    ax.set_ylabel("captured mass (chunk 6, step 3)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def key_of(spec: dict) -> str:
    return f"L{spec['layer']:02d}_h{spec['head']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p1")
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run
    out_dir = ROOT / "deep_dive" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    measurements = {
        "ref_matrix": lambda spec: ref_matrix(run_dir, spec, args.step, args.device),
        "mass_transfer": lambda spec: mass_transfer(
            run_dir, spec, args.step, args.device
        ),
        "local_window": lambda spec: local_window(
            run_dir, spec, args.step, args.device
        ),
        "frame_mass": lambda spec: frame_mass(run_dir, spec, args.step, args.device),
        "step_consistency": lambda spec: step_consistency(
            run_dir, spec, args.device
        ),
    }
    for name, measure in measurements.items():
        results = {}
        for spec in SPECS:
            results[key_of(spec)] = measure(spec)
            print(f"[deep] {name} {key_of(spec)}: done", flush=True)
        (out_dir / f"{name}.json").write_text(json.dumps(results, indent=2))
    plot_dir = ROOT / "plots" / args.run
    plot_ref_matrix(
        json.loads((out_dir / "ref_matrix.json").read_text()),
        plot_dir / "ref_matrix.png",
    )
    plot_mass_transfer(
        json.loads((out_dir / "mass_transfer.json").read_text()),
        plot_dir / "mass_transfer.png",
    )
    plot_local_window(
        json.loads((out_dir / "local_window.json").read_text()),
        plot_dir / "local_window.png",
    )
    print(f"[deep] results -> {out_dir}, figures -> {plot_dir}")


if __name__ == "__main__":
    main()
