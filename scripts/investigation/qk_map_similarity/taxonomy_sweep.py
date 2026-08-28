# SPDX-License-Identifier: Apache-2.0
"""Classify every head of every layer into the four sparse-execution families.

    python taxonomy_sweep.py [--runs p1_sweep,p4_sweep] [--tau 0.85]

Needs the sweep capture (``run.py --spec sweep --chunks 0,3,6 --steps 3``).
For each of the 30 x 12 = 360 heads and each prompt, from the exact chunk-6
last-step softmax:

- ``local_r``: window mass within Chebyshev radius r of the query's own grid
  position, replicated over all frames (r in 1/2/4/9)
- ``lastm``: mass of the newest m frames (the static short-window execution
  set: own chunk + recent), m in 3/4/5
- ``frozen10``: mass captured by chunk-0 self-map top-10% per-query positions
- ``own10 / own20``: same-chunk (refreshed) top-10% / top-20% mass

A head joins a family only if it clears the threshold on EVERY prompt
(conservative boundary, per the content-independence check), in cheapest-first
order: local (r<=4) -> short-window (m<=5) -> local r=9 -> frozen -> diffuse
(own10 <= 0.25: near-uniform rows, suited to subsampling) -> content-dependent
(runtime selection at a 20% budget). Each head also gets its policy's density
and per-prompt captured mass, so the sweep directly yields the family shares,
the fraction of heads needing runtime planning, and the fleet mean density.

Output: results/investigation/qk_map_similarity/deep_dive/taxonomy.json
        + plots/taxonomy_map.png
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
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from run import NUM_HEADS, NUM_LAYERS  # noqa: E402

ROOT = results_dir("qk_map_similarity")
RADII = (1, 2, 4, 9)
LAST_M = (3, 4, 5)
REF_CHUNK, DEPLOY_CHUNK, STEP = 0, 6, 3
FAMILIES = ("local", "shortwin", "frozen", "diffuse", "content")
FAMILY_COLORS = {
    "local": "#4c72b0",
    "shortwin": "#55a868",
    "frozen": "#8172b3",
    "diffuse": "#ccb974",
    "content": "#c44e52",
}
CONTENT_BUDGET = 0.20


def load_layer(run_dir: pathlib.Path, layer: int, chunk: int) -> dict:
    data = np.load(run_dir / "qk" / f"qk_L{layer:02d}_c{chunk}_s{STEP}.npz")
    return {
        "query": torch.from_numpy(data["query"]).float(),  # [Sq, 12, d]
        "key": torch.from_numpy(data["key"]).float(),
        "t": int(data["frame_seqlen"]),
        "grid": (int(data["grid_height"]), int(data["grid_width"])),
    }


@torch.no_grad()
def self_topk(q: torch.Tensor, k: torch.Tensor, t: int, fraction: float) -> list:
    """Per-query top positions of each query frame's self map -> 3 x [T, k]."""
    scale = q.shape[-1] ** -0.5
    frames = k.shape[0] // t
    out = []
    for i in range(3):
        q_frame = q[i * t : (i + 1) * t]
        k_frame = k[(frames - 3 + i) * t : (frames - 2 + i) * t]
        probs = torch.softmax((q_frame @ k_frame.T) * scale, dim=-1)
        out.append(probs.topk(round(fraction * t), dim=1).indices)
    return out


@torch.no_grad()
def head_metrics(
    q6: torch.Tensor, k6: torch.Tensor, ref_topk: dict, t: int, grid: tuple, device
) -> dict:
    """All classifier metrics of one head from one softmax pass per query frame."""
    scale = q6.shape[-1] ** -0.5
    height, width = grid
    frames = k6.shape[0] // t
    own_topk = {f: self_topk(q6, k6, t, f) for f in (0.10, 0.20)}
    offsets = torch.arange(frames, device=device)[None, :, None] * t
    frame_dist = torch.zeros(frames, device=device)
    window = {r: 0.0 for r in RADII}
    gathered = {"frozen10": 0.0, "own10": 0.0, "own20": 0.0}
    y = torch.arange(t, device=device) // width
    x = torch.arange(t, device=device) % width
    rows = torch.arange(t, device=device)
    for i in range(3):
        probs = torch.softmax(
            (q6[i * t : (i + 1) * t] @ k6.T) * scale, dim=-1
        )  # [T, kv]
        frame_dist += probs.view(t, frames, t).sum((0, 2))
        summed = probs.view(t, frames, height, width).sum(1)
        integral = torch.nn.functional.pad(summed.cumsum(1).cumsum(2), (1, 0, 1, 0))
        for r in RADII:
            y0, y1 = (y - r).clamp(min=0), (y + r + 1).clamp(max=height)
            x0, x1 = (x - r).clamp(min=0), (x + r + 1).clamp(max=width)
            window[r] += float(
                (
                    integral[rows, y1, x1]
                    - integral[rows, y0, x1]
                    - integral[rows, y1, x0]
                    + integral[rows, y0, x0]
                ).mean()
            )
        for name, topk in (
            ("frozen10", ref_topk[i]),
            ("own10", own_topk[0.10][i]),
            ("own20", own_topk[0.20][i]),
        ):
            index = (topk[:, None, :] + offsets).reshape(t, -1)
            gathered[name] += float(probs.gather(1, index).sum(1).mean())
    frame_dist /= 3 * t
    return {
        **{f"local_r{r}": round(v / 3, 4) for r, v in window.items()},
        **{f"lastm{m}": round(float(frame_dist[-m:].sum()), 4) for m in LAST_M},
        **{name: round(v / 3, 4) for name, v in gathered.items()},
    }


def classify(per_run: dict[str, dict], tau: float) -> tuple[str, dict]:
    """Family + policy params; a head qualifies only on EVERY prompt."""
    everywhere = lambda metric, low: all(  # noqa: E731
        m[metric] >= low for m in per_run.values()
    )
    for r in (1, 2, 4):
        if everywhere(f"local_r{r}", tau):
            return "local", {"r": r, "density": min((2 * r + 1) ** 2, 3600) / 3600}
    for m in LAST_M:
        if everywhere(f"lastm{m}", tau):
            return "shortwin", {"m": m, "density": m / 21}
    if everywhere("local_r9", tau):
        return "local", {"r": 9, "density": 361 / 3600}
    if everywhere("frozen10", tau):
        return "frozen", {"density": 0.10}
    if all(m["own10"] <= 0.25 for m in per_run.values()):
        return "diffuse", {"density": 0.10}
    return "content", {"density": CONTENT_BUDGET}


def policy_recall(family: str, params: dict, metrics: dict) -> float:
    if family == "local":
        return metrics[f"local_r{params['r']}"]
    if family == "shortwin":
        return metrics[f"lastm{params['m']}"]
    if family == "frozen":
        return metrics["frozen10"]
    if family == "diffuse":
        return metrics["own10"]
    return metrics["own20"]


def plot(records: dict, out: pathlib.Path) -> None:
    family_index = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=int)
    for key, record in records.items():
        layer, head = int(key[1:3]), int(key.split("_h")[1])
        family_index[layer, head] = FAMILIES.index(record["family"])
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.2), dpi=150, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    cmap = ListedColormap([FAMILY_COLORS[f] for f in FAMILIES])
    axes[0].imshow(family_index.T, cmap=cmap, vmin=0, vmax=len(FAMILIES) - 1,
                   aspect="auto", interpolation="nearest")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("head")
    axes[0].set_title("family map (30 layers x 12 heads)", fontsize=10)
    axes[0].set_xticks(range(0, NUM_LAYERS, 2))
    axes[0].tick_params(labelsize=8)
    bottom = np.zeros(NUM_LAYERS)
    for index, family in enumerate(FAMILIES):
        counts = (family_index == index).sum(axis=1)
        axes[1].bar(range(NUM_LAYERS), counts, bottom=bottom,
                    color=FAMILY_COLORS[family], width=0.85, label=family)
        bottom += counts
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("heads")
    axes[1].set_title("family counts per layer", fontsize=10)
    axes[1].legend(
        handles=[Patch(color=FAMILY_COLORS[f], label=f) for f in FAMILIES],
        fontsize=8, loc="upper right",
    )
    fig.suptitle(
        "360-head taxonomy (conservative: threshold met on every prompt)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="p1_sweep,p4_sweep")
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument(
        "--export",
        default="",
        help="also write a backend-consumable taxonomy JSON (family + params "
        "per head) to this path, for --sparse-attention msa",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    runs = args.runs.split(",")

    records: dict[str, dict] = {}
    for layer in range(NUM_LAYERS):
        per_head_runs: dict[int, dict[str, dict]] = {h: {} for h in range(NUM_HEADS)}
        for run in runs:
            run_dir = ROOT / "runs" / run
            ref = load_layer(run_dir, layer, REF_CHUNK)
            deploy = load_layer(run_dir, layer, DEPLOY_CHUNK)
            for head in range(NUM_HEADS):
                q0 = ref["query"][:, head].to(args.device)
                k0 = ref["key"][:, head].to(args.device)
                q6 = deploy["query"][:, head].to(args.device)
                k6 = deploy["key"][:, head].to(args.device)
                ref_topk = self_topk(q0, k0, ref["t"], 0.10)
                per_head_runs[head][run] = head_metrics(
                    q6, k6, ref_topk, deploy["t"], deploy["grid"], args.device
                )
        for head in range(NUM_HEADS):
            family, params = classify(per_head_runs[head], args.tau)
            records[f"L{layer:02d}_h{head}"] = {
                "family": family,
                **params,
                "recall": {
                    run: round(policy_recall(family, params, metrics), 4)
                    for run, metrics in per_head_runs[head].items()
                },
                "metrics": per_head_runs[head],
            }
        print(f"[tax] layer {layer}: "
              + ",".join(records[f"L{layer:02d}_h{h}"]["family"][0]
                         for h in range(NUM_HEADS)), flush=True)

    shares = {
        family: sum(1 for r in records.values() if r["family"] == family)
        for family in FAMILIES
    }
    mean_density = sum(r["density"] for r in records.values()) / len(records)
    mean_recall = {
        run: sum(r["recall"][run] for r in records.values()) / len(records)
        for run in runs
    }
    summary = {
        "tau": args.tau,
        "runs": runs,
        "family_counts": shares,
        "planning_free_share": round(1 - shares["content"] / len(records), 4),
        "mean_density": round(mean_density, 4),
        "mean_policy_recall": {k: round(v, 4) for k, v in mean_recall.items()},
    }
    out = ROOT / "deep_dive" / "taxonomy.json"
    out.write_text(json.dumps({"summary": summary, "heads": records}, indent=2))
    plot(records, ROOT / "plots" / "taxonomy_map.png")
    print(json.dumps(summary, indent=2))
    print(f"[tax] wrote {out}")
    if args.export:
        # Execution-aware gate: the backend executes a local head as a
        # row-quantized window, (2r + ~2)/grid_height of every frame — for
        # r >= 4 that is denser than the content heads' runtime-selected
        # budget, so exporting those as "local" would make them strictly
        # worse than runtime selection. They ship as content instead.
        grid_height = 45
        compact = {}
        for key, record in records.items():
            entry = {
                name: value
                for name, value in record.items()
                if name in ("family", "r", "m")
            }
            if (
                entry["family"] == "local"
                and (2 * entry["r"] + 2) / grid_height >= CONTENT_BUDGET
            ):
                entry = {"family": "content"}
            # Diffuse heads also ship as content: executing them as a frame
            # subsample is either statistically crude (few frames at short
            # context) or dense (within-frame tiles at long context), and at
            # 10/360 heads the runtime selector handles their uniform rows
            # for free under the already-amortized per-chunk planning.
            if entry["family"] == "diffuse":
                entry = {"family": "content"}
            compact[key] = entry
        export = pathlib.Path(args.export)
        export.write_text(
            json.dumps({"summary": summary, "heads": compact}, indent=2)
        )
        print(f"[tax] exported backend taxonomy -> {export}")


if __name__ == "__main__":
    main()
