# SPDX-License-Identifier: Apache-2.0
"""Per-head recall: LightForcing's actual mask vs the hybrid taxonomy policies.

    python lf_compare.py [--lf-tags <tag1,tag2>] [--chunk 6] [--step 3]

LightForcing side: ``osa_recall/run.py --method lightforcing`` instruments the
real backend and reports, per (chunk, layer, step), the softmax mass its kept
block set captures for every head (``recall.jsonl``). Hybrid side: each head's
assigned-policy captured mass from ``taxonomy_sweep.py`` (same prompt, chunk 6,
step 3). Both are read masses of an actual key subset against the same dense
softmax, so the distributions are directly comparable; densities differ per
side and are annotated. Caveat: the hybrid's content-dependent heads are
scored with the same-chunk frame-replicated top-20% (a proxy for runtime
selection), not an implemented selector.

Output: deep_dive/lf_compare.json + plots/lf_compare.png
"""

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from taxonomy_sweep import FAMILIES, FAMILY_COLORS  # noqa: E402

ROOT = results_dir("qk_map_similarity")
RECALL_ROOT = results_dir("osa_recall")


def lf_recalls(tag: str, chunk: int, step: int) -> tuple[np.ndarray, float]:
    """Per-(layer, head) mask recall at (chunk, step) + mean per-call density."""
    by_layer: dict[int, list[float]] = {}
    densities = []
    for line in (RECALL_ROOT / tag / "recall.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["chunk"] == chunk and record["step"] == step:
            by_layer[record["layer"]] = record["recall"]
            densities.append(record["density"])
    assert by_layer, f"no (c{chunk}, s{step}) records in {tag}"
    layers = sorted(by_layer)
    return np.array([by_layer[l] for l in layers]), float(np.mean(densities))


def hybrid_recalls(run: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-head policy recall + family index + mean density from the taxonomy."""
    data = json.loads((ROOT / "deep_dive" / "taxonomy.json").read_text())
    heads = data["heads"]
    keys = sorted(heads)
    recalls = np.array([heads[k]["recall"][run] for k in keys])
    families = np.array([FAMILIES.index(heads[k]["family"]) for k in keys])
    return recalls, families, data["summary"]["mean_density"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lf-tags",
        default=(
            "self_forcing_lightforcing_p1_forest_d0.2_81f,"
            "self_forcing_lightforcing_p1_forest_d0.3_81f"
        ),
    )
    parser.add_argument("--run", default="p1_sweep")
    parser.add_argument("--chunk", type=int, default=6)
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()

    hybrid, families, hybrid_density = hybrid_recalls(args.run)
    curves = [(f"hybrid (mean density {hybrid_density:.2f})", hybrid, "#2ca02c")]
    # The actually-proposed system: static families as assigned, and the
    # content-dependent heads served by LightForcing's OWN selection (its
    # measured per-head recall at d0.2), planned once per chunk.
    taxonomy = json.loads((ROOT / "deep_dive" / "taxonomy.json").read_text())
    keys = sorted(taxonomy["heads"])
    lf_grid, lf_density = lf_recalls(args.lf_tags.split(",")[0], args.chunk, args.step)
    composed = np.empty(len(keys))
    composed_density = 0.0
    content_index = FAMILIES.index("content")
    for position, key in enumerate(keys):
        head = taxonomy["heads"][key]
        layer, head_id = int(key[1:3]), int(key.split("_h")[1])
        if FAMILIES.index(head["family"]) == content_index:
            composed[position] = lf_grid[layer, head_id]
            composed_density += lf_density
        else:
            composed[position] = head["recall"][args.run]
            composed_density += head["density"]
    composed_density /= len(keys)
    curves.append(
        (
            f"hybrid+LF content heads (mean density {composed_density:.2f})",
            composed,
            "#d62728",
        )
    )
    summary = {
        "hybrid": {
            "mean_density": hybrid_density,
            "mean_recall": round(float(hybrid.mean()), 4),
            "p10_recall": round(float(np.percentile(hybrid, 10)), 4),
            "share_below_0.5": round(float((hybrid < 0.5).mean()), 4),
        },
        "hybrid_lf_content": {
            "mean_density": round(composed_density, 4),
            "mean_recall": round(float(composed.mean()), 4),
            "p10_recall": round(float(np.percentile(composed, 10)), 4),
            "share_below_0.5": round(float((composed < 0.5).mean()), 4),
        },
    }
    for tag, color in zip(args.lf_tags.split(","), ("#1f77b4", "#9467bd")):
        recalls, density = lf_recalls(tag, args.chunk, args.step)
        flat = recalls.flatten()
        knob = tag.split("_d")[1].split("_")[0]
        curves.append((f"LightForcing d{knob} (per-call density {density:.2f})",
                       flat, color))
        summary[f"lf_d{knob}"] = {
            "per_call_density": round(density, 4),
            "mean_recall": round(float(flat.mean()), 4),
            "p10_recall": round(float(np.percentile(flat, 10)), 4),
            "share_below_0.5": round(float((flat < 0.5).mean()), 4),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for label, values, color in curves:
        ordered = np.sort(values)
        axes[0].plot(ordered, np.linspace(0, 1, len(ordered)), color=color,
                     linewidth=1.8, label=label)
    axes[0].set_xlabel("per-head captured mass (chunk 6, step 3, p1)")
    axes[0].set_ylabel("fraction of heads below")
    axes[0].set_title("recall CDF over the 360 heads", fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    lf_ref = curves[1][1].reshape(30, 12).flatten()
    for index, family in enumerate(FAMILIES):
        keep = families == index
        axes[1].scatter(lf_ref[keep], hybrid[keep], s=12, alpha=0.7,
                        color=FAMILY_COLORS[family], label=family)
    axes[1].plot([0, 1], [0, 1], color="gray", linewidth=0.8)
    axes[1].set_xlabel(curves[1][0])
    axes[1].set_ylabel("hybrid policy recall")
    axes[1].set_title("per-head: hybrid vs LightForcing", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    out_png = ROOT / "plots" / "lf_compare.png"
    fig.savefig(out_png)
    plt.close(fig)

    out = ROOT / "deep_dive" / "lf_compare.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[lf] wrote {out} and {out_png}")


if __name__ == "__main__":
    main()
