# SPDX-License-Identifier: Apache-2.0
"""Render the temporal-consistency figure from similarity.py's temporal tables.

    python plot_temporal.py [--run p1] [--ref-chunk 0] [--step 3]

One panel per (layer, head) pick, ordered by depth: x = key frame j (global),
y = cos(A_{C,i,self}, A_{c,i,j}) averaged over the 3 query frames i, one line
per generation chunk c. A flat, high bundle of lines means the pattern
measured at chunk C still describes the head's attention through the whole
video; lines sinking with c mean the calibration goes stale.

Output: results/investigation/qk_map_similarity/plots/<run>/temporal_ref{C}_s{s}.png
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import results_dir  # noqa: E402

from run import EXTRA_HEAD_SPECS, HEAD_SPECS  # noqa: E402

ROOT = results_dir("qk_map_similarity")
CHUNK_COLORS = {0: "#4c72b0", 2: "#dd8452", 4: "#55a868", 6: "#c44e52"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p1")
    parser.add_argument("--ref-chunk", type=int, default=0)
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()

    specs = sorted(
        [*HEAD_SPECS, *EXTRA_HEAD_SPECS], key=lambda s: (s["layer"], s["head"])
    )
    sim_dir = ROOT / "similarity" / args.run
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), dpi=150, sharey=True)
    for ax, spec in zip(axes.flat, specs):
        name = (
            f"temporal_L{spec['layer']:02d}_h{spec['head']}"
            f"_ref{args.ref_chunk}_s{args.step}.json"
        )
        record = json.loads((sim_dir / name).read_text())
        for chunk_str, table in sorted(record["chunks"].items(), key=lambda kv: int(kv[0])):
            chunk = int(chunk_str)
            key_frames = len(table[0])
            mean_over_i = [
                sum(row[j] for row in table) / len(table) for j in range(key_frames)
            ]
            ax.plot(
                range(key_frames),
                mean_over_i,
                color=CHUNK_COLORS[chunk],
                linewidth=1.6,
                label=f"chunk {chunk}",
            )
        ax.set_title(f"L{spec['layer']} · h{spec['head']}", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("key frame j (global)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("cosine vs chunk-0 self map", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"Temporal consistency: cos(A_{{C={args.ref_chunk},i,self}}, A_{{c,i,j}}), "
        f"step {args.step}, mean over query frames",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "plots" / args.run / f"temporal_ref{args.ref_chunk}_s{args.step}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
