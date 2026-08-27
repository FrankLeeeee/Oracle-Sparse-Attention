# SPDX-License-Identifier: Apache-2.0
"""Per-chunk recall at density 0.2 and 0.1: frozen vs replan vs LightForcing.

    python fig_lowd.py
"""

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("osa_recall")


def per_chunk(tag: str, field: str) -> dict[int, float]:
    rows = [json.loads(line) for line in (ROOT / tag / "recall.jsonl").open()]
    by_chunk = collections.defaultdict(list)
    for row in rows:
        by_chunk[row["chunk"]].append(np.mean(row[field]))
    return {chunk: float(np.mean(v)) for chunk, v in sorted(by_chunk.items())}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, tier in zip(axes, ("d02", "d01")):
        frozen = per_chunk(f"sf20x_frozen_{tier}", "recall_frozen")
        replan = per_chunk(f"sf20x_replan_{tier}", "recall_frozen")
        free = per_chunk(f"sf20x_replan_{tier}", "recall_free")
        lf = per_chunk(f"sf20_lf_{tier}", "recall")
        lfo = per_chunk(f"sf20_lf_{tier}", "recall_oracle")
        ax.plot(list(replan), list(replan.values()), "-o", ms=3,
                color="#15803d", label="OSA replan_each_chunk")
        ax.plot(list(frozen), list(frozen.values()), "-o", ms=3,
                color="#c2410c", label="OSA frozen")
        ax.plot(list(lf), list(lf.values()), "-s", ms=3,
                color="#1d4ed8", label="LightForcing")
        ax.plot(list(free), list(free.values()), "--", color="#15803d",
                alpha=0.45, label="free oracle (OSA budget)")
        ax.plot(list(lfo), list(lfo.values()), "--", color="#1d4ed8",
                alpha=0.45, label="LF oracle")
        knob = "0.2" if tier == "d02" else "0.1"
        ax.set_title(f"density {knob} (per-call, steady state)", fontsize=10)
        ax.set_xlabel("chunk")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("attention mass recall")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "lf_vs_osa_lowd.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
