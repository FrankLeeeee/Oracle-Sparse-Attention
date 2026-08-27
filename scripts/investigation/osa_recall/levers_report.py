# SPDX-License-Identifier: Apache-2.0
"""Aggregate the exact-measurement lever runs into one per-chunk table.

    python levers_report.py

Columns per run: mean per-call density and mean recall_frozen; the base run
also shows recall_free (the free per-row oracle at the same budget).
"""

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("osa_recall")
TAGS = ["sf_exact_base", "sf_exact_dw", "sf_exact_sched", "sf_exact_dw_sched"]


def per_chunk(tag: str) -> dict[int, dict]:
    rows = [json.loads(line) for line in (ROOT / tag / "recall.jsonl").open()]
    by_chunk = collections.defaultdict(list)
    for row in rows:
        by_chunk[row["chunk"]].append(row)
    return {
        chunk: {
            "density": float(np.mean([r["density"] for r in group])),
            "frozen": float(np.mean([np.mean(r["recall_frozen"]) for r in group])),
            "free": float(np.mean([np.mean(r["recall_free"]) for r in group])),
        }
        for chunk, group in sorted(by_chunk.items())
    }


def main() -> None:
    data = {tag: per_chunk(tag) for tag in TAGS if (ROOT / tag / "recall.jsonl").exists()}
    chunks = sorted(next(iter(data.values())))
    header = f"{'chunk':>5} |"
    for tag in data:
        short = tag.replace("sf_exact_", "")
        header += f" {short + ' d':>10} {'frozen':>7} {'free':>6} |"
    print(header)
    for chunk in chunks:
        line = f"{chunk:>5} |"
        for tag in data:
            e = data[tag].get(chunk)
            if e:
                line += f" {e['density']:>10.3f} {e['frozen']:>7.4f} {e['free']:>6.3f} |"
            else:
                line += f" {'-':>10} {'-':>7} {'-':>6} |"
        print(line)
    print("\n== steady state (chunks >= 10) ==")
    for tag, chunks_data in data.items():
        late = [e for c, e in chunks_data.items() if c >= 10]
        print(
            f"{tag:>20}: density {np.mean([e['density'] for e in late]):.3f}"
            f"  frozen {np.mean([e['frozen'] for e in late]):.4f}"
            f"  free {np.mean([e['free'] for e in late]):.4f}"
        )


if __name__ == "__main__":
    main()
