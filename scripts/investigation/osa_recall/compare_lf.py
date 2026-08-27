# SPDX-License-Identifier: Apache-2.0
"""OSA vs LightForcing attention-mass recall, per chunk at matched density.

    python compare_lf.py [--osa self_forcing_d0.3_153f] \
        [--lf sf_lf_const_d0.3,sf_lf_sched_d0.3]

Both hooks report, for every sparse call, the fraction of the dense softmax
mass the method's kept key set captures (measured on a strided query subset).
OSA rows carry ``recall_frozen`` (chunk-0-frozen pattern) and
``recall_refreshed`` (same-structure per-chunk oracle); LightForcing rows
carry ``recall`` (its per-step mean-pool top-k) and ``recall_oracle`` (exact
per-(head, query block) block top-k at the same budget). Per-call densities
ride along, so methods are compared at what they actually read.
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("osa_recall")


def per_chunk(tag: str, fields: dict[str, str]) -> dict[int, dict]:
    path = ROOT / tag / "recall.jsonl"
    rows = [json.loads(line) for line in path.open()]
    by_chunk = collections.defaultdict(list)
    for row in rows:
        by_chunk[row["chunk"]].append(row)
    out = {}
    for chunk, group in sorted(by_chunk.items()):
        entry = {"density": float(np.mean([r["density"] for r in group]))}
        for out_name, field in fields.items():
            # The exact-measurement runs carry recall_free instead of the
            # legacy replicated-oracle recall_refreshed.
            live = field if field in group[0] else "recall_free"
            entry[out_name] = float(np.mean([np.mean(r[live]) for r in group]))
        out[chunk] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osa", default="self_forcing_d0.3_153f")
    parser.add_argument("--lf", default="sf_lf_const_d0.3,sf_lf_sched_d0.3")
    parser.add_argument("--out", default="lf_vs_osa")
    args = parser.parse_args()

    osa = per_chunk(args.osa, {"recall": "recall_frozen", "oracle": "recall_refreshed"})
    lfs = {
        tag: per_chunk(tag, {"recall": "recall", "oracle": "recall_oracle"})
        for tag in args.lf.split(",")
        if (ROOT / tag / "recall.jsonl").exists()
    }

    print(f"{'chunk':>5} | {'OSA d':>6} {'frozen':>7} {'refr':>6} |", end="")
    for tag in lfs:
        short = "LF-const" if "const" in tag else "LF-sched"
        print(f" {short + ' d':>10} {'recall':>7} {'oracle':>7} |", end="")
    print()
    chunks = sorted(osa)
    for chunk in chunks:
        o = osa[chunk]
        print(
            f"{chunk:>5} | {o['density']:>6.3f} {o['recall']:>7.4f}"
            f" {o['oracle']:>6.4f} |",
            end="",
        )
        for tag, data in lfs.items():
            if chunk in data:
                l = data[chunk]
                print(
                    f" {l['density']:>10.3f} {l['recall']:>7.4f}"
                    f" {l['oracle']:>7.4f} |",
                    end="",
                )
            else:
                print(f" {'-':>10} {'-':>7} {'-':>7} |", end="")
        print()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax.plot(chunks, [osa[c]["recall"] for c in chunks], "-o", ms=3,
            color="#c2410c", label="OSA frozen")
    ax.plot(chunks, [osa[c]["oracle"] for c in chunks], "--",
            color="#c2410c", alpha=0.5, label="OSA refreshed (same structure)")
    colors = {"const": "#1d4ed8", "sched": "#0e7490"}
    for tag, data in lfs.items():
        kind = "const" if "const" in tag else "sched"
        cs = sorted(data)
        ax.plot(cs, [data[c]["recall"] for c in cs], "-s", ms=3,
                color=colors[kind], label=f"LightForcing ({kind})")
        ax.plot(cs, [data[c]["oracle"] for c in cs], "--",
                color=colors[kind], alpha=0.5, label=f"LF oracle ({kind})")
    ax.set_xlabel("chunk")
    ax.set_ylabel("attention mass recall")
    ax.set_title("recall per chunk (annotate density from table)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax2.scatter([osa[c]["density"] for c in chunks],
                [osa[c]["recall"] for c in chunks],
                s=18, color="#c2410c", label="OSA frozen")
    for tag, data in lfs.items():
        kind = "const" if "const" in tag else "sched"
        cs = sorted(data)
        ax2.scatter([data[c]["density"] for c in cs],
                    [data[c]["recall"] for c in cs],
                    s=18, marker="s", color=colors[kind], label=f"LF ({kind})")
    ax2.set_xlabel("per-call density")
    ax2.set_ylabel("recall")
    ax2.set_title("recall vs what was actually read")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out_path = ROOT / f"{args.out}.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
