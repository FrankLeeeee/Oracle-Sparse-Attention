# SPDX-License-Identifier: Apache-2.0
"""Aggregate the per-call recall dumps into per-chunk tables and a figure.

    python report.py [--tags self_forcing_d0.3_165f,...]

`recall_frozen` is the attention mass OSA's chunk-0-frozen tile set captures at
that chunk; `recall_refreshed` is what the best tile set for that chunk would
capture at the same token budget. The gap between them is what re-calibrating
(满窗刷新 / 每 K chunk 重测) could recover; the decay of `recall_frozen` across
chunks is how fast the chunk-0 pattern goes stale.
"""

import argparse
import collections
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("osa_recall")
LABELS = {
    "self_forcing": "Self-Forcing (full context)",
    "causal_forcing": "Causal Forcing (21-frame window)",
    "rolling_forcing": "Rolling Forcing (rolling window)",
}


def load(tag: str) -> list[dict]:
    path = ROOT / tag / "recall.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def per_chunk(rows: list[dict]) -> list[dict]:
    by_chunk: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_chunk[row["chunk"]].append(row)
    out = []
    for chunk in sorted(by_chunk):
        group = by_chunk[chunk]
        frozen = [v for r in group for v in r["recall_frozen"]]
        refreshed = [v for r in group for v in r["recall_refreshed"]]
        whole = [v for r in group for v in r["whole_only"]]
        # Per-layer means, so the worst layer is a layer and not a single head.
        by_layer: dict[int, list[float]] = collections.defaultdict(list)
        for row in group:
            by_layer[row["layer"]].extend(row["recall_frozen"])
        out.append(
            {
                "chunk": chunk,
                "kv_frames": group[0]["kv_frames"],
                "tiles_kept": group[0]["tiles_kept"],
                "num_tiles": group[0]["num_tiles"],
                "density": group[0]["density"],
                "frozen": st.mean(frozen),
                "refreshed": st.mean(refreshed),
                "whole_only": st.mean(whole),
                "frozen_worst_layer": min(st.mean(v) for v in by_layer.values()),
            }
        )
    # Whole-kept frames are a floor neither policy can move, so the tile
    # selection is only responsible for the mass above it. `tile_eff` is the
    # fraction of the *achievable* tile mass the frozen set actually gets.
    for row in out:
        head_room = row["refreshed"] - row["whole_only"]
        got = row["frozen"] - row["whole_only"]
        row["tile_eff"] = (got / head_room) if head_room > 1e-9 else float("nan")
    return out


def table(tag: str, rows: list[dict]) -> str:
    lines = [
        f"### {LABELS.get(tag.split('_d')[0], tag)}  [{tag}]",
        "",
        f"{'chunk':>5} {'kv_f':>5} {'tiles':>7} {'density':>8} {'whole':>7}"
        f" {'frozen':>7} {'refresh':>8} {'gap':>7} {'tile_eff':>8} {'worst_L':>8}",
    ]
    for row in rows:
        lines.append(
            f"{row['chunk']:>5} {row['kv_frames']:>5}"
            f" {row['tiles_kept']:>3}/{row['num_tiles']:<3}"
            f" {row['density']:>8.3f} {row['whole_only']:>7.4f}"
            f" {row['frozen']:>7.4f} {row['refreshed']:>8.4f}"
            f" {row['refreshed'] - row['frozen']:>7.4f}"
            f" {row['tile_eff']:>8.3f}"
            f" {row['frozen_worst_layer']:>8.4f}"
        )
    return "\n".join(lines)


def plot(series: dict[str, list[dict]], out_path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = {k: v for k, v in series.items() if v}
    fig, axes = plt.subplots(
        2, len(live), figsize=(5.2 * len(live), 7.4), squeeze=False
    )
    for col, (tag, rows) in enumerate(live.items()):
        model = tag.split("_d")[0]
        chunks = [r["chunk"] for r in rows]
        frozen = [r["frozen"] for r in rows]
        refreshed = [r["refreshed"] for r in rows]
        whole = [r["whole_only"] for r in rows]
        top = axes[0][col]
        top.plot(chunks, refreshed, "--", color="#1b7f5a", label="refreshed (oracle)")
        top.plot(chunks, frozen, "-o", ms=3, color="#c2410c", label="frozen at chunk 0")
        top.plot(chunks, whole, ":", color="#64748b", label="whole frames only")
        top.fill_between(chunks, frozen, refreshed, color="#c2410c", alpha=0.15)
        top.set_title(LABELS.get(model, model), fontsize=10)
        top.set_xlabel("chunk")
        top.set_ylabel("attention mass recall")
        top.grid(alpha=0.3)
        top.legend(fontsize=8)
        bottom = axes[1][col]
        gap = [r["refreshed"] - r["frozen"] for r in rows]
        bottom.plot(chunks, gap, "-o", ms=3, color="#7c3aed")
        bottom.set_xlabel("chunk")
        bottom.set_ylabel("refreshed − frozen")
        bottom.set_title("staleness of the chunk-0 pattern", fontsize=9)
        bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", default=None)
    args = parser.parse_args()
    tags = (
        args.tags.split(",")
        if args.tags
        else sorted(p.name for p in ROOT.iterdir() if (p / "recall.jsonl").exists())
    )
    series = {}
    chunks_out = []
    for tag in tags:
        rows = per_chunk(load(tag))
        series[tag] = rows
        if rows:
            chunks_out.append(table(tag, rows))
    report = "\n\n".join(chunks_out)
    (ROOT / "summary.txt").write_text(report + "\n")
    print(report)
    (ROOT / "per_chunk.json").write_text(json.dumps(series, indent=2))
    if any(series.values()):
        plot(series, ROOT / "recall_vs_chunk.png")


if __name__ == "__main__":
    main()
