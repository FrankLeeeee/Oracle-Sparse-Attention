# SPDX-License-Identifier: Apache-2.0
"""How similar is the frame-to-frame section map across chunks and steps?

    python similarity.py [--runs sf_sections,sf_L3,sf_L27] [--layers 3,15,27]

Reads the ``section_L{l}_c{c}_s{s}_{group}.npy`` dumps
(``[heads, q_tile, k_tile]``) that hook/sitecustomize.py writes and quantifies
OSA's core premise — one frame-to-frame pattern per head, reusable across
denoising steps and chunk indices — with three per-head metrics on map pairs:

``cosine``
    cosine similarity of the flattened maps (each map row-normalized first, so
    heads with mass concentrated in few rows don't dominate).
``overlap@k``
    per query tile, |top-k(A) ∩ top-k(B)| / k, averaged over rows.
``transfer@k``
    the functional metric: mass of B captured by A's top-k tiles, divided by
    the mass B's own top-k captures at the same budget. 1.0 means freezing A
    loses nothing relative to re-measuring at B.

Chunk 0 has no history; its map is ``self + cross`` (exactly what OSA's
calibration folds together), so the c0→cN comparisons measure the actual
calibrate-then-deploy transfer.
"""

import argparse
import collections
import itertools
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import results_dir  # noqa: E402

ROOT = results_dir("osa_recall")
NAME = re.compile(r"section_L(\d+)_c(\d+)_s(\d+)_(\w+)\.npy")


def load_maps(run_dirs: list[pathlib.Path]) -> dict[tuple[int, int, int], np.ndarray]:
    """(layer, chunk, step) -> [heads, q_tiles, k_tiles] comparable map.

    Chunk 0's map is self + cross (the calibration measurement); later chunks
    use the history group only (the part the frozen pattern is deployed on).
    """
    groups: dict[tuple[int, int, int], dict[str, np.ndarray]] = (
        collections.defaultdict(dict)
    )
    for run in run_dirs:
        for path in sorted((run / "sections").glob("section_*.npy")):
            match = NAME.match(path.name)
            if not match:
                continue
            layer, chunk, step, group = match.groups()
            groups[(int(layer), int(chunk), int(step))][group] = np.load(path)
    maps = {}
    for key, parts in groups.items():
        if "history" in parts:
            maps[key] = parts["history"]
        elif "self" in parts and "cross" in parts:
            maps[key] = parts["self"] + parts["cross"]
    return maps


def row_normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(m.sum(axis=-1, keepdims=True), 1e-12, None)


def head_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine of row-normalized maps, per head -> [heads]."""
    fa = row_normalize(a).reshape(a.shape[0], -1)
    fb = row_normalize(b).reshape(b.shape[0], -1)
    num = (fa * fb).sum(-1)
    den = np.linalg.norm(fa, axis=-1) * np.linalg.norm(fb, axis=-1)
    return num / np.clip(den, 1e-12, None)


def topk_sets(m: np.ndarray, k: int) -> np.ndarray:
    """Boolean kept-mask of the per-row top-k -> [heads, q_tiles, k_tiles]."""
    order = np.argsort(m, axis=-1)[..., ::-1][..., :k]
    mask = np.zeros(m.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask

def head_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    inter = (topk_sets(a, k) & topk_sets(b, k)).sum(-1)  # [heads, q_tiles]
    return (inter / k).mean(-1)


def head_transfer(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Mass of B captured by A's top-k over B's own top-k mass -> [heads]."""
    got = np.where(topk_sets(a, k), b, 0.0).sum(-1)  # [heads, q_tiles]
    best = np.where(topk_sets(b, k), b, 0.0).sum(-1)
    total_got = got.sum(-1)
    total_best = best.sum(-1)
    return total_got / np.clip(total_best, 1e-12, None)


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": round(float(values.mean()), 4),
        "min": round(float(values.min()), 4),
        "argmin": int(values.argmin()),
    }


def compare(
    maps: dict, key_a: tuple, key_b: tuple, k: int
) -> dict:
    a, b = maps[key_a], maps[key_b]
    return {
        "a": key_a,
        "b": key_b,
        "cosine": summarize(head_cosine(a, b)),
        f"overlap@{k}": summarize(head_overlap(a, b, k)),
        f"transfer@{k}": summarize(head_transfer(a, b, k)),
    }


def fmt(result: dict, k: int) -> str:
    (la, ca, sa), (lb, cb, sb) = result["a"], result["b"]
    cos, ov, tr = result["cosine"], result[f"overlap@{k}"], result[f"transfer@{k}"]
    return (
        f"L{la:>2} c{ca:>2}s{sa} -> c{cb:>2}s{sb}   "
        f"cos {cos['mean']:.3f} (min {cos['min']:.3f} h{cos['argmin']})   "
        f"overlap {ov['mean']:.3f} (min {ov['min']:.3f})   "
        f"transfer {tr['mean']:.3f} (min {tr['min']:.3f} h{tr['argmin']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="sf_sections,sf_L3,sf_L27")
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--out", default="similarity.json")
    args = parser.parse_args()

    run_dirs = [ROOT / r for r in args.runs.split(",")]
    maps = load_maps(run_dirs)
    layers = sorted({k[0] for k in maps})
    print(f"loaded {len(maps)} maps, layers {layers}")
    results = []

    print(f"\n== cross-STEP, same chunk (budget k={args.k}) ==")
    for layer in layers:
        chunks = sorted({c for (l, c, s) in maps if l == layer})
        for chunk in chunks:
            steps = sorted({s for (l, c, s) in maps if (l, c) == (layer, chunk)})
            for sa, sb in itertools.combinations(steps, 2):
                r = compare(maps, (layer, chunk, sa), (layer, chunk, sb), args.k)
                results.append(r)
                print(fmt(r, args.k))

    print("\n== cross-CHUNK, last step (s3 -> s3) ==")
    for layer in layers:
        chunks = sorted({c for (l, c, s) in maps if l == layer})
        for ca, cb in itertools.combinations(chunks, 2):
            steps_a = {s for (l, c, s) in maps if (l, c) == (layer, ca)}
            steps_b = {s for (l, c, s) in maps if (l, c) == (layer, cb)}
            s = max(steps_a & steps_b)
            r = compare(maps, (layer, ca, s), (layer, cb, s), args.k)
            results.append(r)
            print(fmt(r, args.k))

    print("\n== DEPLOYMENT transfer: chunk-0 last-step calibration -> chunk c step s ==")
    for layer in layers:
        cal_steps = [s for (l, c, s) in maps if (l, c) == (layer, 0)]
        if not cal_steps:
            continue
        cal = (layer, 0, max(cal_steps))
        for key in sorted(k for k in maps if k[0] == layer and k[1] > 0):
            r = compare(maps, cal, key, args.k)
            results.append(r)
            print(fmt(r, args.k))

    out = ROOT / args.out
    out.write_text(json.dumps([
        {**r, "a": list(r["a"]), "b": list(r["b"])} for r in results
    ], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
