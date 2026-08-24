# SPDX-License-Identifier: Apache-2.0
"""Calibrate every method's knob to the target read densities, per model.

Each (model, method, target) does a monotone secant search on the method's
scalar knob — except STA, whose knob is a discrete (kernel_t, kernel_h,
kernel_w) triple walked along a density-sorted ladder. Densities are measured
as the run-final cumulative density of a 480p / 20 s generation (LingBot: one
realtime session per server config), the same chunk trajectory as the 720p
target runs.

    python calibrate.py --model self_forcing [--gpus 0,1] [--methods ...]

Writes/updates configs.json: {model: {method: {tier: config}}}.
"""

import argparse
import fcntl
import itertools
import json
import pathlib
import queue
import sys
import threading

from common import (
    MODELS,
    ROOT,
    GpuContended,
    GpuPool,
    LingbotServer,
    method_base_config,
    run_generate,
    run_lingbot_session,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

MAX_ITERS = 5
TOLERANCE = 0.02

# Rebuild configs.json from the measurement caches without touching a GPU.
# Used to recover entries after a lost update; a missing measurement is an
# error rather than a silent GPU run.
REPLAY_ONLY = False


class CacheMiss(RuntimeError):
    pass


# Full-context Self-Forcing reaches lower cumulative densities than the
# capped-window models (their own-chunk + sink fixed keeps are a large share
# of a ~21-frame view), so it gets the extra 0.1 tier.
def targets_for(model: str) -> list[float]:
    return (
        [0.5, 0.4, 0.3, 0.2, 0.1] if model == "self_forcing" else [0.5, 0.4, 0.3, 0.2]
    )


# knob -> config; two seed values bracket each search. All knobs are monotone
# in density (band width, kept-mass thresholds, decay length, 1 - sparsity).
SCALAR_METHODS = {
    "osa": {"knob": "density", "seeds": (0.45, 0.12), "bounds": (0.02, 0.95)},
    "xattention": {"knob": "threshold", "seeds": (0.95, 0.5), "bounds": (0.01, 0.9999)},
    "svg1": {"knob": "band_frames", "seeds": (40.0, 8.0), "bounds": (0.5, 80.0)},
    "svg2": {"knob": "top_p", "seeds": (0.95, 0.5), "bounds": (0.01, 0.9999)},
    "radial": {"knob": "decay_factor", "seeds": (2.0, 0.25), "bounds": (0.01, 64.0)},
    "lightforcing": {"knob": "sparsity", "seeds": (0.5, 0.9), "bounds": (0.0, 0.995)},
}


# ---------------------------------------------------------------------------
# STA ladder: discrete kernel triples sorted by predicted cumulative density
# ---------------------------------------------------------------------------


def _pick_tile(grid_h: int, grid_w: int) -> tuple[int, int]:
    from paths import REPO

    sys.path.insert(0, str(REPO / "python"))
    from sglang.multimodal_gen.runtime.layers.attention.sparse.sta import pick_tile

    return pick_tile(grid_h, grid_w)


def _chunk_visible_frames(model: str) -> list[int]:
    """Visible latent frames per chunk of the 20 s run (approximate for RF)."""
    spec = MODELS[model]
    latents = spec["latents_20s"]
    window = spec["window_frames"]
    per_chunk = 8 if model == "longlive2" else 3
    chunks = latents // per_chunk
    out = []
    for c in range(chunks):
        visible = (c + 1) * per_chunk
        if window > 0:
            visible = min(visible, window)
        out.append(visible)
    return out


def sta_candidates(model: str, *, grid_h: int, grid_w: int) -> list[tuple[dict, float]]:
    """(config, predicted cumulative density) sorted by density, balanced shapes.

    The prediction treats chunk 0 (and any all-kept chunk) as density 1.0 and
    later chunks as (kernel_t / visible) * spatial fraction — good enough to
    order the ladder; the measurement picks the actual rung.
    """
    tile_h, tile_w = _pick_tile(grid_h, grid_w)
    tiles_h, tiles_w = grid_h // tile_h, grid_w // tile_w
    visibles = _chunk_visible_frames(model)
    max_t = max(visibles)

    def predict(kt: int, kh: int, kw: int) -> float:
        spatial = (min(kh, tiles_h) * min(kw, tiles_w)) / (tiles_h * tiles_w)
        per_chunk = []
        for i, visible in enumerate(visibles):
            if i == 0:
                per_chunk.append(1.0)  # ramp-up declines to dense
                continue
            temporal = min(2 * (kt // 2) + 1, visible) / visible
            density = temporal * spatial
            per_chunk.append(1.0 if density >= 0.999 else density)
        return sum(per_chunk) / len(per_chunk)

    candidates: list[tuple[dict, float, float]] = []
    kts = sorted({2 * k + 1 for k in range((max_t + 1) // 2 + 1)} | {max_t})
    for kt, kh, kw in itertools.product(
        kts,
        range(1, tiles_h + 1, 2) if tiles_h > 1 else [1],
        range(1, tiles_w + 1, 2) if tiles_w > 1 else [1],
    ):
        density = predict(kt, kh, kw)
        if density >= 0.999:
            continue
        spatial = (min(kh, tiles_h) * min(kw, tiles_w)) / (tiles_h * tiles_w)
        temporal = min(kt, max_t) / max_t
        import math

        balance = abs(math.log(max(temporal, 1e-6)) - math.log(max(spatial, 1e-6)))
        candidates.append(
            ({"kernel_t": kt, "kernel_h": kh, "kernel_w": kw}, density, balance)
        )

    # For near-equal densities keep the most balanced shape.
    candidates.sort(key=lambda item: (round(item[1], 2), item[2]))
    ladder: list[tuple[dict, float]] = []
    seen_bins = set()
    for config, density, _ in candidates:
        bin_key = round(density, 2)
        if bin_key in seen_bins:
            continue
        seen_bins.add(bin_key)
        ladder.append((config, density))
    ladder.sort(key=lambda item: item[1])
    return ladder


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_generate(
    model: str,
    method: str,
    config: dict,
    gpu: int,
    port_base: int,
    tag: str,
    res: str = "480p",
) -> float:
    out_dir = ROOT / model / "calibration" / method / f"{res}_{tag}"
    cache = out_dir / "result.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("config") == config and "density" in cached:
            return cached["density"]
    if REPLAY_ONLY:
        raise CacheMiss(f"{model}/{method} {tag} not measured yet")
    result = run_generate(
        model=model,
        out_dir=out_dir,
        gpu=gpu,
        port_base=port_base,
        duration=20,
        res=res,
        method=method,
        method_config=config,
    )
    if result["returncode"] != 0 or "density" not in result:
        raise RuntimeError(
            f"{model}/{method} {config} failed (rc={result['returncode']})"
        )
    cache.write_text(json.dumps(result, indent=2))
    return result["density"]


def measure_lingbot(
    method: str, config: dict, gpu: int, port_base: int, tag: str
) -> float:
    server_dir = ROOT / "lingbot_world_v2" / "calibration" / method / tag
    cache = server_dir / "result.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("config") == config and "density" in cached:
            return cached["density"]
    if REPLAY_ONLY:
        raise CacheMiss(f"lingbot/{method} {tag} not measured yet")
    with LingbotServer(
        gpu=gpu,
        port_base=port_base,
        server_dir=server_dir,
        method=method,
        method_config=config,
    ) as server:
        server.wait_ready()
        result = run_lingbot_session(
            server, out_dir=server_dir / "session", duration=20, res="480p"
        )
    if result["returncode"] != 0 or "density" not in result:
        raise RuntimeError(
            f"lingbot/{method} {config} failed (rc={result['returncode']})"
        )
    result["config"] = config
    cache.write_text(json.dumps(result, indent=2))
    return result["density"]


CALIBRATION_RES = "480p"


def measure(
    model: str, method: str, config: dict, gpu: int, port_base: int, tag: str
) -> float:
    if MODELS[model]["kind"] == "realtime":
        return measure_lingbot(method, config, gpu, port_base, tag)
    return measure_generate(
        model, method, config, gpu, port_base, tag, res=CALIBRATION_RES
    )


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------


def calibrate_scalar(
    model: str, method: str, gpu: int, port_base: int
) -> dict[str, dict]:
    spec = SCALAR_METHODS[method]
    knob, (seed_hi, seed_lo) = spec["knob"], spec["seeds"]
    lo_bound, hi_bound = spec["bounds"]
    base = method_base_config(method, model)

    observations: list[tuple[float, float]] = []  # (knob value, density)

    def run_one(value: float, tag: str) -> float:
        value = min(max(value, lo_bound), hi_bound)
        for known_value, known_density in observations:
            if abs(known_value - value) < 1e-9:
                return known_density
        density = measure(model, method, {**base, knob: value}, gpu, port_base, tag)
        observations.append((value, density))
        print(
            f"[{model}/{method}] {knob}={value:.4f} -> density {density:.3f}",
            flush=True,
        )
        return density

    run_one(seed_hi, "seed_hi")
    run_one(seed_lo, "seed_lo")

    chosen: dict[str, dict] = {}
    for target in targets_for(model):
        best = None
        for iteration in range(MAX_ITERS):
            observations.sort(key=lambda pair: pair[1])
            densities = [pair[1] for pair in observations]
            best = min(observations, key=lambda pair: abs(pair[1] - target))
            if abs(best[1] - target) <= TOLERANCE:
                break
            # Secant between the two observations bracketing the target (or
            # extrapolate from the nearest pair).
            import bisect

            index = bisect.bisect_left(densities, target)
            if index == 0:
                a, b = observations[0], observations[1]
            elif index >= len(observations):
                a, b = observations[-2], observations[-1]
            else:
                a, b = observations[index - 1], observations[index]
            if abs(b[1] - a[1]) < 1e-6:
                break
            value = a[0] + (target - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            run_one(value, f"t{target}_i{iteration}")
        best = min(observations, key=lambda pair: abs(pair[1] - target))
        # A floor: if even the sparsest observation sits well above the
        # target, the method cannot reach this tier.
        floor = min(pair[1] for pair in observations)
        if best[1] - target > 2 * TOLERANCE and floor - target > 2 * TOLERANCE:
            chosen[str(target)] = {
                "config": {**base, knob: min(observations, key=lambda p: p[1])[0]},
                "achieved_480p": floor,
                "floored": True,
            }
            continue
        chosen[str(target)] = {
            "config": {**base, knob: best[0]},
            "achieved_480p": best[1],
            "floored": False,
        }
    return chosen


def calibrate_sta(model: str, gpu: int, port_base: int) -> dict[str, dict]:
    width, height = MODELS[model]["resolutions"][CALIBRATION_RES]
    downsample = MODELS[model]["token_downsample"]
    grid_h, grid_w = height // downsample, width // downsample
    ladder = sta_candidates(model, grid_h=grid_h, grid_w=grid_w)
    base = method_base_config("sta", model)
    measured: dict[int, float] = {}

    def run_rung(index: int, tag: str) -> float:
        index = min(max(index, 0), len(ladder) - 1)
        if index in measured:
            return measured[index]
        config = {**base, **ladder[index][0]}
        density = measure(model, "sta", config, gpu, port_base, tag)
        measured[index] = density
        print(
            f"[{model}/sta] {ladder[index][0]} (predicted {ladder[index][1]:.3f}) "
            f"-> density {density:.3f}",
            flush=True,
        )
        return density

    chosen: dict[str, dict] = {}
    for target in targets_for(model):
        predictions = [density for _, density in ladder]
        import bisect

        index = min(range(len(ladder)), key=lambda i: abs(predictions[i] - target))
        for iteration in range(3):
            density = run_rung(index, f"t{target}_i{iteration}")
            if abs(density - target) <= TOLERANCE:
                break
            # Walk the ladder by the measured/predicted offset.
            offset = density - predictions[min(index, len(ladder) - 1)]
            wanted = target - offset
            new_index = min(
                range(len(ladder)), key=lambda i: abs(predictions[i] - wanted)
            )
            if new_index == index:
                new_index = index + (1 if density < target else -1)
            if not 0 <= new_index < len(ladder) or new_index in measured:
                break
            index = new_index
        best_index = min(measured, key=lambda i: abs(measured[i] - target))
        floor = min(measured.values())
        chosen[str(target)] = {
            "config": {**base, **ladder[best_index][0]},
            "achieved_480p": measured[best_index],
            "floored": bool(
                measured[best_index] - target > 2 * TOLERANCE
                and floor - target > 2 * TOLERANCE
            ),
        }
    return chosen


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def record_config(path: pathlib.Path, model: str, method: str, result: dict) -> None:
    """Merge one method's calibration into configs.json, atomically.

    Several calibration processes run at once (one model's sweep alongside the
    next model's calibration), so the file has to be re-read under an
    inter-process lock before writing: holding a copy from process start and
    writing it back loses whatever another process recorded in between — which
    is exactly how a finished model's entry once disappeared.
    """
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            configs = json.loads(path.read_text()) if path.exists() else {}
            configs.setdefault(model, {})[method] = result
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(configs, indent=2))
            temporary.replace(path)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["osa", "lightforcing", "radial", "svg1", "svg2", "xattention", "sta"],
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--port-base", type=int, default=29800)
    # STA's knob is a tile-grid kernel, and the tile grid differs per
    # resolution, so its calibration must run at the sweep's resolution;
    # scalar knobs are resolution-relative and stay on cheap 480p runs.
    parser.add_argument("--res", default="480p", choices=["480p", "720p"])
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="rebuild configs.json from cached measurements; never run a GPU",
    )
    args = parser.parse_args()
    global CALIBRATION_RES, REPLAY_ONLY
    CALIBRATION_RES = args.res
    REPLAY_ONLY = args.replay_only

    configs_path = ROOT / "configs.json"
    pool = GpuPool([int(gpu) for gpu in args.gpus.split(",")])
    work: "queue.Queue[str]" = queue.Queue()
    for method in args.methods:
        work.put(method)
    lock = threading.Lock()

    def worker(worker_index: int) -> None:
        port_base = args.port_base + 20 * worker_index
        while True:
            try:
                method = work.get_nowait()
            except queue.Empty:
                return
            for attempt in range(10):
                gpu = -1 if REPLAY_ONLY else pool.acquire()
                try:
                    if method == "sta":
                        result = calibrate_sta(args.model, gpu, port_base)
                    else:
                        result = calibrate_scalar(args.model, method, gpu, port_base)
                    with lock:
                        record_config(configs_path, args.model, method, result)
                    print(f"[{args.model}/{method}] calibrated", flush=True)
                    break
                except GpuContended:
                    # Completed measurements are cached; move to another GPU
                    # and resume the method there.
                    print(
                        f"[{args.model}/{method}] gpu{gpu} contended, moving on",
                        flush=True,
                    )
                except Exception as error:  # noqa: BLE001
                    print(f"[{args.model}/{method}] FAILED: {error}", flush=True)
                    break
                finally:
                    if gpu >= 0:
                        pool.release(gpu)

    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(args.workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"configs written to {configs_path}", flush=True)


if __name__ == "__main__":
    main()
