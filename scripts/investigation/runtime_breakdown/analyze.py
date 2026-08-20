# SPDX-License-Identifier: Apache-2.0
"""Extract per-config runtime breakdowns from the sweep runs.

For every runs/<model>/<res>_<dur>s/ config:
- timing.log (one-shot models) -> stage walltimes + e2e ("Pixel data
  generated successfully"); profile runs are never used for walltimes.
- timing_session.json + server_<res>.log (LingBot realtime) -> session wall,
  per-chunk stats, and stage walltimes sliced out of the shared server log by
  the session's start/end timestamps.
- trace/*.trace.json.gz -> GPU kernel time by category over the profiled
  window, used as *shares* to split the real denoise walltime into
  components. LingBot durations without their own trace reuse the
  resolution's 20s trace (the working window is capped, so steady-state
  per-chunk kernel mix is duration-independent).

Output: breakdown.json next to this script.
"""

import collections
import datetime
import glob
import gzip
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("runtime_breakdown")
RUNS = ROOT / "runs"

MODELS = ["self_forcing", "rolling_forcing", "longlive2", "lingbot_world_v2"]
RESOLUTIONS = ["480p", "720p"]
DURATIONS = [5, 10, 20, 30]

STAGE_BUCKETS = [
    ("DenoisingStage", "denoise_s"),
    ("DecodingStage", "vae_decode_s"),
    ("TextEncodingStage", "text_encode_s"),
    ("ImageVAEEncodingStage", "input_prep_s"),
    ("ImageEncodingStage", "input_prep_s"),
    ("LatentPreparationStage", "input_prep_s"),
    ("ConditioningStage", "input_prep_s"),
    ("InputValidationStage", "input_prep_s"),
]
STAGE_LINE = re.compile(r"\[(\w+Stage)\] finished in ([0-9.]+) seconds")
TIMESTAMP = re.compile(r"^\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\]")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _bucket(stage_name: str) -> str:
    for suffix, bucket in STAGE_BUCKETS:
        if stage_name.endswith(suffix):
            return bucket
    return "other_stages_s"


def parse_generate_log(path: pathlib.Path) -> dict:
    out: dict = {}
    text = _strip_ansi(path.read_text())
    stage_totals: dict[str, float] = collections.defaultdict(float)
    for stage_name, seconds in STAGE_LINE.findall(text):
        stage_totals[_bucket(stage_name)] += float(seconds)
    out.update({k: round(v, 2) for k, v in stage_totals.items()})
    e2e = re.findall(r"Pixel data generated successfully in ([0-9.]+) seconds", text)
    if e2e:
        out["e2e_s"] = float(e2e[-1])
    peak = re.findall(r"Max peak: ([0-9.]+) MB", text)
    if peak:
        out["peak_memory_gb"] = round(float(peak[-1]) / 1024, 1)
    return out


def _log_line_epoch(line: str, year: int) -> float | None:
    match = TIMESTAMP.match(line)
    if not match:
        return None
    month, day, hour, minute, second = (int(g) for g in match.groups())
    return datetime.datetime(year, month, day, hour, minute, second).timestamp()


def parse_realtime_session(
    session_path: pathlib.Path, server_log: pathlib.Path
) -> dict:
    session = json.loads(session_path.read_text())
    out = {
        "e2e_s": round(session["session_wall_s"], 2),
        "time_to_first_chunk_s": round(session["time_to_first_chunk_s"], 2),
        "received_chunks": session["received_chunks"],
        "frames_received": session["frames_received"],
    }
    stats = session.get("chunk_stats", [])
    if stats:
        forward_ms = [s["scheduler_forward_ms"] for s in stats]
        steady = forward_ms[2:] or forward_ms
        out["chunk_forward_avg_ms"] = round(sum(steady) / len(steady), 1)
        out["chunk_forward_total_s"] = round(sum(forward_ms) / 1e3, 2)
    # Slice this session's stage lines out of the shared server log.
    if server_log.exists():
        year = datetime.datetime.fromtimestamp(session["session_start"]).year
        start = session["session_start"] - 1.5
        end = session["session_end"] + 1.5
        stage_totals: dict[str, float] = collections.defaultdict(float)
        for line in _strip_ansi(server_log.read_text()).splitlines():
            stamp = _log_line_epoch(line, year)
            if stamp is None or not start <= stamp <= end:
                continue
            for stage_name, seconds in STAGE_LINE.findall(line):
                stage_totals[_bucket(stage_name)] += float(seconds)
        out.update({k: round(v, 2) for k, v in stage_totals.items()})
    return out


def categorize(kernel_name: str) -> str:
    name = kernel_name.lower()
    if "flash" in name or "fmha" in name or "attention" in name:
        return "attention"
    if re.search(r"gemm|matmul|cutlass|nvjet", name) and "norm" not in name:
        return "GEMM"
    if "memcpy" in name or "memset" in name:
        return "memcpy"
    return "elementwise/norm/other"


def parse_trace(trace_dir: pathlib.Path) -> dict | None:
    paths = sorted(glob.glob(str(trace_dir / "*.trace.json.gz")), key=os.path.getmtime)
    if not paths:
        return None
    with gzip.open(paths[-1], "rb") as handle:
        data = json.loads(handle.read())
    kernels = [
        event
        for event in data.get("traceEvents", [])
        if event.get("cat") in ("kernel", "gpu_memcpy") and "dur" in event
    ]
    if not kernels:
        return None
    categories: dict[str, float] = collections.defaultdict(float)
    for event in kernels:
        categories[categorize(event["name"])] += event["dur"]
    total = sum(categories.values())
    wall = max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)
    return {
        "gpu_ms": {
            k: round(v / 1e3, 1)
            for k, v in sorted(categories.items(), key=lambda kv: -kv[1])
        },
        "gpu_total_ms": round(total / 1e3, 1),
        "window_wall_ms": round(wall / 1e3, 1),
        "busy_fraction": round(total / wall, 4),
    }


def attach_denoise_split(entry: dict, trace: dict) -> None:
    shares = {k: v / trace["window_wall_ms"] for k, v in trace["gpu_ms"].items()}
    shares["host/launch gap"] = 1.0 - trace["busy_fraction"]
    if "denoise_s" in entry:
        entry["denoise_split_s"] = {
            k: round(v * entry["denoise_s"], 2) for k, v in shares.items()
        }


def main() -> None:
    result: dict = {}
    for model in MODELS:
        model_entry: dict = {}
        lingbot_trace_by_res: dict[str, dict] = {}
        for res in RESOLUTIONS:
            for duration in DURATIONS:
                config = f"{res}_{duration}s"
                config_dir = RUNS / model / config
                if not config_dir.exists():
                    continue
                entry: dict = {}
                timing_log = config_dir / "timing.log"
                session_json = config_dir / "timing_session.json"
                if session_json.exists():
                    entry.update(
                        parse_realtime_session(
                            session_json, RUNS / model / f"server_{res}.log"
                        )
                    )
                elif timing_log.exists():
                    entry.update(parse_generate_log(timing_log))
                trace = parse_trace(config_dir / "trace")
                if trace is not None:
                    entry["trace"] = trace
                    if model == "lingbot_world_v2":
                        lingbot_trace_by_res[res] = trace
                if entry:
                    model_entry[config] = entry
        # LingBot: reuse the resolution's profiled trace for its untraced
        # durations (capped window -> duration-independent kernel mix).
        for config, entry in model_entry.items():
            res = config.split("_")[0]
            trace = entry.get("trace") or (
                lingbot_trace_by_res.get(res) if model == "lingbot_world_v2" else None
            )
            if trace is not None:
                attach_denoise_split(entry, trace)
        result[model] = model_entry

    out_path = ROOT / "breakdown.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
