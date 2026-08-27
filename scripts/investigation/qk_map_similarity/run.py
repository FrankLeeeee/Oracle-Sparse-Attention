# SPDX-License-Identifier: Apache-2.0
"""Generate the five 5-second 720p dense Self-Forcing videos and capture Q/K.

    python run.py [--gpus 2,6] [--prompts p1,p2] [--no-capture]

One exclusive-GPU ``sglang generate`` run per prompt (prompts from
``scripts/investigation/prompts.json``), dense attention throughout. Each run
also carries the Q/K dump hook (``hook/sitecustomize.py``), which saves the raw
post-RoPE query/key tensors of the study's four (layer, head) picks at the
0/33/66/100th-percentile chunks for every denoising step — the input to
``plot_maps.py`` (attention maps) and ``similarity.py`` (frame-to-frame
pattern similarity).

Output: results/investigation/qk_map_similarity/runs/<prompt>/{run.log,*.mp4,qk/}.
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paths import REPO, results_dir  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/investigation/sparse_baselines"))
from common import (  # noqa: E402
    MODELS,
    SEED,
    GpuPool,
    GpuWatchdog,
    base_env,
    compute_pids,
    parse_log,
)

ROOT = results_dir("qk_map_similarity")
PROMPTS = json.loads((HERE.parent / "prompts.json").read_text())

MODEL = "self_forcing"
DURATION_S = 5
RES = "720p"
# The study's (layer, head) picks: heads 0/1 in layer 0, head 2 in the middle
# layer, head 3 in the last layer (Self-Forcing 1.3B: 30 layers, 12 heads).
NUM_LAYERS = 30
MIDDLE_LAYER = (NUM_LAYERS - 1) // 2
LAST_LAYER = NUM_LAYERS - 1
HEAD_SPECS = (
    {"task": "2.1", "layer": 0, "head": 0},
    {"task": "2.2", "layer": 0, "head": 1},
    {"task": "2.3", "layer": MIDDLE_LAYER, "head": 2},
    {"task": "2.4", "layer": LAST_LAYER, "head": 3},
)
QKDUMP_SPEC = f"0:0,1;{MIDDLE_LAYER}:2;{LAST_LAYER}:3"
# 0/33/66/100th percentile of the 7 chunks of a 5 s video (21 latent frames,
# 3 per chunk), and every one of Self-Forcing's 4 denoising steps.
NUM_CHUNKS = 7
CHUNK_IDS = sorted({round(p / 100 * (NUM_CHUNKS - 1)) for p in (0, 33, 66, 100)})
STEP_IDS = (0, 1, 2, 3)


def run_one(
    *, prompt_id: str, prompt: str, gpu: int, port_base: int, capture: bool
) -> dict:
    spec = MODELS[MODEL]
    width, height = spec["resolutions"][RES]
    out_dir = ROOT / "runs" / prompt_id
    out_dir.mkdir(parents=True, exist_ok=True)
    qk_dir = out_dir / "qk"
    cmd = [
        "sglang",
        "generate",
        "--model-path",
        spec["path"],
        "--prompt",
        prompt,
        "--width",
        str(width),
        "--height",
        str(height),
        "--num-frames",
        str(spec["frames"][DURATION_S]),
        "--seed",
        str(SEED),
        "--save-output",
        "--master-port",
        str(port_base),
        "--scheduler-port",
        str(port_base + 1),
        "--port",
        str(port_base + 2),
    ]
    extra = {}
    if capture:
        extra = {
            "PYTHONPATH": ":".join([str(HERE / "hook"), str(REPO / "python")]),
            # Enable the probe so `record()` is called, but make its own body
            # a no-op (QK_ONLY + no QK_CHUNKS): the sitecustomize hook is the
            # only writer.
            "SGLANG_DIFFUSION_ATTENTION_MAP_DIR": str(out_dir / "probe_meta"),
            "SGLANG_DIFFUSION_ATTENTION_MAP_QK_ONLY": "1",
            "QKDUMP_DIR": str(qk_dir),
            "QKDUMP_SPEC": QKDUMP_SPEC,
            "QKDUMP_CHUNKS": ",".join(str(c) for c in CHUNK_IDS),
            "QKDUMP_STEPS": ",".join(str(s) for s in STEP_IDS),
        }
    log = out_dir / "run.log"
    started = time.time()
    parked = compute_pids(gpu)
    with open(log, "w") as handle:
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=base_env(gpu, extra),
            cwd=out_dir,
            start_new_session=True,
        )
        watchdog = GpuWatchdog(gpu, proc, preexisting=parked)
        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        finally:
            watchdog.stop()
    result = parse_log(log)
    result.update(
        returncode=proc.returncode,
        contended=watchdog.contended,
        wall_s=round(time.time() - started, 1),
        gpu=gpu,
        dumps=len(list(qk_dir.glob("qk_*.npz"))) if capture else 0,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--port-base", type=int, default=29800)
    args = parser.parse_args()

    pool = GpuPool([int(g) for g in args.gpus.split(",")])
    expected = len(CHUNK_IDS) * len(STEP_IDS) * len({s["layer"] for s in HEAD_SPECS})
    results = {}
    for index, prompt_id in enumerate(args.prompts.split(",")):
        prompt = PROMPTS[prompt_id]["prompt"]
        for attempt in range(3):
            gpu = pool.acquire()
            print(f"[{prompt_id}] attempt {attempt + 1} on gpu {gpu}", flush=True)
            try:
                result = run_one(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    gpu=gpu,
                    port_base=args.port_base + 10 * index,
                    capture=not args.no_capture,
                )
            finally:
                pool.release(gpu)
            if not result["contended"]:
                break
            print(f"[{prompt_id}] contended, re-queueing", flush=True)
        results[prompt_id] = result
        print(f"[{prompt_id}] {result}", flush=True)
        if result["returncode"] != 0:
            log_tail = (ROOT / "runs" / prompt_id / "run.log").read_text()[-3000:]
            print(log_tail, flush=True)
            raise SystemExit(f"{prompt_id} failed rc={result['returncode']}")
        if not args.no_capture and result["dumps"] != expected:
            raise SystemExit(
                f"{prompt_id}: expected {expected} Q/K dumps, got {result['dumps']}"
            )
    (ROOT / "runs" / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[done] {len(results)} runs -> {ROOT / 'runs'}")


if __name__ == "__main__":
    main()
