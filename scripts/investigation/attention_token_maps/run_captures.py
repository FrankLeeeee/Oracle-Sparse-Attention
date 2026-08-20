# SPDX-License-Identifier: Apache-2.0
"""Capture query x key attention token maps for the four models, 720p / 20s.

Per model: one generation with the attention-map probe enabled, dumping raw
QK matrices for three chunks (early / middle / late), three denoising steps
and three layers (shallow / middle / deep), all heads. Then renders the
requested figures (5 heads x 3 steps, middle layer) with
tools/plot_attention_token_maps.py.

    python run_captures.py [--models m1,m2] [--gpus 0,1,6,7] [--plot-only]

Output lands under results/investigation/attention_token_maps/:
    captures  <model>/<ModelTag>-<timestamp>/qk_chunk_*.npz
    figures   <model>/<ModelTag>-<timestamp>/token_map_plots/*.png
"""

import argparse
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from paths import REPO, results_dir  # noqa: E402

ROOT = results_dir("attention_token_maps")
PROMPT = "A red fox trotting across a snowy field, camera slowly tracking sideways"
FIRST_FRAME = REPO / "inputs/uploads/a816103ba740450f9ded724ea1bf11e7_first_frame"
TIMEOUT_S = 14400

# 720p 20s per model. Chunk/layer/head grids follow each model's geometry:
# Self/Rolling Forcing: 27 chunks of 3 latent frames, 30 layers, 12 heads.
# LongLive-2: 15 chunks of 8 latent frames, 30 layers, 24 heads.
# LingBot v2: 27 chunks of 3 latent frames, 40 layers, 40 heads.
MODELS = {
    "self_forcing": {
        "path": "/data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers-fullctx-null",
        "kind": "generate",
        "width": 1280,
        "height": 720,
        "num_frames": 321,
        "qk_chunks": "2,13,25",
        "qk_steps": "0,1,3",
        "qk_layers": "0,15,29",
        "plot_layer": 15,
        "plot_heads": "0,3,6,9,11",
        "plot_steps": "0,1,3",
    },
    "rolling_forcing": {
        "path": "frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers",
        "kind": "generate",
        "width": 1280,
        "height": 720,
        "num_frames": 321,
        # A Rolling Forcing dump is one *window* snapshot (5 staggered-noise
        # blocks denoised jointly), keyed by the window's oldest chunk; only
        # the ramp-up windows (key 0) revisit the same key across steps. To
        # see chunk c at several of its denoising steps, dump the windows
        # starting at c-4..c: e.g. chunk 13 appears at its first / middle /
        # last step in the windows keyed 9 / 11 / 13.
        "qk_chunks": "0,9,11,13,21,23,25",
        "qk_steps": "0,2,4",
        "qk_layers": "0,15,29",
        "plot_layer": 15,
        "plot_heads": "0,3,6,9,11",
        "plot_steps": "0,2,4",
    },
    "longlive2": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "kind": "generate",
        "width": 1280,
        "height": 704,
        "num_frames": 477,
        "qk_chunks": "2,7,13",
        "qk_steps": "0,1,3",
        "qk_layers": "0,15,29",
        "plot_layer": 15,
        "plot_heads": "0,6,12,18,23",
        "plot_steps": "0,1,3",
    },
    "lingbot_world_v2": {
        "path": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "kind": "realtime",
        "width": 1280,
        "height": 720,
        "num_frames": 321,
        "qk_chunks": "2,13,25",
        "qk_steps": "0,1,3",
        "qk_layers": "0,20,39",
        "plot_layer": 20,
        "plot_heads": "0,10,20,30,39",
        "plot_steps": "0,1,3",
    },
}


def probe_env(model: str, spec: dict, gpu: int) -> dict:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO / "python"),
        FLASHINFER_DISABLE_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        SGLANG_DIFFUSION_ATTENTION_MAP_DIR=str(ROOT / model),
        SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS=spec["qk_chunks"],
        SGLANG_DIFFUSION_ATTENTION_MAP_QK_STEPS=spec["qk_steps"],
        SGLANG_DIFFUSION_ATTENTION_MAP_QK_LAYERS=spec["qk_layers"],
    )
    return env


def run_capture(model: str, gpu: int, port_base: int) -> None:
    spec = MODELS[model]
    out_dir = ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    env = probe_env(model, spec, gpu)
    log = out_dir / "capture.log"
    print(f"[gpu{gpu}] START capture {model}", flush=True)
    started = time.time()
    if spec["kind"] == "generate":
        args = [
            "sglang",
            "generate",
            "--model-path",
            spec["path"],
            "--prompt",
            PROMPT,
            "--width",
            str(spec["width"]),
            "--height",
            str(spec["height"]),
            "--num-frames",
            str(spec["num_frames"]),
            "--seed",
            "42",
            "--master-port",
            str(port_base),
            "--scheduler-port",
            str(port_base + 1),
            "--port",
            str(port_base + 2),
        ]
        with open(log, "w") as handle:
            proc = subprocess.run(
                args,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=out_dir,
                timeout=TIMEOUT_S,
            )
        returncode = proc.returncode
    else:
        port = port_base + 2
        server_args = [
            "sglang",
            "serve",
            "--model-path",
            spec["path"],
            "--master-port",
            str(port_base),
            "--scheduler-port",
            str(port_base + 1),
            "--port",
            str(port),
        ]
        with open(log, "w") as handle:
            server = subprocess.Popen(
                server_args,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=out_dir,
            )
        try:
            _wait_for_server(port, server, log)
            client_args = [
                "python",
                str(
                    REPO
                    / "scripts/investigation/runtime_breakdown/lingbot_ws_client.py"
                ),
                "--port",
                str(port),
                "--model-path",
                spec["path"],
                "--prompt",
                PROMPT,
                "--first-frame",
                str(FIRST_FRAME),
                "--size",
                f"{spec['width']}x{spec['height']}",
                "--num-frames",
                str(spec["num_frames"]),
                "--out",
                str(out_dir / "capture_session.json"),
            ]
            with open(out_dir / "client.log", "w") as handle:
                proc = subprocess.run(
                    client_args,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=TIMEOUT_S,
                )
            returncode = proc.returncode
            # The probe flushes on session dispose; give the server a moment.
            time.sleep(30)
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
    status = "OK" if returncode == 0 else f"RC={returncode}"
    print(
        f"[gpu{gpu}] DONE  capture {model} {status} "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )


def _wait_for_server(port: int, proc: subprocess.Popen, log) -> None:
    import urllib.request

    deadline = time.time() + 1800
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died, see {log}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"server on port {port} not healthy in time, see {log}")


def render_plots(model: str) -> None:
    spec = MODELS[model]
    run_dirs = sorted(
        d
        for d in (ROOT / model).iterdir()
        if d.is_dir() and list(d.glob("qk_chunk_*.npz"))
    )
    if not run_dirs:
        print(f"no qk dumps for {model}, skipping plots", flush=True)
        return
    run_dir = run_dirs[-1]
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    subprocess.run(
        [
            "python",
            "-m",
            "sglang.multimodal_gen.tools.plot_attention_token_maps",
            str(run_dir),
            "--chunks",
            spec["qk_chunks"],
            "--steps",
            spec["plot_steps"],
            "--layers",
            str(spec["plot_layer"]),
            "--heads",
            spec["plot_heads"],
        ],
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--gpus", default="0,1,6,7")
    parser.add_argument("--plot-only", action="store_true")
    # Concurrent invocations must not share a port base.
    parser.add_argument("--port-base", type=int, default=33000)
    args = parser.parse_args()

    models = args.models.split(",")
    if not args.plot_only:
        gpus = [int(g) for g in args.gpus.split(",")]
        jobs: queue.Queue = queue.Queue()
        for model in models:
            jobs.put(model)

        def worker(index: int, gpu: int) -> None:
            port_base = args.port_base + index * 20
            while True:
                try:
                    model = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    run_capture(model, gpu, port_base)
                except Exception as error:
                    print(f"[gpu{gpu}] FAIL capture {model}: {error}", flush=True)
                finally:
                    jobs.task_done()

        threads = [
            threading.Thread(target=worker, args=(i, g), daemon=True)
            for i, g in enumerate(gpus)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    for model in models:
        render_plots(model)
    print("CAPTURES DONE", flush=True)


if __name__ == "__main__":
    main()
