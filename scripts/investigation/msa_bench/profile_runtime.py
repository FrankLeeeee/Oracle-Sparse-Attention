# SPDX-License-Identifier: Apache-2.0
"""Runtime execution profiling: MSA vs LightForcing (vs dense).

    python profile_runtime.py --stage runs     # probed real runs (b1, 5s + 20s)
    python profile_runtime.py --stage micro    # per-call component microbench
    python profile_runtime.py --stage report   # figures + summary JSON

Two views of the same question:

- ``runs``: real generations of b1 with the chunk-timing probe
  (``SGLANG_DIFFUSION_CHUNK_TIMING_DIR``): CUDA-event brackets around every
  attention module, summed per chunk and pass kind. This is the ground truth
  of where denoise time goes, per chunk, per method.
- ``micro``: steady-state ``attend()`` component breakdown on an exclusively
  idle GPU at synthetic shapes (6..81 visible frames): MSA's static kernel /
  content planning / content kernel, plan-hit vs plan-miss calls, and
  LightForcing's plan vs execution, plus effective TFLOP/s per kernel.

Output: results/investigation/msa_bench/profiling/{chunk_timing_*.json,
micro.json, profile_curves.png, profile_components.png}
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
)

from run_bench import PROMPTS, method_flags  # noqa: E402

ROOT = results_dir("msa_bench")
PROF = ROOT / "profiling"
MODEL = "self_forcing"
METHODS = ("dense", "msa", "lightforcing")
FRAME_SHAPES = (6, 12, 21, 41, 81)


def probed_run(*, method: str, seconds: int, gpu: int, port_base: int) -> None:
    spec = MODELS[MODEL]
    width, height = spec["resolutions"]["720p"]
    out_dir = PROF / f"run_{method}_{seconds}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = out_dir / "timing"
    cmd = [
        "sglang", "generate",
        "--model-path", spec["path"],
        "--prompt", PROMPTS["b1"]["prompt"],
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(spec["frames"][seconds]),
        "--seed", str(SEED),
        "--master-port", str(port_base),
        "--scheduler-port", str(port_base + 1),
        "--port", str(port_base + 2),
    ] + method_flags(method)
    env = base_env(gpu, {"SGLANG_DIFFUSION_CHUNK_TIMING_DIR": str(timing_dir)})
    with open(out_dir / "run.log", "w") as handle:
        proc = subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=env,
            cwd=out_dir, start_new_session=True,
        )
        watchdog = GpuWatchdog(gpu, proc, preexisting=compute_pids(gpu))
        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        finally:
            watchdog.stop()
    assert proc.returncode == 0 and not watchdog.contended, (method, seconds)
    # newest chunk_timing.json of this run
    timing = sorted(timing_dir.glob("*/chunk_timing.json"), key=os.path.getmtime)[-1]
    (PROF / f"chunk_timing_{method}_{seconds}s.json").write_text(timing.read_text())
    print(f"[prof] {method} {seconds}s done", flush=True)


def stage_runs(args) -> None:
    pool = GpuPool([int(g) for g in args.gpus.split(",")])
    index = 0
    for seconds in (5, 20):
        for method in METHODS:
            target = PROF / f"chunk_timing_{method}_{seconds}s.json"
            if target.exists():
                print(f"skip {method} {seconds}s (already done)", flush=True)
                continue
            gpu = pool.acquire()
            try:
                probed_run(
                    method=method, seconds=seconds, gpu=gpu,
                    port_base=args.port_base + 10 * index,
                )
            finally:
                pool.release(gpu)
            index += 1


_MICRO_SCRIPT = r"""
import json, sys, torch
from sglang.multimodal_gen.runtime.layers.attention.sparse import build_sparse_attention_backend
from sglang.multimodal_gen.runtime.layers.attention.sparse.base import SparseAttentionCall
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import ChunkGeometry
from sglang.multimodal_gen.runtime.layers.attention.sparse.msa_kernel import (
    msa_content_attention, msa_static_attention,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (
    frame_aligned_block_bounds,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_segment_mask, sparse_attention,
)

def timed(fn, iters=40):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) / iters, 4)

T, H, D = 3600, 12, 128
device = "cuda"
msa_cfg, lf_cfg, shapes = json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3])
msa = build_sparse_attention_backend("msa", msa_cfg)
lf = build_sparse_attention_backend("lightforcing", lf_cfg)
out = {}
for frames in shapes:
    q = torch.randn(1, 3 * T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, frames * T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    geom = ChunkGeometry(frame_seqlen=T, frames_per_block=3,
                         query_token_start=(frames - 3) * T,
                         grid_height=45, grid_width=80)
    call = SparseAttentionCall(layer_index=14, query=q, key=k, value=v,
                               key_segments=((0, frames * T),), head_start=0,
                               num_local_heads=H, softmax_scale=D ** -0.5)
    entry = {}
    for name, backend in (("msa", msa), ("lf", lf)):
        backend.begin_forward(geom)
        result = backend.attend(call)
        assert result is not None
        entry[f"{name}_attend_hit_ms"] = timed(lambda b=backend: b.attend(call))
    # plan-miss: clear MSA's per-chunk plan cache before every call
    def msa_miss():
        msa._content_cache.clear()
        msa.attend(call)
    entry["msa_attend_miss_ms"] = timed(msa_miss)
    layout = msa._layout(call)
    static, content = msa._split_heads(14, head_start=0, num_local_heads=H)
    splan = msa._static_plan(14, static, layout=layout, q_len=q.shape[1], device=device)
    buf = torch.empty_like(q)
    entry["msa_static_kernel_ms"] = timed(lambda: msa_static_attention(
        query=q, key=k, value=v, out=buf, head_ids=splan.head_ids,
        frame_lo=splan.frame_lo, frame_step=splan.frame_step,
        frame_tail=splan.frame_tail, ranges=splan.ranges, counts=splan.counts,
        num_frames=frames, frame_seqlen=T, block_m=128,
        softmax_scale=call.softmax_scale))
    entry["msa_content_plan_ms"] = timed(
        lambda: msa._content_mask(call, layout, content)
    )
    mask, topk, kv_blocks = msa._content_mask(call, layout, content)
    lo, hi = frame_aligned_block_bounds(num_frames=frames, frame_seqlen=T,
                                        block=128, device=device)
    cplan = plan_from_segment_mask(mask, segment_starts=lo, segment_ends=hi, block_m=128)
    ids = torch.tensor(content, dtype=torch.int32, device=device)
    entry["msa_content_kernel_ms"] = timed(lambda: msa_content_attention(
        query=q, key=k, value=v, out=buf, head_ids=ids,
        range_starts=cplan.range_starts, range_ends=cplan.range_ends,
        range_counts=cplan.range_counts, block_m=128,
        softmax_scale=call.softmax_scale))
    entry["msa_num_static_heads"] = len(static)
    lf.begin_forward(geom)
    execution = lf.prepare(call, layout)
    entry["lf_plan_ms"] = timed(lambda: lf.prepare(call, layout))
    entry["lf_exec_kernel_ms"] = timed(lambda: sparse_attention(
        query=execution.query, key=execution.key, value=execution.value,
        plan=execution.plan, softmax_scale=call.softmax_scale))
    sq, sk, sv = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    entry["dense_sdpa_ms"] = timed(
        lambda: torch.nn.functional.scaled_dot_product_attention(sq, sk, sv)
    )
    out[str(frames)] = entry
    print(f"[micro] frames={frames} done", file=sys.stderr)
print(json.dumps(out))
"""


def stage_micro(args) -> None:
    pool = GpuPool([int(g) for g in args.gpus.split(",")])
    gpu = pool.acquire()
    try:
        msa_cfg = {
            "taxonomy_path": str(
                HERE.parent / "qk_map_similarity" / "msa_taxonomy_self_forcing.json"
            ),
            "content_density": 0.2,
        }
        lf_cfg = json.loads(
            (results_dir("sparse_baselines") / "configs.json").read_text()
        )[MODEL]["lightforcing"]["0.2"]["config"]
        env = base_env(gpu)
        env["PYTHONPATH"] = str(REPO / "python")
        proc = subprocess.run(
            [
                sys.executable, "-c", _MICRO_SCRIPT,
                json.dumps(msa_cfg), json.dumps(lf_cfg), json.dumps(FRAME_SHAPES),
            ],
            capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0:
            print(proc.stderr[-3000:])
            raise SystemExit("micro benchmark failed")
        PROF.mkdir(parents=True, exist_ok=True)
        (PROF / "micro.json").write_text(proc.stdout.strip().splitlines()[-1])
        print("[prof] micro done")
    finally:
        pool.release(gpu)


def stage_report(_args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    colors = {"dense": "#7f7f7f", "msa": "#2ca02c", "lightforcing": "#1f77b4"}
    summary: dict = {"runs": {}, "micro": {}}
    for ax, seconds in zip(axes, (5, 20)):
        for method in METHODS:
            timing = json.loads(
                (PROF / f"chunk_timing_{method}_{seconds}s.json").read_text()
            )
            chunks = [c["chunk"] for c in timing["chunks"]]
            denoise = [c["denoise"]["self_attn_ms"] for c in timing["chunks"]]
            cache = [
                c.get("cache_update", {}).get("self_attn_ms", 0.0)
                for c in timing["chunks"]
            ]
            ax.plot(chunks, denoise, color=colors[method], linewidth=1.7,
                    marker="o", markersize=3, label=method)
            summary["runs"][f"{method}_{seconds}s"] = {
                "self_attn_denoise_total_ms": round(sum(denoise), 1),
                "self_attn_cache_update_total_ms": round(sum(cache), 1),
                "forward_denoise_total_ms": round(
                    sum(c["denoise"]["forward_ms"] for c in timing["chunks"]), 1
                ),
            }
        ax.set_title(f"{seconds}s video (b1)", fontsize=10)
        ax.set_xlabel("chunk index")
        ax.set_ylabel("self-attention ms per chunk (denoise, all layers/steps)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Per-chunk self-attention time (chunk-timing probe, real runs)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PROF / "profile_curves.png")
    plt.close(fig)

    micro = json.loads((PROF / "micro.json").read_text())
    summary["micro"] = micro
    frames = [int(f) for f in micro]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    msa_parts = ("msa_static_kernel_ms", "msa_content_kernel_ms", "msa_content_plan_ms")
    part_labels = ("static kernel", "content kernel", "content planning (miss only)")
    part_colors = ("#55a868", "#2ca02c", "#c44e52")
    width = 0.38
    for index, f in enumerate(frames):
        entry = micro[str(f)]
        bottom = 0.0
        for part, color in zip(msa_parts, part_colors):
            axes[0].bar(index - width / 2, entry[part], width, bottom=bottom,
                        color=color)
            bottom += entry[part]
        axes[0].bar(index + width / 2, entry["lf_plan_ms"], width,
                    bottom=entry["lf_exec_kernel_ms"], color="#c44e52", alpha=0.55)
        axes[0].bar(index + width / 2, entry["lf_exec_kernel_ms"], width,
                    color="#1f77b4")
    axes[0].set_xticks(range(len(frames)))
    axes[0].set_xticklabels([f"{f}f" for f in frames])
    axes[0].set_ylabel("ms per attention call (layer 14)")
    axes[0].set_title("components: MSA (left bar) vs LightForcing (right bar)",
                      fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               part_colors + ("#1f77b4",)]
    axes[0].legend(handles, part_labels + ("LF exec kernel (plan hatched on top)",),
                   fontsize=7)
    axes[0].grid(alpha=0.25, axis="y")
    for name, key, color in (
        ("MSA plan-hit", "msa_attend_hit_ms", "#2ca02c"),
        ("MSA plan-miss", "msa_attend_miss_ms", "#98df8a"),
        ("LightForcing", "lf_attend_hit_ms", "#1f77b4"),
        ("dense SDPA", "dense_sdpa_ms", "#7f7f7f"),
    ):
        axes[1].plot(frames, [micro[str(f)][key] for f in frames], color=color,
                     marker="o", markersize=3, linewidth=1.7, label=name)
    axes[1].set_xlabel("visible frames")
    axes[1].set_ylabel("ms per attend() call")
    axes[1].set_title("whole-call latency vs context length", fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Per-call component microbenchmark (layer 14, exclusive GPU)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PROF / "profile_components.png")
    plt.close(fig)

    (PROF / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["runs"], indent=2))
    print(f"[prof] figures -> {PROF}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["runs", "micro", "report"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port-base", type=int, default=29990)
    args = parser.parse_args()
    PROF.mkdir(parents=True, exist_ok=True)
    {"runs": stage_runs, "micro": stage_micro, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
