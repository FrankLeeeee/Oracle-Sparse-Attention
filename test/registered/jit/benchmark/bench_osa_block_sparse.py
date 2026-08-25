import torch

from sglang.jit_kernel.benchmark import marker
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


def _plan(heads, num_frames, frame_seqlen, band, device):
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        build_block_plan,
    )

    k_tiles = frame_seqlen // 64
    q_tiles = (frame_seqlen + 127) // 128
    tiles = torch.zeros(heads, q_tiles, band, dtype=torch.int32)
    for h in range(heads):
        for q in range(q_tiles):
            centre = int(q * 2)
            lo = min(max(centre - band // 2, 0), max(k_tiles - band, 0))
            tiles[h, q] = torch.arange(lo, lo + band, dtype=torch.int32)
    return build_block_plan(
        tiles=tiles.to(device),
        hist_offsets=torch.arange(num_frames, dtype=torch.int32, device=device)
        * frame_seqlen,
        whole_offsets=torch.zeros(0, dtype=torch.int32, device=device),
        frame_seqlen=frame_seqlen,
        query_tile=128,
        key_tile=64,
        kv_len=num_frames * frame_seqlen,
    )


def _run(impl, q, k, v, plan, scale):
    if impl == "triton":
        from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
            block_sparse_attention,
        )

        return block_sparse_attention(
            query=q[None], key=k[None], value=v[None], plan=plan, softmax_scale=scale
        )
    from sglang.jit_kernel.osa_block_sparse import osa_block_sparse_attention

    return osa_block_sparse_attention(
        query=q,
        key=k,
        value=v,
        starts=plan.starts,
        q_tiles_per_frame=plan.q_tiles_per_frame,
        num_q_frames=3,
        frame_seqlen=plan.frame_seqlen,
        softmax_scale=scale,
    )


# Self-Forcing 720p shapes: 3 query frames of 3600 tokens, growing KV.
@marker.parametrize("kv_frames", [27, 54, 81], [27])
@marker.benchmark("impl", ["triton", "cuda_sm90"], unit="ms")
def benchmark(kv_frames: int, impl: str):
    if impl == "cuda_sm90" and torch.cuda.get_device_capability()[0] != 9:
        return marker.skip("SM90 only")
    device = torch.device("cuda")
    heads, dim, fs = 12, 128, 3600
    plan = _plan(heads, kv_frames, fs, 15, device)
    q = torch.randn(3 * fs, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(kv_frames * fs, heads, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return marker.do_bench(
        _run,
        input_args=("triton" if impl == "triton" else "cuda_sm90", q, k, v, plan, dim**-0.5),
        graph_clone_args=(1, 2, 3),
        memory_args=(q, k, v),
    )


if __name__ == "__main__":
    benchmark.run()
