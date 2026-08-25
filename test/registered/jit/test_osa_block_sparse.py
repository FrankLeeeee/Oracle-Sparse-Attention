import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _skip_unless_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("osa_block_sparse requires SM90")


def _diagonal_plan(heads, num_frames, frame_seqlen, band, whole, qt, device):
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        build_block_plan,
    )

    k_tiles = frame_seqlen // 64
    q_tiles = (frame_seqlen + qt - 1) // qt
    tiles = torch.zeros(heads, q_tiles, band, dtype=torch.int32)
    generator = torch.Generator().manual_seed(0)
    for h in range(heads):
        for q in range(q_tiles):
            centre = int(q * qt / 64)
            lo = min(max(centre - band // 2, 0), max(k_tiles - band, 0))
            near = torch.arange(lo, lo + band - 1, dtype=torch.int32)
            far = torch.randperm(k_tiles, generator=generator)[:1].to(torch.int32)
            tiles[h, q] = torch.cat([near, far])
    whole_frames = [0] + list(range(num_frames - whole + 1, num_frames)) if whole else []
    hist = [f for f in range(num_frames) if f not in set(whole_frames)]
    return build_block_plan(
        tiles=tiles.to(device),
        hist_offsets=torch.tensor(
            [f * frame_seqlen for f in hist], dtype=torch.int32, device=device
        ),
        whole_offsets=torch.tensor(
            [f * frame_seqlen for f in whole_frames], dtype=torch.int32, device=device
        ),
        frame_seqlen=frame_seqlen,
        query_tile=qt,
        key_tile=64,
        kv_len=num_frames * frame_seqlen,
    )


@pytest.mark.parametrize("num_frames,query_frames,band,whole", [
    (6, 1, 3, 0),   # fully sparse, single frame
    (6, 2, 3, 0),   # even frame fold
    (9, 3, 4, 2),   # odd frame tail + whole frames
])
def test_osa_cuda_matches_triton(num_frames, query_frames, band, whole):
    _skip_unless_sm90()
    from sglang.jit_kernel.osa_block_sparse import osa_block_sparse_attention
    from sglang.multimodal_gen.runtime.layers.attention.sparse.block_kernel import (
        block_sparse_attention,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, dim, frame_seqlen = 4, 128, 384
    plan = _diagonal_plan(heads, num_frames, frame_seqlen, band, whole, 128, device)
    q_len = query_frames * frame_seqlen
    kv_len = num_frames * frame_seqlen
    q = torch.randn(1, q_len, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, heads, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, kv_len, heads, dim, device=device, dtype=torch.bfloat16)

    ref = block_sparse_attention(query=q, key=k, value=v, plan=plan, softmax_scale=dim**-0.5)
    got = osa_block_sparse_attention(
        query=q[0],
        key=k[0],
        value=v[0],
        starts=plan.starts,
        q_tiles_per_frame=plan.q_tiles_per_frame,
        num_q_frames=query_frames,
        frame_seqlen=frame_seqlen,
        softmax_scale=dim**-0.5,
    )
    torch.testing.assert_close(got, ref[0], atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
