"""JIT-compiled SM90 block-sparse attention for OSA's replicated plans.

One CTA folds two query frames over a single K/V stream (the plan row is
frame-invariant), halving K/V DRAM traffic relative to the Triton kernel;
both matmuls run as SM90 GMMA. bf16, head_dim 128 only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_osa_module() -> Module:
    with override_jit_cuda_arch(9, 0, "a"):
        return load_jit(
            "osa_block_sparse_sm90",
            cuda_files=["diffusion/osa_block_sparse_sm90.cuh"],
            cuda_wrappers=[("run", "osa_sm90::osa_block_sparse")],
            extra_cuda_cflags=[
                "-O3",
                "-DNDEBUG",
                "-DCUTE_USE_PACKED_TUPLE=1",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "--use_fast_math",
            ],
            extra_dependencies=["cutlass"],
        )


def is_osa_cuda_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def osa_block_sparse_attention(
    *,
    query: torch.Tensor,  # [q_len, heads, dim] bf16
    key: torch.Tensor,  # [kv_len, heads, dim] bf16
    value: torch.Tensor,
    starts: torch.Tensor,  # [heads, q_tiles_per_frame, n_blocks] int32
    q_tiles_per_frame: int,
    num_q_frames: int,
    frame_seqlen: int,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.dtype != torch.bfloat16:
        raise RuntimeError(f"osa_block_sparse: bf16 only, got {query.dtype}")
    if out is None:
        out = torch.empty_like(query)
    module = _jit_osa_module()
    module.run(
        out,
        query,
        key,
        value,
        starts,
        q_tiles_per_frame,
        num_q_frames,
        frame_seqlen,
        softmax_scale,
    )
    return out
