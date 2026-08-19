# SPDX-License-Identifier: Apache-2.0
"""Gather-then-flash execution for OSA's replicate policy.

The replicate policy reads, per head, the same within-frame tile set in every
history frame plus a few whole frames — and by construction every head keeps
the *same number* of tokens. Expressed as generic key ranges, the shared
range-walking kernel degenerates to one 64-token tile per range (no software
pipelining, ~300 TFLOP/s on H200). Two alternatives were measured at the real
720p / 20 s shapes:

- a specialized flat tile-list Triton kernel: no better (~290 TFLOP/s) — the
  per-iteration scalar indirection blocks pipelining just like ranges;
- **gather the kept K/V rows into a compact contiguous buffer and run the
  ordinary FA3 varlen kernel over it**: ~607 TFLOP/s end to end (gather cost
  included), 2.0x the range kernel. Per-head key sets become varlen batch
  entries (nheads=1, batch=heads), which FA parallelizes identically.

This module implements the second. The gather indices are frozen per
(layer, chunk) and reused across that chunk's denoising steps; the gather
itself re-runs per call because the own chunk's K/V change every step.
"""

import msgspec
import torch

from sglang.jit_kernel.flash_attention import flash_attn_varlen_func


class ReplicateGatherPlan(msgspec.Struct, frozen=True):
    """Frozen per-(layer, chunk) gather of the kept keys.

    ``indices[h]`` are the kept token positions of head ``h`` in the KV view,
    sorted ascending; every head keeps the same count, so one varlen batch of
    ``heads`` sequences covers the call. ``density`` is the exact fraction of
    the view read (for accounting, no device sync needed).
    """

    indices: torch.Tensor  # [heads, kept] int64, sorted per head
    cu_seqlens_q: torch.Tensor  # [heads + 1] int32
    cu_seqlens_k: torch.Tensor  # [heads + 1] int32
    q_len: int
    kept: int
    density: float


def build_gather_plan(
    *,
    indices: torch.Tensor,  # [heads, kept] int64, sorted per head
    q_len: int,
    kv_len: int,
) -> ReplicateGatherPlan:
    heads, kept = indices.shape
    device = indices.device
    arange = torch.arange(0, heads + 1, device=device, dtype=torch.int32)
    return ReplicateGatherPlan(
        indices=indices.contiguous(),
        cu_seqlens_q=arange * q_len,
        cu_seqlens_k=arange * kept,
        q_len=q_len,
        kept=kept,
        density=kept / kv_len,
    )


def replicate_gather_attention(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    value: torch.Tensor,
    plan: ReplicateGatherPlan,
    softmax_scale: float,
) -> torch.Tensor:
    batch, q_len, heads, head_dim = query.shape
    expanded = plan.indices[:, :, None].expand(heads, plan.kept, head_dim)
    outputs = []
    for element in range(batch):
        keys = key[element].permute(1, 0, 2)  # [heads, kv, dim]
        values = value[element].permute(1, 0, 2)
        k_compact = torch.gather(keys, 1, expanded).reshape(-1, 1, head_dim)
        v_compact = torch.gather(values, 1, expanded).reshape(-1, 1, head_dim)
        q_flat = (
            query[element].permute(1, 0, 2).reshape(-1, 1, head_dim).contiguous()
        )
        out = flash_attn_varlen_func(
            q_flat,
            k_compact,
            v_compact,
            plan.cu_seqlens_q,
            plan.cu_seqlens_k,
            q_len,
            plan.kept,
            softmax_scale=softmax_scale,
        )
        outputs.append(out.view(heads, q_len, head_dim).permute(1, 0, 2))
    return torch.stack(outputs, dim=0)
