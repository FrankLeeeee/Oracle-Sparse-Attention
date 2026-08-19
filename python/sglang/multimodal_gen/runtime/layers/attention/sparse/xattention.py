# SPDX-License-Identifier: Apache-2.0
"""X-Attention baseline — faithful port of the antidiagonal estimator.

Reproduces https://github.com/mit-han-lab/x-attention (commit e379887): the
``select_mode="inverse"`` reduction of ``xattn/src/Xattention.py::xattn_estimate``
followed by ``xattn/src/utils.py::find_blocks_chunked``. Parity against verbatim
copies of those is asserted in ``test/unit/realtime/test_xattention_parity.py``.

The estimator is a strided *reshape*, not a pooling. With stride ``S``, upstream
builds

    reshaped_key   = cat([K[a::S] for a in range(S)], dim=-1)        # [Lk/S, S·d]
    reshaped_query = cat([Q[S-1-a::S] for a in range(S)], dim=-1)    # [Lq/S, S·d]

and multiplies them. Element ``(u, v)`` of that product is

    Σ_a  q[u·S + (S−1−a)] · k[v·S + a]

which is **the antidiagonal of the ``S x S`` sub-block at ``(u, v)``** of the true
score matrix — ``S`` entries, one per row and one per column of that sub-block.
(Every term shares the global antidiagonal index ``(u+v)·S + S−1``, but that
global antidiagonal crosses other sub-blocks whose entries are not included.) So
the estimator samples ``S`` entries per sub-block and ``block/S`` sub-blocks per
axis, touching every row and every column of a block exactly once. The reduced
matrix is scaled by ``1/(√d · S)``,
softmaxed along the reduced key axis, and only then summed into
``(query block, key block)`` totals. That ordering matters: softmax-then-sum
gives an estimate of block *probability mass*, which is what the threshold rule
consumes.

An earlier version of this file summed the antidiagonals first and softmaxed the
totals. That is a different estimator — it aggregates every antidiagonal whose
index is congruent mod ``S`` into one number — and it selected noticeably
different blocks.

Two quirks of upstream's selection are reproduced deliberately, because they
change the result: the threshold is applied to ``threshold × row_sum`` rather
than to a normalized distribution, and rejected slots collapse to index 0, which
means **key block 0 is always selected**.
"""

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    SparseAttentionBackend,
    SparseAttentionCall,
    SparseAttentionExecution,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.blocks import (
    block_bounds,
    own_block_mask,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import VisibleLayout
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_block_mask,
)


class XAttentionConfig(msgspec.Struct, frozen=True):
    block: int = 128
    stride: int = 16
    # Upstream's sparsity knob: the fraction of estimated mass a query block's
    # selected key blocks must cover. Lower is sparser.
    threshold: float = 0.9


def _strided_reshape(
    tensor: torch.Tensor, *, stride: int, reverse: bool, padded_length: int
) -> torch.Tensor:
    """``[batch, len, heads, dim]`` → ``[heads, padded/stride, stride*dim]``.

    Upstream's ``cat([x[a::stride] for a in ...], dim=-1)``; ``reverse`` selects
    the query side's ``stride - 1 - a`` residue order, which is what makes each
    output element a single antidiagonal. Row ``u`` of upstream's cat is token
    block ``u*stride .. (u+1)*stride`` flattened (reversed within the block on
    the query side), so the whole reshape is one contiguous view — upstream's
    default Triton path fuses it into the GEMM the same way; spelling it as 16
    strided cat-copies (as an earlier version here did) costs ~3.5 ms per call
    on a 280k-token KV and dominated the estimator.
    """
    batch, length, heads, dim = tensor.shape
    # Batch is averaged rather than kept: the plan has no batch axis, and under
    # CFG both elements want the same blocks.
    source = (
        tensor.permute(2, 0, 1, 3).mean(dim=1)
        if batch > 1
        else tensor[0].permute(1, 0, 2)
    )
    # Zero-pad to the block grid inside the transpose copy: F.pad would copy
    # the whole KV view a second time, and this is the KV-sized data movement
    # that dominates the estimator.
    merged = tensor.new_empty(heads, padded_length, dim)
    merged[:, length:].zero_()
    merged[:, :length].copy_(source)
    blocks = merged.view(heads, padded_length // stride, stride, dim)
    if reverse:
        blocks = blocks.flip(2)
    return blocks.reshape(heads, padded_length // stride, stride * dim)


def antidiagonal_block_scores(
    *,
    query: torch.Tensor,  # [batch, q_len, heads, head_dim]
    key: torch.Tensor,  # [batch, kv_len, heads, head_dim]
    block: int,
    stride: int,
) -> torch.Tensor:
    """``[heads, q_blocks, key_blocks]`` estimated block mass, upstream's recipe.

    Rows do **not** sum to one: they sum to the number of reduced query rows per
    block, exactly as upstream's ``attn_sum`` does, because the selection rule
    normalizes by the row sum itself.
    """
    head_dim = query.shape[-1]
    q_len = query.shape[1]
    padded_q_len = -(-q_len // block) * block
    padded_kv_len = -(-key.shape[1] // block) * block
    reduced_query = _strided_reshape(
        query, stride=stride, reverse=True, padded_length=padded_q_len
    )
    reduced_key = _strided_reshape(
        key, stride=stride, reverse=False, padded_length=padded_kv_len
    )
    scores = torch.bmm(reduced_query, reduced_key.transpose(1, 2)).float()
    scores /= head_dim**0.5 * stride
    probabilities = torch.softmax(scores, dim=-1)
    # Zero the reduced rows made entirely of query padding (upstream's
    # ``sum_mask`` rule, applied in both its eager and Triton paths): the
    # zero-padded rows would otherwise contribute uniform softmax mass to the
    # last query block's scores whenever q_len is not a block multiple.
    pad_rows = (padded_q_len - q_len) // stride
    if pad_rows:
        probabilities[:, -pad_rows:, :] = 0

    rows_per_block = block // stride
    heads, reduced_q, reduced_k = probabilities.shape
    return (
        probabilities.view(
            heads,
            reduced_q // rows_per_block,
            rows_per_block,
            reduced_k // rows_per_block,
            rows_per_block,
        )
        .sum(dim=-1)
        .sum(dim=-2)
    )


def select_blocks_by_cumulative_mass(
    scores: torch.Tensor, *, threshold: float
) -> torch.Tensor:
    """Upstream ``find_blocks_chunked`` with ``causal=False``.

    Takes key blocks in descending estimated mass until the *exclusive*
    cumulative sum reaches ``threshold`` of the row total. Key block 0 is always
    selected — upstream folds rejected slots onto index 0 before scattering, and
    that side effect is load-bearing enough in practice to keep.
    """
    scores = scores.double()
    required = scores.sum(dim=-1, keepdim=True) * threshold
    ordered, order = torch.sort(scores, dim=-1, descending=True)
    exclusive = torch.cat(
        [torch.zeros_like(ordered[..., :1]), ordered[..., :-1]], dim=-1
    ).cumsum(dim=-1)
    keep_ordered = exclusive < required
    order = torch.where(keep_ordered, order, torch.zeros_like(order))
    keep = torch.zeros_like(scores, dtype=torch.bool)
    return keep.scatter(-1, order, True)


class XAttention(SparseAttentionBackend):
    name = "xattention"

    def __init__(self, config: XAttentionConfig) -> None:
        super().__init__()
        self._config = config

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        config = self._config
        q_len = call.query.shape[1]
        kv_len = call.key.shape[1]
        if kv_len <= q_len:
            return None  # the first chunk sees only itself

        scores = antidiagonal_block_scores(
            query=call.query,
            key=call.key,
            block=config.block,
            stride=config.stride,
        )
        keep = select_blocks_by_cumulative_mass(scores, threshold=config.threshold)

        device = call.query.device
        q_lo, q_hi = block_bounds(q_len, config.block, device=device)
        k_lo, k_hi = block_bounds(kv_len, config.block, device=device)
        keep |= own_block_mask(
            q_lo=q_lo,
            q_hi=q_hi,
            k_lo=k_lo,
            k_hi=k_hi,
            query_offset_in_view=kv_len - q_len,
        )[None]
        if bool(keep.all()):
            return None
        plan = plan_from_block_mask(
            keep, block_n=config.block, kv_len=kv_len, block_m=config.block
        )
        return SparseAttentionExecution(
            plan=plan, query=call.query, key=call.key, value=call.value
        )
