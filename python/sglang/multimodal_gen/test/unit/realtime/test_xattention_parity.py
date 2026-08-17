# SPDX-License-Identifier: Apache-2.0
"""X-Attention: parity against the upstream estimator and block selection.

Compared against ``reference/xattention_reference.py``, which holds verbatim
copies of ``xattn_estimate`` and ``find_blocks_chunked`` from
mit-han-lab/x-attention @ e379887. Upstream's Triton path self-disables on this
device (it name-checks for ``"100"`` in the device name, and this is an H200), so
the reference runs its pure-PyTorch branch, and the stubs in the reference module
turn any accidental Triton call into an assertion failure rather than a silent
divergence.

Shapes are chosen so upstream does no padding — ``chunk_size == q_len`` and
``kv_len`` a whole number of chunks. That matters: upstream's non-causal branch
softmaxes over padded key positions too, and zero-padded keys score 0, which
would put a chunk-sized lump of probability mass into the comparison and make any
mismatch impossible to attribute.

What this file does *not* check is the sparse attention kernel itself. Upstream
executes its selection with ``block_sparse_attn_func``, a CUDA extension that is
not installed and not buildable here; only the estimator and the selection rule
are reproducible. The kernel is covered against masked SDPA in
``test_sparse_attention.py`` instead.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.xattention import (
    antidiagonal_block_scores,
    select_blocks_by_cumulative_mass,
)

from .reference import xattention_reference

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

BLOCK = 128
STRIDE = 16
HEADS = 4
HEAD_DIM = 64
Q_LEN = 1024  # == chunk_size, so upstream pads nothing
KV_LEN = 4096  # a whole number of chunks


def _inputs(device, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    query = torch.randn(1, Q_LEN, HEADS, HEAD_DIM, device=device, dtype=dtype)
    key = torch.randn(1, KV_LEN, HEADS, HEAD_DIM, device=device, dtype=dtype)
    return query, key


def _upstream(query, key, *, threshold):
    """``(block_scores, selected_mask)`` from the vendored upstream estimator."""
    sums, masks = xattention_reference.xattn_estimate(
        query.transpose(1, 2),  # upstream wants [batch, heads, len, dim]
        key.transpose(1, 2),
        block_size=BLOCK,
        stride=STRIDE,
        norm=1,
        softmax=True,
        threshold=threshold,
        chunk_size=Q_LEN,
        select_mode="inverse",
        use_triton=False,
        causal=False,
        kdb=1,
    )
    return sums[0].float(), masks[0]


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_block_scores_match_upstream_estimator(dtype):
    """Our estimator must produce upstream's ``attn_sum``, element for element.

    This is the claim that matters: the reduced matrix is a strided sample of the
    true scores (one antidiagonal per stride-sized sub-block) and the softmax
    happens *before* the block sum. An implementation that sums antidiagonals
    first and softmaxes the totals passes no version of this test.
    """
    device = torch.device("cuda")
    query, key = _inputs(device, dtype=dtype)
    ours = antidiagonal_block_scores(
        query=query, key=key, block=BLOCK, stride=STRIDE
    )
    theirs, _ = _upstream(query, key, threshold=0.9)
    assert ours.shape == theirs.shape == (HEADS, Q_LEN // BLOCK, KV_LEN // BLOCK)
    tolerance = 2e-3 if dtype is torch.bfloat16 else 1e-5
    torch.testing.assert_close(ours, theirs, atol=tolerance, rtol=tolerance)


@requires_cuda
@pytest.mark.parametrize("threshold", [0.5, 0.8, 0.9, 0.95])
def test_selection_matches_upstream_find_blocks(threshold):
    """Same scores in, same blocks out — including upstream's index-0 quirk."""
    device = torch.device("cuda")
    query, key = _inputs(device, dtype=torch.float32)
    scores = antidiagonal_block_scores(
        query=query, key=key, block=BLOCK, stride=STRIDE
    )
    ours = select_blocks_by_cumulative_mass(scores, threshold=threshold)
    theirs = xattention_reference.find_blocks_chunked(
        scores[None],
        KV_LEN // BLOCK - Q_LEN // BLOCK,
        threshold,
        None,
        decoding=False,
        mode="prefill",
        causal=False,
    )[0]
    torch.testing.assert_close(ours.int(), theirs.int())


@requires_cuda
@pytest.mark.parametrize("threshold", [0.5, 0.9])
def test_end_to_end_mask_matches_upstream(threshold):
    """Estimator and selection composed, against upstream composed the same way."""
    device = torch.device("cuda")
    query, key = _inputs(device, dtype=torch.float32)
    ours = select_blocks_by_cumulative_mass(
        antidiagonal_block_scores(query=query, key=key, block=BLOCK, stride=STRIDE),
        threshold=threshold,
    )
    _, theirs = _upstream(query, key, threshold=threshold)
    torch.testing.assert_close(ours.int(), theirs.int())


@requires_cuda
def test_key_block_zero_is_always_selected():
    """Upstream folds rejected slots onto index 0, so block 0 survives everywhere.

    Pinned as a test because it is a behaviour, not an accident: it gives every
    query block an unconditional attention sink at the start of the visible
    window, and dropping it would change generated video.
    """
    device = torch.device("cuda")
    query, key = _inputs(device, dtype=torch.float32)
    scores = antidiagonal_block_scores(
        query=query, key=key, block=BLOCK, stride=STRIDE
    )
    for threshold in (0.01, 0.5, 0.99):
        keep = select_blocks_by_cumulative_mass(scores, threshold=threshold)
        assert keep[..., 0].all(), threshold


@requires_cuda
def test_reduced_element_is_the_antidiagonal_of_its_stride_subblock():
    """The structural claim behind the estimator, checked by brute force.

    Element ``(u, v)`` of the reduced matrix equals the sum of the true scores
    along the antidiagonal of the ``stride x stride`` sub-block at ``(u, v)`` —
    the entries with ``(i - u*stride) + (j - v*stride) == stride - 1``.

    The restriction to the sub-block is the part that is easy to state wrongly.
    Every term does share one global antidiagonal index ``(u+v)*stride +
    stride-1``, but that global antidiagonal also crosses other sub-blocks, and
    those terms are *not* in this element. So the estimator samples exactly
    ``stride`` of the block's entries, one per sub-block, not a whole diagonal
    band of it.
    """
    device = torch.device("cuda")
    stride, length_q, length_k, dim = 4, 16, 32, 8
    query = torch.randn(1, length_q, 1, dim, device=device)
    key = torch.randn(1, length_k, 1, dim, device=device)

    reduced_query = torch.cat(
        [query[:, stride - 1 - a :: stride] for a in range(stride)], dim=-1
    )[0, :, 0]
    reduced_key = torch.cat(
        [key[:, a::stride] for a in range(stride)], dim=-1
    )[0, :, 0]
    reduced = reduced_query @ reduced_key.T

    exact = query[0, :, 0] @ key[0, :, 0].T
    for u in range(reduced.shape[0]):
        for v in range(reduced.shape[1]):
            sub_block = exact[
                u * stride : (u + 1) * stride, v * stride : (v + 1) * stride
            ]
            expected = torch.flip(sub_block, dims=[1]).diagonal().sum()
            torch.testing.assert_close(
                reduced[u, v], expected, atol=1e-4, rtol=1e-4
            )
