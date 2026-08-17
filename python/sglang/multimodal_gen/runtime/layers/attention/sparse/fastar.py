# SPDX-License-Identifier: Apache-2.0
"""FAST-AR baseline — TempCache + AnnSA, reproduced for Self-Forcing.

Reproduction of the self-attention half of "Fast Autoregressive Video Diffusion
and World Models with Temporal Cache Compression and Sparse Attention"
(arXiv:2602.01801). No code was released, so this follows the paper's
description; the two mechanisms it specifies for self-attention are both here,
and the cross-attention pruning (AnnCA) is deliberately left out so that the
comparison against the other baselines stays a comparison of *self*-attention.

**TempCache.** Video keys repeat: the same content sits at the same spatial
position in frame after frame. TempCache finds those temporal correspondences
and keeps one representative per group. The paper's Lemma 5.1 makes the merge
*exact* when the grouped keys are identical — attend to the representative with
the group's mean value and a ``log(group size)`` logit bias — so the merge is
lossless in the limit and controlled by a cosine threshold (0.9) away from it.
Here correspondence is tracked down the temporal axis at fixed spatial
position, and merged runs collapse into their newest frame.

**AnnSA.** Each query attends only to semantically matched keys, found with a
lightweight ANN rather than a score matrix: sign-of-random-projection LSH puts
queries and keys into buckets, and a query block reads the buckets its queries
landed in plus their Hamming-1 neighbours. Keys are stored bucket-sorted so
each bucket is one contiguous range for the block-sparse kernel.

Both stages depend only on the *history* keys, which do not change across the
denoising steps of a chunk, so the whole compaction is built once per
``(chunk, layer)``. The chunk's own keys stay dense and unpermuted at the end
of the key axis.
"""

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    SparseAttentionBackend,
    SparseAttentionCall,
    SparseAttentionExecution,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import VisibleLayout
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_segment_mask,
)


class FastArConfig(msgspec.Struct, frozen=True):
    block: int = 128
    # TempCache: cosine similarity above which two temporally corresponding
    # keys are treated as the same key.
    merge_threshold: float = 0.9
    # AnnSA: bits of the sign-LSH, and how many bits a probe may differ by.
    hash_bits: int = 6
    probe_hamming: int = 1
    # The paper leaves the earliest denoising steps dense; with four steps per
    # chunk that is the first one.
    dense_steps: int = 1
    seed: int = 0


class CompressedHistory(msgspec.Struct, frozen=True):
    """TempCache + AnnSA view of one layer's history keys.

    ``key``/``value`` are ``[kept, heads, head_dim]``: the bucket-sorted
    surviving history keys, *history only*. The chunk's own keys are
    concatenated fresh on every call — they are rewritten at every denoising
    step, so caching them here would make steps 2..N attend to step 1's keys.
    Heads keep different numbers of history keys, so the region is padded to
    the longest head; the padding is unreachable because a head's bucket
    segments only span its own keys.
    """

    key: torch.Tensor
    value: torch.Tensor
    logit_bias: torch.Tensor  # [heads, kept + q_len], zero over the own chunk
    bucket_starts: torch.Tensor  # [heads, 1, buckets + 1] int32
    bucket_ends: torch.Tensor  # [heads, 1, buckets + 1] int32
    projection: torch.Tensor  # [heads, head_dim, bits]
    kept: int

    def with_current_chunk(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[batch, kept + q_len, heads, head_dim]`` key/value for this step."""
        batch = key.shape[0]
        history_key = self.key[None].expand(batch, -1, -1, -1)
        history_value = self.value[None].expand(batch, -1, -1, -1)
        return (
            torch.cat([history_key, key], dim=1),
            torch.cat([history_value, value], dim=1),
        )


def temporal_merge(
    keys: torch.Tensor,  # [heads, frames, frame_seqlen, head_dim]
    values: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TempCache groups: ``(keep, group_size, merged_values)``.

    A key merges into the key at the same spatial position in the next frame
    when their cosine similarity clears ``threshold``; runs of such merges
    collapse into the run's newest frame, whose value becomes the run mean.
    """
    normed = torch.nn.functional.normalize(keys.float(), dim=-1)
    merges = (normed[:, :-1] * normed[:, 1:]).sum(-1) >= threshold  # [H, F-1, P]

    group_size = torch.ones(keys.shape[:3], device=keys.device, dtype=torch.float32)
    value_sum = values.float().clone()
    for frame in range(1, keys.shape[1]):
        carry = merges[:, frame - 1]
        group_size[:, frame] += carry * group_size[:, frame - 1]
        value_sum[:, frame] += carry[..., None] * value_sum[:, frame - 1]

    keep = torch.ones_like(group_size, dtype=torch.bool)
    keep[:, :-1] = ~merges
    merged_values = value_sum / group_size[..., None]
    return keep, group_size, merged_values.to(values.dtype)


def sign_lsh_buckets(
    vectors: torch.Tensor, projection: torch.Tensor
) -> torch.Tensor:
    """``[heads, n]`` bucket id from the sign pattern of random projections."""
    bits = (vectors.float() @ projection) > 0
    weights = 2 ** torch.arange(projection.shape[-1], device=vectors.device)
    return (bits * weights).sum(-1)


def hamming_expand(present: torch.Tensor, *, bits: int, radius: int) -> torch.Tensor:
    """OR a ``[..., buckets]`` mask with its Hamming-``radius`` neighbours."""
    expanded = present
    for _ in range(radius):
        neighbours = expanded
        bucket_ids = torch.arange(present.shape[-1], device=present.device)
        for bit in range(bits):
            neighbours = neighbours | expanded[..., bucket_ids ^ (1 << bit)]
        expanded = neighbours
    return expanded


class FastArAttention(SparseAttentionBackend):
    name = "fastar"

    def __init__(self, config: FastArConfig) -> None:
        super().__init__()
        self._config = config
        self._history: dict[int, tuple[tuple, CompressedHistory]] = {}
        self._step_of_layer: dict[int, tuple[int, int]] = {}

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        config = self._config
        q_len = call.query.shape[1]
        history_len = call.key.shape[1] - q_len
        if history_len <= 0:
            return None
        if self._advance_step(call, layout) < config.dense_steps:
            return None

        history = self._compressed_history(call, layout, history_len)
        query, permutation, keep = self._permute_and_select(call, history)
        keys, values = history.with_current_chunk(
            call.key[:, history_len:], call.value[:, history_len:]
        )
        plan = plan_from_segment_mask(
            keep,
            segment_starts=history.bucket_starts,
            segment_ends=history.bucket_ends,
            block_m=config.block,
            logit_bias=history.logit_bias,
        )
        return SparseAttentionExecution(
            plan=plan,
            query=query,
            key=keys,
            value=values,
            query_permutation=permutation,
        )

    def _advance_step(self, call: SparseAttentionCall, layout: VisibleLayout) -> int:
        chunk, step = self._step_of_layer.get(call.layer_index, (-1, -1))
        if chunk != layout.query_chunk_index:
            chunk, step = layout.query_chunk_index, 0
        else:
            step += 1
        self._step_of_layer[call.layer_index] = (chunk, step)
        return step

    def _projection(self, call: SparseAttentionCall) -> torch.Tensor:
        generator = torch.Generator(device=call.query.device)
        generator.manual_seed(self._config.seed * 1013 + call.layer_index)
        return torch.randn(
            call.num_local_heads,
            call.head_dim,
            self._config.hash_bits,
            device=call.query.device,
            generator=generator,
        )

    def _compressed_history(
        self, call: SparseAttentionCall, layout: VisibleLayout, history_len: int
    ) -> CompressedHistory:
        signature = (call.key_segments, history_len, call.num_local_heads)
        cached = self._history.get(call.layer_index)
        if cached is not None and cached[0] == signature:
            return cached[1]
        history = self._build_compressed_history(call, layout, history_len)
        self._history[call.layer_index] = (signature, history)
        return history

    def _build_compressed_history(
        self, call: SparseAttentionCall, layout: VisibleLayout, history_len: int
    ) -> CompressedHistory:
        config = self._config
        num_heads, head_dim = call.num_local_heads, call.head_dim
        frame_seqlen = layout.frame_seqlen
        num_frames = history_len // frame_seqlen
        shape = (num_heads, num_frames, frame_seqlen, head_dim)
        keys = call.key[0, :history_len].permute(1, 0, 2).reshape(shape)
        values = call.value[0, :history_len].permute(1, 0, 2).reshape(shape)

        keep, group_size, merged_values = temporal_merge(
            keys, values, threshold=config.merge_threshold
        )
        flat_keys = keys.reshape(num_heads, history_len, head_dim)
        flat_values = merged_values.reshape(num_heads, history_len, head_dim)
        keep = keep.reshape(num_heads, history_len)
        bias = group_size.reshape(num_heads, history_len).log()

        projection = self._projection(call)
        buckets = sign_lsh_buckets(flat_keys, projection)
        num_buckets = 1 << config.hash_bits
        # Dropped keys sort past every real bucket, so each head's buckets stay
        # contiguous and the padded tail is unreachable.
        order = torch.where(keep, buckets, num_buckets).argsort(dim=1, stable=True)
        kept_per_head = keep.sum(1)
        kept = int(kept_per_head.max().item())
        order = order[:, :kept]

        gather = order[..., None].expand(num_heads, kept, head_dim)
        packed_keys = flat_keys.gather(1, gather)
        packed_values = flat_values.gather(1, gather)
        packed_bias = bias.gather(1, order)
        packed_bias = torch.where(
            torch.arange(kept, device=order.device)[None, :] < kept_per_head[:, None],
            packed_bias,
            torch.zeros_like(packed_bias),
        )

        counts = torch.zeros(
            num_heads, num_buckets, device=order.device, dtype=torch.long
        ).scatter_add_(1, buckets * keep, keep.long())
        ends = counts.cumsum(1)
        starts = ends - counts
        own = torch.full((num_heads, 1), kept, device=order.device, dtype=torch.long)

        own_len = call.key.shape[1] - history_len
        return CompressedHistory(
            key=packed_keys.permute(1, 0, 2).contiguous(),
            value=packed_values.permute(1, 0, 2).contiguous(),
            logit_bias=torch.cat(
                [
                    packed_bias,
                    torch.zeros(
                        num_heads, own_len, device=order.device, dtype=packed_bias.dtype
                    ),
                ],
                dim=1,
            ).contiguous(),
            bucket_starts=torch.cat([starts, own], dim=1)[:, None].to(torch.int32),
            bucket_ends=torch.cat([ends, own + own_len], dim=1)[:, None].to(
                torch.int32
            ),
            projection=projection,
            kept=kept,
        )

    def _permute_and_select(
        self, call: SparseAttentionCall, history: CompressedHistory
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bucket-sort the queries and pick each block's buckets.

        Sorting matters: a spatially contiguous block of 128 queries scatters
        across most buckets, which would select nearly everything. Bucket-sorted
        blocks are near-homogeneous, so a block reads one or two buckets plus
        their Hamming neighbours.
        """
        config = self._config
        q_len = call.query.shape[1]
        num_blocks = -(-q_len // config.block)
        num_buckets = 1 << config.hash_bits
        queries = call.query[0].permute(1, 0, 2)  # [heads, q_len, head_dim]
        buckets = sign_lsh_buckets(queries, history.projection)  # [heads, q_len]

        order = buckets.argsort(dim=1, stable=True)
        buckets = buckets.gather(1, order)
        permutation = order.T[None].expand(call.query.shape[0], -1, -1)
        query = call.query.gather(
            1, permutation[..., None].expand_as(call.query)
        )

        block_of_query = (
            torch.arange(q_len, device=buckets.device) // config.block
        )[None, :].expand_as(buckets)
        present = torch.zeros(
            call.num_local_heads,
            num_blocks,
            num_buckets,
            dtype=torch.bool,
            device=buckets.device,
        )
        present[
            torch.arange(call.num_local_heads, device=buckets.device)[:, None],
            block_of_query,
            buckets,
        ] = True
        present = hamming_expand(
            present, bits=config.hash_bits, radius=config.probe_hamming
        )
        own = torch.ones_like(present[..., :1])
        return query, permutation, torch.cat([present, own], dim=-1)
