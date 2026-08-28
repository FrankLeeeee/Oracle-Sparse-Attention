# SPDX-License-Identifier: Apache-2.0
"""Mixed Sparse Attention (MSA): per-head static patterns + runtime selection.

MSA executes each attention head by the *family* an offline taxonomy assigned
it (``scripts/investigation/qk_map_similarity/taxonomy_sweep.py``, measured on
dense runs of the calibration prompts):

``local(r)``
    a static within-frame window of grid rows around the query's own row,
    replicated over every visible frame — zero runtime planning.
``shortwin(m)``
    only the newest ``m`` visible frames (own chunk + recent), read fully —
    zero runtime planning.
``diffuse``
    every k-th history frame read in full (plus the whole own chunk): these
    heads' attention rows are near-uniform, so a structured subsample
    approximates their output (a mean) without any selection — and whole
    frames keep the key walks contiguous, unlike within-frame striding.
``content`` (and ``frozen``, folded in)
    per-call top-k of mean-pooled key blocks per query block — the same
    estimator family LightForcing uses, restricted to the heads that actually
    need it.

Static-family descriptors are built once per (layer, visible layout) and
cached; only the content heads pay planning per call. Execution runs the two
specialized kernels of ``msa_kernel.py`` into one shared output tensor.

Config: ``taxonomy_path`` (required) points at the exported taxonomy JSON;
``content_density`` is the content heads' kept fraction of key blocks.
"""

import json
import pathlib

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    LayoutCache,
    SparseAttentionBackend,
    SparseAttentionCall,
    SparseAttentionExecution,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import VisibleLayout
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    plan_from_segment_mask,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (
    frame_aligned_block_bounds,
    lightforcing_block_mask,
    mean_pool_blocks,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.msa_kernel import (
    msa_content_attention,
    msa_static_attention,
)

STATIC_FAMILIES = ("local", "shortwin", "diffuse")
_MAX_STATIC_RANGES = 8


class MsaConfig(msgspec.Struct, frozen=True):
    # Path to the exported per-head taxonomy JSON (family + params per head).
    taxonomy_path: str = ""
    # Kept fraction of pooled key blocks for the content-dependent heads.
    content_density: float = 0.2
    # Kept fraction of every frame for the diffuse heads' strided tiles.
    diffuse_density: float = 0.10
    # Query block of both kernels and of the content estimator's query pooling.
    block_q: int = 128
    # Key block of the content estimator's pooled, frame-aligned scoring.
    block_k: int = 128
    # Two-stage eligibility of the content selection (LightForcing semantics):
    # stage 1 keeps the top ``keep_frames`` past frames per query block plus
    # the sink, the near window and the own chunk; stage 2 takes the block
    # top-k inside them. Besides matching the measured composed system, the
    # restriction makes the kept blocks contiguous — merged ranges execute
    # ~2x faster than the same density scattered over every frame.
    keep_frames: int = 6
    keep_sink: int = 1
    keep_near: int = 2
    # Plan the content heads once per (layer, chunk) — at the chunk's first
    # denoising step — and reuse the plan for the remaining steps and the
    # cache refresh. LightForcing replans on every call; the study's
    # step-consistency and prev-chunk-recall measurements showed the selection
    # barely moves within a chunk, so this amortizes most of MSA's remaining
    # planning cost. Set True to replan per call (LightForcing parity).
    replan_each_step: bool = False


class HeadSpec(msgspec.Struct, frozen=True):
    family: str
    r: int = 0  # local: Chebyshev radius in grid rows/cols
    m: int = 0  # shortwin: newest visible frames kept


class MsaStaticPlan(msgspec.Struct, frozen=True):
    """Cached static-head descriptors for one (layer, visible layout)."""

    head_ids: torch.Tensor  # [groups] int32
    frame_lo: torch.Tensor  # [groups] int32
    frame_step: torch.Tensor  # [groups] int32
    frame_tail: torch.Tensor  # [groups] int32, own-chunk frames always read
    ranges: torch.Tensor  # [groups, q_blocks, max_ranges, 2] int32
    counts: torch.Tensor  # [groups, q_blocks] int32
    # Mean over (head, q_block) of kept keys / kv_len, summed over the groups —
    # the static side of the call's analytic density.
    fraction_sum: float


def load_taxonomy(path: str) -> dict[str, HeadSpec]:
    if not path:
        raise ValueError("msa needs taxonomy_path in --sparse-attention-config")
    raw = json.loads(pathlib.Path(path).read_text())
    heads = raw["heads"] if "heads" in raw else raw
    out: dict[str, HeadSpec] = {}
    for key, record in heads.items():
        family = record["family"]
        if family == "frozen":
            family = "content"  # runtime selection strictly dominates
        out[key] = HeadSpec(
            family=family,
            r=int(record.get("r", 0)),
            m=int(record.get("m", 0)),
        )
    return out


def within_frame_query_spans(
    q_block: int, *, block_m: int, q_len: int, frame_seqlen: int
) -> list[tuple[int, int]]:
    """Within-frame token spans a query block covers (2 when it straddles)."""
    lo = q_block * block_m
    hi = min(q_len, (q_block + 1) * block_m)
    spans = []
    while lo < hi:
        step = min(hi, (lo // frame_seqlen + 1) * frame_seqlen)
        spans.append((lo % frame_seqlen, (step - 1) % frame_seqlen + 1))
        lo = step
    return spans


def local_ranges(
    q_block: int,
    *,
    radius: int,
    block_m: int,
    q_len: int,
    frame_seqlen: int,
    grid_height: int,
    grid_width: int,
) -> list[tuple[int, int]]:
    """The row-window token ranges of a local head's query block."""
    ranges = []
    for span_lo, span_hi in within_frame_query_spans(
        q_block, block_m=block_m, q_len=q_len, frame_seqlen=frame_seqlen
    ):
        row_lo = max(0, span_lo // grid_width - radius)
        row_hi = min(grid_height - 1, (span_hi - 1) // grid_width + radius)
        ranges.append((row_lo * grid_width, (row_hi + 1) * grid_width))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def diffuse_frame_step(density: float) -> int:
    """The history-frame stride keeping ~``density`` of the history frames."""
    return max(1, round(1.0 / density))


class MixedSparseAttention(SparseAttentionBackend):
    name = "msa"

    def __init__(self, config: MsaConfig) -> None:
        super().__init__()
        self._config = config
        self._taxonomy = load_taxonomy(config.taxonomy_path)
        self._static_cache = LayoutCache()
        self._pooled_history = LayoutCache()
        self._content_cache = LayoutCache()
        self._content_ids: dict[tuple[int, int], torch.Tensor] = {}
        # layer -> (static [(local head, spec)], content local-head ids); the
        # split depends only on the taxonomy, resolved per layer on first use.
        self._layer_split: dict[tuple[int, int, int], tuple[list, list[int]]] = {}

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        """Unused: MSA executes through its own :meth:`attend`."""
        return None

    def _split_heads(
        self, layer_index: int, *, head_start: int, num_local_heads: int
    ) -> tuple[list, list[int]]:
        key = (layer_index, head_start, num_local_heads)
        cached = self._layer_split.get(key)
        if cached is not None:
            return cached
        static: list[tuple[int, HeadSpec]] = []
        content: list[int] = []
        for local_head in range(num_local_heads):
            head_key = f"L{layer_index:02d}_h{head_start + local_head}"
            spec = self._taxonomy.get(head_key)
            if spec is None:
                raise ValueError(f"taxonomy {self._config.taxonomy_path} lacks {head_key}")
            if spec.family in STATIC_FAMILIES:
                static.append((local_head, spec))
            else:
                content.append(local_head)
        self._layer_split[key] = (static, content)
        return static, content

    def _static_plan(
        self,
        layer_index: int,
        static: list[tuple[int, HeadSpec]],
        *,
        layout: VisibleLayout,
        q_len: int,
        device: torch.device,
    ) -> MsaStaticPlan:
        config = self._config
        geometry = self.geometry
        signature = (layout.num_frames, layout.frame_seqlen, q_len)
        hit, cached = self._static_cache.get(layer_index, signature)
        if hit:
            return cached
        block_m = config.block_q
        q_blocks = -(-q_len // block_m)
        frame_seqlen = layout.frame_seqlen
        num_frames = layout.num_frames
        groups = len(static)
        head_ids = torch.empty(groups, dtype=torch.int32)
        frame_lo = torch.zeros(groups, dtype=torch.int32)
        frame_step = torch.ones(groups, dtype=torch.int32)
        frame_tail = torch.zeros(groups, dtype=torch.int32)
        ranges = torch.zeros(
            (groups, q_blocks, _MAX_STATIC_RANGES, 2), dtype=torch.int32
        )
        counts = torch.zeros((groups, q_blocks), dtype=torch.int32)
        fraction_sum = 0.0
        for group, (local_head, spec) in enumerate(static):
            head_ids[group] = local_head
            kept_per_frame = 0.0
            if spec.family == "shortwin":
                frame_lo[group] = max(0, num_frames - spec.m)
                head_ranges = {
                    q_block: [(0, frame_seqlen)] for q_block in range(q_blocks)
                }
            elif spec.family == "diffuse":
                frame_step[group] = diffuse_frame_step(config.diffuse_density)
                frame_tail[group] = min(layout.query_frames, num_frames)
                head_ranges = {
                    q_block: [(0, frame_seqlen)] for q_block in range(q_blocks)
                }
            else:  # local
                head_ranges = {
                    q_block: local_ranges(
                        q_block,
                        radius=spec.r,
                        block_m=block_m,
                        q_len=q_len,
                        frame_seqlen=frame_seqlen,
                        grid_height=geometry.grid_height,
                        grid_width=geometry.grid_width,
                    )
                    for q_block in range(q_blocks)
                }
            for q_block, block_ranges in head_ranges.items():
                assert len(block_ranges) <= _MAX_STATIC_RANGES
                counts[group, q_block] = len(block_ranges)
                for index, (lo, hi) in enumerate(block_ranges):
                    ranges[group, q_block, index, 0] = lo
                    ranges[group, q_block, index, 1] = hi
                kept_per_frame += sum(hi - lo for lo, hi in block_ranges)
            tail = int(frame_tail[group])
            visible = tail + len(
                range(int(frame_lo[group]), num_frames - tail, int(frame_step[group]))
            )
            fraction_sum += (kept_per_frame / q_blocks) * visible / layout.kv_len
        plan = MsaStaticPlan(
            head_ids=head_ids.to(device),
            frame_lo=frame_lo.to(device),
            frame_step=frame_step.to(device),
            frame_tail=frame_tail.to(device),
            ranges=ranges.to(device),
            counts=counts.to(device),
            fraction_sum=fraction_sum,
        )
        self._static_cache.put(layer_index, signature, plan)
        return plan

    def _content_mask(
        self,
        call: SparseAttentionCall,
        layout: VisibleLayout,
        content: list[int],
    ) -> tuple[torch.Tensor, int, int]:
        """Kept-block mask for the content heads + (topk, kv_blocks)."""
        config = self._config
        frame_seqlen = layout.frame_seqlen
        blocks_per_frame = -(-frame_seqlen // config.block_k)
        kv_blocks = layout.num_frames * blocks_per_frame
        topk = min(kv_blocks, max(1, int(config.content_density * kv_blocks)))

        query = call.query.mean(dim=0) if call.query.shape[0] > 1 else call.query[0]
        key = call.key.mean(dim=0) if call.key.shape[0] > 1 else call.key[0]
        pooled_query = mean_pool_blocks(query, block=config.block_q)

        history_len = (layout.num_frames - layout.query_frames) * frame_seqlen
        signature = (call.key_segments, layout.kv_len, frame_seqlen)
        hit, pooled_history = self._pooled_history.get(call.layer_index, signature)
        if not hit:
            pooled_history = mean_pool_blocks(
                key[:history_len], block=config.block_k, group=frame_seqlen
            )
            self._pooled_history.put(call.layer_index, signature, pooled_history)
        pooled_own = mean_pool_blocks(
            key[history_len:], block=config.block_k, group=frame_seqlen
        )
        pooled_key = torch.cat([pooled_history, pooled_own], dim=0)

        index = torch.tensor(content, device=query.device)
        mask = lightforcing_block_mask(
            pooled_query=pooled_query.index_select(1, index).permute(1, 0, 2),
            pooled_key=pooled_key.index_select(1, index).permute(1, 0, 2),
            blocks_per_frame=blocks_per_frame,
            past_frames=layout.num_frames - layout.query_frames,
            keep_frames=config.keep_frames,
            keep_sink=config.keep_sink,
            keep_near=config.keep_near,
            topk=topk,
        )
        return mask, topk, kv_blocks

    def attend(self, call: SparseAttentionCall) -> torch.Tensor | None:
        layout = self._layout(call)
        if layout is None:
            return None
        kv_len = call.key.shape[1]
        # Cache-update forwards are sparsified too — LightForcing does the
        # same, and the static families are content-independent by design.
        if kv_len <= call.query.shape[1]:
            self._record_density(None, kv_len=kv_len)
            return None
        geometry = self.geometry
        if geometry.grid_height * geometry.grid_width != layout.frame_seqlen:
            self.warn_dense_once("latent grid does not match the frame length")
            self._record_density(None, kv_len=kv_len)
            return None
        static, content = self._split_heads(
            call.layer_index,
            head_start=call.head_start,
            num_local_heads=call.num_local_heads,
        )
        query, key, value = call.query, call.key, call.value
        if query.stride(-1) != 1:
            query = query.contiguous()
        if key.stride(-1) != 1:
            key = key.contiguous()
        if value.stride(-1) != 1:
            value = value.contiguous()
        q_len = query.shape[1]
        out = torch.empty_like(query)
        fraction = 0.0

        if static:
            plan = self._static_plan(
                call.layer_index, static, layout=layout, q_len=q_len, device=query.device
            )
            msa_static_attention(
                query=query,
                key=key,
                value=value,
                out=out,
                head_ids=plan.head_ids,
                frame_lo=plan.frame_lo,
                frame_step=plan.frame_step,
                frame_tail=plan.frame_tail,
                ranges=plan.ranges,
                counts=plan.counts,
                num_frames=layout.num_frames,
                frame_seqlen=layout.frame_seqlen,
                block_m=self._config.block_q,
                softmax_scale=call.softmax_scale,
            )
            fraction += plan.fraction_sum

        if content:
            signature = (call.key_segments, kv_len)
            hit, cached = self._content_cache.get(call.layer_index, signature)
            if hit and not self._config.replan_each_step:
                content_plan, topk, kv_blocks = cached
            else:
                mask, topk, kv_blocks = self._content_mask(call, layout, content)
                block_lo, block_hi = frame_aligned_block_bounds(
                    num_frames=layout.num_frames,
                    frame_seqlen=layout.frame_seqlen,
                    block=self._config.block_k,
                    device=query.device,
                )
                content_plan = plan_from_segment_mask(
                    mask,
                    segment_starts=block_lo,
                    segment_ends=block_hi,
                    block_m=self._config.block_q,
                )
                self._content_cache.put(
                    call.layer_index, signature, (content_plan, topk, kv_blocks)
                )
            ids_key = (call.layer_index, call.head_start)
            content_ids = self._content_ids.get(ids_key)
            if content_ids is None:
                content_ids = torch.tensor(
                    content, dtype=torch.int32, device=query.device
                )
                self._content_ids[ids_key] = content_ids
            msa_content_attention(
                query=query,
                key=key,
                value=value,
                out=out,
                head_ids=content_ids,
                range_starts=content_plan.range_starts,
                range_ends=content_plan.range_ends,
                range_counts=content_plan.range_counts,
                block_m=self._config.block_q,
                softmax_scale=call.softmax_scale,
            )
            fraction += len(content) * topk / kv_blocks

        num_heads = call.num_local_heads
        self._record_density(None, kv_len=kv_len, fraction=fraction / num_heads)
        return out
