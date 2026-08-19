# SPDX-License-Identifier: Apache-2.0
"""Radial Attention baseline — faithful port of the upstream mask.

Reproduces https://github.com/mit-han-lab/radial-attention (commit 72788d4),
specifically ``radial_attn/attn_mask.py::gen_log_mask_shrinked`` with
``model_type="wan"``. The four pieces of that construction are reproduced here
one-for-one, and
``test/unit/realtime/test_radial_parity.py`` asserts equality against a verbatim
copy of the upstream functions on an aligned geometry.

The mask encodes "spatiotemporal energy decay" — attention narrows as temporal
distance grows — and it does so in *two* stages, which is the part that is easy
to get wrong:

1. **A narrowing diagonal band.** For frame distance ``d``, the half-width is
   ``2^bit_length(frame_seqlen) / 2^bit_length(d) * decay_factor`` tokens,
   floored at the block size. Note it is an absolute token count derived from the
   *power of two above* the frame length, not a fraction of the frame: at
   ``frame_seqlen = 1560`` the reference length is 2048, so d=2,3 gives 512
   tokens, d=4..7 gives 256, d=8..15 gives 128, and everything beyond is
   clamped to 128.
2. **Frame decimation.** Once the band would fall below the block size, upstream
   stops narrowing it and starts dropping whole frame pairs instead: with
   ``split_factor = block_size / decay_length``, the pair is kept only when
   ``d % split_factor == 0``. So at large distances the mask keeps a
   block-wide band on every second frame, then every fourth, and so on. Without
   this the mask never actually gets sparse at long range.

Key frame 0 of the *video* is exempt from both and always fully attended —
upstream's attention sink. In the block-causal setting it leaves the KV window
after a few chunks, at which point that clause stops applying.

The mask depends only on geometry, so it is built once per visible layout and
shared by every layer, head and denoising step.
"""

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    LayoutCache,
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


class RadialConfig(msgspec.Struct, frozen=True):
    block: int = 128
    # Upstream's sparsity knob: scales the band half-width at every distance.
    # 1.0 is what `wan_t2v_inference.py` passes; lower is sparser.
    decay_factor: float = 1.0
    # Whether video frame 0 stays fully attended (upstream's `j == 0` clause).
    dense_sink_frames: int = 1


def radial_window_width(
    distance: int, *, frame_seqlen: int, decay_factor: float, block: int
) -> float:
    """Band half-width in tokens at temporal ``distance``.

    Mirrors upstream ``get_window_width(..., model_type="wan")``: distances 0 and
    1 keep the whole frame, and beyond that the width halves per power of two of
    the distance but never drops below one block.
    """
    if distance <= 1:
        return float(frame_seqlen)
    decay_length = (
        2 ** frame_seqlen.bit_length() / 2 ** distance.bit_length() * decay_factor
    )
    return decay_length if decay_length >= block else float(block)


def radial_frame_is_kept(distance: int, *, frame_seqlen: int, block: int) -> bool:
    """Upstream ``get_diagonal_split_mask``: is this frame pair kept at all?

    Once the band would be narrower than a block, upstream keeps the band at one
    block and decimates frames instead — every ``block / decay_length``-th frame.

    Note there is no ``decay_factor`` here: upstream applies that knob only to the
    band width, not to the decimation schedule, so the set of *frames* a query
    reaches is fixed and only the width within them shrinks. Passing the factor in
    here (as an earlier version did) makes the two disagree from d=8 onward.
    """
    if distance <= 1:
        return True
    decay_length = 2 ** frame_seqlen.bit_length() / 2 ** distance.bit_length()
    if decay_length >= block:
        return True
    return distance % int(block / decay_length) == 0


def shrink_mask_strict(mask: torch.Tensor, *, block: int) -> torch.Tensor:
    """Token mask → block mask, upstream's ``shrinkMaskStrict``.

    A block survives when more than 60% of its columns that have *any* coverage
    are covered on more than a third of their rows. This is markedly stricter
    than "any overlap": a band clipping the corner of a block does not keep it.
    """
    length = mask.shape[0]
    blocks = length // block
    view = mask[: blocks * block, : blocks * block].view(blocks, block, blocks, block)
    column_densities = view.sum(dim=1) / block
    non_zero = column_densities > 0
    high_density = column_densities > 1 / 3
    fraction = high_density.sum(-1) / (non_zero.sum(-1) + 1e-9)
    return fraction > 0.6


def build_radial_block_mask(
    *,
    layout: VisibleLayout,
    q_len: int,
    kv_len: int,
    config: RadialConfig,
    device: torch.device,
) -> torch.Tensor:
    """``[q_blocks, key_blocks]`` radial mask for one visible layout.

    Assembled frame pair by frame pair exactly as upstream's
    ``gen_log_mask_shrinked`` does, but only over the query frames of the current
    chunk — which is the block-row slice of upstream's square whole-clip mask
    that the causal setting actually needs.
    """
    block = config.block
    frame_seqlen = layout.frame_seqlen
    q_lo, _ = block_bounds(q_len, block, device=device)
    k_lo, k_hi = block_bounds(kv_len, block, device=device)
    mask = torch.zeros(
        (q_lo.numel(), k_lo.numel()), dtype=torch.bool, device=device
    )

    global_frames = [int(f) for f in layout.global_frame_ids]
    query_frames = global_frames[-layout.query_frames :]
    rows = torch.arange(frame_seqlen, device=device).view(-1, 1)
    columns = torch.arange(frame_seqlen, device=device).view(1, -1)
    ones = torch.ones((frame_seqlen, frame_seqlen), dtype=torch.bool, device=device)

    for query_index, query_frame in enumerate(query_frames):
        for key_index, key_frame in enumerate(global_frames):
            if key_frame < config.dense_sink_frames:
                local = ones
            else:
                distance = abs(query_frame - key_frame)
                if not radial_frame_is_kept(
                    distance, frame_seqlen=frame_seqlen, block=block
                ):
                    continue
                width = radial_window_width(
                    distance,
                    frame_seqlen=frame_seqlen,
                    decay_factor=config.decay_factor,
                    block=block,
                )
                local = (columns - rows).abs() <= width

            # Upstream pads the frame pair onto a block-aligned canvas so that a
            # frame boundary falling inside a block is accounted for, shrinks
            # that, then ORs it in at the block offset.
            row_remainder = (query_index * frame_seqlen) % block
            column_remainder = (key_index * frame_seqlen) % block
            canvas_rows = row_remainder + -(-frame_seqlen // block) * block
            canvas_columns = column_remainder + -(-frame_seqlen // block) * block
            canvas = torch.zeros(
                (canvas_rows, canvas_columns), dtype=torch.bool, device=device
            )
            canvas[
                row_remainder : row_remainder + frame_seqlen,
                column_remainder : column_remainder + frame_seqlen,
            ] = local
            shrunk = shrink_mask_strict(canvas, block=block)

            row_start = (query_index * frame_seqlen) // block
            column_start = (key_index * frame_seqlen) // block
            row_stop = min(row_start + shrunk.shape[0], mask.shape[0])
            column_stop = min(column_start + shrunk.shape[1], mask.shape[1])
            mask[row_start:row_stop, column_start:column_stop] |= shrunk[
                : row_stop - row_start, : column_stop - column_start
            ]

    # A query block must always be able to see its own tokens; upstream gets this
    # from the d=0 full-frame case, but our block grid is offset from its global
    # one so the diagonal is pinned explicitly.
    q_lo_full, q_hi_full = block_bounds(q_len, block, device=device)
    mask |= own_block_mask(
        q_lo=q_lo_full,
        q_hi=q_hi_full,
        k_lo=k_lo,
        k_hi=k_hi,
        query_offset_in_view=kv_len - q_len,
    )
    return mask


class RadialAttention(SparseAttentionBackend):
    name = "radial"

    def __init__(self, config: RadialConfig) -> None:
        super().__init__()
        self._config = config
        self._plans = LayoutCache()

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        q_len = call.query.shape[1]
        kv_len = call.key.shape[1]
        if kv_len <= q_len:
            return None

        # Head- and value-independent, so one entry per visible layout serves
        # every layer and every denoising step.
        signature = (call.key_segments, q_len, call.num_local_heads)
        hit, cached = self._plans.get(0, signature)
        if not hit:
            mask = build_radial_block_mask(
                layout=layout,
                q_len=q_len,
                kv_len=kv_len,
                config=self._config,
                device=call.query.device,
            )
            cached = None
            if not bool(mask.all()):
                cached = plan_from_block_mask(
                    mask[None].expand(call.num_local_heads, -1, -1),
                    block_n=self._config.block,
                    kv_len=kv_len,
                    block_m=self._config.block,
                )
            self._plans.put(0, signature, cached)
        if cached is None:
            return None
        return SparseAttentionExecution(
            plan=cached, query=call.query, key=call.key, value=call.value
        )
