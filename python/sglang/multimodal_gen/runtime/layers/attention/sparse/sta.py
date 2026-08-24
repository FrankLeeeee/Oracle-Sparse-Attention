# SPDX-License-Identifier: Apache-2.0
"""Sliding Tile Attention baseline — faithful port of the upstream mask.

Reproduces https://github.com/hao-ai-lab/fastvideo — specifically the STA mask
of ``fastvideo-kernel/tests/support_flex_sta.py::generate_sta_mask`` (the flex
reference the upstream kernels are tested against; the full pipeline lives on
the archived ``sta_do_not_delete`` branch). ``test/unit/realtime/
test_sta_parity.py`` asserts equality against a verbatim copy of that function
on an aligned geometry.

STA's idea is NATTEN made hardware-friendly: partition each latent frame into
spatial **tiles**, reorder tokens tile-major, and let every query tile attend a
3D window of key tiles — ``kernel_t`` frames by ``kernel_h x kernel_w`` tiles —
*centered on the query tile but clamped to the canvas edges*. The clamp is the
load-bearing part: every query tile keeps exactly the same number of key tiles,
and a kept tile is a contiguous token run, so the mask maps onto dense tile
compute with no ragged partial blocks.

Upstream is bidirectional over a whole clip. The block-causal adaptation is the
same one SVG uses: the query side is the chunk being generated, the key side is
the visible KV view, and the temporal axis of the canvas is the *visible frame
axis* — the window centers on the query frame's position among the visible
frames and clamps there, which for a tail-of-history query degenerates to "the
trailing ``kernel_t`` visible frames", exactly the row-slice of upstream's
square mask that the causal setting needs. Like the other baselines it declines
the ramp-up calls where the view holds nothing but the query itself.

The mask depends only on geometry, so it is built once per visible layout and
shared by every layer, head and denoising step. The per-frame tile permutation
is one cached index; applying it to K/V is a single gather per call.
"""

import msgspec
import numpy as np
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


class StaConfig(msgspec.Struct, frozen=True):
    # 3D window size: latent frames (on the visible-frame axis) by spatial
    # tiles. Odd values center cleanly; an even value behaves as the next odd
    # one, which is upstream's ``abs(center - x) <= k // 2`` arithmetic.
    kernel_t: int = 5
    kernel_h: int = 3
    kernel_w: int = 3
    # Spatial tile shape in latent-grid cells. 0 auto-picks divisors of the
    # model's post-patch grid targeting upstream's 8x8 tiles (~64 tokens).
    tile_h: int = 0
    tile_w: int = 0
    # Query rows per kernel program. The mask is defined per query *tile*, but
    # tile token counts are rarely powers of two (Triton's requirement), so
    # execution quantizes on this grid: a query block keeps the union of the
    # windows of the tiles it covers. Adjacent tiles' clamped windows overlap
    # in all but one tile column, so the union costs a few percent, and the
    # reported density is the executed one.
    #
    # 128 rather than a tile-sized 64: the kernel's ``tl.dot`` wants at least
    # 128 query rows to reach full tensor-core throughput, and halving it
    # costs about a factor of two on the attention itself — far more than the
    # few percent of extra keys that unioning ~2 tiles per block adds.
    block: int = 128


def pick_tile(extent: int, other_extent: int) -> tuple[int, int]:
    """Divisor pair ``(tile_h, tile_w)`` for a ``extent x other_extent`` grid.

    Upstream fixes 8x8 tiles because its grids divide by 8; ours (45x80 at Wan
    720p, 30x52 at 480p, 22x40 on LongLive-2) mostly do not, so the tile must
    be a divisor pair. Score = distance from upstream's 64-token tile area
    plus a penalty on anisotropy, so a near-square ~64-token tile wins when
    one exists (45x80 -> 9x8, 30x52 -> 5x13, 22x40 -> 11x5).
    """

    def divisors(n: int) -> list[int]:
        return [d for d in range(1, n + 1) if n % d == 0]

    # A 1-row (or 1-column) grid cannot have square tiles, so the anisotropy
    # penalty would only push the tile toward a single cell there.
    degenerate = min(extent, other_extent) == 1
    best: tuple[float, int, int] | None = None
    for th in divisors(extent):
        for tw in divisors(other_extent):
            ratio = 1.0 if degenerate else max(th, tw) / min(th, tw)
            score = abs(th * tw - 64) + 2.0 * (ratio - 1.0)
            if best is None or score < best[0]:
                best = (score, th, tw)
    assert best is not None
    return best[1], best[2]


def tile_major_permutation(
    *, grid_h: int, grid_w: int, tile_h: int, tile_w: int
) -> np.ndarray:
    """``perm[new] = old``: one frame's row-major tokens in tile-major order.

    Tile-major means upstream's canvas order: tiles ordered row-major over the
    tile grid, tokens row-major within each tile.
    """
    rows = np.arange(grid_h * grid_w) // grid_w
    cols = np.arange(grid_h * grid_w) % grid_w
    tiles_w = grid_w // tile_w
    tile_id = (rows // tile_h) * tiles_w + cols // tile_w
    within = (rows % tile_h) * tile_w + cols % tile_w
    return np.argsort(tile_id * (tile_h * tile_w) + within, kind="stable")


def clamped_window(positions: np.ndarray, *, extent: int, kernel: int) -> np.ndarray:
    """``[positions, extent]`` bool: upstream's clamped centered window.

    ``center = clamp(p, k // 2, extent - 1 - k // 2)``, keep ``|center - x| <=
    k // 2``. When the canvas is no wider than the kernel everything is kept
    (the clamp bounds cross and the window covers the whole axis either way).
    """
    if extent <= kernel:
        return np.ones((positions.size, extent), dtype=bool)
    half = kernel // 2
    centers = np.clip(positions, half, extent - 1 - half)
    axis = np.arange(extent)
    return np.abs(centers[:, None] - axis[None, :]) <= half


def build_sta_tile_mask(
    *,
    layout: VisibleLayout,
    grid_h: int,
    grid_w: int,
    tile_h: int,
    tile_w: int,
    config: StaConfig,
) -> np.ndarray:
    """``[q_tiles, key_tiles]`` STA mask over tile-major token order.

    Query tile ``f * tiles_per_frame + t`` is frame ``f`` of the query, tile
    ``t`` of the ``(grid_h / tile_h) x (grid_w / tile_w)`` tile grid; key tiles
    run over the visible frames the same way. Each axis of the 3D window is
    upstream's clamped centered interval; the temporal canvas is the visible
    frame axis, on which the query frames are the last ``query_frames``.
    """
    tiles_h = grid_h // tile_h
    tiles_w = grid_w // tile_w
    tiles_per_frame = tiles_h * tiles_w
    num_frames = layout.num_frames
    query_frames = layout.query_frames

    time_keep = clamped_window(
        np.arange(num_frames - query_frames, num_frames),
        extent=num_frames,
        kernel=config.kernel_t,
    )  # [query_frames, num_frames]
    row_keep = clamped_window(
        np.arange(tiles_h), extent=tiles_h, kernel=config.kernel_h
    )  # [tiles_h, tiles_h]
    col_keep = clamped_window(
        np.arange(tiles_w), extent=tiles_w, kernel=config.kernel_w
    )  # [tiles_w, tiles_w]

    spatial = (row_keep[:, None, :, None] & col_keep[None, :, None, :]).reshape(
        tiles_per_frame, tiles_per_frame
    )
    return (time_keep[:, None, :, None] & spatial[None, :, None, :]).reshape(
        query_frames * tiles_per_frame, num_frames * tiles_per_frame
    )


class StaAttention(SparseAttentionBackend):
    name = "sta"

    def __init__(self, config: StaConfig) -> None:
        super().__init__()
        self._config = config
        self._plans = LayoutCache()
        # (grid_h, grid_w) -> (tile_h, tile_w, frame permutation on device)
        self._tiling: tuple[tuple[int, int], tuple[int, int, torch.Tensor]] | None = (
            None
        )

    def _frame_tiling(
        self, *, grid_h: int, grid_w: int, device: torch.device
    ) -> tuple[int, int, torch.Tensor] | None:
        if self._tiling is not None and self._tiling[0] == (grid_h, grid_w):
            return self._tiling[1]
        tile_h, tile_w = self._config.tile_h, self._config.tile_w
        if tile_h <= 0 or tile_w <= 0:
            tile_h, tile_w = pick_tile(grid_h, grid_w)
        if grid_h % tile_h or grid_w % tile_w:
            return None
        permutation = torch.from_numpy(
            tile_major_permutation(
                grid_h=grid_h, grid_w=grid_w, tile_h=tile_h, tile_w=tile_w
            )
        ).to(device)
        self._tiling = ((grid_h, grid_w), (tile_h, tile_w, permutation))
        return self._tiling[1]

    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        q_len = call.query.shape[1]
        kv_len = call.key.shape[1]
        if kv_len <= q_len:
            return None  # ramp-up: the view holds nothing beyond the query
        geometry = self.geometry
        assert geometry is not None
        grid_h, grid_w = geometry.grid_height, geometry.grid_width
        if grid_h * grid_w != layout.frame_seqlen:
            self.warn_dense_once("frame tokens do not form the stamped grid")
            return None
        tiling = self._frame_tiling(
            grid_h=grid_h, grid_w=grid_w, device=call.query.device
        )
        if tiling is None:
            self.warn_dense_once(
                f"configured tile {self._config.tile_h}x{self._config.tile_w} "
                f"does not divide the {grid_h}x{grid_w} grid"
            )
            return None
        tile_h, tile_w, frame_permutation = tiling

        # The mask depends only on the shape of the view (frame count and
        # query frames), not on absolute segments — Rolling Forcing's steady
        # windows repeat the same shape under changing segments, and its
        # denoise/updating layouts alternate, so the slot is the frame count.
        signature = (layout.num_frames, q_len, call.num_local_heads)
        hit, cached = self._plans.get(layout.num_frames, signature)
        if not hit:
            cached = self._build_plan(
                layout=layout,
                grid_h=grid_h,
                grid_w=grid_w,
                tile_h=tile_h,
                tile_w=tile_w,
                frame_permutation=frame_permutation,
                num_heads=call.num_local_heads,
                device=call.query.device,
            )
            self._plans.put(layout.num_frames, signature, cached)
        if cached is None:
            return None
        plan, query_index, key_index = cached

        batch = call.query.shape[0]
        query = call.query.index_select(1, query_index)
        return SparseAttentionExecution(
            plan=plan,
            query=query,
            key=call.key.index_select(1, key_index),
            value=call.value.index_select(1, key_index),
            query_permutation=query_index[None, :, None].expand(
                batch, q_len, call.num_local_heads
            ),
        )

    def _build_plan(
        self,
        *,
        layout: VisibleLayout,
        grid_h: int,
        grid_w: int,
        tile_h: int,
        tile_w: int,
        frame_permutation: torch.Tensor,
        num_heads: int,
        device: torch.device,
    ):
        tile_tokens = tile_h * tile_w
        block = self._config.block
        tile_mask = build_sta_tile_mask(
            layout=layout,
            grid_h=grid_h,
            grid_w=grid_w,
            tile_h=tile_h,
            tile_w=tile_w,
            config=self._config,
        )
        if tile_mask.all():
            return None

        # Tile-defined mask, block-quantized execution: query block b keeps
        # the union of the windows of the tiles its rows fall in.
        q_len = layout.query_frames * layout.frame_seqlen
        q_blocks = -(-q_len // block)
        row_tile = np.minimum(np.arange(q_len) // tile_tokens, tile_mask.shape[0] - 1)
        mask = np.zeros((q_blocks, tile_mask.shape[1]), dtype=bool)
        for b in range(q_blocks):
            tiles = row_tile[b * block : (b + 1) * block]
            mask[b] = tile_mask[tiles[0] : tiles[-1] + 1].any(axis=0)

        key_tiles = tile_mask.shape[1]
        tile_ids = torch.arange(key_tiles, device=device, dtype=torch.int32)
        plan = plan_from_segment_mask(
            torch.from_numpy(mask).to(device)[None].expand(num_heads, -1, -1),
            segment_starts=tile_ids * tile_tokens,
            segment_ends=(tile_ids + 1) * tile_tokens,
            block_m=block,
        )
        frame_seqlen = layout.frame_seqlen
        query_index = (
            torch.arange(layout.query_frames, device=device)[:, None] * frame_seqlen
            + frame_permutation[None, :]
        ).reshape(-1)
        key_index = (
            torch.arange(layout.num_frames, device=device)[:, None] * frame_seqlen
            + frame_permutation[None, :]
        ).reshape(-1)
        return plan, query_index, key_index
