# SPDX-License-Identifier: Apache-2.0
"""Geometry of one block-causal DiT attention call.

A sparse-attention method needs to know what the key axis *means* before it can
drop any of it. In the Self-Forcing family that mapping is not obvious from the
tensor shapes: the visible keys are a rolling view of a KV cache, so key index
0 is whichever latent frame the window happens to start at, and the query is a
whole chunk of ``frames_per_block`` latent frames rather than a single token.

:class:`ChunkGeometry` is stamped once per DiT forward (all layers share it) and
:func:`visible_layout` turns it plus the layer's visible key segments into
frame- and chunk-indexed coordinates that every method in this package speaks.
"""

from collections.abc import Sequence

import msgspec
import numpy as np

KeySegments = tuple[tuple[int, int], ...]


class ChunkGeometry(msgspec.Struct, frozen=True):
    """Token layout shared by every attention layer of one DiT forward."""

    frame_seqlen: int
    frames_per_block: int
    query_token_start: int
    grid_height: int
    grid_width: int

    @property
    def chunk_tokens(self) -> int:
        return self.frame_seqlen * self.frames_per_block

    @property
    def query_chunk_index(self) -> int:
        return self.query_token_start // self.chunk_tokens


class VisibleLayout(msgspec.Struct, frozen=True):
    """The visible key axis, in latent-frame and chunk coordinates.

    ``global_frame_ids[f]`` is the video-global latent frame index of view-local
    frame ``f``, whose tokens are ``[f * frame_seqlen, (f + 1) * frame_seqlen)``
    of the key axis. ``chunk_offsets[f]`` is that frame's chunk index minus the
    query's chunk index, so ``0`` is the query's own chunk and ``-1`` the one
    before it.
    """

    global_frame_ids: np.ndarray  # int64 [num_frames]
    chunk_offsets: np.ndarray  # int64 [num_frames]
    frame_seqlen: int
    frames_per_block: int
    query_frames: int
    query_chunk_index: int

    @property
    def num_frames(self) -> int:
        return int(self.global_frame_ids.size)

    @property
    def kv_len(self) -> int:
        return self.num_frames * self.frame_seqlen

    @property
    def own_frames(self) -> np.ndarray:
        """Bool mask of the frames belonging to the query's own chunk."""
        return self.chunk_offsets == 0

    def frames_of_offset(self, offset: int) -> np.ndarray:
        return self.chunk_offsets == offset

    def sink_frames(self, num_sink_frames: int) -> np.ndarray:
        """Bool mask of the first ``num_sink_frames`` frames of the video."""
        return self.global_frame_ids < num_sink_frames

    @property
    def num_past_chunks(self) -> int:
        """How many whole past chunks the view holds (offsets -1, -2, ...)."""
        offsets = self.chunk_offsets[self.chunk_offsets < 0]
        return 0 if offsets.size == 0 else int(-offsets.min())


def visible_layout(
    key_segments: Sequence[tuple[int, int]],
    *,
    geometry: ChunkGeometry,
    query_tokens: int,
) -> VisibleLayout | None:
    """Frame/chunk coordinates of the visible keys, or ``None`` if unmappable.

    ``None`` means the view is not aligned to whole latent frames — the caller
    must fall back to dense attention rather than guess.
    """
    frame_seqlen = geometry.frame_seqlen
    if (
        geometry.query_token_start % frame_seqlen != 0
        or query_tokens % frame_seqlen != 0
    ):
        return None
    for token_start, length in key_segments:
        if token_start % frame_seqlen != 0 or length % frame_seqlen != 0:
            return None

    global_frame_ids = np.concatenate(
        [
            np.arange(start // frame_seqlen, (start + length) // frame_seqlen)
            for start, length in key_segments
        ]
    )
    frames_per_block = geometry.frames_per_block
    query_chunk = geometry.query_chunk_index
    chunk_offsets = global_frame_ids // frames_per_block - query_chunk
    return VisibleLayout(
        global_frame_ids=global_frame_ids,
        chunk_offsets=chunk_offsets,
        frame_seqlen=frame_seqlen,
        frames_per_block=frames_per_block,
        query_frames=query_tokens // frame_seqlen,
        query_chunk_index=query_chunk,
    )


def frame_mask_to_ranges(
    keep: np.ndarray,
    *,
    frame_seqlen: int,
) -> list[list[tuple[int, int]]]:
    """Per-head token ranges of a ``[heads, frames]`` keep mask.

    Runs of adjacent kept frames collapse into one range, which is what lets a
    frame-granular method run at nearly dense throughput on its kept window.
    """
    num_heads, num_frames = keep.shape
    padded = np.concatenate(
        [np.zeros((num_heads, 1), dtype=bool), keep, np.zeros((num_heads, 1), dtype=bool)],
        axis=1,
    )
    edges = np.diff(padded.astype(np.int8), axis=1)
    ranges: list[list[tuple[int, int]]] = []
    for head in range(num_heads):
        starts = np.flatnonzero(edges[head] == 1)
        ends = np.flatnonzero(edges[head] == -1)
        ranges.append(
            [
                (int(start) * frame_seqlen, int(end) * frame_seqlen)
                for start, end in zip(starts, ends, strict=True)
            ]
        )
    return ranges
