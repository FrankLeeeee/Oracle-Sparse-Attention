# SPDX-License-Identifier: Apache-2.0
"""Intra-chunk frame-similarity probe for block-causal video DiTs.

Answers "how alike are the latent frames *inside* one chunk, and where in the
network does that change?" A block-causal model denoises 3 (or 8) latent
frames jointly, so the frames of a chunk share a single forward and can end up
near-duplicates of each other — which bounds how much temporal detail a chunk
can carry, and how much redundancy a sparse-attention or caching scheme could
exploit.

Enabled by setting ``SGLANG_DIFFUSION_FRAME_SIMILARITY_DIR``; disabled (one
``None`` check per DiT forward) otherwise. When on, the model reports the
hidden states entering every transformer block plus the output of the last
one, and the probe reduces each to one cosine similarity per frame pair:

    sim(i, j) = mean over spatial positions p of  cos(h_i[p], h_j[p])

i.e. the same spatial position is compared across frames and the per-position
cosines are averaged. Flattening a whole frame into a single vector instead
would weight positions by their norm and let a handful of high-norm tokens
speak for the frame.

:meth:`FrameSimilarityRecorder.flush` writes one ``frame_similarity.npz`` per
video with ``sim [chunks, steps, layers, pairs]`` (NaN where a chunk has fewer
steps) plus the pair index list. Only the denoising passes are recorded; the
KV-cache refresh pass is skipped so it does not masquerade as an extra step.

The probe is meant for single-GPU debugging runs: it only records on world
rank 0.
"""

import json
import logging
import pathlib
import time

import numpy as np
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.probe_pass_kind import (
    DENOISE_PASS,
    current_pass_kind,
)

logger = logging.getLogger(__name__)


def frame_pairs(num_frames: int) -> list[tuple[int, int]]:
    """Every unordered pair of distinct frames, in a stable order."""
    return [(i, j) for i in range(num_frames) for j in range(i + 1, num_frames)]


@torch.no_grad()
def pairwise_frame_cosine(frames: torch.Tensor) -> torch.Tensor:
    """Mean per-position cosine similarity between every pair of frames.

    ``frames`` is ``[num_frames, frame_seqlen, dim]``. Returns ``[pairs]`` in
    :func:`frame_pairs` order. The cast to float32 matters: a bf16 dot product
    over thousands of channels loses more precision than the differences being
    measured.
    """
    normalized = frames.float()
    normalized /= normalized.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack(
        [
            (normalized[i] * normalized[j]).sum(-1).mean()
            for i, j in frame_pairs(frames.shape[0])
        ]
    )


class FrameSimilarityRecorder:
    """Buffers per-(chunk, step, layer) frame similarities, then writes them out.

    The model calls :meth:`begin_forward` once per DiT forward, then
    :meth:`record_layer` for every block input and once more for the final
    output; the denoising stage calls :meth:`flush` when the video is done.
    """

    def __init__(self, *, output_dir: str) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.enabled = True
        self._frame_seqlen = 0
        self._frames_per_block = 0
        self._chunk_base = 0
        self._recording = False
        # (chunk, layer) -> how many denoise records this layer has seen for
        # that chunk; indexes the chunk's denoising steps
        self._step_counter: dict[tuple[int, int], int] = {}
        # (chunk, step, layer) -> [pairs] on device, moved to host at flush
        self._values: dict[tuple[int, int, int], torch.Tensor] = {}
        self._num_pairs = 0

    def begin_forward(
        self,
        *,
        frame_seqlen: int,
        num_frames_per_block: int,
        query_token_start: int,
    ) -> None:
        self._recording = self.enabled and current_pass_kind() == DENOISE_PASS
        if not self._recording:
            return
        self._frame_seqlen = frame_seqlen
        self._frames_per_block = num_frames_per_block
        self._chunk_base = query_token_start // (num_frames_per_block * frame_seqlen)

    def end_forward(self) -> None:
        self._recording = False

    @torch.no_grad()
    def record_layer(self, *, layer_index: int, hidden_states: torch.Tensor) -> None:
        """Record one layer boundary of the forward that is currently open.

        ``hidden_states`` is ``[batch, tokens, dim]``; only batch element 0 is
        recorded. A forward can cover several chunks at once (Rolling Forcing
        denoises a five-block window jointly), so the token axis is split into
        ``num_frames_per_block``-sized groups and each group is attributed to
        its own chunk.
        """
        if not self._recording:
            return
        tokens = hidden_states[0]
        num_frames = tokens.shape[0] // self._frame_seqlen
        block_frames = self._frames_per_block
        if num_frames < block_frames or num_frames % block_frames:
            # e.g. an independent first frame appended to the chunk; the group
            # split would not line up with chunk boundaries, so skip it.
            return
        block_tokens = block_frames * self._frame_seqlen
        for group in range(num_frames // block_frames):
            frames = tokens[group * block_tokens : (group + 1) * block_tokens].view(
                block_frames, self._frame_seqlen, -1
            )
            values = pairwise_frame_cosine(frames)
            self._num_pairs = values.shape[0]
            chunk = self._chunk_base + group
            step_key = (chunk, layer_index)
            step = self._step_counter.get(step_key, 0)
            self._step_counter[step_key] = step + 1
            self._values[(chunk, step, layer_index)] = values

    def flush(self, *, model_tag: str, meta: dict | None = None) -> str | None:
        """Write ``frame_similarity.npz`` for this video and reset the state."""
        values = self._values
        num_pairs = self._num_pairs
        frames_per_block = self._frames_per_block
        self._values = {}
        self._step_counter = {}
        self._recording = False
        if not values:
            return None

        num_chunks = max(key[0] for key in values) + 1
        num_steps = max(key[1] for key in values) + 1
        num_layers = max(key[2] for key in values) + 1
        sim = np.full(
            (num_chunks, num_steps, num_layers, num_pairs), np.nan, dtype=np.float32
        )
        for (chunk, step, layer), value in values.items():
            sim[chunk, step, layer] = value.float().cpu().numpy()

        run_dir = self.output_dir / f"{model_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_dir / "frame_similarity.npz",
            sim=sim,
            pairs=np.array(frame_pairs(frames_per_block), dtype=np.int32),
        )
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "model_tag": model_tag,
                    "layout": "[chunks, steps, layers, pairs]",
                    "layers_note": (
                        "layer i is the hidden state entering block i; the last "
                        "index is the output of the final block"
                    ),
                    "similarity": (
                        "mean over spatial positions of the per-position cosine "
                        "between the two frames"
                    ),
                    "num_chunks": num_chunks,
                    "num_steps": num_steps,
                    "num_layers": num_layers,
                    "num_frames_per_block": frames_per_block,
                    "pairs": frame_pairs(frames_per_block),
                    "meta": meta or {},
                },
                indent=2,
            )
        )
        logger.info("Frame similarity probe wrote %s", run_dir)
        return str(run_dir)


_recorder: FrameSimilarityRecorder | None = None
_recorder_resolved = False


def get_frame_similarity_recorder() -> FrameSimilarityRecorder | None:
    """The process-wide recorder, or ``None`` when the probe is disabled."""
    global _recorder, _recorder_resolved
    if _recorder_resolved:
        return _recorder
    _recorder_resolved = True
    output_dir = envs.SGLANG_DIFFUSION_FRAME_SIMILARITY_DIR
    if output_dir is None:
        return None

    from sglang.multimodal_gen.runtime.distributed import get_world_rank

    if get_world_rank() != 0:
        return None
    _recorder = FrameSimilarityRecorder(output_dir=output_dir)
    logger.info("Frame similarity probe enabled (dir=%s)", output_dir)
    return _recorder
