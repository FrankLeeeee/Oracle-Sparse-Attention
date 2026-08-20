# SPDX-License-Identifier: Apache-2.0
"""Per-chunk forward / attention wall-time probe for block-causal video DiTs.

Answers "how long does chunk *c* take, and how much of that is attention?" for
the Self-Forcing family (Causal Forcing, Rolling Forcing, LongLive-2, LingBot
World). A block-causal model generates a video chunk by chunk, so its cost
profile is a curve over the chunk index rather than a single number: models
with a growing context get slower per chunk, window-capped models flatten out.
The stage-level timings that ``SGLANG_DIFFUSION_STAGE_LOGGING`` reports cannot
show this — they aggregate the whole denoising stage.

Enabled by setting ``SGLANG_DIFFUSION_CHUNK_TIMING_DIR``; disabled (one
``None`` check per DiT forward and per attention call) otherwise. When on, the
probe brackets

* every DiT forward, and
* the self-attention and cross-attention module of every transformer block

with CUDA events, and sums them per chunk over that chunk's denoising steps.
Events are only resolved when the chunk changes, so nothing synchronizes inside
a chunk and the measured latency stays close to an unprobed run. Forwards that
refresh the KV cache after a chunk is finished are tagged ``cache_update`` and
kept apart from the ``denoise`` steps.

:meth:`ChunkTimingRecorder.flush` writes one ``chunk_timing.json`` per
generated video. The probe is meant for single-GPU debugging runs: it only
records on world rank 0.
"""

import json
import logging
import pathlib
import time
from contextlib import contextmanager, nullcontext

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
    CACHE_UPDATE_PASS,
    DENOISE_PASS,
)

logger = logging.getLogger(__name__)

SELF_ATTENTION = "self_attn"
CROSS_ATTENTION = "cross_attn"
FORWARD = "forward"

_NULL_SCOPE = nullcontext()


class ChunkTimingRecorder:
    """Buffers per-chunk CUDA-event timings across a generation, then writes them.

    The model calls :meth:`begin_forward` / :meth:`end_forward` once per DiT
    forward and wraps its attention modules in :meth:`region`; the denoising
    stage calls :meth:`flush` when the video is done.
    """

    def __init__(self, *, output_dir: str) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.current_pass_kind = DENOISE_PASS
        self.enabled = True
        # (chunk, pass_kind) -> region -> accumulated milliseconds
        self._totals: dict[tuple[int, str], dict[str, float]] = {}
        # (chunk, pass_kind) -> per-forward milliseconds, one entry per step
        self._forward_steps: dict[tuple[int, str], list[float]] = {}
        # not-yet-resolved (key, region, start_event, end_event) triples
        self._pending: list[
            tuple[tuple[int, str], str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._key: tuple[int, str] | None = None
        self._forward_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._num_layers = 0

    @contextmanager
    def pass_kind_scope(self, pass_kind: str):
        """Tag the forwards run inside the block (e.g. KV cache refreshes)."""
        previous = self.current_pass_kind
        self.current_pass_kind = pass_kind
        try:
            yield
        finally:
            self.current_pass_kind = previous

    @contextmanager
    def recording_scope(self, enabled: bool):
        """Gate the forwards run inside the block (e.g. a warmup pass).

        Nested scopes only ever narrow, matching the attention-map probe.
        """
        previous = self.enabled
        self.enabled = previous and enabled
        try:
            yield
        finally:
            self.enabled = previous

    def begin_forward(self, *, chunk_index: int, pass_kind: str | None = None) -> None:
        if not self.enabled or not torch.cuda.is_available():
            self._key = None
            return
        key = (chunk_index, pass_kind or self.current_pass_kind)
        if self._key is not None and key != self._key:
            # Chunk boundary: the pipeline synchronizes here anyway.
            self._resolve()
        self._key = key
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._forward_events = (start, end)

    def end_forward(self) -> None:
        if self._key is None or self._forward_events is None:
            return
        start, end = self._forward_events
        end.record()
        self._pending.append((self._key, FORWARD, start, end))
        self._forward_events = None

    @contextmanager
    def region(self, name: str):
        """Time one attention module of the forward that is currently open."""
        key = self._key
        if key is None:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((key, name, start, end))

    def note_layer_count(self, num_layers: int) -> None:
        self._num_layers = max(self._num_layers, num_layers)

    def _resolve(self) -> None:
        """Turn recorded events into milliseconds; called at chunk boundaries."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for key, region, start, end in self._pending:
            elapsed = start.elapsed_time(end)
            regions = self._totals.setdefault(key, {})
            regions[region] = regions.get(region, 0.0) + elapsed
            if region == FORWARD:
                self._forward_steps.setdefault(key, []).append(elapsed)
        self._pending = []

    def flush(self, *, model_tag: str, meta: dict | None = None) -> str | None:
        """Write ``chunk_timing.json`` for this video and reset the state."""
        self._resolve()
        totals = self._totals
        forward_steps = self._forward_steps
        num_layers = self._num_layers
        self._totals = {}
        self._forward_steps = {}
        self._key = None
        self._forward_events = None
        if not totals:
            return None

        chunks: dict[int, dict] = {}
        for (chunk_index, pass_kind), regions in totals.items():
            entry = chunks.setdefault(chunk_index, {"chunk": chunk_index})
            steps = forward_steps.get((chunk_index, pass_kind), [])
            entry[pass_kind] = {
                "steps": len(steps),
                "forward_ms": regions.get(FORWARD, 0.0),
                "self_attn_ms": regions.get(SELF_ATTENTION, 0.0),
                "cross_attn_ms": regions.get(CROSS_ATTENTION, 0.0),
                "forward_ms_per_step": steps,
            }

        run_dir = self.output_dir / f"{model_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_tag": model_tag,
            "num_layers": num_layers,
            "pass_kinds": [DENOISE_PASS, CACHE_UPDATE_PASS],
            "region_note": (
                "forward_ms brackets the whole DiT forward; self_attn_ms and "
                "cross_attn_ms bracket the attention modules inside it, summed "
                "over layers and over the chunk's steps"
            ),
            "meta": meta or {},
            "chunks": [chunks[index] for index in sorted(chunks)],
        }
        path = run_dir / "chunk_timing.json"
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Chunk timing probe wrote %s", path)
        return str(run_dir)


_recorder: ChunkTimingRecorder | None = None
_recorder_resolved = False


def get_chunk_timing_recorder() -> ChunkTimingRecorder | None:
    """The process-wide recorder, or ``None`` when the probe is disabled."""
    global _recorder, _recorder_resolved
    if _recorder_resolved:
        return _recorder
    _recorder_resolved = True
    output_dir = envs.SGLANG_DIFFUSION_CHUNK_TIMING_DIR
    if output_dir is None:
        return None

    from sglang.multimodal_gen.runtime.distributed import get_world_rank

    if get_world_rank() != 0:
        return None
    _recorder = ChunkTimingRecorder(output_dir=output_dir)
    logger.info("Chunk timing probe enabled (dir=%s)", output_dir)
    return _recorder


def attention_timing(name: str):
    """Context manager timing one attention module, or a no-op when disabled."""
    recorder = get_chunk_timing_recorder()
    if recorder is None:
        return _NULL_SCOPE
    return recorder.region(name)


@contextmanager
def timing_pass_kind_scope(pass_kind: str):
    """Tag a block of forwards, whether or not the probe is enabled."""
    recorder = get_chunk_timing_recorder()
    if recorder is None:
        yield
        return
    with recorder.pass_kind_scope(pass_kind):
        yield
