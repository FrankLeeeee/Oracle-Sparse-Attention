# SPDX-License-Identifier: Apache-2.0
"""The contract every sparse-attention method implements.

A method sees one self-attention call of a block-causal video DiT and either
returns an output tensor or declines (``None``), in which case the caller runs
its ordinary dense attention. Declining is always safe and always available:
methods decline for the first chunks (nothing to drop yet), for KV-cache
layouts they cannot map to latent frames, and whenever their own selection
would keep everything anyway.
"""

import abc
from typing import ClassVar

import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.sparse.context import (
    ChunkGeometry,
    KeySegments,
    VisibleLayout,
    visible_layout,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.kernel import (
    SparseAttentionPlan,
    sparse_attention,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Density accounting is *sampled*. Measuring a plan's density is a handful of
# tiny CUDA launches, and a Self-Forcing 20-second run makes ~4000 attention
# calls, so doing it on every one costs ~0.2 ms per call — a fifth of the
# attention it is supposed to be measuring. One sample in 64 is statistically
# ample over thousands of calls and costs nothing.
_DENSITY_SAMPLE_INTERVAL = 64
# How often the running figure is logged. One report costs a host sync.
_DENSITY_REPORT_INTERVAL = 512


class SparseAttentionCall(msgspec.Struct, frozen=True):
    """One layer's self-attention call, in the caller's layout.

    ``query`` is ``[batch, q_len, heads, head_dim]``; ``key``/``value`` are the
    visible KV-cache view ``[batch, kv_len, heads, head_dim]``.
    ``key_segments`` gives the global ``(token_start, length)`` ranges the view
    covers, in key order.
    """

    layer_index: int
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_segments: KeySegments
    head_start: int
    num_local_heads: int
    softmax_scale: float

    @property
    def head_dim(self) -> int:
        return self.query.shape[-1]


class SparseAttentionExecution(msgspec.Struct, frozen=True):
    """What a method decided to run: a plan plus the tensors it applies to.

    Methods that reorder the sequence (SVG2's semantic clustering, FAST-AR's
    LSH buckets) or rewrite the cache (FAST-AR's TempCache) hand back their own
    ``query``/``key``/``value``; ``query_permutation[b, i, h]`` is then the
    original row that permuted row ``i`` of head ``h`` came from, and the output
    is scattered back before it leaves :meth:`SparseAttentionBackend.attend`.
    """

    plan: SparseAttentionPlan
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    query_permutation: torch.Tensor | None = None  # [batch, q_len, heads] long


class SparseAttentionBackend(abc.ABC):
    """Base for the methods: geometry stamping, guards, and dense fallback."""

    name: ClassVar[str]

    def __init__(self) -> None:
        self._geometry: ChunkGeometry | None = None
        self._warned: set[str] = set()
        self._sampled_density_sum: torch.Tensor | None = None
        self._density_samples = 0
        self._sparse_calls = 0
        self._dense_calls = 0
        self._calls_since_report = 0

    def begin_forward(self, geometry: ChunkGeometry) -> None:
        """Stamp the geometry shared by every layer of one DiT forward."""
        self._geometry = geometry
        self._on_begin_forward(geometry)

    def _on_begin_forward(self, geometry: ChunkGeometry) -> None:
        """Hook for methods with per-forward state."""

    @property
    def geometry(self) -> ChunkGeometry | None:
        return self._geometry

    def warn_dense_once(self, reason: str) -> None:
        if reason in self._warned:
            return
        self._warned.add(reason)
        logger.warning("%s attention falls back to dense: %s", self.name, reason)

    def attend(self, call: SparseAttentionCall) -> torch.Tensor | None:
        """Sparse output for this call, or ``None`` to let the caller go dense."""
        layout = self._layout(call)
        if layout is None:
            return None
        execution = self.prepare(call, layout)
        if execution is None:
            self._record_density(None, kv_len=call.key.shape[1])
            return None
        # Normalised by the *caller's* key count, not the execution's: FAST-AR
        # hands back a compacted cache, and the work it saved by compacting is
        # exactly what the density is supposed to show.
        self._record_density(execution.plan, kv_len=call.key.shape[1])
        out = sparse_attention(
            query=execution.query,
            key=execution.key,
            value=execution.value,
            plan=execution.plan,
            softmax_scale=call.softmax_scale,
        )
        if execution.query_permutation is None:
            return out
        restored = torch.empty_like(out)
        index = execution.query_permutation[..., None].expand_as(out)
        restored.scatter_(1, index, out)
        return restored

    @abc.abstractmethod
    def prepare(
        self, call: SparseAttentionCall, layout: VisibleLayout
    ) -> SparseAttentionExecution | None:
        """Decide what to run for this call, or ``None`` for dense."""

    def _record_density(self, plan: SparseAttentionPlan | None, *, kv_len: int) -> None:
        """Accumulate the fraction of keys actually read, and log it now and then.

        The density a method *achieves on real activations* is the number every
        comparison turns on, and it is not predictable from the config: it falls
        out of the calibration, the estimator or the clustering. Declined calls
        count as fully dense, so the running figure covers the whole run rather
        than only its sparse part. Accumulating on device costs a few
        microseconds; the host sync happens once every
        ``_DENSITY_REPORT_INTERVAL`` calls.
        """
        if plan is None:
            self._dense_calls += 1
        else:
            self._sparse_calls += 1
            if self._sparse_calls % _DENSITY_SAMPLE_INTERVAL == 1:
                heads, q_blocks = plan.range_counts.shape
                fraction = plan.kept_tokens().sum().float() / float(
                    heads * q_blocks * kv_len
                )
                fraction = fraction.clamp(max=1.0)
                if self._sampled_density_sum is None:
                    self._sampled_density_sum = torch.zeros_like(fraction)
                self._sampled_density_sum += fraction
                self._density_samples += 1
        self._calls_since_report += 1
        if self._calls_since_report < _DENSITY_REPORT_INTERVAL:
            return
        self._calls_since_report = 0
        logger.info(
            "%s attention density so far: %.3f over %d calls (%d dense)",
            self.name,
            self.density(),
            self._sparse_calls + self._dense_calls,
            self._dense_calls,
        )

    def density(self) -> float:
        """Fraction of keys read so far, dense fallbacks counted as 1.0."""
        calls = self._sparse_calls + self._dense_calls
        if calls == 0:
            return 1.0
        sampled = (
            1.0
            if self._density_samples == 0
            else float(self._sampled_density_sum.item()) / self._density_samples
        )
        return (self._dense_calls + self._sparse_calls * sampled) / calls

    def _layout(self, call: SparseAttentionCall) -> VisibleLayout | None:
        geometry = self._geometry
        if geometry is None:
            self.warn_dense_once("no forward geometry was stamped")
            return None
        if call.key_segments is None:
            self.warn_dense_once("this KV-cache view is not mapped to latent frames")
            return None
        head_dim = call.head_dim
        if head_dim & (head_dim - 1) != 0 or head_dim < 16:
            self.warn_dense_once(f"head_dim {head_dim} is not a power of two >= 16")
            return None
        kv_len = call.key.shape[1]
        if sum(length for _, length in call.key_segments) != kv_len:
            self.warn_dense_once("key segments disagree with the KV view length")
            return None
        layout = visible_layout(
            call.key_segments, geometry=geometry, query_tokens=call.query.shape[1]
        )
        if layout is None:
            self.warn_dense_once("the KV view is not aligned to whole latent frames")
            return None
        return layout


class LayoutCache:
    """Per-layer memo of anything derived only from the visible layout.

    The block-causal pipelines run several denoising steps per chunk over an
    identical visible layout, so a method whose selection does not depend on the
    tensor values (OSA once calibrated, Radial, SVG1's two candidate masks) pays
    for planning once per ``(chunk, layer)`` instead of once per step.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[tuple, object]] = {}

    def get(self, layer_index: int, signature: tuple) -> tuple[bool, object]:
        entry = self._entries.get(layer_index)
        if entry is not None and entry[0] == signature:
            return True, entry[1]
        return False, None

    def put(self, layer_index: int, signature: tuple, value: object) -> None:
        self._entries[layer_index] = (signature, value)

    def clear(self) -> None:
        self._entries.clear()
