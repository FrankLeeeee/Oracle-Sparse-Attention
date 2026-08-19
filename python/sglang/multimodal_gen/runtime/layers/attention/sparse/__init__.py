# SPDX-License-Identifier: Apache-2.0
"""Sparse attention for block-causal video DiTs, selected by ``--sparse-attention``.

``--sparse-attention`` unset keeps the dense path untouched — nothing in this
package is constructed or called. Otherwise it names one method:

======================  ====================================================
``osa``                 Oracle Sparse Attention: per-head within-frame tile
                        pattern, calibrated on chunk 0 at runtime and
                        replicated across every history frame.
``xattention``          X-Attention's antidiagonal block scoring.
``svg1`` / ``svg2``     Sparse VideoGen v1 (spatial/temporal heads) and v2
                        (semantic clustering).
``radial``              Radial Attention's log-decaying spatiotemporal mask.
``fastar``              FAST-AR's TempCache + AnnSA.
``lightforcing``        Light Forcing's chunk-aware sparsity schedule +
                        hierarchical frame/block top-k selection.
======================  ====================================================

``--sparse-attention-config`` passes the method's knobs as JSON or ``k=v``
pairs; the keys are the fields of that method's config struct.
"""

from collections.abc import Mapping
from typing import Any

import msgspec

from sglang.multimodal_gen.runtime.layers.attention.sparse.base import (
    SparseAttentionBackend,
    SparseAttentionCall,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.context import ChunkGeometry
from sglang.multimodal_gen.runtime.layers.attention.sparse.fastar import (
    FastArAttention,
    FastArConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing import (
    LightForcingAttention,
    LightForcingConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.osa import (
    OracleSparseAttention,
    OsaConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.radial import (
    RadialAttention,
    RadialConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.svg import (
    Svg1Attention,
    Svg1Config,
    Svg2Attention,
    Svg2Config,
)
from sglang.multimodal_gen.runtime.layers.attention.sparse.xattention import (
    XAttention,
    XAttentionConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_METHODS: dict[str, tuple[type[SparseAttentionBackend], type[msgspec.Struct]]] = {
    "osa": (OracleSparseAttention, OsaConfig),
    "xattention": (XAttention, XAttentionConfig),
    "svg1": (Svg1Attention, Svg1Config),
    "svg2": (Svg2Attention, Svg2Config),
    "radial": (RadialAttention, RadialConfig),
    "fastar": (FastArAttention, FastArConfig),
    "lightforcing": (LightForcingAttention, LightForcingConfig),
}

SPARSE_ATTENTION_METHODS = tuple(_METHODS)

__all__ = [
    "ChunkGeometry",
    "SPARSE_ATTENTION_METHODS",
    "SparseAttentionBackend",
    "SparseAttentionCall",
    "build_sparse_attention_backend",
    "get_sparse_attention_backend",
    "set_sparse_attention_backend",
]


def build_sparse_attention_backend(
    method: str | None, config: Mapping[str, Any] | None = None
) -> SparseAttentionBackend | None:
    """Instantiate the named method, or ``None`` for dense attention."""
    if method is None or method == "none":
        return None
    if method not in _METHODS:
        raise ValueError(
            f"unknown --sparse-attention {method!r}; expected one of "
            f"{', '.join(SPARSE_ATTENTION_METHODS)}"
        )
    backend_cls, config_cls = _METHODS[method]
    fields = msgspec.structs.fields(config_cls)
    known = {field.name for field in fields}
    unknown = set(config or {}) - known
    if unknown:
        # msgspec.convert ignores unknown keys, which would turn a typo in
        # --sparse-attention-config into a silently unapplied setting.
        raise ValueError(
            f"unknown --sparse-attention-config key(s) for {method}: "
            f"{', '.join(sorted(unknown))}; expected any of "
            f"{', '.join(sorted(known))}"
        )
    return backend_cls(msgspec.convert(dict(config or {}), config_cls, strict=False))


_backend: SparseAttentionBackend | None = None
_resolved = False


def get_sparse_attention_backend() -> SparseAttentionBackend | None:
    """The process-wide backend, or ``None`` when the feature is off.

    Resolved once from the server args; the DiT calls this on every attention
    layer, so the miss path must stay a single boolean check.
    """
    global _backend, _resolved
    if _resolved:
        return _backend
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    server_args = get_global_server_args()
    _backend = build_sparse_attention_backend(
        server_args.sparse_attention, server_args.sparse_attention_config
    )
    _resolved = True
    if _backend is not None:
        logger.info(
            "Sparse attention enabled: %s %s",
            _backend.name,
            server_args.sparse_attention_config or "(defaults)",
        )
    return _backend


def set_sparse_attention_backend(backend: SparseAttentionBackend | None) -> None:
    """Install a backend directly, bypassing the server args (tests, tools)."""
    global _backend, _resolved
    _backend = backend
    _resolved = True


def reset_sparse_attention_backend() -> None:
    global _backend, _resolved
    _backend = None
    _resolved = False
