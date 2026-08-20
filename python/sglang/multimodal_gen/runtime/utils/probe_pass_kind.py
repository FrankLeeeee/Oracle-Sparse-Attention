# SPDX-License-Identifier: Apache-2.0
"""Which kind of DiT forward is currently running, for the debugging probes.

A block-causal video model runs two kinds of forward per chunk: the denoising
steps themselves, and a pass that re-runs the finished chunk at a fixed low
timestep to refresh its KV cache. Every probe needs to tell them apart — a
cache refresh is not a denoising step, and counting it as one shifts each
chunk's step indices and pollutes per-step statistics.

The denoising stage marks the refresh with :func:`pass_kind_scope`; probes read
:func:`current_pass_kind`. Kept in its own module so a probe does not have to
depend on another probe to learn this.
"""

from contextlib import contextmanager

from sglang.multimodal_gen.runtime.utils.attention_map_probe import (
    CACHE_UPDATE_PASS,
    DENOISE_PASS,
)

__all__ = [
    "CACHE_UPDATE_PASS",
    "DENOISE_PASS",
    "current_pass_kind",
    "pass_kind_scope",
]

_current = DENOISE_PASS


def current_pass_kind() -> str:
    return _current


@contextmanager
def pass_kind_scope(pass_kind: str):
    """Tag the forwards run inside the block (e.g. KV cache refreshes)."""
    global _current
    previous = _current
    _current = pass_kind
    try:
        yield
    finally:
        _current = previous
