# SPDX-License-Identifier: Apache-2.0
"""Dump raw post-RoPE Q/K for selected (layer, head, chunk, step) attention calls.

``sglang generate`` runs the DiT in a spawned worker, so the dump is injected
the only way that reaches every child interpreter: this module sits on
``PYTHONPATH`` and Python imports it at startup. It installs a post-import
hook on the attention-map probe module and wraps
:meth:`ChunkAttentionRecorder.record` — the one place the dense attention path
hands over the exact post-RoPE query/key tensors together with the chunk/step
geometry. The probe itself must be enabled (``SGLANG_DIFFUSION_ATTENTION_MAP_DIR``
set) so that ``record`` is called at all; pair it with
``SGLANG_DIFFUSION_ATTENTION_MAP_QK_ONLY=1`` and no ``..._QK_CHUNKS`` so the
built-in probe body is a no-op and this hook is the only thing that writes.

Raw Q/K is dumped instead of softmax maps because it is ~50x smaller (the
full probability matrix of the last 720p chunk is ~1.6 GB per head, fp16 Q/K
~25 MB) and lets every downstream artifact — full-key-axis softmax maps at any
display pooling, and the per-(query frame, key frame) softmax of the
similarity study — be recomputed offline, exactly.

Env:
  ``QKDUMP_DIR``     output directory (required; hook inert without it)
  ``QKDUMP_SPEC``    per-layer head ids, e.g. ``"0:0,1;14:2;29:3"``
  ``QKDUMP_CHUNKS``  comma-separated chunk indices to keep
  ``QKDUMP_STEPS``   comma-separated denoising-step indices to keep
"""

import importlib.util
import os
import pathlib
import sys
from importlib.abc import MetaPathFinder

TARGET = "sglang.multimodal_gen.runtime.utils.attention_map_probe"
DUMP_DIR = os.environ.get("QKDUMP_DIR", "")
SPEC = os.environ.get("QKDUMP_SPEC", "")
CHUNKS = {int(c) for c in os.environ.get("QKDUMP_CHUNKS", "").split(",") if c.strip()}
STEPS = {int(s) for s in os.environ.get("QKDUMP_STEPS", "").split(",") if s.strip()}


def _install(module) -> None:
    import numpy as np
    import torch

    recorder_cls = module.ChunkAttentionRecorder
    original_record = recorder_cls.record
    head_spec = module.parse_head_spec(SPEC) or {}
    out_dir = pathlib.Path(DUMP_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # (chunk, layer) -> denoise records seen so far == this call's step index.
    step_counter: dict[tuple[int, int], int] = {}
    failed: list[str] = []

    @torch.no_grad()
    def dump(scope, *, layer_index, chunk, step, query, key, key_segments) -> None:
        heads = head_spec.get(layer_index, head_spec.get(-1))
        index = torch.as_tensor(heads, device=query.device)
        np.savez(
            out_dir / f"qk_L{layer_index:02d}_c{chunk}_s{step}.npz",
            query=query[0].index_select(1, index).to(torch.float16).cpu().numpy(),
            key=key[0].index_select(1, index).to(torch.float16).cpu().numpy(),
            head_ids=np.asarray(heads),
            layer=layer_index,
            chunk=chunk,
            step=step,
            frame_seqlen=scope.frame_seqlen,
            num_frames_per_block=scope.num_frames_per_block,
            query_token_start=scope.query_token_start,
            grid_height=scope.grid_height,
            grid_width=scope.grid_width,
            key_segments=np.asarray(list(key_segments), dtype=np.int64),
        )

    def record(self, *, layer_index, query, key, key_segments):
        scope = self._scope
        if (
            scope is not None
            and scope.pass_kind == module.DENOISE_PASS
            and (layer_index in head_spec or -1 in head_spec)
        ):
            chunk = scope.query_token_start // scope.chunk_tokens
            counter_key = (chunk, layer_index)
            step = step_counter.get(counter_key, 0)
            step_counter[counter_key] = step + 1
            if chunk in CHUNKS and step in STEPS:
                try:
                    dump(
                        scope,
                        layer_index=layer_index,
                        chunk=chunk,
                        step=step,
                        query=query,
                        key=key,
                        key_segments=key_segments,
                    )
                except Exception as exc:  # noqa: BLE001 - never break the run
                    if not failed:
                        failed.append(str(exc))
                        print(f"[qkdump] dump failed: {exc!r}", file=sys.stderr)
        return original_record(
            self,
            layer_index=layer_index,
            query=query,
            key=key,
            key_segments=key_segments,
        )

    recorder_cls.record = record
    print(f"[qkdump] instrumented attention probe, writing {out_dir}", file=sys.stderr)


class _PostImportHook(MetaPathFinder):
    """Let the normal machinery load the target module, then patch it."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


if DUMP_DIR:
    sys.meta_path.insert(0, _PostImportHook())
