# SPDX-License-Identifier: Apache-2.0
"""Measure OSA's per-chunk attention recall, without touching production code.

``sglang generate`` runs the DiT in a spawned worker, so the measurement is
injected the only way that reaches every child interpreter: this module sits on
``PYTHONPATH`` and Python imports it at startup. It installs a post-import hook
on the OSA backend and wraps :meth:`OracleSparseAttention.attend`.

For every sparse call it recomputes ``softmax(q k^T)`` on a strided subset of
queries and folds the key axis into the same per-frame spatial tiles OSA
selects over, which gives two numbers at the same token budget:

``recall_frozen``
    attention mass the *chunk-0-frozen* tile set actually captures.
``recall_refreshed``
    mass the best tile set *for this chunk* would capture — i.e. what a
    re-calibration at this chunk would buy.

Their gap is the staleness of the chunk-0 pattern, which is what the design
doc's 满窗刷新 / 周期重校准 proposals are about. Whole-kept frames (own chunk,
sink, recent) are counted in both, since neither policy can drop them.
"""

import importlib.util
import json
import os
import sys
import threading
from importlib.abc import MetaPathFinder

TARGET = "sglang.multimodal_gen.runtime.layers.attention.sparse.osa"
OUT_PATH = os.environ.get("OSA_RECALL_OUT")
QUERY_STRIDE = int(os.environ.get("OSA_RECALL_QUERY_STRIDE", "32"))
QUERY_TILE = int(os.environ.get("OSA_RECALL_QUERY_TILE", "32"))
# Optional: dump one layer's raw per-tile mass profile alongside the summary.
TILE_DUMP_LAYER = int(os.environ.get("OSA_RECALL_TILE_LAYER", "-1"))
TILE_DUMP_PATH = os.environ.get("OSA_RECALL_TILE_OUT", "")
# Optional: dump the full frame-to-frame *section* map [heads, q_tile, k_tile],
# which keeps the query axis the tile profile marginalises away.
SECTION_LAYER = int(os.environ.get("OSA_SECTION_LAYER", "-1"))
SECTION_CHUNKS = {
    int(c) for c in os.environ.get("OSA_SECTION_CHUNKS", "").split(",") if c.strip()
}
SECTION_DIR = os.environ.get("OSA_SECTION_DIR", "")


def _install(module) -> None:
    import torch

    frame_ages = module.frame_ages
    backend_cls = module.OracleSparseAttention
    original_attend = backend_cls.attend
    write_lock = threading.Lock()
    steps: dict[tuple[int, int], int] = {}
    section_steps: dict[int, int] = {}
    failed: list[str] = []

    def emit(record: dict) -> None:
        with write_lock:
            with open(OUT_PATH, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    @torch.no_grad()
    def dump_section(self, call, layout) -> None:
        """Block-reduced frame-to-frame map, per query tile AND key tile.

        Splits key frames into the group OSA replicates tiles over and the
        group it keeps whole, so the deployment-time target is separable from
        the own chunk. At chunk 0 there is no history, so the split is instead
        self-sections (query frame == key frame) vs cross-sections.
        """
        import numpy as np

        query = call.query[:1]
        key = call.key[:1]
        heads, q_len = query.shape[2], query.shape[1]
        frame_seqlen = layout.frame_seqlen
        num_frames = layout.num_frames
        tile = self._config.spatial_tile
        num_tiles = frame_seqlen // tile
        query_frames = q_len // frame_seqlen
        chunk_index = int(layout.query_chunk_index)

        if chunk_index == 0:
            groups = {"self": None, "cross": None}  # decided per query frame
        else:
            own = layout.frames_of_offset(0)
            ages = frame_ages(layout, query_chunk_offset=0)
            full = self._whole_frame_mask(layout, own, ages)
            groups = {"history": ~full, "whole": full}

        maps = {
            name: torch.zeros(
                heads, num_tiles, num_tiles, dtype=torch.float64, device=query.device
            )
            for name in groups
        }
        # Cap the score buffer at roughly 800 MB.
        per_query = heads * num_frames * frame_seqlen * 4
        tiles_per_block = max(1, min(8, int(8e8 // (per_query * tile))))
        for q_frame in range(query_frames):
            base = q_frame * frame_seqlen
            for group_start in range(0, num_tiles, tiles_per_block):
                count = min(tiles_per_block, num_tiles - group_start)
                start = base + group_start * tile
                block = query[:, start : start + count * tile]
                scores = torch.einsum("bqhd,bkhd->hqk", block, key).float()
                probs = torch.softmax(scores * call.softmax_scale, dim=-1)
                per_frame = probs.view(heads, count * tile, num_frames, frame_seqlen)
                body = per_frame[..., : num_tiles * tile].reshape(
                    heads, count * tile, num_frames, num_tiles, tile
                ).sum(-1)
                for name in groups:
                    if chunk_index == 0:
                        keep = np.arange(num_frames) == q_frame
                        if name == "cross":
                            keep = ~keep
                    else:
                        keep = groups[name]
                    selected = body[:, :, torch.from_numpy(keep).to(body.device), :]
                    reduced = selected.sum(2).view(heads, count, tile, num_tiles).sum(2)
                    maps[name][:, group_start : group_start + count, :] += reduced.double()

        for name, value in maps.items():
            path = (
                f"{SECTION_DIR}/section_L{call.layer_index}"
                f"_c{chunk_index}_s{section_steps.get(chunk_index, 0)}_{name}.npy"
            )
            np.save(path, value.cpu().numpy().astype("float32"))
        section_steps[chunk_index] = section_steps.get(chunk_index, 0) + 1

    @torch.no_grad()
    def measure_chunk_2d(self, call, layout, plan) -> None:
        """Recall of the 2-D plan vs a per-query-tile refreshed oracle."""
        import numpy as np

        query = call.query[:1]
        key = call.key[:1]
        heads = query.shape[2]
        frame_seqlen = layout.frame_seqlen
        num_frames = layout.num_frames
        tile = self._config.spatial_tile
        num_tiles = frame_seqlen // tile
        q_tiles = plan.q_tiles_per_frame
        query_tile = plan.query_tile
        device = query.device

        own = layout.frames_of_offset(0)
        ages = frame_ages(layout, query_chunk_offset=0)
        full = self._whole_frame_mask(layout, own, ages)
        num_full = int(full.sum())
        num_other = num_frames - num_full
        if num_other <= 0 or num_tiles <= 0:
            return
        band = plan.starts.shape[2] - num_full * ((frame_seqlen + tile - 1) // tile)
        band = int(band // max(num_other, 1))
        full_mask = torch.from_numpy(full).to(device)

        # The frozen per-(head, q_tile) history tile sets, from the plan itself.
        hist_start = plan.starts.shape[2] - band * num_other
        hist_entries = plan.starts[:, :, hist_start:]  # [heads, q_tiles, band*n_hist]
        within = (hist_entries % frame_seqlen) // tile
        frozen_tiles = within[:, :, :band].long()  # replicated: first frame's tiles

        positions = torch.arange(0, query.shape[1], QUERY_STRIDE, device=device)
        sampled = query[:, ::QUERY_STRIDE]
        bucket = ((positions % frame_seqlen) // query_tile).long()
        num_sampled = sampled.shape[1]
        mass_full = torch.zeros(heads, dtype=torch.float64, device=device)
        mass_qt = torch.zeros(
            heads, q_tiles, num_tiles, dtype=torch.float64, device=device
        )
        for start in range(0, num_sampled, QUERY_TILE):
            chunk = sampled[:, start : start + QUERY_TILE]
            scores = torch.einsum("bqhd,bkhd->hqk", chunk, key).float()
            probs = torch.softmax(scores * call.softmax_scale, dim=-1)
            per_frame = probs.view(heads, -1, num_frames, frame_seqlen)
            mass_full += per_frame[:, :, full_mask, :].sum((1, 2, 3)).double()
            other = per_frame[:, :, ~full_mask, :]
            body = other[..., : num_tiles * tile].reshape(
                heads, other.shape[1], other.shape[2], num_tiles, tile
            ).sum((2, 4)).double()  # [heads, nq, num_tiles]
            mass_qt.index_add_(1, bucket[start : start + QUERY_TILE], body)

        # Per query tile: mass its frozen tiles capture vs its own top-band.
        # mass_qt sums over the bucket's sampled queries, so summing buckets
        # and dividing by the sample count gives the per-query mean directly.
        frozen = mass_qt.gather(2, frozen_tiles).sum((1, 2))  # [heads]
        refreshed = mass_qt.topk(band, dim=2).values.sum((1, 2))
        scale = float(num_sampled)
        whole_only = (mass_full / scale).cpu()
        recall_frozen = ((mass_full + frozen) / scale).cpu()
        recall_refreshed = ((mass_full + refreshed) / scale).cpu()

        chunk_index = int(layout.query_chunk_index)
        key_id = (call.layer_index, chunk_index)
        step = steps.get(key_id, 0)
        steps[key_id] = step + 1
        emit(
            {
                "chunk": chunk_index,
                "layer": int(call.layer_index),
                "step": step,
                "kv_frames": int(num_frames),
                "whole_frames": num_full,
                "tiles_kept": band,
                "num_tiles": int(num_tiles),
                "density": float(plan.density),
                "recall_frozen": [round(v, 6) for v in recall_frozen.tolist()],
                "recall_refreshed": [round(v, 6) for v in recall_refreshed.tolist()],
                "whole_only": [round(v, 6) for v in whole_only.tolist()],
            }
        )

    def attend(self, call):
        if SECTION_DIR and call.layer_index == SECTION_LAYER and not self.in_cache_update:
            try:
                layout = self._layout(call)
                if layout is not None and layout.query_chunk_index in SECTION_CHUNKS:
                    dump_section(self, call, layout)
            except Exception as exc:  # noqa: BLE001
                print(f"[osa-recall] section dump failed: {exc!r}", file=sys.stderr)
        out = original_attend(self, call)
        if not OUT_PATH or out is None or self.in_cache_update:
            return out
        try:
            layout = self._layout(call)
            if layout is None or layout.query_chunk_index < 1:
                return out
            if call.layer_index not in self._section_order:
                return out
            # By now the plan is in the per-(layer, chunk) cache, so this is a
            # pure lookup — no re-calibration, no state change.
            plan = self._prepare_plan(call, layout)
            if plan is None:
                return out
            if isinstance(plan, module.BlockSparsePlan):
                measure_chunk_2d(self, call, layout, plan)
            # Rolling-window tuples are not measured; the recall study runs on
            # single-chunk models.
        except Exception as exc:  # noqa: BLE001 - never break the run
            if not failed:
                failed.append(str(exc))
                print(f"[osa-recall] measurement disabled: {exc!r}", file=sys.stderr)
        return out

    backend_cls.attend = attend
    print(f"[osa-recall] instrumented OSA, writing {OUT_PATH}", file=sys.stderr)


class _PostImportHook(MetaPathFinder):
    """Let the normal machinery load the OSA module, then patch it."""

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


if OUT_PATH:
    sys.meta_path.insert(0, _PostImportHook())
