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
LF_TARGET = "sglang.multimodal_gen.runtime.layers.attention.sparse.lightforcing"
OUT_PATH = os.environ.get("OSA_RECALL_OUT")
LF_OUT_PATH = os.environ.get("LF_RECALL_OUT")
QUERY_STRIDE = int(os.environ.get("OSA_RECALL_QUERY_STRIDE", "32"))
QUERY_TILE = int(os.environ.get("OSA_RECALL_QUERY_TILE", "32"))
# Optional: dump one layer's raw per-tile mass profile alongside the summary.
TILE_DUMP_LAYER = int(os.environ.get("OSA_RECALL_TILE_LAYER", "-1"))
TILE_DUMP_PATH = os.environ.get("OSA_RECALL_TILE_OUT", "")
# Optional: dump the full frame-to-frame *section* map [heads, q_tile, k_tile],
# which keeps the query axis the tile profile marginalises away. Accepts a
# comma-separated layer list.
SECTION_LAYERS = {
    int(l) for l in os.environ.get("OSA_SECTION_LAYER", "-1").split(",") if l.strip()
} - {-1}
SECTION_CHUNKS = {
    int(c) for c in os.environ.get("OSA_SECTION_CHUNKS", "").split(",") if c.strip()
}
SECTION_DIR = os.environ.get("OSA_SECTION_DIR", "")
# Optional: analysis granularity (tokens) of the section map, decoupled from
# the config's spatial_tile so a fine map (e.g. 4-token granules) can be
# dumped while OSA executes at its normal 64-token kernel tile.
SECTION_TILE = int(os.environ.get("OSA_SECTION_TILE", "0"))
# Optional: per-chunk attention mass and plan recall split by frame group
# (sink / own chunk / recent / history), for the anchoring analysis.
GROUP_OUT = os.environ.get("OSA_GROUP_OUT", "")
# Exact plan-based measurement instead of the legacy uniform-band one. Works
# for demand-weighted / scheduled plans, whose per-frame tile counts vary; the
# legacy path assumes a uniform band and would silently mis-slice them.
EXACT = os.environ.get("OSA_RECALL_EXACT", "") == "1"


def _install(module) -> None:
    import torch

    frame_ages = module.frame_ages
    backend_cls = module.OracleSparseAttention
    original_attend = backend_cls.attend
    write_lock = threading.Lock()
    steps: dict[tuple[int, int], int] = {}
    section_steps: dict[tuple[int, int], int] = {}
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
        tile = SECTION_TILE or self._config.spatial_tile
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

        step_key = (call.layer_index, chunk_index)
        for name, value in maps.items():
            path = (
                f"{SECTION_DIR}/section_L{call.layer_index}"
                f"_c{chunk_index}_s{section_steps.get(step_key, 0)}_{name}.npy"
            )
            np.save(path, value.cpu().numpy().astype("float32"))
        section_steps[step_key] = section_steps.get(step_key, 0) + 1

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
        band = plan.starts.shape[2] - num_full * (frame_seqlen // tile)
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

    @torch.no_grad()
    def measure_plan_exact(self, call, layout, plan) -> None:
        """Recall of the plan as built, plus a free per-row oracle.

        ``recall_frozen``: the mass the plan's actual key blocks capture,
        summed straight off ``plan.starts`` — exact for any allocation
        (uniform, demand-weighted, scheduled). ``recall_free``: the best
        per-(head, query tile) set of the same *number* of 64-token tiles,
        chosen freely over the whole visible KV (whole-frame exemptions and
        replication both lifted) — the structural headroom at OSA's own
        selection granularity.
        """
        query = call.query[:1]
        key = call.key[:1]
        heads = query.shape[2]
        kv_len = key.shape[1]
        frame_seqlen = layout.frame_seqlen
        key_tile = self._config.spatial_tile
        n_blocks = plan.starts.shape[2]
        q_tiles = plan.starts.shape[1]
        device = query.device

        positions = torch.arange(0, query.shape[1], QUERY_STRIDE, device=device)
        sampled = query[:, ::QUERY_STRIDE]
        bucket = ((positions % frame_seqlen) // plan.query_tile).clamp(
            max=q_tiles - 1
        ).long()
        num_sampled = sampled.shape[1]
        k_tiles_glob = kv_len // key_tile

        frozen = torch.zeros(heads, dtype=torch.float64, device=device)
        tile_mass = torch.zeros(
            heads, num_sampled, k_tiles_glob, dtype=torch.float32, device=device
        )
        starts = plan.starts.long()  # [heads, q_tiles, n_blocks]
        for start in range(0, num_sampled, QUERY_TILE):
            chunk = sampled[:, start : start + QUERY_TILE]
            scores = torch.einsum("bqhd,bkhd->hqk", chunk, key).float()
            probs = torch.softmax(scores * call.softmax_scale, dim=-1)
            cum = torch.nn.functional.pad(probs.cumsum(-1), (1, 0))
            idx = starts[:, bucket[start : start + QUERY_TILE]]  # [h, nq, n]
            got = cum.gather(2, (idx + key_tile).clamp(max=kv_len)) - cum.gather(
                2, idx
            )
            frozen += got.sum((1, 2)).double()
            tile_mass[:, start : start + QUERY_TILE] = (
                probs[..., : k_tiles_glob * key_tile]
                .view(heads, -1, k_tiles_glob, key_tile)
                .sum(-1)
            )

        # Free oracle: per (head, query tile) the top-n_blocks aligned tiles
        # of the bucket-mean mass, evaluated on each query's own mass.
        free = torch.zeros(heads, dtype=torch.float64, device=device)
        for b in torch.unique(bucket):
            rows = tile_mass[:, bucket == b]  # [h, nb, k_tiles_glob]
            best = torch.topk(
                rows.mean(1), min(n_blocks, k_tiles_glob), dim=-1
            ).indices
            free += rows.gather(
                2, best[:, None, :].expand(-1, rows.shape[1], -1)
            ).sum((1, 2)).double()

        chunk_index = int(layout.query_chunk_index)
        key_id = (call.layer_index, chunk_index)
        step = steps.get(key_id, 0)
        steps[key_id] = step + 1
        emit(
            {
                "chunk": chunk_index,
                "layer": int(call.layer_index),
                "step": step,
                "kv_frames": int(layout.num_frames),
                "n_blocks": int(n_blocks),
                "density": float(plan.density),
                "recall_frozen": [
                    round(v, 6) for v in (frozen / num_sampled).tolist()
                ],
                "recall_free": [round(v, 6) for v in (free / num_sampled).tolist()],
            }
        )

    @torch.no_grad()
    def measure_groups(self, call, layout, plan) -> None:
        """Dense demand vs plan recall, per frame group.

        ``demand[g]``: fraction of a query's softmax mass that lands on group
        g's frames. ``recall[g]``: the fraction of that group's mass the plan
        actually reads. Sink anchoring shows up as high sink demand with low
        sink recall.
        """
        import numpy as np

        query = call.query[:1]
        key = call.key[:1]
        heads = query.shape[2]
        frame_seqlen = layout.frame_seqlen
        num_frames = layout.num_frames
        tile = self._config.spatial_tile
        num_tiles = frame_seqlen // tile
        q_tiles = plan.q_tiles_per_frame
        device = query.device

        own = layout.frames_of_offset(0)
        ages = frame_ages(layout, query_chunk_offset=0)
        sink = layout.sink_frames(self._sink_frames())
        recent = (ages > 0) & (ages <= self._config.num_recent_frames) & ~sink
        hist = ~(own | sink | recent)
        groups = {"sink": sink, "own": own & ~sink, "recent": recent, "hist": hist}

        # The plan's kept tiles per (head, q_tile), and which frames are whole.
        full = self._whole_frame_mask(layout, own, ages)
        num_full = int(full.sum())
        k_tiles_pad = frame_seqlen // tile
        band = (plan.starts.shape[2] - num_full * k_tiles_pad) // max(
            num_frames - num_full, 1
        )
        hist_entries = plan.starts[:, :, num_full * k_tiles_pad :]
        kept = ((hist_entries % frame_seqlen) // tile)[:, :, :band].long()

        positions = torch.arange(0, query.shape[1], QUERY_STRIDE, device=device)
        sampled = query[:1, ::QUERY_STRIDE]
        bucket = ((positions % frame_seqlen) // plan.query_tile).long()
        num_sampled = sampled.shape[1]
        demand = {g: 0.0 for g in groups}
        captured = {g: 0.0 for g in groups}
        full_t = torch.from_numpy(full).to(device)
        for start in range(0, num_sampled, QUERY_TILE):
            chunk = sampled[:, start : start + QUERY_TILE]
            scores = torch.einsum("bqhd,bkhd->hqk", chunk, key).float()
            probs = torch.softmax(scores * call.softmax_scale, dim=-1)
            per_frame = probs.view(heads, -1, num_frames, frame_seqlen)
            tiles = per_frame[..., : num_tiles * tile].reshape(
                heads, per_frame.shape[1], num_frames, num_tiles, tile
            ).sum(-1)
            kept_rows = kept[:, bucket[start : start + QUERY_TILE]]  # [h, nq, band]
            for name, mask in groups.items():
                mask_t = torch.from_numpy(np.asarray(mask)).to(device)
                demand[name] += float(per_frame[:, :, mask_t].sum())
                # captured: whole frames in the group count fully, patterned
                # frames count their kept tiles.
                whole_in = mask_t & full_t
                pattern_in = mask_t & ~full_t
                got = float(per_frame[:, :, whole_in].sum())
                if int(pattern_in.sum()) and band > 0:
                    group_tiles = tiles[:, :, pattern_in]  # [h, nq, f, num_tiles]
                    got += float(
                        group_tiles.gather(
                            3,
                            kept_rows[:, :, None, :].expand(
                                -1, -1, int(pattern_in.sum()), -1
                            ),
                        ).sum()
                    )
                captured[name] += got
        scale = num_sampled * heads
        record = {"chunk": int(layout.query_chunk_index), "layer": int(call.layer_index)}
        for name in groups:
            d = demand[name] / scale
            c = captured[name] / scale
            record[f"demand_{name}"] = round(d, 6)
            record[f"recall_{name}"] = round(c / d, 4) if d > 1e-9 else None
        with write_lock:
            with open(GROUP_OUT, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def attend(self, call):
        if SECTION_DIR and call.layer_index in SECTION_LAYERS and not self.in_cache_update:
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
            # pure lookup — no re-calibration, no state change. Since the
            # flat-row commit the entry is a (plan, flat_token_index) tuple.
            entry = self._prepare_plan(call, layout)
            if entry is None:
                return out
            plan, _flat_index = entry
            if isinstance(plan, module.BlockSparsePlan):
                if EXACT:
                    measure_plan_exact(self, call, layout, plan)
                else:
                    measure_chunk_2d(self, call, layout, plan)
                if GROUP_OUT:
                    measure_groups(self, call, layout, plan)
            # Rolling-window tuples are not measured; the recall study runs on
            # single-chunk models.
        except Exception as exc:  # noqa: BLE001 - never break the run
            if not failed:
                failed.append(str(exc))
                print(f"[osa-recall] measurement disabled: {exc!r}", file=sys.stderr)
        return out

    backend_cls.attend = attend
    print(f"[osa-recall] instrumented OSA, writing {OUT_PATH}", file=sys.stderr)


def _install_lightforcing(module) -> None:
    """Measure the attention mass LightForcing's per-step block mask captures.

    Wraps :meth:`LightForcingAttention.prepare`. The kept-block mask is
    captured by shadowing the module-level ``lightforcing_block_mask`` the
    method calls; the wrapper then recomputes ``softmax(q k^T)`` on a strided
    subset of queries, folds the key axis into LF's frame-aligned key blocks,
    and reports per head:

    ``recall``
        mass the mask actually captures (fraction of total, like OSA's
        ``recall_frozen``).
    ``recall_oracle``
        mass the best per-(head, query block) block set at the same block
        budget would capture — the gap to it is LF's mean-pool estimator cost.
    """
    import torch

    backend_cls = module.LightForcingAttention
    original_prepare = backend_cls.prepare
    original_mask_fn = module.lightforcing_block_mask
    write_lock = threading.Lock()
    steps: dict[tuple[int, int], int] = {}
    failed: list[str] = []
    holder: dict[str, object] = {}

    def capture_mask(**kwargs):
        mask = original_mask_fn(**kwargs)
        holder["mask"] = mask
        return mask

    module.lightforcing_block_mask = capture_mask

    @torch.no_grad()
    def measure(self, call, layout, mask) -> None:
        config = self._config
        query = call.query[:1]
        key = call.key[:1]
        heads = query.shape[2]
        kv_len = key.shape[1]
        frame_seqlen = layout.frame_seqlen
        blocks_per_frame = -(-frame_seqlen // config.block_k)
        kv_blocks = layout.num_frames * blocks_per_frame
        device = query.device

        # Frame-aligned key block of every kv token, [kv_len].
        token = torch.arange(kv_len, device=device)
        block_of = (
            token // frame_seqlen * blocks_per_frame
            + (token % frame_seqlen) // config.block_k
        )
        positions = torch.arange(0, query.shape[1], QUERY_STRIDE, device=device)
        sampled = query[:, ::QUERY_STRIDE]
        bucket = (positions // config.block_q).long()  # LF's query blocks
        num_sampled = sampled.shape[1]

        captured = torch.zeros(heads, dtype=torch.float64, device=device)
        block_mass = torch.zeros(
            heads, num_sampled, kv_blocks, dtype=torch.float32, device=device
        )
        for start in range(0, num_sampled, QUERY_TILE):
            chunk = sampled[:, start : start + QUERY_TILE]
            scores = torch.einsum("bqhd,bkhd->hqk", chunk, key).float()
            probs = torch.softmax(scores * call.softmax_scale, dim=-1)
            folded = torch.zeros(
                heads, probs.shape[1], kv_blocks, dtype=torch.float32, device=device
            )
            folded.index_add_(2, block_of, probs)
            block_mass[:, start : start + QUERY_TILE] = folded
            kept = mask[:, bucket[start : start + QUERY_TILE], :]  # [h, nq, kb]
            captured += (folded * kept).sum((1, 2)).double()

        # Oracle at the same per-(head, query block) block budget: top-k of the
        # bucket-mean true block mass.
        topk = int(mask[0, 0].sum())
        buckets = torch.unique(bucket)
        oracle = torch.zeros(heads, dtype=torch.float64, device=device)
        for b in buckets:
            rows = block_mass[:, bucket == b]  # [h, nb, kv_blocks]
            mean = rows.mean(1)
            best = torch.topk(mean, topk, dim=-1).indices  # [h, topk]
            kept_mass = rows.gather(
                2, best[:, None, :].expand(-1, rows.shape[1], -1)
            ).sum((1, 2))
            oracle += kept_mass.double()

        sizes = torch.zeros(kv_blocks, dtype=torch.float64, device=device)
        sizes.index_add_(0, block_of, torch.ones_like(token, dtype=torch.float64))
        density = float(
            (mask.double() * sizes).sum() / (mask.shape[0] * mask.shape[1] * kv_len)
        )

        chunk_index = int(layout.query_chunk_index)
        key_id = (call.layer_index, chunk_index)
        step = steps.get(key_id, 0)
        steps[key_id] = step + 1
        record = {
            "chunk": chunk_index,
            "layer": int(call.layer_index),
            "step": step,
            "kv_frames": int(layout.num_frames),
            "kv_blocks": int(kv_blocks),
            "topk": topk,
            "density": round(density, 6),
            "recall": [round(v, 6) for v in (captured / num_sampled).tolist()],
            "recall_oracle": [round(v, 6) for v in (oracle / num_sampled).tolist()],
        }
        with write_lock:
            with open(LF_OUT_PATH, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def prepare(self, call, layout):
        holder.pop("mask", None)
        execution = original_prepare(self, call, layout)
        if execution is None or self.in_cache_update:
            return execution
        mask = holder.pop("mask", None)
        if mask is None:
            return execution
        try:
            measure(self, call, layout, mask)
        except Exception as exc:  # noqa: BLE001 - never break the run
            if not failed:
                failed.append(str(exc))
                print(f"[lf-recall] measurement disabled: {exc!r}", file=sys.stderr)
        return execution

    backend_cls.prepare = prepare
    print(f"[lf-recall] instrumented LightForcing, writing {LF_OUT_PATH}", file=sys.stderr)


class _PostImportHook(MetaPathFinder):
    """Let the normal machinery load the target module, then patch it."""

    def __init__(self, target: str, install) -> None:
        self._target = target
        self._install = install

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        real_exec = spec.loader.exec_module
        install = self._install

        def exec_module(module):
            real_exec(module)
            install(module)

        spec.loader.exec_module = exec_module
        return spec


if OUT_PATH:
    sys.meta_path.insert(0, _PostImportHook(TARGET, _install))
if LF_OUT_PATH:
    sys.meta_path.insert(0, _PostImportHook(LF_TARGET, _install_lightforcing))
