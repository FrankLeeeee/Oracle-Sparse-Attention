# Per-Chunk Attention Maps: Causal Forcing & Rolling Forcing

Date: 2026-07-21
Scope: visualize, per generated chunk and per transformer layer, how a
block-causal Wan DiT distributes its self-attention over all chunks of the
video. Companion to [`baseline integration.md`](baseline%20integration.md).

## Why it needs runtime support

Both baselines run fused attention kernels (FA / SDPA) that never materialize
the probability matrix, and the keys they see are a *rolling view of a KV
cache*, not the plain token sequence — Rolling Forcing additionally prepends a
re-roped attention sink and appends the in-flight window. So the mapping from
"key row *i* of this attention call" to "chunk *j* of the video" only exists
inside the model. The probe therefore lives in the runtime, is off unless an
env var is set, and dumps compact per-chunk arrays that a separate CLI plots.

## Usage

```bash
export PYTHONPATH=/data/projects/vision-gen/sglang/python

# 1. generate with the probe on (any Causal-Forcing / Rolling-Forcing model)
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_maps \
sglang generate \
  --model-path /data/projects/vision-gen/models/RollingForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 81 --seed 42 --save-output

# -> .../attn_maps/RollingForcingWanTransformer3DModel-<timestamp>/
#      chunk_000.npz ... chunk_006.npz, meta.json

# 2. render
python -m sglang.multimodal_gen.tools.plot_chunk_attention_maps \
  /data/projects/vision-gen/attn_maps/RollingForcingWanTransformer3DModel-<timestamp>
# -> <run dir>/plots/chunk_000.png ... chunk_006.png, summary.png
```

Options: `SGLANG_DIFFUSION_ATTENTION_MAP_QUERY_STRIDE` (default 8) trades probe
cost against sampling noise; the plotter takes `--color-scale {log,linear}`
(default log — the off-diagonal mass is 1-2 orders of magnitude below the
diagonal), `--pass-kind {denoise,cache_update}`, `--out-dir`, `--dpi`.

## What is recorded

For every self-attention call the probe recomputes `softmax(q k^T)` for every
`stride`-th query, averages over heads and over the queries of a chunk, and
sums the probabilities of each key chunk. One row = the attention mass that one
chunk's queries put on every chunk, so **each row sums to 1**.

`chunk_<c>.npz` holds `[num_denoising_steps, num_layers, num_chunks]` under
`denoise`, plus the KV-cache-refresh forward under `cache_update`. Chunks that
were never visible stay `NaN` (rendered light grey).

Per-chunk figure: layer x chunk heatmap (mean over steps, dashed outline on the
chunk itself) above per-step layer-mean curves. `summary.png` is the
chunk-to-chunk map averaged over layers and steps.

Step indices are per chunk: for Causal Forcing they are the 4 DMD steps of that
block; for Rolling Forcing they are the 5 window passes the block takes part in
as the window slides over it (window pass *k* over block *b* is step `k - b`).

## Observations from the 81-frame runs (seed 42, same prompt)

- **Causal Forcing** is strictly block-causal — future chunks are empty. Mass is
  dominated by the chunk itself (~0.63 at chunk 4), decays into the recent past
  (0.19, 0.06, 0.04) and then *rises again on chunk 0* (0.085): the first block
  behaves as an emergent attention sink even though `sink_size=0`. Layers 15-27
  put visibly more mass on distant chunks than layers 0-10.
- **Rolling Forcing** is causal only at chunk granularity *across* windows —
  within a window all 5 blocks attend to each other, so early chunks show mass
  on chunks that are generated later (chunk 0: 0.20 on chunk 1, 0.08 on chunk 2,
  ...). The explicit sink keeps chunk 0 at ~0.14 even for the last chunk of the
  video, an order of magnitude above the neighbouring old chunks.

## Implementation

```
new:
  python/sglang/multimodal_gen/runtime/utils/attention_map_probe.py
  python/sglang/multimodal_gen/tools/plot_chunk_attention_maps.py
  python/sglang/multimodal_gen/test/unit/realtime/test_attention_map_probe.py
modified:
  python/sglang/multimodal_gen/envs.py                    (2 env vars)
  python/sglang/multimodal_gen/runtime/models/dits/causal_wanvideo.py
  python/sglang/multimodal_gen/runtime/models/dits/rolling_forcing_wanvideo.py
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/causal_denoising.py
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/rolling_forcing_denoising.py
```

- `ChunkAttentionRecorder` is a process-wide singleton resolved once from
  `SGLANG_DIFFUSION_ATTENTION_MAP_DIR`; disabled it costs one `None` check per
  attention call. The DiT calls `begin_forward()` once per forward (the key
  layout is shared by all layers) and each attention layer calls `record()`; the
  denoising stage calls `flush()` when the request finishes.
- The global token ranges behind the visible keys are derived by
  `visible_key_segments()` in each DiT module — a trailing cache window for
  Causal Forcing, `sink + working cache + current window` for Rolling Forcing
  (with the sink slots mapped back to chunk 0 and the rolled slots offset by
  `global_end - local_end`). Attention-sink variants of the plain causal cache
  are not mappable and are skipped with a warning.
- Unit tests cover the softmax/segment math against a brute-force reference and
  replay both cache schedules (incl. eviction) to check the reconstructed
  positions match the keys the cache actually returns.

A 201-frame Rolling Forcing run (17 chunks, denoise stage 60 s) exercises cache
eviction: chunk 16 sees the sink (0.10 on chunk 0), exactly zero on the evicted
chunks 1-10, and a rising ramp over the 5 cached chunks 11-15 into itself
(0.48). The `summary.png` band structure — bright column 0, black evicted
region, ~5-chunk diagonal band — is the clearest single picture of the scheme.

## Limitations

- Records on world rank 0 only; under TP the reported mass covers rank 0's
  heads. Both baselines currently run replicated on one GPU, so this is moot.
- Overhead at stride 8 (81 frames, denoise stage): Causal Forcing 23.6 s vs
  ~23 s without the probe, Rolling Forcing 31 s vs ~15 s (its windows attend to
  far more keys per forward, and every window pass is probed).
- Evicted chunks record exact 0 and are painted black; chunks that were never
  co-visible are `NaN` and painted light grey.
- Head-averaged only; per-head maps would need a schema change.
