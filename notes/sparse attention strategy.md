# Sparse Attention Strategy for the Video DiTs

Date: 2026-07-23
Status: proposal — measurements done, no implementation yet.

Grounded in the 10-second six-model probe sweep of 2026-07-23
([`attention sparsity analysis.md`](attention%20sparsity%20analysis.md), section
"The 10-second six-model revision") and the 2026-07-21 displacement study in
the same note. Models covered: Self-Forcing, Causal Forcing, Rolling Forcing
(Wan2.1-T2V-1.3B), LingBot-World v2 (14B), LongVie 2 (Wan2.1-I2V-14B),
LongLive-2.0 (Wan2.2-TI2V-5B). Evidence artifacts:
`/data/projects/vision-gen/attn_token_10s/sparsity_analysis/`.

## The measurements the strategy stands on

| finding | number | consequence |
|---|---|---|
| per-head concentration | half of a head's mass in 5–10% of visible keys, all six models | sparse attention is viable everywhere |
| head heterogeneity | retention@4% spans 14–85% across heads; within-layer gap 0.56–0.77 | budgets must be per-(layer,head) |
| granularity | frame-level selection ≈ token oracle at equal budget; 4×4 spatial blocks lose 25–45% relative | select whole frames + single cells, never mid-size spatial blocks |
| sink usage | ~20% of heads put <2% mass on pinned sinks; tail to 0.79 | per-head sink dropping is free |
| registers | spatially anchored; Wan-1.3B: static positional stripe lattice (cell-Jaccard 0.63–0.86); 14B: partly content-bound | offline lattice for 1.3B, runtime harvest for 14B |
| temporal profiles | own block 0.41–0.78, prev 0.09–0.17, sink ~0.07–0.09; LongVie 2: ±4 frames = 0.74, history 0.25× uniform | ~⅓ of frames holds ≳90% of mass |
| checkerboard | even/odd column bias 1.08–1.19× in all models | align spatial blocking to even columns |
| step-invariance (07-21 study) | per-head frame masks stable across steps, r = 0.92–0.98 | calibrate once, reuse across denoising steps |
| cross-model transfer (07-21) | CF↔RF per-(layer,head) spatial locality r = 0.945 | calibrate per backbone, not per finetune |

## The strategy

Compose a per-(layer, head) attention mask from four layers, calibrated once
per backbone. For the KV-cache models the mask doubles as a cache-eviction
policy; for the full-attention models it is a FlexAttention BlockMask.

1. **Static per-head temporal frame mask** — the primary lever. Each head
   keeps its calibrated top-m latent frames: own block, a recency ramp, sink
   if used. Frames are contiguous rows of 390–1560 tokens, so the mask is
   kernel-friendly and only changes at chunk boundaries. Rolling Forcing's
   mask must be *window-relative* (it denoises a window jointly — its mass is
   not concentrated on the newest frame). For LongVie 2, this includes
   dropping history keys after the first ~2 generated latent frames
   (history is attended at 0.25× uniform) and a ±4-frame band within clips.
2. **Per-head sink policy.** Drop sink keys for the ~20% of (layer,head)
   sites measured at <2% sink mass; always keep them for the >30% tail.
   Pure KV-memory/bandwidth win with zero measured risk.
3. **Register-cell columns.** Keep ~1–2% of grid cells visible in *all*
   frames, including dropped ones — the isolated 10–670× cells are the tail
   the frame mask cannot catch. Wan-1.3B: the cells form a static positional
   lattice → calibrate offline. 14B models: harvest each chunk's top cells at
   its first denoising step (full attention), reuse for remaining steps —
   justified by the r = 0.92–0.98 step-invariance.
4. **Head-adaptive budgets + dense fallback.** Allocate m and the cell budget
   from each head's calibrated retention curve; leave the most diffuse
   ~10–20% of heads fully dense. The head spread is where the FLOPs come from.

Granularity rules: whole-frame rows and single-cell columns only; no mid-size
spatial blocks; any spatial partitioning aligned to even column boundaries.

## Expected gains (from the oracle numbers, upper bounds)

- Block-causal models (CF/SF/LongLive-2.0/LingBot): keep ~⅓ of cached frames
  per head at ≳90% retained mass → ~2–3× attention FLOP/KV reduction, more
  once sink dropping and per-head budgets compound.
- LongVie 2 (full attention, 50 steps — the most expensive model of the six):
  most concentrated of all (top-4% of tokens = 45% of mass, ±4-frame band =
  74%); a per-head frame band + register columns is the first thing to try.

## Validation plan

1. **Mask emulation** at the probe's softmax call sites (they already
   recompute attention): apply the candidate mask, report retained mass vs
   the dumps per (layer,head) — no kernel work needed.
2. **End-to-end replay**: same seeds, masked vs unmasked; score pixel deltas
   and, for LongVie 2, the depth-L1 / subject-IoU harness in
   `outputs/longvie2_ar_verification/measure_ar.py`.
3. **Kernel**: FlexAttention BlockMask per (layer,head) for the causal
   models (mask static within a chunk); measure wall-clock, not just FLOPs.
4. Ablate the four layers independently; a second prompt/seed to confirm the
   Wan-1.3B register lattice is prompt-independent before trusting offline
   calibration.

## Caveats

- All retention numbers are *oracle* (post-hoc top-k) — upper bounds for any
  real selector; retained mass is a proxy, end-to-end quality is the gate.
- Received-mass view: per-query structure inside a chunk is averaged away;
  re-check the frame masks per query block for Rolling Forcing especially.
- n = 1 prompt/seed per model.

Related notes: [`attention sparsity analysis.md`](attention%20sparsity%20analysis.md)
(measurements), [`token attention maps.md`](token%20attention%20maps.md) (probe
mechanics + per-model maps), [`attention map visualization.md`](attention%20map%20visualization.md)
(chunk-level probe).
