# Attention Sparsity: Where Causal / Rolling Forcing Actually Spend Attention

Date: 2026-07-21
Scope: measure the temporal *and* spatial structure of self-attention in the
block-causal Wan DiTs, per `(layer, head)`, and decide what kind of sparse
attention the measurements justify. Companion to
[`attention map visualization.md`](attention%20map%20visualization.md) (which
covers the chunk-level probe) and
[`baseline integration.md`](baseline%20integration.md).

## Framing: the frame axis is already bounded

Both models cap self-attention at **21 latent frames**
(`sliding_window_num_frames: 21` for Causal Forcing; `max_attention_num_frames:
21` plus `sink_size: 3` for Rolling Forcing). Confirmed in the dumps: RF's chunk
16 of 17 sees exactly 21 frames and records exact `0` on the evicted chunks
1–10.

So per-forward attention cost is already O(1) in video length, and the frame
axis is only **21 wide** while the spatial axis is **1560 tokens** (30x52 latent
grid at 480p). Any large win has to come from the spatial axis — which the
original probe summed away.

## Part 1 — Temporal axis

`dt` = query frame − key frame, over the steady-state chunks (CF):

| dt | 0 | 1 | 2 | ≥3 | <0 |
|---|---|---|---|---|---|
| mass | 0.412 | 0.161 | 0.078 | 0.258 | 0.091 |

`dt<0` is real: these are block-*causal*, so within a 3-frame chunk attention is
bidirectional.

**Frame-granular sparsity is weak.** 90% of a row's mass needs 0.59 (CF) / 0.55
(RF) of the cached frames. Mass retained at a matched budget:

| budget | oracle | static per-(L,H) | static shared | recent+sink |
|---|---|---|---|---|
| CF 20% | 0.573 | **0.548** | 0.494 | 0.491 |
| CF 50% | 0.848 | **0.820** | 0.777 | 0.785 |
| RF 20% | 0.591 | **0.566** | 0.512 | 0.214 |
| RF 50% | 0.831 | **0.809** | 0.764 | 0.513 |

Three conclusions:

- **A calibrated static per-`(layer,head)` frame mask tracks the oracle within
  2–3 points.** No dynamic top-k is needed on the frame axis. It is also
  step-invariant (profile correlation 0.979 CF / 0.923 RF across denoising
  steps; step 0 vs later steps 0.99+), so one mask serves all steps.
- **Naive StreamingLLM (recent window + sink) is the wrong pattern**, badly so
  for RF (0.31 vs 0.67 at 30% budget) — its window is bidirectional within a
  pass, so "most recent" misses the mass. 72% (CF) / 97% (RF) of sites need an
  offset ≥15 frames back, so a pure sliding window fails too.
- **Savings are capped by the existing 21-frame limit**: at ≥97% retained mass,
  **CF ~31%**, **RF ~10%** of attention FLOPs.

Temporal head structure: 137/360 CF sites and 78/360 RF sites are ≥95% local
(own chunk ±1). The RF set is a **strict subset** of the CF set, so a mask
calibrated on RF is safe to apply to CF but not the reverse. Layers **15, 18,
22, 23, 24** are the globally-attending ones in both models. Per-`(layer,head)`
temporal locality correlates **0.862** between the two models.

## Part 2 — Spatial axis

Recorded as mass by spatial displacement `(dy, dx)` from the query's own latent
position, split by `dt` bucket.

**Attention is not spatially local in aggregate.** A 9x9 window (4.6% of a
frame) holds only **0.548** of same-frame mass (CF); cross-frame it drops to
0.360. Row bands beat column bands at equal cost (|dy|≤1, 9.8% cost → 0.619 vs
|dx|≤2, 9.4% cost → 0.500): attention spreads horizontally more than
vertically, matching the 52x30 landscape grid.

But the per-head spread is enormous — same-frame locality ranges **0.04 → 1.00**
across the 360 sites (layer 29/head 9 puts all its mass on one cell; layer
1/head 6 is a near-flat plateau).

**The decisive measurement.** Per-query top-k selection dominates every static
shape at matched cost (CF, same-frame):

| budget | oracle top-k | best static shape |
|---|---|---|
| 2.1% | 0.657 | 0.429 |
| 4.1% | 0.734 | 0.548 |
| 8.2% | 0.806 | 0.628 |
| 16.4% | 0.873 | 0.704 |

The mass *is* concentrated per query; it just is not concentrated at a fixed
displacement. **A static spatial window is the wrong primitive.** Absolute-
position maps rule out the other static hypothesis too — top-10% of absolute
frame cells hold only 0.306 (CF) / 0.359 (RF), barely above uniform, so heads
are not fixating on a fixed frame region either.

Two further facts:

- **Per-head budgets nearly halve the cost.** 95% same-frame mass costs
  **0.263** of the frame with per-head budgets vs **0.444** with one uniform
  budget (CF). The distribution is bimodal: 112/360 sites need <5% of the frame,
  180/360 need <15%, but 85/360 need >50%.
- **Spatial and temporal locality select different heads** (r = +0.43): only
  62/360 sites are local in both, 157/360 in either. One head classification
  will not serve both axes.
- **Spatial locality is a backbone property, not a scheme property**:
  correlation **0.945** between CF and RF. Calibrate once, reuse across both.

## Proposed method

Three parts, each justified by a specific measurement above:

1. **Static temporal frame mask, per `(layer, head)`.** Calibrated offline,
   baked as a 30x12 table. Near-oracle (§1), zero runtime selection cost,
   handles the sink and the evicted region naturally. The cheap, safe win.
2. **Dynamic block selection within the surviving frames.** Score mean-pooled
   query blocks against mean-pooled key blocks, keep top-B, union with a small
   fixed local window as a floor. Dynamic is *required*, not preferred — the
   static gap is 0.18–0.20 of mass at equal budget (§2).
3. **Per-head budgets + head gating.** Each site gets the budget its calibrated
   concentration curve demands. The ~110 strongly-local sites bypass selection
   entirely (fixed small window, no scoring overhead); the ~85 diffuse sites
   stay dense.

Rough combined estimate: **~5x attention FLOP reduction at ~0.92 retained mass**
(0.69 temporal x ~0.30 spatial).

## Caveats — read before building

- **Retained mass is not quality.** 0.92 mass may be visually fine or may
  destroy temporal coherence. Needs a perceptual check against the dense
  baseline before the number means anything.
- **Block granularity will erode the oracle.** The oracle above is per-cell; a
  real kernel selects 64/128-token blocks, and with a 52-wide grid a 64-token
  block is ~1.2 rows. Measuring block-level selection quality is the next probe
  change — cheap now that the plumbing exists.
- **One prompt, one seed**, chunks ≥2, spatial query stride 32. The head
  classification should be calibrated across several prompts before being baked
  into a config.
- Rank-0 heads only under TP (both baselines currently run replicated).

## Implementation

```
modified:
  runtime/utils/attention_map_probe.py   (spatial accumulators + pure functions)
  envs.py                                (2 env vars)
  runtime/models/dits/causal_wanvideo.py (grid dims -> begin_forward)
  test/unit/realtime/test_attention_map_probe.py  (2 tests)
```

`RollingForcingWanTransformer3DModel` subclasses `CausalWanTransformer3DModel`
and its `forward` delegates to `super().forward()`, so the single
`begin_forward` change covers **both** models — no RF-side edit needed.

Enable with `SGLANG_DIFFUSION_ATTENTION_MAP_SPATIAL=true` on top of the existing
`SGLANG_DIFFUSION_ATTENTION_MAP_DIR`; `SGLANG_DIFFUSION_ATTENTION_MAP_SPATIAL_QUERY_STRIDE`
(default 32) trades cost against noise independently of the frame probe's stride.

```bash
export PYTHONPATH=/data/projects/vision-gen/sglang/python
CUDA_VISIBLE_DEVICES=7 \
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_spatial2 \
SGLANG_DIFFUSION_ATTENTION_MAP_SPATIAL=true \
sglang generate \
  --model-path /data/projects/vision-gen/models/CausalForcing-Wan2.1-T2V-1.3B-chunkwise-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 81 --seed 42
```

Three accumulators are written to `spatial_displacement.npz` alongside the
per-chunk frame dumps:

- `displacement` `[layers, heads, dt_bucket, dy, dx]` — mass by spatial offset
- `absolute` `[layers, heads, dt_bucket, y, x]` — same mass in frame coordinates,
  to catch heads that fixate on a fixed region rather than near themselves
- `concentration` `[layers, heads, dt_bucket, top_k]` — per-query mass held by
  its top-k cells, which measures sparsity *independently of whether it has any
  geometric structure*. This is the array that decided static vs dynamic.

`dt` buckets are `("dt=0", "dt=1", "dt=2", "dt>=3", "dt<0")`. The scatter is
done one query *frame* at a time so the key-side dt bucket is constant within a
tile, making the per-key scatter `buckets x frame_seqlen` instead of one bin per
(query, key) pair — without that it is ~50x more expensive and impractical.
`spatial_displacement_mass` is unit-tested against a brute-force per-(query,key)
reference.

Analysis scripts used for the tables above are not checked in; they live in the
session scratchpad (`spatial_analysis2.py`, `plot_spatial_summary.py`,
`plot_site_attention_maps.py` — the last renders per-`(layer,head)` site maps
and is the companion to `tools/plot_chunk_attention_maps.py`). Promote them into
`tools/` if this line of work continues.

## Follow-ups

- Block-granularity probe (score at 64/128-token block resolution) to replace
  the per-cell oracle with an achievable bound.
- Multi-prompt calibration of the head tables before baking any config.
- Perceptual A/B of a prototype mask against the dense baseline.

---

# The 10-second six-model revision (2026-07-23)

Re-analysis over the fresh 10 s token-score dumps (`attn_token_10s/`: Self/
Causal/Rolling Forcing, LingBot-World v2, LongVie 2 clip 2 with history,
LongLive-2.0). Stats + figures in `attn_token_10s/sparsity_analysis/`
(`stats.json`, `block_gap.json`, `retention_curves.png`,
`concentration_by_depth.png`, `sink_mass_per_head.png`,
`register_cell_maps.png`); scan scripts in the 2026-07-23 session scratchpad
(`sparsity_scan.py`, `sparsity_figs.py`).

## Findings

1. **Concentration is universal and strong.** Median per-(chunk,layer,head):
   half of a head's mass sits in 5.1–9.7% of its visible keys; 90% needs
   30–50%. An oracle top-4% keeps 30–45% of mass (uniform would keep 4%);
   top-16% keeps 63–76%. LongVie 2 is the most concentrated (n50 = 5.1%,
   4% → 45%) — full attention hides the most exploitable structure.
2. **Head heterogeneity is the dominant axis, again.** p10–p90 of
   retention@4% spans ~14% to ~85% in every model; the within-layer
   head gap has median 0.56–0.77. Any uniform budget wastes most of the win.
3. **Frame granularity ≈ token granularity; 4×4 spatial blocks lose.** At an
   equal ~4% token budget (last chunk, per head): token oracle 0.34–0.48,
   whole-latent-frame selection 0.24–0.45 (within a few points for the Wan
   models, budget-rounding favours it slightly), but 4×4 spatial blocks only
   0.19–0.30 — 25–45% relative loss vs tokens. Attention mass is organized in
   *frames* plus *isolated cells*, and mid-size spatial blocks straddle both.
   (The even/odd column checkerboard, 1.08–1.19× in all six models, is one
   reason: any even-width block mixes biased and unbiased columns.)
4. **Sink usage is zero-inflated.** Mass on the pinned sink per
   (chunk,layer,head) at steady state: RF median 0.085 (17–21% of heads
   < 2%, tail to 0.79), LingBot 0.089 (17% < 2%), LongLive-2.0 0.074
   (23% < 2%, 8% > 30%). Roughly a fifth of sites never read the sink.
5. **Registers are spatially anchored — and in the Wan-1.3B family they form
   a static positional lattice.** Token-level register positions (>10×
   uniform at layer mean) barely persist across chunks (Jaccard 0.03–0.18;
   LingBot 0.39), but their *grid cells* do: cell-Jaccard 0.63–0.86 for
   SF/CF/RF. The register_cell_maps figure shows why: SF/CF/RF registers sit
   on a periodic vertical-stripe + checkerboard lattice covering the grid —
   position-locked (patchify/RoPE geometry), not content-locked, hence
   calibratable offline. The 14B models differ: LingBot's registers partly
   track content (a cluster over the subject), LongLive-2.0's are sparse and
   scattered — those need runtime discovery.
6. **Temporal structure by model** (steady-state medians): LongLive-2.0 own
   block 0.78 / prev 0.086 / sink 0.074 (0.94 in three row-blocks); CF/SF own
   0.56–0.58 / prev 0.17; LingBot own 0.41 / prev 0.17 / sink 0.089.
   Rolling Forcing spreads mass across its jointly-denoised window (its
   "own chunk" is not the newest frame), so its masks must be window-relative.
   LongVie 2: ±4 frames ≈ 0.74 of mass within a clip, history at 0.25×
   uniform after the first generated frames.

## Proposed sparse-attention strategy

Promoted to its own note:
[`sparse attention strategy.md`](sparse%20attention%20strategy.md) — per-head
static frame masks + per-head sink dropping + register-cell columns +
head-adaptive budgets with a dense fallback, with expected gains and the
validation plan.

## Caveats

- Received-mass view: per-query structure inside a chunk is averaged; the
  frame-mask calibration should be re-checked per query block before trusting
  it for RF's joint window.
- n = 1 prompt/seed/model; the Wan-1.3B register lattice looks positional
  (stripes + checkerboard) but prompt-independence is unverified — one more
  prompt would settle it.
- Oracle retention is an upper bound for any selector, and retained mass is a
  proxy — end-to-end quality is the real gate.
