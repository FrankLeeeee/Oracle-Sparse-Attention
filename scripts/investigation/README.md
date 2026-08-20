# Investigation: runtime breakdown + attention token maps

Layout: the scripts live here (`scripts/investigation/<topic>/`) and are
version controlled; everything they produce goes to
`results/investigation/<topic>/`, which is gitignored. `paths.py` holds that
convention — a script reads its own assets relative to its own directory and
writes output under `results_dir(<topic>)`. Run them from anywhere:

```bash
python scripts/investigation/chunk_runtime/run_sweep.py --gpus 7
```

Date: 2026-08-18 · Hardware: 1×H200 per run (idle GPUs 0/1/6/7) · Prompt:
"A red fox trotting across a snowy field, camera slowly tracking sideways",
seed 42 (the per-chunk runtime round in section 3 uses a different prompt —
see below).

Models:

| model | checkpoint | causal geometry |
|---|---|---|
| Self-Forcing 1.3B | `models/SelfForcing-…-Diffusers-fullctx-null` (config-patched `sliding_window_num_frames: null` so full-context holds at every duration) | 3-frame chunks, 4 steps, full context |
| Rolling Forcing 1.3B | `frankleeeee/RollingForcing-…-Diffusers` | 3-frame blocks, 5-block joint window, 5 steps, 3-frame sink + 24-frame cache |
| LongLive-2.0 5B | `Rabinovich/LongLive-2.0-5B-Diffusers` | 8-frame chunks, 4 steps, 8-frame sink + 32-frame window |
| LingBot-World v2 14B | `robbyant/lingbot-world-v2-14b-…` (realtime WebSocket sessions — one-shot generate does a single chunk) | 3-frame chunks, 4 steps, I2V condition |

**LingBot is I2V — its condition image must match the prompt.** It is the only
image-conditioned model of the four, and the condition image dominates the
scene: pointing it at a leftover picture produces that picture's world with the
text prompt only bleeding in, while the other three models render the prompt.
`chunk_runtime/run_sweep.py:lingbot_first_frame()` derives it from frame 0 of
the Self-Forcing run for the same prompt, seed and resolution. The older
`attention_token_maps/run_captures.py` still points at
`inputs/uploads/a816103ba740450f9ded724ea1bf11e7_first_frame`, which is the red
fox — correct for that round's fox prompt, wrong for any other prompt.

## 1. Runtime breakdown (`runtime_breakdown/`)

Sweep: 4 models × {480p, 720p} × {5, 10, 20, 30}s. Per config a clean
`timing` run (stage walltimes, `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`)
plus a separate `profile` run (torch-profiler, 40 denoising steps) used only
for GPU-kernel *shares* that split the real denoise walltime. 720p is
1280×720 (1280×704 for LongLive-2, its supported grid). Durations in pixel
frames: 81/165/321/477 @16 fps (LingBot durations count streamed frames),
125/253/477/733 @24 fps for LongLive-2.

Figures: `e2e_stages_{480p,720p}.png`, `denoise_components_{480p,720p}.png`,
`denoise_scaling.png`. Raw numbers: `breakdown.json`; logs+traces under
`runs/<model>/<res>_<dur>s/`. Rerun: `run_sweep.py` → `analyze.py` →
`plot_breakdown.py`.

Headlines (720p):

- **Denoise dominates everywhere**; VAE decode is the second component and
  becomes comparable to denoise only for LongLive-2 (30 s: 31 s denoise vs
  37 s decode). Text encode / input prep are negligible (<1–2 s).
- **Scaling with duration**: Self-Forcing (full context) is the only
  super-linear model — denoise 11→24→72→143 s for 5→10→20→30 s (attention is
  74 % of its denoise at 20 s: 53.3 of 71.7 s). The window-capped models are
  ~linear: Rolling Forcing 14→26→48→70 s, LongLive-2 13→18→25→31 s,
  LingBot 40→83→170→257 s (~8.2 s per 3-frame chunk steady-state at 720p,
  ~2.6 s at 480p).
- **Component mix at 720p/20 s** (denoise split by profiled kernel shares):
  Self-Forcing 53s attention / 12s GEMM; Rolling Forcing 23s / 15s;
  LongLive-2 12s / 10s; LingBot 88s / 62s. I.e. only full-context
  Self-Forcing is attention-bound; the capped-window models are close to
  compute-balanced between attention and GEMM.
- LingBot session extras (from ws `chunk_stats`): time-to-first-chunk ≈
  8–27 s (grows with requested length), steady per-chunk forward ≈ 2.5 s
  (480p) / 8.2 s (720p).

Caveats: LingBot's per-duration component splits reuse the resolution's 20 s
profiled trace (window is capped, so the steady-state kernel mix is
duration-independent). Profile-run walltimes are never reported.

## 2. Attention token maps (`attention_token_maps/`)

720p / 20 s per model, attention-map probe with per-step QK dumps
(`SGLANG_DIFFUSION_ATTENTION_MAP_QK_{CHUNKS,STEPS,LAYERS}`): 3 chunks
(early/mid/late) × 3 denoising steps × 3 layers (shallow/mid/deep), all
heads, query stride 8 / key stride 16. Each `qk_chunk_<c>_step_<s>.npz`
holds `scores [layers, heads, queries, keys]` plus `coverage` — per query
row, the **minimum number of top-ranked key tokens whose probabilities sum
past 0.9**, computed on the full un-strided key axis at capture time.

Figures (5 heads × 3 steps × 3 chunks, mid layer) in
`<model>/<ModelTag>-<timestamp>/token_map_plots/`; x = key token index,
y = query token index, log color = attention probability, right panel =
per-row top-k>0.9 counts. Re-render with other heads/steps/layers/chunks:

```bash
python -m sglang.multimodal_gen.tools.plot_attention_token_maps <run_dir> \
    --chunks 13 --steps 0,1,3 --layers 29 --heads 0-11
```

Rolling Forcing semantics: one dump = one *window* (5 staggered-noise blocks
jointly denoised), keyed by the window's oldest chunk; chunk c at its
first/middle/last step lives in the windows keyed c-4 / c-2 / c (captured:
0, 9/11/13, 21/23/25).

Notes from a first pass over the maps (mid layer, median over query rows of
the top-k>0.9 count): every model needs only ~4–8 % of its visible keys at
step 0, and later steps concentrate further — Self-Forcing 6.5 %→4.6 %
(chunk 25, 281 k visible keys, full context), LingBot v2 7.4 %→4.8 %
(9-frame sink + 9-frame recent window = 65 k keys), Rolling Forcing ~4 %
(3-frame re-roped sink + 18-frame window), LongLive-2 4.6 %→1.6 % (the
sharpest). Per-row spread is huge (p10 vs p90 ≈ 100×). The heatmap x-axis is the
visible-key *column* index with global-token tick labels (jumping across the
sink/window gap), with green frame-boundary and white chunk-boundary lines
computed from global frame ids — correct even for disjoint segments.

## Probe/tooling changes made for this investigation

- `SGLANG_DIFFUSION_ATTENTION_MAP_QK_STEPS` / `_QK_LAYERS` env vars; QK
  dumps are per (chunk, step) files with full-key top-k coverage.
- New `tools/plot_attention_token_maps.py`; removed the four superseded
  plot tools (`plot_chunk/token/qk_attention_maps`, `plot_token_attention_bars`).
- `--profile` now advances its schedule on Rolling Forcing and LongLive-2
  (missing `step_profile()` in their overridden loops dumped empty traces).

## 3. Per-chunk runtime (`chunk_runtime/`)

New probe `SGLANG_DIFFUSION_CHUNK_TIMING_DIR` (`runtime/utils/chunk_timing_probe.py`):
CUDA events around every DiT forward and around each layer's self- and
cross-attention, summed per chunk over that chunk's denoising steps; events
resolve only at chunk boundaries so nothing synchronizes mid-chunk. The
KV-cache-refresh forward is tagged `cache_update` and kept separate.

Sweep: 4 models x {480p, 720p} x {5, 10, 20}s, **serial on one idle H200**
(concurrency inflates wall time). Prompt: the forest-rainstorm one, seed 42.
Rerun: `run_sweep.py --gpus <id>` then `plot.py`; `doc_update.py` writes the
figures + 24 videos into the Feishu doc's Runtime Breakdown section
(`--stage text|figures|videos` resumes a partially applied update).

Headlines (720p / 20s, steady state = median of the middle third of chunks):

| model | forward/chunk | attention/chunk | attention share | shape |
|---|---|---|---|---|
| Self-Forcing 1.3B | 0.50 -> 3.66 s | 0.29 -> 3.45 s | 59 -> 94 % | linear growth |
| Rolling Forcing 1.3B | 1.40 s | 1.16 s | 83 % | flat (ramp-up + drain) |
| LongLive-2.0 5B | 0.88 s | 0.52 s | 59 % | flat after 3 chunks |
| LingBot-World v2 14B | 5.90 s | 3.81 s | 65 % | flat after 5 chunks |

- Per-chunk linear (Self-Forcing) == quadratic over the video; that is exactly
  where its super-linear duration scaling comes from.
- Attention share rises with resolution (480p: 81/69/48/45 %), since attention
  is quadratic in tokens and GEMM only linear.
- Cross-attention is small and constant (0.03-0.31 s/chunk); the cache-refresh
  forward is not (13.6 s for Self-Forcing, 38.4 s for LingBot over 20 s).
- Consistency with section 1: per-chunk forward + cache refresh sums to
  71.9 / 47.0 / 20.0 s vs the 72 / 48 / 25 s denoise walltimes there.
- Caveat: a one-shot `sglang generate` pays CUDA kernel autotuning on chunk 0
  (4-6 s vs 0.2-1.4 s steady), so plots drop chunk 0 when it exceeds 1.5x
  chunk 1. A realtime server warms up first, so LingBot's chunk 0 is real.

## 4. Intra-chunk frame similarity (`frame_similarity/`)

New probe `SGLANG_DIFFUSION_FRAME_SIMILARITY_DIR`
(`runtime/utils/frame_similarity_probe.py`): for every (chunk, denoising step,
layer boundary, frame pair), the mean per-spatial-position cosine between the
two latent frames of the chunk. Layer *i* is the hidden state *entering* block
*i*; the last index is the final block's output. The KV-cache refresh pass is
excluded via `probe_pass_kind`.

Same 24 configs as section 3, but no wall time is measured so it fans out over
several GPUs. Rerun: `run_sweep.py --gpus 4,7` then `plot.py`; `doc_update.py`
writes the section. The probe does not perturb generation — its 24 videos are
byte-identical to section 3's, which is why the doc does not re-upload them.

Headlines (720p / 20s, last denoising step, mean over chunks and pairs):

| model | chunk frames | input latents | after block 0 | body | final output |
|---|---|---|---|---|---|
| Self-Forcing 1.3B | 3 | 0.42 | 0.85 | 0.75 | 0.90 |
| Rolling Forcing 1.3B | 3 | 0.46 | 0.88 | 0.77 | 0.89 |
| LongLive-2.0 5B | 8 | 0.27 | 0.90 | 0.59 | 0.78 |
| LingBot-World v2 14B | 3 | 0.66 | 0.88 | 0.71 | 0.95 |

- **Block 0 erases the difference**: whatever the input similarity, every model
  lands at 0.85-0.90 after the first block, independent of scale (1.3B-14B),
  depth (30/40), chunk frames (3/8), resolution and duration. The models only
  differ in the body.
- Variation lives along depth, not time: the std across layers is 2.2x
  (LingBot) to 10.3x (Rolling Forcing) the std across chunks.
- Frames converge as denoising proceeds (body 0.58 -> 0.75 for Self-Forcing),
  **except LongLive-2**, whose 0.593 -> 0.585 is flat — the one model with
  8-frame chunks.
- Similarity falls with frame distance but saturates: LongLive-2's output-layer
  similarity is 0.793 at delta=1 and 0.765 from delta=6 on.
- LingBot starts highest (0.66) because it is I2V — all frames of a chunk share
  one condition image.

## 5. Chunk 0's attention forming (`attention_chunk0/`) — task 3.1

Chunk 0 is the only chunk with no prior context, so whatever attention
structure it has must form inside its own frames. Dumps every denoising step
of chunk 0 at 5 depth percentiles x 4 heads per layer, using two new probe
switches: `SGLANG_DIFFUSION_ATTENTION_MAP_QK_HEADS` (flat or per-layer head
lists) and `..._QK_ONLY` (skips the always-on per-frame mass pass). Heads are
drawn per (model, layer) from a seeded RNG, so every figure of a model is
comparable. Rerun: `run_captures.py --gpus 2,4,7` then `plot.py`
(`--no-sheets` refreshes only summary.json and the formation curves).

Headlines (720p / 20s; key share = median fraction of keys needed for 90% of
a row's mass, first step -> last):

- **Denoising tightens attention**: middle layers sparsify sharply over the
  steps — Self-Forcing layer 22 goes 11.6% -> 1.6%, LongLive-2 layer 7
  17.8% -> 3.2%. Sparse attention saves least on the first step, most on the
  last.
- **The dense layers are the first and last**; the middle is where it is
  sparse. Layer 0 stays at 16-49% and barely moves; LingBot's layer 29 needs
  only 1.2% of its keys.
- **The pattern forms in the middle of the network.** Correlating each step's
  map with the final one (log space): the shallowest and deepest layers are
  already at 0.76-0.84 on step 1, while middle layers start far away —
  LongLive-2's layer 14 at 0.125, essentially unrelated to what it converges
  to.
- Rolling Forcing cannot be correlated this way: its ramp-up grows the joint
  window 1 -> 5 blocks, so each step's matrix has a different shape. Its key
  concentration follows the same trend (layer 0: 17.3% -> 5.0%).

Two traps worth remembering: `np.percentile` over a large float16 array
overflows its own index arithmetic and returns NaN (cast to float32 first),
and float16 underflows small probabilities to exactly 0, which LogNorm masks
and would draw white — i.e. looking like the high end.

## 6. Attention over the video and over depth (`attention_layers/`) — tasks 3.2 + 3.3

Same capture for both subtasks (they differ only in which layers are read), so
the head selection is identical: 5 depth percentiles x 6 chunk percentiles x
the same 4 seeded heads per layer as task 3.1, at each chunk's **last**
denoising step only. Rerun: `run_captures.py --gpus 2,4,6,7` then `plot.py`.

Geometry cross-check (720p/20s, last chunk, visible tokens / tokens-per-frame):
Self-Forcing 81 latent frames in one contiguous segment (full context), Rolling
Forcing 21 frames in 2 segments, LongLive-2 32 in 2, LingBot 18 in 2 — the sink
plus window split shows up as the second segment, matching the geometry table.

Headlines (720p/20s, last chunk, share of visible keys for 90% of a row's mass):

| model | 0% depth | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Self-Forcing 1.3B | 40.6% | 1.4% | 4.7% | 3.9% | 1.9% |
| Rolling Forcing 1.3B | 2.5% | 1.6% | 1.3% | 0.8% | 1.1% |
| LongLive-2.0 5B | 30.9% | 1.0% | 4.2% | 4.1% | 20.7% |
| LingBot-World v2 14B | 4.4% | 5.4% | 2.8% | 0.4% | 13.5% |

- **Capped models are flat in steady state**; only chunk 0 differs (it can see
  nothing but itself). Self-Forcing's *share* also barely moves (42.0% ->
  40.6% at layer 0) but its visible set grows 27x, so the absolute cost grows
  from ~4.5k to ~118k tokens — the attention-side counterpart of its linear
  per-chunk slowdown in section 3.
- **First and last layers dense, middle sparse**, across the whole video, not
  just chunk 0.
- Self-Forcing's middle layers get *wider* with context (layer 14: 2.8% ->
  4.7%), so their absolute key count grows faster than linearly. The capped
  models show no such drift.

Caveat: the key axis is sampled at stride 32, so shares are estimates; very
concentrated heads (a few 0.1%) rest on few samples.

Gotcha found here: `extra_env` used to be called with `durations[0]`, but a
realtime model serves every duration from **one** server process, so LingBot
got the 5s chunk percentiles for all three durations. It now receives the whole
duration list and requests the union; the plotter filters back down.

## 7. Bidirectional baseline (`attention_bidirectional/`) — task 4

Wan2.1-T2V-1.3B attends over the whole video at once — no chunks, no KV cache,
no causal mask — and is the model the causal ones are distilled from (identical
transformer shape as Self-Forcing: 30 layers x 12 heads). 6 configs x 5 depth
percentiles x 4 seeded heads x the 0/25/50/75/100% steps of its 50-step
schedule. The sampling stride adapts to the sequence length (16 at 480p/5s,
256 at 720p/20s) since the matrix is (total tokens)^2.

- **The depth profile is inverted versus the causal models.** Layer 0 is the
  *sparsest* here — 10 to 57 tokens (0.02-0.03%), nearly constant as the
  sequence grows from 32k to 292k tokens, i.e. a strictly local diagonal band.
  The dense layers are the middle ones (layer 7: 5.7-11.8%). The causal models
  are the other way round (layer 0 at 40-58%, middle sparse). Same architecture,
  same parent model — causal distillation moves where attention concentrates.
- **Denoising tightens attention**, exactly as in task 3.1 (layer 7: 27-31% ->
  5.7%), so that property comes from diffusion itself, not from block-causality.
- **Longer sequences need a larger fraction**: layer 7's final share is 5.7% ->
  6.8% -> 11.8% for 5/10/20s at 480p, so its absolute count grows
  super-linearly (1.9k -> 4.5k -> 14.9k tokens).

Caveat: 720p/20s is far out of distribution for a 480p-native 1.3B model (9x
the tokens of 480p/5s) and is the one config where the middle layers *re-widen*
at the last steps (layer 7: 21.9% -> 28.0%). Treat it as a stress test; the
other five agree. It took 43 min on one GPU.
