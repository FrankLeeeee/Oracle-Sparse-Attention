# Token-Level Attention Maps: LingBot-World v2 & LongVie 2

Date: 2026-07-22
Scope: render attention at *latent-token* resolution — for every key token, how
much attention mass it receives — for the two 14B video DiTs, and compare a
block-causal world model against a full-attention I2V model on the same
conditioning image and grid. Companion to
[`attention map visualization.md`](attention%20map%20visualization.md) (chunk
granularity) and [`attention sparsity analysis.md`](attention%20sparsity%20analysis.md)
(displacement statistics).

The probe already kept the key axis unreduced behind
`SGLANG_DIFFUSION_ATTENTION_MAP_TOKEN_SCORES`; what was missing was a plotter,
instrumentation for full-attention Wan DiTs, and a way to record a realtime
session. All three are added here.

## Usage

```bash
export PYTHONPATH=/data/projects/vision-gen/sglang/python

SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_token \
SGLANG_DIFFUSION_ATTENTION_MAP_TOKEN_SCORES=true \
SGLANG_DIFFUSION_ATTENTION_MAP_QUERY_STRIDE=16 \
sglang generate --model-path /data/projects/vision-gen/models/LongVie2-Diffusers ...

RUN=/data/projects/vision-gen/attn_token/LongVie2Transformer3DModel-<timestamp>

python -m sglang.multimodal_gen.tools.plot_token_attention_maps $RUN
# -> $RUN/token_plots/chunk_XXX_frames.png, chunk_XXX_layers.png,
#    layer_summary.png

python -m sglang.multimodal_gen.tools.plot_token_attention_bars $RUN \
  --layers 0,6,11,17,22,28,33,39 --out-dir $RUN/token_bars
# -> $RUN/token_bars/chunk_XXX_layer_YY.png

# the chunk-level plotter also applies, and for a full-attention run its
# summary.png IS the frame-to-frame matrix (one query group per latent frame)
python -m sglang.multimodal_gen.tools.plot_chunk_attention_maps $RUN
# -> $RUN/plots/chunk_XXX.png, summary.png
```

Three figures per run: `chunk_XXX_frames.png` (one latent-frame panel per
frame, head- and layer-averaged), `chunk_XXX_layers.png` (layer x frame small
multiples, head-averaged, subsampled by `--max-layers` / `--max-frames`), and
`layer_summary.png` (each layer's frame-averaged spatial preference).

Colour is **multiples of uniform**: each row is a distribution over the tokens
visible to that chunk, so the plotter rescales by the visible-token count and
1.0 means "exactly what flat attention would give". Never-visible tokens are
light grey, evicted tokens (exact 0) are black.

## The two runs

Same conditioning image (`outputs/longvie2_verification/condition_image.png`,
832x480), same prompt, same seed 1234, so both land on a **30x52 latent grid**
and the maps are directly comparable.

| | LingBot-World v2 14B causal-fast | LongVie 2 (Wan2.1-I2V-14B) |
|---|---|---|
| attention | block-causal, 3-frame blocks, 9-frame sink, 18-frame window | full, bidirectional *within a clip* |
| context | 27 latent frames over 9 chunks | 21 latent frames — **clip 1 only, no history** |
| query groups | 9 chunks (the model's own blocking) | 21, one per latent frame |
| steps | 4 DMD steps per chunk | 50, conditional CFG branch only |
| driver | realtime websocket session (below) | `sglang generate` with both control videos |
| dump | `attn_token/CausalLingBotWorldTransformer3DModel-20260722-060914` | `attn_token/LongVie2Transformer3DModel-20260722-071218` |

The superseded LongVie 2 dump `...-20260722-061518` is the single-query-group
run whose flat temporal profile is corrected below; keep it only as the record
of that mistake.

## Earlier dumps: `attn_tokens/` (1.3B baselines, 2026-07-21)

`/data/projects/vision-gen/attn_tokens/` is a **separate, earlier** experiment —
the same `token_scores` probe run on the two 1.3B baselines rather than the 14B
models, before either the plotter or the full-attention instrumentation existed:

- `CausalWanTransformer3DModel-20260721-145621` and
  `RollingForcingWanTransformer3DModel-20260721-145820`
- plain `sglang generate` on the local converted checkpoints, T2V (no image),
  832x480, 81 frames, seed 42, same fox prompt; 7 chunks, 30 layers, 12 heads,
  default `QUERY_STRIDE=8`, spatial probe off
- 210 PNGs each, rendered by what is now
  `tools/plot_token_attention_bars.py` (see below)

Its `meta.json` predates the recorded grid, so pass `--grid 30x52` to render
those dumps with `plot_token_attention_maps`.

## The two views

`plot_token_attention_bars` was written for the run above and lived only in a
session scratchpad; it is now checked in, unchanged except for deriving the
frame/chunk token counts instead of requiring `num_token_per_frame` and
`num_frames_per_block` in `meta.json` (a full-attention run records neither).
The port reproduces the 2026-07-21 figures **byte-for-byte** — `md5
478d0693f6a9f5ff4b2b20dcc225ec30` for RF chunk 6 / layer 13.

The two plotters read the same array and are complementary:

| | `..._maps` | `..._bars` |
|---|---|---|
| figure | per chunk | per (chunk, layer) |
| head axis | averaged | one panel per head |
| token axis | folded onto the latent grid | global token index, one bar each |
| answers | *where in the frame* | *which head, and where in the sequence* |

The bar view is the one that shows head divergence and chunk boundaries; the
map view is the one that shows registers and sink bands as image structure.
Neither subsumes the other, so today's runs are rendered both ways —
`token_plots/` and `token_bars/` under each run directory.

Bar figures cost ~37 s each at 40 heads x 32760 tokens (vs 12 heads in the 1.3B
run), so today's are a subset: layers 0, 6, 11, 17, 22, 28, 33, 39, and for
LingBot chunks 0, 4, 8. Pass `--chunks` / `--layers` to widen it.

## What the maps show

**LingBot-World v2 — the cache schedule is visible directly.** Chunk 8's frame
panel is the clearest single picture: frames 0-8 bright (the pinned sink),
frames 9-17 light grey (never in this chunk's cache), frames 18-26 a rising
ramp into the chunk itself. Frame mass in units of uniform:

| frames | 0 | 1-8 (sink) | 9-17 | 18-22 | 23 | 24-26 (self) |
|---|---|---|---|---|---|---|
| mass | 0.95 | 0.45-0.58 | not visible | 0.58-0.91 | 1.50 | 2.36-2.75 |

Frame 0 gets ~2x its sink neighbours — an attention sink *inside* the pinned
block, the same emergent first-frame preference the chunk-level probe found in
Causal Forcing. Attention is also markedly concentrated: 50% of the mass sits
in 16.6% of the visible tokens, and the strongest layer (L8) needs only 51% of
tokens for 90% of its mass.

**LongVie 2 — strongly local in time, within a clip.** An earlier version of
this note reported the frame profile as "nearly flat, 0.90-1.33x uniform,
nothing to exploit". That was wrong, and wrong for a boring reason: the probe
had been told to treat the whole clip as one query group, so every frame's
queries were averaged together. With full attention and equal queries per frame,
total received mass is then near-uniform *by construction* — the number measured
the aggregation, not the model. Re-run with one query group per latent frame
(`num_frames_per_block=1`), the frame-to-frame matrix is anything but flat:

| | dt=0 | \|dt\|<=1 | \|dt\|<=2 | \|dt\|<=4 |
|---|---|---|---|---|
| share of mass | **0.321** | 0.529 | 0.631 | 0.741 |

A frame puts **32% of its attention on itself** where uniform would be 4.8% —
6.7x — and 74% within +-4 frames. Frame 10 is typical: 0.296 on itself,
0.104/0.097 on its neighbours, 0.024 on frame 0, 0.019 on frame 20. Attention is
close to symmetric (0.309 forward vs 0.370 backward), as bidirectional attention
should be. **So the temporal axis is very much exploitable within a clip** — the
opposite of what this note previously claimed.

The conditioning frame behaves differently from a sink. It attends to *itself*
with 0.625, and the mass other frames send it decays fast and then sits slightly
*below* uniform:

```
frame  0     1     2     3     4     5     6     8    10    14    18    20
->f0   .625  .131  .073  .050  .041  .036  .033  .026  .024  .022  .026  .028
```

Averaged over frames 1-20 that is 0.035 against a uniform 0.048. LingBot's
pinned sink is a genuine attractor; LongVie 2's conditioning frame is just a
strong local neighbour that fades by about frame 8, with a slight uptick at the
tail.

The spatial reading is unchanged by the regrouping: more diffuse than LingBot at
the layer+head mean (90% of mass needs 84.8% of tokens vs 71.1%), with the
per-layer and per-head structure below.

**Both models put mass on isolated "register" tokens.** These are single latent
cells that receive 10-170x uniform while their neighbours sit near 1. LingBot
chunk 8 has 98 tokens above 10x (peak 174x at layer 28); LongVie 2 has 43
(peak 82x at layer 22). They are a per-layer phenomenon — the layer+head mean
peaks at only 16x / 17x — and they are the sparse-attention-relevant structure
that a chunk-level or displacement-level view averages away entirely.

**Head averaging hides the sparsity — badly.** This is what the bar view is
for, and it changes the headline. Share of a head's mass held by its top-2048
key tokens (6.3% / 7.3% of the visible sequence):

| | min | median | max | sites >90% | sites >99% |
|---|---|---|---|---|---|
| LongVie 2, chunk 0 | 6.6% | 22.5% | 100.0% | 9 / 1600 | 3 / 1600 |
| LingBot, chunk 8 | 8.0% | 46.9% | 100.0% | 56 / 1600 | 8 / 1600 |

The saturated sites are a thin tail, not the norm — but the spread is the point.
**Within a single layer** the head-to-head gap has a median of 57 points
(LongVie 2) / 67 points (LingBot) and reaches 93 / 92 points at layer 38; at
LongVie 2 layer 22 the heads run from 11.9% (h0) to 74.5% (h19). So the
head-averaged "90% of mass needs 84.8% of tokens" quoted above is an artifact of
averaging heterogeneous heads, not a statement that any head is diffuse — the
same conclusion the displacement study reached on the spatial axis, now visible
on the token axis.

The same view splits LingBot's heads by *what they read*. Share of a site's
chunk-8 mass on the pinned sink (frames 0-8) vs the sliding window (18-26):

| | min | median | max |
|---|---|---|---|
| sink | 0.0% | 27.3% | 88.0% |
| window | 12.0% | 72.7% | 100.0% |

**212 of 1600 sites put >95% in the window and ignore the sink entirely**, while
728 put >30% on it. Within layer 22 alone the sink share runs 0.9% (h30) to
66.4% (h33). A head-uniform cache policy has to serve both populations; a
per-head one could drop the sink for an eighth of the sites outright.

Curiously the most concentrated site is **layer 38 head 33 in both models** —
90% of its mass in 1844 tokens (LongVie 2, 5.6% of the sequence) and 527
(LingBot, 1.9%). Both are 40-layer/40-head Wan-family backbones so the indices
are comparable, but they were trained separately; whether this is coincidence or
a backbone property needs more than n=2.

**A sink *band*, not just sink tokens.** In LongVie 2 the top row of latent
frame 1 averages **4.49x uniform** against 1.32x for that frame overall. The
top row is mildly preferred everywhere in both models (1.18-1.21x), but this
one band is a different scale of effect.

**Patchification leaves a checkerboard.** Even latent columns average 1.04x /
1.08x uniform and odd columns 0.96x / 0.92x in the two models — the vertical
striping visible in every figure. It is an artifact of the 2x2 patch embedding,
not scene content, and it means any spatial block selection at odd granularity
will straddle a systematic bias.

**Depth changes the pattern qualitatively.** The registers are a middle-layer
phenomenon in both models — mean peak token by layer group is 8x / 34x / 18x of
uniform for LongVie 2 (L0-9 / L10-31 / L32-39) and 14x / 40x / 14x for LingBot.
Early layers instead trace the scene: LongVie 2's layer 6 map has the fox
silhouette legible in it, and LingBot's layers 0-9 hold a soft blob on the
subject.

Layers also disagree sharply about the conditioning frame. Frame 0's mass
ranges from 0.59x uniform at layer 17 to 2.87x at layer 36 in LongVie 2, and
0.41x (L15) to 2.71x (L36) in LingBot; roughly half the layers sit below
uniform on it in each model. The head-and-layer-averaged 1.33x headline is a
cancellation of two opposite groups, not a consensus.

## Implementation

```
new:
  python/sglang/multimodal_gen/tools/plot_token_attention_maps.py
  python/sglang/multimodal_gen/tools/plot_token_attention_bars.py  (promoted)
modified:
  runtime/utils/attention_map_probe.py            (recording_scope, grid in meta)
  runtime/models/dits/wanvideo.py                 (probe the full-attention path)
  runtime/pipelines_core/stages/denoising.py      (CFG/warmup gating + flush)
  runtime/pipelines_core/stages/causal_denoising.py  (deferred flush)
  runtime/pipelines_core/stages/model_specific_stages/lingbot_world/
      lingbot_world_causal_denoising.py           (use the deferred flush)
  runtime/realtime/states/causal.py               (on_dispose hook)
```

Three things had to change beyond writing the plotter.

**Full-attention Wan DiTs were not instrumented.** Only the causal variants
were. `WanTransformer3DModel.forward` now calls `begin_forward` with the whole
video as a single chunk, and `WanTransformerBlock` records with
`key_segments=[(0, seq_len)]` — every key is a plain video token, so the
segment machinery that exists for rolled KV caches degenerates to one range.
Blocks get their depth from the transformer after the list is built (the same
pattern `CausalWanTransformer3DModel` already used), which means LongVie 2's
control towers — which reuse `WanTransformerBlock` — keep `layer_index == -1`
and are never recorded. Recording only the main stream is what makes the map
mean "where the video attends", not "where the video and two side networks
attend".

**CFG doubled the cost for nothing.** The unconditional branch runs the same
DiT over the same geometry with an empty prompt. `recording_scope` gates the
probe around a block of forwards and composes by narrowing only, so the
denoising loop can disable it for warmup and `predict_fn` can enable it only
for the conditional branch without the inner scope re-enabling a warmup pass.

**LingBot generates one chunk per request.** The one-shot CLI path emits a
single 3-frame block regardless of `--num-frames`, so the block-causal structure
never appears — a realtime *session* is the only way to get a multi-chunk video.
But the flush was gated on `not persist_state`, so a session never flushed at
all. `RealtimeCausalDiTState` grew an `on_dispose` hook that the stage sets when
probing, so the whole session lands in one run directory when the client
disconnects. Driving it is a websocket loop (`init` payload, N frame batches,
close); the throwaway script lives in the session scratchpad.

## Cost

At `QUERY_STRIDE=16`, LongVie 2 (81 frames, 50 steps, 40 layers, 32760 tokens):
**10.8 s/step with the probe vs 7.11 s/step without**, i.e. +51% on the denoise
stage (5.9 min -> 8.9 min). The query grouping does not affect this — 21 groups
cost the same per step as 1, since the softmax is shared and only the scatter
destination changes. LingBot's 9-chunk session took 78 s end-to-end.

Memory and disk *do* scale with the group count, because `token_scores` is
`[groups, layers, heads, tokens]`:

| run | groups | `token_scores.npz` | GPU peak |
|---|---|---|---|
| LongVie 2, one group (superseded) | 1 | 186 MB | ~40 GB |
| LongVie 2, per frame | 21 | **4.0 GB** | ~87 GB |
| LingBot, 9 chunks | 9 | 347 MB | — |

Per-frame grouping on a longer video would grow linearly from here, so a 5x
longer clip needs either a coarser grouping or a bigger card.

## Limitations

- **LongVie 2 is autoregressive, and this run does not show that.** Upstream
  (`inference.py`) loops clips, carrying an 8-frame `history` and a persistent
  `noise` between them, and the pipeline prepends `history_latents` to the token
  sequence (`x = torch.cat([history, x], dim=1)`, positions from
  `freqs_history`); the paper calls it "an end-to-end autoregressive framework"
  for 3-5 minute videos. Attention is full and bidirectional only *within* a
  clip, over `[history ++ current]`. Our integration implements a single clip
  with no history (`runtime/models/dits/longvie.py` still raises when the
  control and main token counts differ), so **the history->current attention
  that carries LongVie 2's temporal consistency is absent from the dump
  entirely**. What was probed is clip 1 of a rollout, not the model.
- **Received mass, not a token-to-token map.** Each row is averaged over the
  sampled queries of a chunk, so this answers "which tokens are attended to"
  and not "which token attends to which".
- **The query grouping is a probe parameter, and it silently decides what a
  "flat" result means.** `wanvideo.py` originally grouped the whole clip into
  one row, which makes any full-attention model look temporally uniform no
  matter what it does. It now groups per latent frame. Cost: `token_scores`
  scales with the group count — 186 MB at 1 group, 4.0 GB at 21 (GPU peak ~87 GB
  including the model). Anything reading these dumps should check
  `num_frames_per_block` in `meta.json` before interpreting a temporal claim.
- One prompt, one seed, one conditioning image per model. The register-token
  and sink-band findings are n=1.
- Query stride 8 (LingBot) / 16 (LongVie 2) — the two runs sample queries at
  different rates, which is fine for a mean but not for comparing variance.
- Rank-0 heads only under TP; both runs were single-GPU and replicated.
- LingBot's `meta.json` reports `num_frames: 21` from the last chunk request,
  which is the per-request value, not the 27 latent frames actually generated.

## Follow-ups

- Per-`(layer, head)` register-token *maps*. The bar view already answers "how
  concentrated is each head" (very, and unevenly); folding those same heads onto
  the latent grid individually would answer "and is it the same cell each time",
  which is what decides pinnable-vs-dynamic.
- Whether register positions are stable across prompts and seeds. If they are,
  they can be pinned like a sink; if they are content-dependent, they force
  dynamic selection (the same conclusion the spatial study reached).
- Extend the same instrumentation to the rest of the full-attention Wan family
  now that `wanvideo.py` carries the hook.
- **Re-probe LongVie 2 once the clip loop and history conditioning land** (phase
  2 of [`longvie2 integration.md`](longvie2%20integration.md)). That is the run
  worth having: with history tokens prepended, `[history | current]` is exactly
  the chunk structure the probe was built for, and it would show whether the
  8-frame history is attended to like LingBot's sink or ignored — the question
  this note set out to answer for LongVie 2 and could not.
