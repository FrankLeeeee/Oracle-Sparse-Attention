# QK attention-map & frame-to-frame pattern similarity (dense Self-Forcing)

Everything here runs the **dense** model — no sparse attention anywhere. The
study generates the five multi-prompt videos, captures the exact post-RoPE Q/K
of a few (layer, head) picks, renders full-softmax attention maps, and
quantifies the frame-to-frame pattern similarity that motivates OSA's
replicate design.

Published to Feishu doc `Rs3sdTCinoc6kqxdiGxcUDIQnfd`（OSA Properties）.

## Setup

- Model: Self-Forcing 1.3B full-context (fullctx-null conversion), 720p
  (1280x720), 5 s = 81 pixel frames = 21 latent frames = 7 chunks of 3, seed
  42, single H200.
- Prompts: `scripts/investigation/prompts.json` (p1–p5, from the SF study's
  multi-prompt validation).
- (layer, head) picks (30 layers, 12 heads): layer 0 heads 0 and 1, middle
  layer 14 head 2, last layer 29 head 3. A second "extra" round (`--spec
  extra`) adds five depth-verification picks L5·h4 / L10·h5 / L15·h6 / L20·h7
  / L25·h8 after the first round showed frame-pattern similarity is high only
  in layer 0 (0.92–0.99 vs 0.03–0.67 elsewhere; mid layers lowest, both ends
  higher).
- Chunks: 0 / 2 / 4 / 6 = the 0 / 33 / 66 / 100th percentiles of 7 chunks;
  all 4 denoising steps of each.

## Pipeline

```bash
cd scripts/investigation/qk_map_similarity
python run.py                      # 5 dense videos + Q/K capture (exclusive GPU)
CUDA_VISIBLE_DEVICES=<idle> python plot_maps.py --run p1     # 64 map PNGs
CUDA_VISIBLE_DEVICES=<idle> python similarity.py --run p1    # 64 cosine tables
python doc_update.py --stage sections   # doc text + tables with placeholders
python doc_update.py --stage media      # upload videos/plots into placeholders
python doc_update.py --stage verify

# depth-verification round (5 extra heads; re-generates p1 deterministically
# into runs/p1_extra/, dumps land in the shared runs/p1/qk/)
python run.py --spec extra --prompts p1
CUDA_VISIBLE_DEVICES=<idle> python similarity.py --run p1 --spec extra
python doc_update.py --stage extra      # verification subsection + 9-row summary

# self-referenced recomputation + temporal consistency
CUDA_VISIBLE_DEVICES=<idle> python similarity.py --run p1 --spec main
CUDA_VISIBLE_DEVICES=<idle> python similarity.py --run p1 --spec extra
python plot_temporal.py --run p1
python doc_update.py --stage resim      # republish intro + all 9 tables + summary
python doc_update.py --stage temporal   # temporal subsection + figure + table
```

- `run.py` launches plain `sglang generate` per prompt with
  `hook/sitecustomize.py` on `PYTHONPATH`. The hook wraps the attention-map
  probe's `ChunkAttentionRecorder.record` and dumps raw fp16 Q/K (selected
  heads only) per (layer, chunk, step) — ~50x smaller than dumping softmax
  maps, and every downstream artifact is recomputed from them *exactly*. The
  probe itself is enabled but body-disabled
  (`SGLANG_DIFFUSION_ATTENTION_MAP_QK_ONLY=1`, no `QK_CHUNKS`), so the hook is
  the only writer; probes were previously verified not to perturb generation
  (byte-identical videos).
- `plot_maps.py` recomputes `softmax(QK^T/sqrt(d))` over the **full** visible
  key axis and renders token-by-token maps: y = query token, x = key token,
  log color scale (mean probability at 75k keys is ~1e-5), mean-pooled
  15x45 tokens/pixel (45 = one latent row, so frame boundaries stay
  pixel-aligned), green latent-frame boundaries on both axes, white vertical
  line where the current chunk's keys start.
- `similarity.py` computes, per query frame i and key frame j,
  `A_ij = softmax(Q_i K_j^T / sqrt(d))` (softmax **within that key frame
  only**, i.e. the standalone pattern of the frame pair) and reports
  `cos(A_{i,self}, A_ij)` flattened, where the reference is the query frame's
  own self map `A_{i, Sk/T-3+i}` (the chunk's frames are the newest 3 of the
  cache) — each row's self column is 1 by construction. It also writes a
  temporal-consistency table per pick: the chunk-`--ref-chunk` (default 0)
  self maps compared against every captured chunk's maps,
  `cos(A_{C,i,self}, A_{c,i,j})`; `plot_temporal.py` renders the 9-panel
  figure.

## Deep dive (sparse-opportunity analysis)

```bash
python run.py --spec all9 --chunks 0,1,2,3,4,5,6 --prompts p1   # all 9 picks, every chunk
CUDA_VISIBLE_DEVICES=<idle> python deep_dive.py --run p1
python doc_update.py --stage deepdive
```

`deep_dive.py` computes five measurements (JSON per measurement under
`deep_dive/p1/`, figures under `plots/p1/`):

- `ref_matrix` — cos(self maps of reference chunk C, pair maps of chunk c) for
  every (C, c): re-calibrating at any C (even C=c-1) does not rescue the
  mid-layer whole-map replicate — their frame-pair maps change every chunk.
- `mass_transfer` — the oracle recall: chunk-0 per-query top-p% *frame-relative*
  key positions replicated over all visible frames; plus refreshed (same
  chunk) and prev (previous chunk — measurable for free in its cache-update
  forward) variants. Key result: top-k mass and whole-map cosine disagree —
  L20·h7 has ~0.4 cosine but 0.97-1.0 mass@10%; L0·h0 has 0.92+ cosine but
  0.16 mass (near-uniform rows). Three head families: geometric/local,
  diffuse, content-dependent.
- `local_window` — mass within Chebyshev radius r of the query's own grid
  position (zero-calibration geometric pattern): L0·h1 hits 0.82 at r=1
  (0.25% density).
- `frame_mass` — per-key-frame distribution: L20·h7 puts 0.99 on its own 3
  frames (skip history entirely); mid layers need 11-17 of 21 frames.
- `step_consistency` — pair-map cosine of steps 0-2 vs 3: stable heads 0.65+,
  mid layers tighten only late in denoising (plan late or in the cache-update
  forward).

Strategy proposal (in the doc, targeting faster-than-LightForcing at matched
quality): offline per-head taxonomy -> static execution for local /
own-chunk / diffuse heads (zero planning), LightForcing-style selection only
for content-dependent heads with the plan measured once per chunk in the
previous chunk's KV-cache-update forward and reused across all 4 steps, and
per-head density budgets instead of one global knob.

## Outputs

`results/investigation/qk_map_similarity/`

- `runs/<p>/` — video, `run.log`, `qk/qk_L{layer}_c{chunk}_s{step}.npz`
  (query `[Sq, heads, d]`, key `[Sk, heads, d]` fp16 + geometry); Q/K captured
  for **all five** prompts, doc figures use p1.
- `plots/p1/L{l}_h{h}_c{c}_s{s}.png` — 64 attention maps.
- `similarity/p1/sim_L{l}_h{h}_c{c}_s{s}.json` — cosine tables
  (`cosine[i][j]`, query frames x key frames, self-referenced; `self_columns`
  marks each row's trivially-1 column).
- `similarity/p1/temporal_L{l}_h{h}_ref0_s3.json` — temporal-consistency
  tables (`chunks[c][i][j]` vs chunk-0 self maps); figure
  `plots/p1/temporal_ref0_s3.png`.

## Gotchas

- GPU use follows the exclusive-idle rule: `GpuPool` acquisition + `GpuWatchdog`
  kill-and-requeue on co-tenants (both imported from `sparse_baselines/common.py`).
- The 720p latent frame is 45x80 = 3600 tokens; a chunk is 3 frames = 10800
  query tokens; the last chunk sees 21 frames = 75600 keys.
- Feishu can't insert media mid-doc: `doc_update.py` writes `[[kind:name]]`
  placeholder paragraphs (including inside table cells), then media-inserts at
  the doc end and `block_move_after`s each file behind its placeholder.

## Content independence check

```bash
python run.py --spec all9 --chunks 0,1,2,3,4,5,6 --prompts p2,p3,p4,p5
CUDA_VISIBLE_DEVICES=<idle> python content_stability.py
python doc_update.py --stage stability
```

`content_stability.py` repeats the taxonomy metrics on all five prompts and
adds the direct test: positions calibrated on prompt A's chunk 0 deployed on
prompt B (`cross@10%`). Result (in the doc's 「画像的内容无关性验证」 section):
static-family heads are strictly content-independent (L0·h1 own 1.00 on every
prompt, position overlap 0.98; L20·h7 0.96-0.99; diffuse heads equally tight),
but the content-dependent heads' metrics swing with content (L10·h5 own
0.44-0.83, local_r9 0.34-0.92; the chaotic p1 is always the hardest). The
strategy's claim was accordingly qualified in place: family assignment is
calibrate-once only under conservative boundaries; borderline heads go to
runtime selection, and content-head budgets must be sized on worst-case
(high-motion) content.

## Next-step experiments (a)(b)(c)

```bash
python run.py --spec sweep --chunks 0,3,6 --steps 3 --prompts p1,p4  # all 360 heads
CUDA_VISIBLE_DEVICES=<idle> python taxonomy_sweep.py                 # (a) classify
# (b) LightForcing per-head mask recall (osa_recall LF hook, p1_forest, 5s):
#     ../osa_recall/run.py --method lightforcing --density 0.2/0.3 --seconds 5 --prompt p1_forest
CUDA_VISIBLE_DEVICES=<idle> python lf_compare.py                     # (b) compare
CUDA_VISIBLE_DEVICES=<idle> python bench_lf_plan.py                  # (c) plan cost
python doc_update.py --stage nextsteps
```

Results (720p/5s, p1, chunk 6, step 3): (a) 131/360 heads (36.4%) are
planning-free (50 local, 63 short-window, 8 frozen, 10 diffuse), fleet mean
density 0.167, static families cluster at the network ends. (b) the composed
system (static families + LF's own selection for the 229 content heads)
reaches mean recall 0.792 at density 0.166 vs LF-d0.2's 0.815 at 0.198 —
parity-level quality proxy at 84% of the keys. (c) LF planning is only
0.34 ms/call (~3% of a 8-14 ms attention call; ~0.29 s per video) — the
amortization lever that decided LongLive-2 is minor here; density is the
lever. Linear fit of LF's measured density-time ladder projects the hybrid
at ~8.0 s denoise = 1.34x over dense, ~1.1x over LF-d0.2 at 5 s (larger at
longer durations). Remaining: shrink the 64% content-head share and build
the real backend for e2e timing.
