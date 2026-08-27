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
  layer 14 head 2, last layer 29 head 3.
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
  `cos(A_i0, A_ij)` flattened — column 0 is 1 by construction.

## Outputs

`results/investigation/qk_map_similarity/`

- `runs/<p>/` — video, `run.log`, `qk/qk_L{layer}_c{chunk}_s{step}.npz`
  (query `[Sq, heads, d]`, key `[Sk, heads, d]` fp16 + geometry); Q/K captured
  for **all five** prompts, doc figures use p1.
- `plots/p1/L{l}_h{h}_c{c}_s{s}.png` — 64 attention maps.
- `similarity/p1/sim_L{l}_h{h}_c{c}_s{s}.json` — 64 cosine tables
  (`cosine[i][j]`, query frames x key frames).

## Gotchas

- GPU use follows the exclusive-idle rule: `GpuPool` acquisition + `GpuWatchdog`
  kill-and-requeue on co-tenants (both imported from `sparse_baselines/common.py`).
- The 720p latent frame is 45x80 = 3600 tokens; a chunk is 3 frames = 10800
  query tokens; the last chunk sees 21 frames = 75600 keys.
- Feishu can't insert media mid-doc: `doc_update.py` writes `[[kind:name]]`
  placeholder paragraphs (including inside table cells), then media-inserts at
  the doc end and `block_move_after`s each file behind its placeholder.
