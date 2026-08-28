# MSA benchmark: dense vs MSA vs LightForcing (Self-Forcing, 720p)

MSA (mixed sparse attention, `--sparse-attention msa`) executes each head by
its offline family: local heads (static row window, frame-replicated) and
short-window heads (newest m frames) run with zero runtime planning through
the frame-replicated Triton kernel in `sparse/msa_kernel.py`; content heads
use LightForcing-style two-stage pooled top-k through a head-indexed range
kernel, planned once per (layer, chunk) and reused across all denoising steps
and the cache refresh. Taxonomy calibrated on the study prompts p1-p5
(`../qk_map_similarity/msa_taxonomy_self_forcing.json`, execution-aware
gates: 87/360 static, 273 content).

This benchmark uses five NEW prompts (`bench_prompts.json`, out of
calibration), 720p/5s, seed 42, exclusive-GPU serial timing, PSNR vs the same
prompt's dense output.

```bash
python run_bench.py                              # dense/msa/lightforcing, 5s
python run_bench.py --methods msa25              # content-density 0.25 point
python run_bench.py --seconds 20 --prompts b1    # 20s timing
python doc_update.py                             # publish section to the doc
```

## Results (held-out prompts, 5s means)

| method | denoise | vs dense | cum density | PSNR mean |
|---|---|---|---|---|
| dense | 10.63 s | 1.00x | 1.0 | - |
| MSA content 0.20 | **8.73 s** | **1.22x** | 0.337 | 17.01 |
| MSA content 0.25 | 8.94 s | 1.19x | 0.369 | 17.72 |
| LightForcing 0.2 | 9.00 s | 1.18x | 0.357 | 17.86 |

MSA@0.25 = LightForcing-level quality (within 0.14 dB mean, <=0.7 dB per
prompt) while slightly faster and reading similar keys; MSA@0.20 is 3% faster
still at ~0.85 dB mean cost. Known gap: at 20s MSA (33.2 s) trails LF
(29.0 s) — the frame-replicated static walks degrade at 81 visible frames
(~250-320 TFLOP/s vs ~500 for merged long ranges); fix path is gather-based
static execution or query-permuted true 2-D windows.

Development notes (benchmark-driven reversals worth remembering): pure global
top-k content selection executes ~2x slower than the two-stage version (
scattered blocks don't merge into ranges); frame-subsampled diffuse execution
is statistically biased at short contexts (b5 -2.8 dB) — diffuse heads ship
as content.
