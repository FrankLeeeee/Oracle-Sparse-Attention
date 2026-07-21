# Baseline Integration: Causal-Forcing & Rolling Forcing

Date: 2026-07-20 / 2026-07-21
Scope: native SGLang Diffusion (`python/sglang/multimodal_gen`) support for two
Self-Forcing-family autoregressive video models on Wan2.1-T2V-1.3B:

- **Causal Forcing** — thu-ml/Causal-Forcing (arXiv 2602.02214, ICML 2026)
- **Rolling Forcing** — TencentARC/RollingForcing (arXiv 2509.25161)

## Background

Both models are KV-cached, block-wise causal, few-step DMD-distilled Wan
generators (descendants of guandeh17/Self-Forcing). Upstream releases are
DiT-only `.pt` training states (`generator` / `generator_ema` keys, original
Wan parameter naming, no diffusers layout):

- `zhuhz22/Causal-Forcing` — `chunkwise/causal_forcing.pt` (3-frame blocks),
  `framewise/causal_forcing.pt` (1-frame blocks). 4 warped denoising steps
  `[1000, 750, 500, 250]`, flow shift 5, no CFG. Inference is identical to
  Self-Forcing (the novelty is training-time: causal-teacher ODE/consistency
  initialization before DMD).
- `TencentARC/RollingForcing` — `checkpoints/rolling_forcing_dmd.pt`. 5 steps
  `[1000, 800, 600, 400, 200]`, rolling window of 5 blocks denoised jointly at
  staggered noise levels, attention-sink KV cache for minute-scale streaming.

## What was added

### Checkpoint conversion

`tools/convert_forcing_to_diffusers.py` — assembles a self-contained
diffusers-layout model dir from an upstream `.pt`:

- presets: `causal-forcing-chunkwise`, `causal-forcing-framewise`,
  `rolling-forcing`
- picks `generator_ema` (default), strips `model.[_fsdp_wrapped_module.]`
  prefixes, renames original-Wan → diffusers keys (reuses
  `wan_repack.TRANSFORMER_KEYS_RENAME_DICT`), casts bf16
- writes `transformer/` (weights + config with `_class_name` and causal
  geometry: `num_frames_per_block`, `sliding_window_num_frames`, `sink_size`,
  `max_attention_num_frames`), copies scheduler/text_encoder/tokenizer/vae
  from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, writes `model_index.json`

Converted dirs on this machine: `/data/projects/vision-gen/models/`
(`CausalForcing-Wan2.1-T2V-1.3B-{chunkwise,framewise}-Diffusers`,
`RollingForcing-Wan2.1-T2V-1.3B-Diffusers`).

### Causal Forcing (reuses existing causal DMD infra)

- `runtime/pipelines/causal_forcing_pipeline.py` — `CausalForcingPipeline`
  subclasses `WanCausalDMDPipeline`; only swaps in
  `SelfForcingFlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)`.
  Denoising runs through the existing `CausalDMDDenoisingStage` (block-wise
  4-step DMD with clean-latent KV-cache refresh). Block size (3 vs 1 frames)
  comes from the converted checkpoint's `transformer/config.json` via
  `update_model_arch`.
- Configs: `CausalForcingWanT2V480PConfig` (pipeline_configs/wan.py),
  `CausalForcingT2VSamplingParams` (sample/wan.py — 4 steps, guidance 1.0,
  no negative prompt, 81 frames, 480×832, fps 16).

### Rolling Forcing (new denoising machinery)

- `runtime/models/dits/rolling_forcing_wanvideo.py` —
  `RollingForcingWanTransformer3DModel` (+ block/attention subclasses).
  Faithful port of upstream streaming attention: only the window's first
  block is written to the KV cache; first-ever block is stored **un-RoPE'd**
  as the attention sink and re-roped on the fly to a relative position just
  before the working cache; separate `updating_cache=True` forward overwrites
  the finished block's slots with clean t=0 features. Per-forward cache index
  math lives in pure `compute_rolling_cache_layout()` (computed once per
  forward, shared by all 30 layers via `RollingForcingSelfAttentionKVCache`).
- `runtime/pipelines_core/stages/rolling_forcing_denoising.py` —
  `RollingForcingDenoisingStage`: window schedule (ramp-up / full / drain),
  staggered per-frame timesteps (oldest block cleanest), joint denoise pass,
  stochastic re-noise of unfinished blocks to their next timestep, clean-cache
  update pass. Pure helpers: `build_rolling_window_bounds`,
  `build_staggered_timesteps`.
- `runtime/pipelines/wan_rolling_forcing_pipeline.py` —
  `WanRollingForcingPipeline`.
- Configs: `RollingForcingWanVideoConfig` (dits config: block 3, cache 24
  frames, sink 3 frames, attention context 21 frames),
  `RollingForcingWanT2V480PConfig`, `RollingForcingT2VSamplingParams`.

### Registry

Both registered in `registry.py` with `hf_model_paths` whose basenames equal
the recommended converted-dir names (so longest-substring resolution beats the
base `Wan2.1-T2V-1.3B-Diffusers` entry) plus `causal-forcing`/`rolling-forcing`
substring detectors.

### Extension hooks added to shared code

- `CausalWanTransformerBlock._self_attn_cls` and
  `CausalWanTransformer3DModel._block_cls` class attributes so subclasses swap
  attention/block classes without duplicating the block code.
- `RollingForcingSelfAttentionKVCache` subclass in
  `layers/kvcache/causal_attention_cache.py`.

## Bugs found and fixed in existing code

The `WanCausalDMDPipeline` / `CausalDMDDenoisingStage` path had bit-rotted at
HEAD (no registered model exercised it):

1. **DiT left on CPU under `dit_cpu_offload`** — the stage never reported the
   DiT use-site to the component residency manager (the base `DenoisingStage`
   does this per step via `_select_and_manage_model`). Fixed: report at
   `forward()` start. Also `SelfForcingWanT2V480PConfig` now declares
   `keep_resident_components=("dit", "vae")` so the 1.3B DiT stays on GPU
   when memory allows.
2. **`crossattn_cache` TypeError** — `CausalWanTransformerBlock` passed
   `crossattn_cache` to `WanT2VCrossAttention.forward`, which doesn't accept
   it. Fixed by porting LingBot's `_cross_attn_with_cache` (text K/V computed
   once per request, reused across steps) into the block.
3. **Inference-tensor in-place crash** — with `dit_cpu_offload=true` the
   executor runs the stage outside `inference_mode` (offload hooks need
   version counters), and the stage's in-place writes into the prep stage's
   inference-tensor latents raised. Fixed: clone the latents when
   `latents.is_inference()` and inference mode is off.

Registry gotcha worth remembering: partial matching treats registered repo
short-names as substrings, so a local dir named `X-Wan2.1-T2V-1.3B-Diffusers`
resolves to the *base Wan* config unless a longer registered path wins.

## Verification (8×H200 host)

- Unit: `test/unit/realtime/test_rolling_forcing_denoising.py` (window
  bounds, staggered timesteps, cache-layout invariants incl. eviction and
  bounded attention context) — all pass; no regressions in the realtime suite
  (4 pre-existing failures at HEAD: lingbot cache-config ×2, realtime
  adapter, webui presets).
- E2E, all visually verified non-noise / prompt-faithful (`--seed` fixed):
  - Causal-Forcing chunk-wise, 81f: ~23–29 s
  - Causal-Forcing frame-wise, 81f: ~22 s
  - Rolling Forcing, 81f: ~15 s (idle GPU)
  - Rolling Forcing, 501f (~31 s video): 45 s idle / 137 s shared GPU —
    exercises cache eviction + sink re-roping; frame 480 remains fully
    coherent with frame 30 (no drift or color collapse).
- Sample videos: `/data/projects/vision-gen/samples/`
  (`causal_forcing_chunkwise`, `causal_forcing_framewise`, `rolling_forcing`,
  `rolling_forcing_long`).

## Usage

```bash
export PYTHONPATH=/data/projects/vision-gen/sglang/python  # this checkout

python -m sglang.multimodal_gen.tools.convert_forcing_to_diffusers \
  --preset rolling-forcing \
  --checkpoint <path to rolling_forcing_dmd.pt> \
  --output-path /data/projects/vision-gen/models/RollingForcing-Wan2.1-T2V-1.3B-Diffusers

sglang generate \
  --model-path /data/projects/vision-gen/models/RollingForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "..." --num-frames 501 --save-output
```

Notes: pixel `num_frames` should map to latent frames divisible by 3
(`num_frames = 12k + 9`); configs auto-round otherwise. Concurrent
`sglang generate` processes need distinct `--master-port` /
`--scheduler-port` / `--port` (default master port 30005 collides).

## Known follow-ups (not done)

- Causal Forcing++ 1/2-step models (`denoising_step_list_first_chunk`
  support in the causal DMD stage).
- Frame-wise Causal-Forcing I2V (first-latent KV warm-up exists in the stage
  but is unwired/untested for this model).
- TP/SP for the Rolling Forcing DiT (currently replicated linears only).
- Seed-matched numerical parity vs the upstream repos (visual parity only so
  far; upstream envs not installed here).
- CI `gpu_cases` / suite entries — blocked on hosting the converted
  checkpoints (upload the three dirs to HF to unblock).
- Realtime/streaming serving integration (per-chunk session API, streaming
  VAE decode) — batch offline generation only for now.

## Changed / new files

```
modified:
  python/sglang/multimodal_gen/configs/pipeline_configs/wan.py
  python/sglang/multimodal_gen/configs/sample/wan.py
  python/sglang/multimodal_gen/registry.py
  python/sglang/multimodal_gen/runtime/layers/kvcache/causal_attention_cache.py
  python/sglang/multimodal_gen/runtime/models/dits/causal_wanvideo.py
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/__init__.py
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/causal_denoising.py
  docs_new/docs/sglang-diffusion/compatibility_matrix.mdx
new:
  python/sglang/multimodal_gen/configs/models/dits/rolling_forcing_wanvideo.py
  python/sglang/multimodal_gen/runtime/models/dits/rolling_forcing_wanvideo.py
  python/sglang/multimodal_gen/runtime/pipelines/causal_forcing_pipeline.py
  python/sglang/multimodal_gen/runtime/pipelines/wan_rolling_forcing_pipeline.py
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/rolling_forcing_denoising.py
  python/sglang/multimodal_gen/test/unit/realtime/test_rolling_forcing_denoising.py
  python/sglang/multimodal_gen/tools/convert_forcing_to_diffusers.py
```
