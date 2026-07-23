# minWM Causal HunyuanVideo 1.5 TI2V Integration

Date: 2026-07-22
Scope: native SGLang Diffusion support for **minWM's causal HY1.5-TI2V (8B)** —
KV-cached autoregressive chunk-by-chunk I2V on the HunyuanVideo-1.5 480p
backbone, trained with the Causal Forcing / Causal Forcing++ recipe.

- upstream code: <https://github.com/shengshu-ai/minWM> (Apache-2.0; the
  HunyuanVideo-derived sources carry the Tencent Hunyuan Community License)
- weights: `MIN-Lab/minWM` (`HY15/TI2V/{bidirectional,ar_diffusion_tf,causal_ode,causal_cd,dmd}`, MIT)
- base components: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`
- papers: Causal Forcing arXiv 2602.02214, Causal Forcing++ arXiv 2605.15141,
  minWM tech report arXiv 2605.30263

**Status: complete and verified.** DiT is bit-exact (fp32) against the minWM
reference with real weights; end-to-end TI2V works via `sglang generate` and
`DiffGenerator`, including multiple sequential requests. ~16.5 s for a
77-frame 480×832 video on one H200 (7.9 s denoise, 4.75 s VAE decode).
Sample outputs: `/data/projects/vision-gen/outputs/minwm-hy15/`.

## What the release actually is

minWM is the official full-stack framework from the Causal Forcing authors
(successor to `thu-ml/Causal-Forcing`, which we already integrated for Wan).
The HY15/TI2V line is a finetune of **tencent/HunyuanVideo-1.5,
`transformer/480p_i2v`, 8B** — and, unlike the Wan `.pt` releases, the
checkpoints are already **diffusers-layout safetensors with original tencent
parameter naming** (`double_blocks.N.img_attn_q`, split q/k/v). We verified the
safetensors header is byte-identical in size/tensor-count/naming to tencent's
480p_i2v file, so **no weight conversion is needed** — the causal modifications
are entirely *weightless* (forward-path restructuring only).

Architecture (from `config.json`, identical across all five stage checkpoints):
hidden 2048, 16 heads × head_dim 128, **54 dual-stream (MMDiT) blocks, 0
single-stream**, MLP ratio 4 (gelu_tanh), per-head RMS qk-norm, RoPE dims
(t,h,w)=(16,56,56) θ=256, patch size [1,1,1], `in_channels` 65 =
32 noisy latent + 32 cond latent + 1 mask, out 32. Encoder stack is stock
HY1.5: Qwen2.5-VL-7B text backbone (3584-d), ByT5-small + Glyph-SDXL-v2
(1472-d), SigLIP so400m (1152-d, 729 tokens), `AutoencoderKLConv3D` VAE
(32ch latents, 16× spatial / 4× temporal, `scaling_factor` 1.03682, no shift).

## The causal rollout algorithm (from `HY15/hy15_inference.py`)

Defaults from `run_infer_causal.sh`: 4 steps, shift 5.0, guidance 1.0 (no
CFG), fps 16, 77 frames → 20 latent frames → **5 chunks × 4 latent frames**.

1. **Text KV prefill** (once per prompt): embed Qwen text (token-refiner,
   `timestep_txt=0`) + ByT5 + SigLIP tokens, each offset by a 3-way
   `cond_type_embedding`, reorder valid-tokens-first and drop invalid tokens
   entirely; run the *txt half* of all 54 blocks, caching each layer's
   post-qk-norm K/V.
2. **Chunk denoising**: per chunk, 4 Euler flow-matching steps
   (`FlowMatchDiscreteScheduler`, SD3 shift warp → sigmas
   [1.0, .9375, .8333, .625, 0], timesteps [1000, 937.5, 833.3, 625]; step is
   `x += v·Δσ`, v-prediction). The *img half* of each block computes
   `K = concat(k_txt, k_vision_cache, k_current)` and runs **plain SDPA, no
   mask** — causality is purely by cache construction. RoPE uses absolute
   temporal positions (freqs computed for the whole sequence so far, sliced to
   the chunk). Timesteps are per latent frame (broadcast per token).
3. **Clean-context re-encode**: after a chunk converges, one extra img-half
   pass over the *clean* chunk at timestep `stabilization_level−1 = 0` appends
   that chunk's K/V to the vision cache. The cache grows without bound (no
   sliding window over the 20 default frames).
4. **TI2V conditioning**: channel-concat `[noisy(32) | cond(32) | mask(1)]`
   where cond carries the VAE-encoded first frame at latent frame 0 only, mask
   is 1 on frame 0. ByT5 gets the **full caption** — the rollout script does
   *not* do the diffusers pipeline's glyph-text extraction.

## Mapping onto the existing causal-forcing machinery

The fit is close but not free. `CausalDMDDenoisingStage` already drives
block-wise KV-cached few-step denoising with a clean-latent cache refresh — the
same skeleton. Differences and how they landed:

| minWM concept | SGLang counterpart |
|---|---|
| per-layer `{k_txt, v_txt}` (prefilled once) | `CrossAttentionKVCache` (`store()` replaces buffers, so variable token counts are fine) |
| per-layer growing `{k_vision, v_vision}` | `CausalSelfAttentionKVCache`, `cache_size` = full sequence (no rolling) |
| noisy-chunk K/V concatenated but *not* persisted | `update_and_get_attention_kv` overwrites the same slots every step; the final clean pass overwrites them again — net cache state identical |
| Euler steps within a chunk | **new** `_denoise_causal_dmd_chunk` override (the base stage's x0-predict + re-noise is minWM's `"cm"` solver, *not* what the shipped scripts use) |
| `forward_txt` prefill | run lazily inside the DiT forward when `crossattn_cache[0].is_init` is False (keeps the stage's call contract untouched) |
| cond channel-concat | full-length `cond_latents` tensor passed via `image_kwargs`; the DiT slices `[start_frame : start_frame+T]` per chunk |

## Files

All under `python/sglang/multimodal_gen/` (uncommitted on `main` as of writing):

- `runtime/models/dits/causal_hunyuanvideo15.py` —
  `CausalHunyuanVideo15Transformer3DModel`. Module names match the checkpoint
  verbatim (`img_in.proj`, `txt_in.individual_token_refiner.blocks.N.*`,
  `byt5_in.fc{1..3}`, `vision_in.proj.{0..4}`, `double_blocks.N.*`,
  `final_layer.adaLN_modulation.1`) → `param_names_mapping = {}` (identity).
  Interleaved (GPT-J style) RoPE implemented inline to match
  `posemb_layers.py` exactly. Replicated only — no TP/SP yet.
- `runtime/pipelines_core/stages/model_specific_stages/causal_hunyuanvideo15.py`
  — `CausalHunyuanVideo15DenoisingStage` (Euler chunk loop, scheduler-driven
  timesteps, TI2V cond assembly, cache sized to the request's latent length)
  and `CausalHunyuanVideo15ImagePreprocessStage` (squash-resize, below).
- `runtime/pipelines/causal_hunyuanvideo15_pipeline.py` — stage wiring:
  InputValidation → ImagePreprocess → TextEncoding(Qwen+ByT5) →
  ImageEncoding(SigLIP) → ImageVAEEncoding → LatentPreparation →
  CausalDenoising → Decoding. `initialize_pipeline` installs
  `FlowMatchDiscreteScheduler(shift=flow_shift)`.
- `runtime/models/schedulers/scheduling_flow_match_discrete.py` — trimmed port
  (Euler + training sigma table).
- `runtime/models/vaes/hunyuanvideo15_vae.py` — `AutoencoderKLHunyuanVideo15`
  ported from diffusers 0.37 on the `ParallelTiledVAE` base; encoder+decoder;
  bit-exact vs diffusers; 218/218 keys.
- `runtime/models/encoders/siglip.py` — `SiglipVisionModel` following the
  `clip.py` pattern (no class token, 384 px floor-divides patch 14 → 729
  tokens; checkpoint `head.*` attention-pool keys skipped); bit-exact vs
  transformers.
- `runtime/models/encoders/qwen2_5vl_text.py` — `TextEncoder` wrapper around
  the existing `qwen2_5vl.py` text backbone, because HY1.5 ships a **bare**
  `Qwen2_5_VLTextModel` (top-level `embed_tokens.*`/`layers.*` keys, no vision
  tower / LM head) that nothing in the registry could resolve.
- `configs/models/dits/hunyuanvideo15.py`, `configs/models/vaes/hunyuanvideo15.py`,
  `configs/models/encoders/siglip.py`, `configs/pipeline_configs/hunyuanvideo15.py`,
  `configs/sample/hunyuanvideo15.py` — arch/pipeline/sampling configs. The DiT
  arch config's field names mirror the checkpoint `config.json`
  (`heads_num`, `mm_double_blocks_depth`, …) so `update_model_arch` overrides
  them directly; causal geometry `num_frames_per_block=4`,
  `sliding_window_num_frames=20` (grown per request as needed).
- `tools/assemble_minwm_hy15.py` — builds the runnable model dir.
- `registry.py` (+ config `__init__`s, `component_loader.py`, compatibility
  matrix docs row) — wiring.

## Checkpoint assembly

There is no single runnable repo: minWM publishes DiT-only folders, the
encoder stack lives in the community diffusers repo. The tool combines them
and casts the fp32 transformer (33.3 GB) to bf16 (15.5 GB):

```bash
hf download MIN-Lab/minWM --include "HY15/TI2V/dmd/*" --local-dir <dl>/minWM
hf download hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v \
    --exclude "transformer/*" --local-dir <dl>/HY15-base   # ~21 GB

python -m sglang.multimodal_gen.tools.assemble_minwm_hy15 \
    --minwm-transformer <dl>/minWM/HY15/TI2V/dmd \
    --hy15-base <dl>/HY15-base \
    --output-path /data/projects/vision-gen/models/minWM-HY15-TI2V-dmd-Diffusers
```

The `causal_cd` / `causal_ode` stage checkpoints are drop-in: same tool,
different `--minwm-transformer` subdir (all five stages share one config).
Raw downloads are kept in `/data/projects/vision-gen/models/_downloads/`.

Run:

```bash
sglang generate \
  --model-path /data/projects/vision-gen/models/minWM-HY15-TI2V-dmd-Diffusers \
  --prompt "..." --image-path input.png --save-output
```

## Verification

- **DiT parity, real weights** (`scratchpad/test_parity.py` pattern): loaded
  minWM's `HunyuanVideo_1_5_DiffusionTransformer` and ours side by side on one
  GPU; compared per-layer prefill txt-KV (layers 0/26/53) and four full
  480×832-shaped chunk forwards spanning two chunks (i.e. including attention
  over the persisted cache). **fp32: max abs diff 0.0 everywhere.** bf16:
  ~2–4 % max-rel, purely rounding accumulation through 54 blocks — confirmed
  by the fp32 result; don't chase it.
- **Component parity**: VAE encode/decode and SigLIP forward bit-exact vs the
  diffusers/transformers references on identical weights; all weight names
  load with zero missing/unexpected (transformer 1793, VAE 218, SigLIP 329).
- **Prompt template**: the flat-string Qwen chat template tokenizes
  id-identically to `apply_chat_template` with minWM's
  `li-dit-encode-video-json` messages, and the crop position after
  `<|im_start|>user\n` is exactly **108** (matches diffusers' constant).
- **End-to-end**: three of minWM's own example images → coherent videos whose
  first frame reproduces the input; two sequential requests (45f then 77f)
  through one engine confirmed cache reset/regrowth between requests.
- **No regressions**: `test/unit/realtime/` shows only the four pre-existing
  failures (lingbot cache-config ×2, realtime adapter, webui presets).

## Gotchas (the non-obvious ones)

1. **`_class_name` resolution ignores `_aliases`.**
   `ModelRegistry.resolve_model_cls` only knows classes registered via
   `EntryClass` under their own names; `_aliases` is a separate module-path
   lookup (hunyuan3d). The assembly tool therefore rewrites the transformer
   `config.json` `_class_name` from `ARHunyuanVideo_1_5_DiffusionTransformer`
   to `CausalHunyuanVideo15Transformer3DModel` (original kept as
   `_original_class_name`).
2. **`InputValidationStage`'s TI2V branch is Wan2.2-specific** — it reads
   `vae_config.arch_config.scale_factor_spatial` (a Wan name), does
   aspect-preserving crop sizing with a hardcoded max area, and converts the
   condition image to a tensor. Bypassed with
   `skip_input_image_preprocess=True` plus a model-specific stage that
   squash-resizes the PIL image to (width, height) bilinear, which is exactly
   minWM's `transforms.Resize((H, W))`.
3. **`ImageVAEEncodingStage` pads with zero *pixel* frames** up to
   `num_frames` before encoding. A causal-conv VAE folds those zeros into the
   first latent. Fixed via the config hook `preprocess_vae_encode` slicing to
   `[:, :, :1]` — encode only the single image frame, as minWM does.
4. **Attention backends don't thread dense masks** (`sdpa.py` hardcodes
   `attn_mask=None`). The token refiner needs a pairwise validity mask, so it
   calls `F.scaled_dot_product_attention` directly (two small layers). The
   main blocks never need masks — invalid condition tokens are dropped before
   prefill (batch 1), matching upstream's `txt[text_mask.bool()]`.
5. **ByT5 gets the full caption**, not extracted glyph text — the diffusers
   HY1.5 pipeline and minWM's rollout differ here; we follow minWM.
6. **Euler vs DMD renoise**: the shipped causal scripts use the `euler`
   solver; the base stage's `pred_noise_to_pred_video` + `add_noise` recipe is
   minWM's `cm` solver. Deliberately overridden.
7. Deliberate deviation: `encode_sample_mode` is `argmax` (posterior mode)
   while minWM `.sample()`s the first-frame latent.

## End-to-end accuracy audit (2026-07-22, after the "poor quality" question)

Seed-matched audit against upstream `run_inference_rollout` (script pattern:
`scratchpad/accuracy_check.py`; fox prompt/image, seed 1234, 77f):

- **Conditioning**: ByT5, SigLIP tokens and the first-frame VAE latent are
  exactly identical; Qwen embeds differ at bf16-rounding level only (max rel
  4.2e-3 ≈ 1 ulp, padded-vs-unpadded kernel tiling). Deliberate deviations:
  VAE posterior mode vs unseeded `.sample()` (6e-4 rel) and a resize-filter
  difference that is moot for 832×480 inputs (resize is a no-op).
- **Rollout**: full 5-chunk × 4-step rollout is **bit-exact in fp32**
  (upstream transformer+loop vs our DiT+scheduler, max diff 0.0), and the
  real `CausalHunyuanVideo15DenoisingStage` is bit-identical (bf16) to the
  upstream-shaped loop on our DiT. bf16 U-vs-S diverges in fine detail only
  (decoded PSNR 18.6 dB) from rounding over 80 forwards — same-quality video
  with the same drift/smudge artifacts on both sides.
- **Verdict**: the artifacts are the checkpoint's, not the port's. Note
  upstream's `run_infer_causal.sh` defaults to **`causal_cd`**, not `dmd` —
  an A/B against that stage is the natural next quality experiment.

Attention probe: the DiT now calls `begin_forward`/`record`/`end_forward`
(vision keys only — the joint attention also sees text KV, but the probe axes
are latent tokens). Token-bar dump at
`attn_token/CausalHunyuanVideo15Transformer3DModel-20260722-105912` confirms
strict block-causality: visible keys end exactly at the current chunk.

## Not done / follow-ups

- **TP/SP** for the DiT (replicated only, like the Rolling Forcing port).
- **Action2V line** (camera/action world model): adds per-block
  `img_attn_prope_proj` (ProPE) + `action_in` (81-class discrete actions);
  TI2V has none of these. The Wan21/Action2V checkpoints are 1.3B `.pt` files
  mirroring the Causal-Forcing code we already have.
- **Other TI2V stages** (`causal_cd`, `causal_ode`) — assembly-only; the
  registry already lists their suggested dir names.
- **CI**: no gpu_cases / suite / accuracy entries (blocked on hosting the
  assembled checkpoint, same situation as the Wan forcing models).
- Seed-matched full-pipeline A/B vs upstream `hy15_inference.py` (would need
  the original tencent-layout encoder checkpoints; component-level parity +
  visual checks stood in for it).
