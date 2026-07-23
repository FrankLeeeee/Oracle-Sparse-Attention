# LongVie 2 Integration (in progress)

Date: 2026-07-21 / 2026-07-23
Scope: native SGLang Diffusion support for **LongVie 2** — multimodal-controllable
ultra-long I2V on a Wan2.1-I2V-14B-480P backbone.

- upstream: <https://github.com/Vchitect/LongVie> (repo `main` is LongVie 2)
- weights: `Vchitect/LongVie2` (`dit.safetensors`, `control.safetensors`)
- papers: LongVie arXiv 2508.03694, LongVie 2 arXiv 2512.13604

**Status: single-chunk control path AND clip-by-clip AR with 8-frame history
work end-to-end and demonstrably steer.** Conversion, loading, control fusion,
controllability, and the long-video clip loop (unified noise, history tokens,
first-frame re-noising) are all verified on GPU against a real driving video.
What remains is production-grade extractors and SP/USP. See the phase table,
"Clip-by-clip AR" and "Next step" below.

## What the release actually is

Upstream is **DiffSynth-Studio**, not Diffusers, and the release is an *overlay*
on stock Wan2.1-I2V-14B-480P rather than a full checkpoint. Two non-obvious
findings, both established from the tensors rather than the README:

- **`dit.safetensors` is self-attention only.** 400 tensors / 4.20B params, all
  under `blocks.*`, containing just `q`/`k`/`v`/`o` + `norm_q`/`norm_k` for all
  40 blocks at dim 5120. Patch embedding, cross-attention, FFN, condition
  embedders and head are unchanged from the base model. Without the base you get
  an unusable directory.
- **Every key maps through the existing rename dict.** DiffSynth uses
  original-Wan naming, which `tools/wan_repack.py::TRANSFORMER_KEYS_RENAME_DICT`
  already covers — 400/400 and 800/800 keys renamed with no collisions,
  including `cross_attn.k_img` → `attn2.add_k_proj` and the `norm3` → `norm2`
  swap.

### Control branch geometry (read off `control.safetensors`)

800 tensors / 2.63B params, half the model width:

| module | shape |
|---|---|
| `control_blocks_{dense,sparse}.0..11` | full Wan DiT blocks, dim 2560, 20 heads, FFN 6912 |
| `control_initial_combine_linear_{dense,sparse}` | 5120 → 2560 |
| `control_combine_linears.0..11` | 2560 → 5120 |
| `control_text_linear`, `control_t_mod` | 5120 → 2560 |

`WanModelDualControl` upstream is only a container — it has no `forward`. The
fusion lives in the *pipeline* (`wan_video_new_longvie.py::model_fn_wan_video`):

```python
dense  = in_proj_dense(patchify(dense_latents))
sparse = in_proj_sparse(patchify(sparse_latents))
control_context, control_t_mod = text_proj(context), time_proj(t_mod)
for i, block in enumerate(dit.blocks):
    x = block(x, context, t_mod, freqs_history)
    if i < 12:
        dense  = blocks_dense[i](dense,  control_context, control_t_mod, freqs)
        sparse = blocks_sparse[i](sparse, control_context, control_t_mod, freqs)
        x[:, -dense.size(1):] += combine_linears[i](dense + sparse)
```

Note the control streams use `freqs` (no history) while the main stream uses
`freqs_history`, and injection targets only the tail (non-history) tokens.

Control latents are **36-channel**, laid out like the Wan I2V input so they pass
through the same patch embedding: `concat[VAE(control video) 16ch,
first-frame mask 4ch, VAE(control first frame) 16ch]`. History latents are
likewise padded to 36 with a 20-channel `ones` block.

## Phase status

| Phase | Status |
|---|---|
| 0 — conversion + backbone runs | **verified** — coherent video, not noise |
| 1 — model loads with control branch | **verified** — byte-identical to baseline |
| 1 — control fusion | **verified** — executes, steers geometry, no quality cost |
| 2 — control encoding stage | **verified** on GPU |
| 2 — sampling params (control paths) | **done** — CLI flags → SamplingParams → Req |
| 2 — depth / track extractors | **proxy works** (DA-V2 + Lucas-Kanade); VideoDepthAnything + SpaTracker remain |
| 3 — clip-by-clip AR: unified noise, chunk loop + 8-frame history | **verified** on GPU (161 frames / 2 clips) |
| 3 — global control normalization | inherent: one long control video is rendered globally, then sliced per clip |

### Evidence

**Load / no-op fallback.** Three separate runs produce
`md5 c79b0e652165cd67e834152f829fe42e`:

1. Phase 0 — LongVie backbone (no control branch) through the stock Wan I2V pipeline
2. the same after the `wanvideo.py` `_run_transformer_blocks` refactor (regression check)
3. Phase 1 — the **full** model, all 2143 tensors including the control branch,
   with no control signal supplied

(3) is the load test: zero missing/unexpected parameters, control branch built,
and the no-control fallback proven to be a true no-op.

**Control fusion** (2026-07-22, 1×H200, 33 frames, 20 steps, seed 42, synthetic
control videos — a swept depth blob and six orbiting dots):

- the encoding stage produces `(1, 36, 9, 78, 78)` for both streams, matching the
  main stream's token count exactly, so the `x[:, -dense.size(1):]` slice — the
  risk flagged below — lines up. The DiT now asserts this rather than assuming it.
- control vs. no-control outputs differ (mean |Δ| 17/255, 42% of pixels by >10)
  and **both are coherent video**, so the branch steers rather than corrupts.
- the injected residual is 4–27% of `‖x‖`, growing monotonically with depth
  (0.053 at layer 0 → 0.274 at layer 11) — the profile of a trained side network,
  neither vanishing nor exploding.
- with *black* control videos the same residual is systematically smaller
  (0.16 at layer 11 vs 0.27), proving the fusion depends on control *content*,
  not merely on control being present.

`test/unit/test_longvie_control_wiring.py` covers the CLI → `Req` path on CPU.

**Quality + controllability** (2026-07-22, 81 frames, 832×480, 50 steps, seed
1234). Driving video: an existing Wan fox clip; depth from Depth-Anything-V2-Small,
tracks from Lucas-Kanade, conditioned on the driving video's first frame. Depth
re-extracted from each output and compared against the control that was fed in:

| run | depth L1 vs control | subject IoU |
|---|---|---|
| with control | **0.0315** | **0.904** |
| no control | 0.1055 | 0.667 |
| with control, wrong depth polarity | 0.1273 | 0.633 |

3.3× lower depth error and IoU 0.90 vs 0.67: the control branch genuinely steers
geometry, not just perturbs it. Visually the controlled run holds the driving
video's framing (fox scale constant, treeline in shot) for all 81 frames, while
the uncontrolled run zooms in and drifts. Both are sharp and artifact-free, so
the control branch does not cost quality.

Artifacts: `/data/projects/vision-gen/outputs/longvie2_verification/`
(videos, `comparison.png`, plus the extraction and measurement scripts).

## Clip-by-clip AR with 8-frame history (2026-07-23)

Upstream (`inference.py` + `wan_video_new_longvie.py`) generates long videos as
consecutive 81-frame I2V clips: `image = video[-1]`, `history = video[-8:]`,
and the same seed-derived noise reused every clip. Our mapping:

- **User surface**: nothing new — request `--num-frames > 81` (config
  `longvie_clip_num_frames`) with both control videos and the loop engages;
  the two control mp4s span the full request and are sliced per clip
  (last-frame padded on the tail clip). Single-clip requests are untouched.
- `LongVie2ControlEncodingStage` detects the multi-clip case, clamps
  `batch.num_frames` to the clip size (so every downstream stage works at clip
  shape), and stashes total frames + raw control pixels in `batch.extra`.
- `LongVie2ClipLoopStage` (registered instead of denoise+decode; owns
  unregistered `DenoisingStage`/`DecodingStage` instances and re-runs them per
  clip) reproduces upstream exactly:
  - previous clip's last frame → `condition_image`, re-run through
    `ImageEncodingStage` (CLIP) + `ImageVAEEncodingStage` (36-ch `y`);
  - previous clip's last 8 pixel frames → VAE → 16-ch history latents,
    padded to 36-ch as `[ones(20) | latents]` (**ones first** — the reverse of
    the control-latent layout) → `Req.longvie_history_latents`;
  - initial noise reused every clip (upstream's returned
    `inputs_shared["noise"]` is a fresh identical seed-derived tensor each
    call, so pristine-noise-per-clip is the faithful reading of its in-place
    mutation), with latent frame 0 re-noised:
    `(1-σ)·history[:, :, -1:] + σ·noise[:, :, :1]`, σ = 0.925926;
  - scheduler rewound by clearing `batch.timesteps` and re-running
    `TimestepPreparationStage` (UniPC `set_timesteps` resets its state).
- In the DiT, history latents are patchified and **prepended** to the main
  stream; RoPE is recomputed over `f_history + f_clip` frames from position 0
  (history at [0, f_h), clip shifted — upstream's `freqs_history`) while the
  control streams keep unshifted freqs; control injection and the final output
  slice target only the tail (`x[:, -control_len:]`), and the history tokens
  are dropped before the head.
- **Upstream's chunked VAE encoder consumes only the first
  `1 + 4*((n-1)//4)` history frames** (5 of the 8; `wan_video_vae.py::encode`
  drops the tail chunk-remainder) — we encode exactly that prefix, which is
  computation-identical. So "8-frame history" is really 2 latent frames from
  pixel frames 0–4 of the 8-frame window.
- Upstream's `first_frame_latents` re-pin at every scheduler step is gated on
  `fuse_vae_embedding_in_latents` (Wan2.2-TI2V-5B only) — correctly absent
  here.

**Verification** (161 frames = 2 clips, 832×480, seed 1234, real driving video
extended boomerang-style to 161 frames, `outputs/longvie2_ar_verification/`):
depth re-extracted from the output vs the control fed in —

| run | clip | depth L1 | subject IoU |
|---|---|---|---|
| 10 steps | frames 0–80 | 0.0363 | 0.895 |
| 10 steps | frames 81–160 | 0.0372 | 0.888 |
| 50 steps | frames 0–80 | 0.0339 | 0.903 |
| 50 steps | frames 81–160 | 0.0389 | 0.878 |

Clip 1 at 50 steps reproduces the single-clip verified numbers (0.0315 / 0.904;
no-control baseline 0.1055 / 0.667), and clip 2 follows its control almost as
well — no AR degradation. The 80→81 boundary frame-diff is *below* the median
adjacent-frame diff (50 steps: 3.95 vs 8.70) — smoother than ordinary motion,
i.e. no cut, and the contact sheet (`ar50_sheet.png`) shows the fox tracking
the boomerang reversal through the boundary with no drift or color shift.
Clip 2 also runs ~3% slower per step (7.08 vs 6.86 s at 10 steps), confirming
the history tokens really extend the sequence.
`test/unit/test_longvie_clip_loop.py` pins the CPU-verifiable invariants
(slicing/padding, VAE history prefix, ones-first layout, control stage
clamping).

Note the residency-manager wrinkle: the loop's inner stages are not registered
with the executor, so `LongVie2ClipLoopStage` reports the union of their
component uses and hands its manager down before each forward
(`_propagate_residency_manager`); `DenoisingStage` needed a `manager is None`
guard for this wrapped mode.

### The depth-rendering convention (cost a full 6-minute run)

`utils/get_depth.py` saves a raw `.npy`; the mp4 the pipeline actually consumes
is rendered by `utils/depth_npy2mp4.py`, and **both** of its details are easy to
get backwards:

```python
p95, p5 = np.percentile(depth, [95, 5])          # GLOBAL over the clip, not per frame
depth = (p95 - np.clip(depth, p5, p95)) / (p95 - p5)   # INVERTED: near -> BLACK
```

A per-frame-normalized, near-white depth map is out of distribution: the control
branch still fires with a healthy residual, but it steers *worse than no control
at all* (row 3 of the table). This is the single most likely reason a future
extractor "runs but does nothing" — check polarity and global scale first. Note
that `get_track.py` re-inverts the depth video when feeding SpaTracker, which is
a good way to talk yourself into the wrong polarity.

## The class-attribute trap (cost an hour)

`WanTransformer3DModel` binds

```python
param_names_mapping = WanVideoConfig().param_names_mapping
```

as a **class attribute at class-definition time**. A subclass silently inherits
the *base* mapping, so any added submodule's rewrite rules never reach the
weight loader — it failed on `control.blocks_dense.3.attn2.to_out.weight`. Any
Wan DiT subclass must re-bind `param_names_mapping`,
`reverse_param_names_mapping` and `lora_param_names_mapping`, exactly as
`CausalLingBotWorldTransformer3DModel` does.

Related: sglang rewrites diffusers names to its own module names
(`blocks.N.attn1.to_q` → `blocks.N.to_q`) with regexes anchored at `^blocks\.`,
which silently skip `control.*`. `configs/models/dits/longvie.py` generates the
control rules by *re-anchoring* the Wan ones rather than hand-copying ~20
regexes, shifting capture-group indices to account for the dense/sparse group.

## Implementation

```
new:
  tools/convert_longvie_to_diffusers.py
  configs/models/dits/longvie.py                     (arch config + re-anchored mappings)
  runtime/models/dits/longvie.py                     (control branch + transformer)
  runtime/pipelines/longvie_pipeline.py
  runtime/pipelines_core/stages/model_specific_stages/longvie.py
  test/unit/test_longvie_control_wiring.py
modified:
  runtime/models/dits/wanvideo.py                    (_run_transformer_blocks hook)
  configs/pipeline_configs/wan.py                    (LongVie2Config)
  configs/sample/wan.py                              (LongVie2SamplingParams)
  configs/sample/sampling_params.py                  (--longvie-{dense,sparse}-video)
  configs/pipeline_configs/__init__.py, registry.py  (export + registration)
  runtime/pipelines_core/schedule_batch.py           (4 Req fields)
```

The `wanvideo.py` change extracts the block loop into an overridable
`_run_transformer_blocks()` so variants that interleave a side network don't
duplicate a ~150-line `forward`. It is a pure refactor — regression-checked
byte-identical.

### Conversion

```bash
python -m sglang.multimodal_gen.tools.convert_longvie_to_diffusers \
    --output /data/projects/vision-gen/models/LongVie2-Diffusers
```

No paths needed: `--longvie` defaults to `Vchitect/LongVie2` (downloading only
the two safetensors) and `--base-model` to `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`;
either accepts a local dir. Output is 82 GB, 2143 tensors (1343 base + 800
control). `verify_complete()` runs at the end and asserts on the *artifact*:
every component `model_index.json` declares exists, the expected control-block
count is present, and the base-only tensors (`patch_embedding.weight`,
`proj_out.weight`, `scale_shift_table`) survived the merge — the specific
failure the overlay structure invites, since a half-converted model still loads
and only the outputs are wrong.

`models/LongVie2-Phase0-BaseOnly/` is a control-free variant (components
symlinked) kept as the regression baseline.

## Running it

```bash
sglang generate --model-path /data/projects/vision-gen/models/LongVie2-Diffusers \
    --prompt "A red fox trotting across a snowy field, camera follows" \
    --image-path cond.png --num-frames 81 --seed 1234 \
    --longvie-dense-video depth.mp4 --longvie-sparse-video track.mp4
```

That is the exact shape of the verified run above. Both control flags must be
given together — one alone raises. With neither, the stage logs and falls
through to plain Wan I2V. The control videos must be renderable by
`utils/vision.py::load_video` and are resampled to the request's frame count and
to whatever resolution `InputValidationStage` settles on.

## Three traps this cost

- **`Req` fields shadow the sampling params.** `Req.__getattr__` delegates to
  `sampling_params` *only for names `Req` does not declare*. Since `Req` declares
  `longvie_dense_video`, `Req.__init__` sets it to `None` and the delegation never
  fires — so `LongVie2SamplingParams.apply_request_extra` is load-bearing, and
  the wiring test exists to keep it that way. (`realtime_chunk_size` is the same
  shape.)
- **`vae.encode` returns a `DiagonalGaussianDistribution`, not a tensor.** The
  first draft of the encoding stage treated it as one. Control latents must go
  through the same `mode()` / `postprocess_vae_encode` / mean-std scale-shift
  path as `ImageVAEEncodingStage`, or they land in a different space than the
  video tokens they steer.
- **Stage order matters.** `InputValidationStage` resolves the real height/width
  from the condition image (for Wan I2V from `max_area` + aspect ratio — a square
  input becomes 624×624 regardless of `--height 480 --width 832`), so the control
  stage must run *after* it or it encodes at the wrong resolution. Hence
  `LongVie2Pipeline` adds validation itself and calls
  `add_standard_ti2v_stages(include_input_validation=False)`.

## Next step

Promote the proxy extractors to real ones. The verification above used
Depth-Anything-V2-Small per frame + Lucas-Kanade dots
(`outputs/longvie2_verification/extract_control.py`), which is already good
enough to steer generation. Upstream uses **VideoDepthAnything** (temporally
consistent, so no per-frame scale flicker) and **SpaTracker**. Swapping those in
is the remaining gap, and the rendering convention is now pinned down.

Then:

1. USP/SP (currently an explicit `NotImplementedError` — the control streams
   are unsharded and history tokens make sharding trickier still; upstream's
   USP branch gathers/rechunks around the tail slice)
2. `gpu_cases` entry; a longer (3+ clip) run to watch for slow drift
3. control-latent caching across clips of the *same* stream is already free
   (encoded per clip from stashed pixels), but the two VAE passes per clip
   (~5 s) could overlap with denoising if it ever matters

## Limitations / risks

- The clip loop is verified at **2 clips** (10 and 50 steps); longer horizons
  (drift over many clips) are unmeasured.
- Clip boundaries duplicate one visual instant (clip N+1's first frame ≈ clip
  N's last frame, the I2V condition) — upstream has the same property since it
  concatenates whole clips. Control slicing is non-overlapping
  (`start = clip_index * 81`), so the boundary control frame is off by one
  from the condition frame, again matching upstream's data convention.
- The converter holds the full merged state dict (~33 GB) in RAM and writes a
  single unsharded safetensors. Fine on this box (2 TB RAM); it will OOM on a
  smaller one. Shard-and-stream if portability matters.
- Controllability is measured **self-consistently** — depth re-extracted from the
  output with the same model and convention as the control — so it proves the
  control steers, not that it matches upstream numerically. There is still no
  side-by-side parity run against the reference DiffSynth implementation.
- The controllability numbers use a single driving clip and a single seed. The
  effect is large (3.3×), but it is n=1.
- Extractor dependency footprint for the *real* extractors (SpaTracker
  especially) has not been assessed; only the DA-V2 + Lucas-Kanade proxy was run.
- The control videos are re-encoded per request (~14 s for two 33-frame clips at
  624×624, dominated by two VAE passes each). Fine for one-shot generation; a
  chunked long-video loop will want them cached.
