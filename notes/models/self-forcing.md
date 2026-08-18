# Self-Forcing（Wan2.1-T2V-1.3B）基线接入

日期：2026-07-23（Self-Forcing 转换）/ 2026-07-20（所复用的 Causal-DMD 通路）
范围：SGLang Diffusion 原生支持 **Self-Forcing**（guandeh17/Self-Forcing，
gdhe17）——Wan2.1-T2V-1.3B 上的块因果自回归 T2V：3 潜帧块、4 步 DMD、
全上下文注意力（`sliding_window_num_frames: None`，无固定 sink；与上游
`self_forcing_dmd.yaml` 的 `local_attn_size=-1` 默认对齐——上游固定的
32760-token cache 恰为整段 21 潜帧视频，并无滚动淘汰。2026-08-18 起转换
preset 与 HF 上的 config 均改为 None，runtime 按整段视频长度分配 KV cache、
从不淘汰；此前误写为 21 帧滚动滑窗，仅在超过 21 潜帧时行为有差异。上游的
`local_attn_size: 21` 属于 local21/720p 另外的 checkpoint）。它是全部
稀疏注意力实验的主试验台（见 [`../experiments/self-forcing.md`](../experiments/self-forcing.md)）。

## 权重转换

上游发布的是 DiT-only 的 `.pt` 训练态（`generator`/`generator_ema` 键、
原始 Wan 命名）。独立入口 `tools/convert_self_forcing_to_diffusers.py`
（2026-08-18 起从共用工具拆出，机制复用
`convert_forcing_to_diffusers.py`）将其组装为自包含的 diffusers 布局：
取 `generator_ema`、重命名 key、写入因果几何配置
（`num_frames_per_block`、`sliding_window_num_frames` 等），并从
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 复制 scheduler/text_encoder/tokenizer/vae：

转换后直接上传到 HuggingFace（`frankleeeee` 名下），不在本地长期保存：

```bash
# 一次性登录（需要 write 权限的 token）
hf auth login

# 下载上游 checkpoint（gdhe17/Self-Forcing 仓库，DiT-only 训练态）
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --local-dir /tmp/self-forcing-upstream

# 转换到临时目录 → 上传 → 清理
python -m sglang.multimodal_gen.tools.convert_self_forcing_to_diffusers \
  --checkpoint /tmp/self-forcing-upstream/checkpoints/self_forcing_dmd.pt \
  --output-path /tmp/SelfForcing-Wan2.1-T2V-1.3B-Diffusers

hf upload frankleeeee/SelfForcing-Wan2.1-T2V-1.3B-Diffusers \
  /tmp/SelfForcing-Wan2.1-T2V-1.3B-Diffusers . --repo-type model
rm -rf /tmp/self-forcing-upstream /tmp/SelfForcing-Wan2.1-T2V-1.3B-Diffusers
```

**仓库名必须保留 `SelfForcing-Wan2.1-T2V-1.3B-Diffusers` 这个 basename**：
注册表按注册路径的最长子串匹配解析模型（登记的是
`gdhe17/SelfForcing-Wan2.1-T2V-1.3B-Diffusers`），改名会落到基础 Wan
配置上。

同家族其它模型用共用工具 `convert_forcing_to_diffusers.py` 的
preset：`causal-forcing-chunkwise` /
`causal-forcing-framewise`（thu-ml/Causal-Forcing，推理路径与 Self-Forcing
完全相同，只是训练配方不同）、`light-forcing` / `light-forcing-long`
（chengtao-lv/LightForcing，训练时即带稀疏注意力）、`rolling-forcing`
（见 [`rolling-forcing.md`](rolling-forcing.md)）——上传方式相同，保留
各自的 basename 即可。

## 生成命令

直接用 HF 仓库路径，权重在首次运行时自动下载：

```bash
sglang generate \
  --model-path frankleeeee/SelfForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 165 --seed 42 --save-output
```

注意：像素帧数应使潜帧数能被 3 整除（`num_frames = 12k + 9`，如
81/165/321），否则配置会自动取整。并发运行多个 `sglang generate` 需要
区分 `--master-port` / `--scheduler-port` / `--port`。

## 接入要点

- 复用已有的 `WanCausalDMDPipeline` + `CausalDMDDenoisingStage`（块因果
  4 步 DMD + 干净潜变量 KV 刷新），只换调度器
  （`SelfForcingFlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)`）；
  块大小从转换出的 `transformer/config.json` 读取。
- 接入时顺带修复了该通路上的三个 bug（当时无注册模型覆盖它）：
  CPU offload 下 DiT 未上 GPU、`crossattn_cache` TypeError、
  inference-tensor 原地写崩溃。
- 注册表按最长子串匹配解析本地目录名——本地目录若叫
  `X-Wan2.1-T2V-1.3B-Diffusers`，会命中基础 Wan 配置，除非注册了更长的
  名字。

## 验证

- 端到端（单 H200，固定 seed）：81/165 帧输出为连贯、符合 prompt 的视频
  （165 帧去噪约 8.8 s，整程约 16 s）；同家族 Causal Forcing chunkwise
  81 帧约 23–29 s、framewise 约 22 s。
- 已知模型特性（非移植问题）：超出训练时长（5 s / 7 chunk）后长视频缓慢
  漂色，20 s 处出现粉色偏色——量化见
  [`../experiments/self-forcing.md`](../experiments/self-forcing.md) §3。
- 同噪声数值 parity（2026-08-18）：把同一组噪声（初始 + 21 次 re-noise）
  注入上游与 sglang，以"上游 FA3 vs 上游 SDPA"为浮点基准对照。block 0
  latent 相对误差 sglang 0.068 vs 基准 0.076；末块 0.629 vs 0.595；视频
  PSNR 18.1 dB vs 基准 19.5 dB（无关样本基线 10.8 dB）；文本编码器输出与
  上游逐位一致。偏差逐块单调增长无突变——即偏差完全在 kernel 级浮点噪声
  被自回归 cache 放大的量级内，实现逻辑与上游一致。已封装为一条命令：
  `python -m sglang.multimodal_gen.tools.verify_self_forcing_parity
  --upstream-repo <Self-Forcing checkout> --model-path <转换后模型>`
  （runtime 侧噪声注入走 `SGLANG_DIFFUSION_TEST_PARITY_DIR` 测试钩子）。

## 未做事项

- Causal Forcing++ 的 1/2 步模型（`denoising_step_list_first_chunk`）。
- Rolling Forcing / Causal Forcing 的同噪声 parity（Self-Forcing 已做，
  方法可复用）。
- CI 条目（等转换后的 checkpoint 上传 HF）。
