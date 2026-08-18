# Rolling Forcing（Wan2.1-T2V-1.3B）基线接入

日期：2026-07-20 / 07-21
范围：SGLang Diffusion 原生支持 **Rolling Forcing**
（TencentARC/RollingForcing，arXiv 2509.25161）——Wan2.1-T2V-1.3B 上的
流式自回归 T2V：5 个 3 帧块组成滚动窗口联合去噪、逐帧错峰噪声等级
（越旧越干净）、attention-sink KV cache（`sink_size=3`、21 帧注意力上下文），
面向分钟级流式生成。

## 权重转换

上游发布 `checkpoints/rolling_forcing_dmd.pt`（DiT-only 训练态）。转换与
Self-Forcing 家族共用同一工具（详见 [`self-forcing.md`](self-forcing.md)）：

转换后直接上传到 HuggingFace（`frankleeeee` 名下），不在本地长期保存：

```bash
hf auth login   # 一次性，需要 write token

# 下载上游 checkpoint（TencentARC/RollingForcing 仓库，DiT-only 训练态）
hf download TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt \
  --local-dir /tmp/rolling-forcing-upstream

python -m sglang.multimodal_gen.tools.convert_forcing_to_diffusers \
  --preset rolling-forcing \
  --checkpoint /tmp/rolling-forcing-upstream/checkpoints/rolling_forcing_dmd.pt \
  --output-path /tmp/RollingForcing-Wan2.1-T2V-1.3B-Diffusers

hf upload frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers \
  /tmp/RollingForcing-Wan2.1-T2V-1.3B-Diffusers . --repo-type model
rm -rf /tmp/rolling-forcing-upstream /tmp/RollingForcing-Wan2.1-T2V-1.3B-Diffusers
```

仓库名必须保留 `RollingForcing-Wan2.1-T2V-1.3B-Diffusers` 这个 basename
（注册表按登记路径 `TencentARC/RollingForcing-Wan2.1-T2V-1.3B-Diffusers`
的最长子串匹配解析）。

## 生成命令

直接用 HF 仓库路径，权重在首次运行时自动下载：

```bash
sglang generate \
  --model-path frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 501 --seed 42 --save-output
```

像素帧数约定与并发端口注意事项同 [`self-forcing.md`](self-forcing.md)。

## 接入要点（新增的去噪机制）

与 Causal/Self-Forcing 不同，滚动窗口需要新的模型与阶段代码：

- `runtime/models/dits/rolling_forcing_wanvideo.py` ——
  `RollingForcingWanTransformer3DModel`，忠实移植上游流式注意力：
  只有窗口的**第一个块**写入 KV cache；首个块以**未加 RoPE** 的形式存为
  attention sink，读取时按相对位置现场重加 RoPE；单独的
  `updating_cache=True` 前向用干净 t=0 特征覆写完成块的槽位。每次前向的
  cache 下标计算收敛在纯函数 `compute_rolling_cache_layout()`（30 层共享）。
- `runtime/pipelines_core/stages/rolling_forcing_denoising.py` ——
  `RollingForcingDenoisingStage`：窗口调度（ramp-up / 满窗 / 收尾）、
  逐帧错峰时间步（最旧块最干净）、联合去噪、未完成块随机再加噪到下一
  时间步、干净 cache 更新。纯辅助函数：`build_rolling_window_bounds`、
  `build_staggered_timesteps`。
- 5 步时间表 `[1000, 800, 600, 400, 200]`。
- 共享代码上加的扩展钩子：`CausalWanTransformerBlock._self_attn_cls` /
  `CausalWanTransformer3DModel._block_cls`（子类换注意力/块类而不复制
  块代码）、`RollingForcingSelfAttentionKVCache` 子类。

## 验证

- 单元测试：`test/unit/realtime/test_rolling_forcing_denoising.py`——
  窗口边界、错峰时间步、cache 布局不变量（含淘汰与有界注意力上下文），
  全部通过。
- 端到端（单 H200，固定 seed）：81 帧约 15 s；**501 帧（约 31 s 视频）
  45 s**——覆盖 cache 淘汰与 sink 重加 RoPE，第 480 帧与第 30 帧保持
  一致，无漂移无偏色（三个 1.3B 模型中长视频保色最好）。

## 未做事项

- DiT 的 TP/SP（当前仅单卡复制式运行）。
- 与上游逐位对齐的数值 parity（仅视觉验证）。
- CI 条目（checkpoint 已于 2026-08-18 上传
  `frankleeeee/RollingForcing-Wan2.1-T2V-1.3B-Diffusers`（private），CI 条目本身仍未加）。
- 流式 serving 集成（逐 chunk 会话 API、流式 VAE 解码）——目前仅离线
  批式生成。
