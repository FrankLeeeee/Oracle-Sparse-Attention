# LingBot-World v2（14B causal-fast）模型接入

模型：**LingBot-World v2 14B causal-fast**——块因果、少步 DMD 蒸馏的
交互式世界模型（I2V 条件），面向 realtime 逐 chunk 会话式生成。

- 权重：直接用现成的 diffusers 转档
  `robbyant/lingbot-world-v2-14b-causal-fast-diffusers`，无需任何
  checkpoint 预处理（v1 对应 `IPostYellow/lingbot-world-fast-diffusers` /
  `robbyant/lingbot-world-fast-diffusers`，同一套管线代码）。
- 管线配置（`configs/pipeline_configs/lingbot_world.py`）：
  `LingBotWorldV2CausalDMDConfig` 继承 I2V 基配置，4 步 DMD
  `[1000, 750, 500, 250]`、`flow_shift 5.0`；架构默认
  （`configs/models/dits/lingbot_world.py`）为 3 潜帧块，sink/窗口大小由
  checkpoint 的 `config.json` 经 `update_model_arch` 覆盖——v2
  causal-fast 在运行时表现为 9 帧固定 sink + 18 帧工作窗（探针实测，见
  [`../experiments/lingbot-world-2.md`](../experiments/lingbot-world-2.md)）。
- 代码：DiT `runtime/models/dits/lingbot_world.py`（其
  `_cross_attn_with_cache`——文本 K/V 每请求算一次、跨步复用——后来被
  移植进共享的 causal Wan 通路）、管线
  `runtime/pipelines/lingbot_world_causal_dmd_pipeline.py`、去噪阶段
  `runtime/pipelines_core/stages/model_specific_stages/lingbot_world/`。

## 运行方式：realtime 会话，而非一次性 CLI

一次性 `sglang generate` 请求只产出**一个 chunk**——多 chunk 的块因果
rollout 只在 realtime 会话里发生：

```bash
sglang serve --model-path robbyant/lingbot-world-v2-14b-causal-fast-diffusers
# websocket 客户端：init payload（条件图 + prompt）→ 若干帧批次请求 → close
# （165 帧 / 14 chunk 的会话约 126 s；每批 chunk 大小由
#  batch.realtime_chunk_size 控制，默认为 num_frames_per_block）
```

会话状态（KV cache、chunk 计数）保存在 `RealtimeCausalDiTState`，客户端
断开时经 `on_dispose` 钩子清理（注意力探针也挂在这里 flush）。

## 交互式生成的专用机制（配置项）

- **交互 KV 窗口**：`interactive_kv_window_enable` +
  `interactive_kv_{still,moving}_window` / `interactive_kv_still_chunks`
  ——静止/运动场景用不同的 KV 保留窗口。
- **惰性 VAE 编码**：`lazy_vae_encode_black_frames`（或环境变量
  `SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES`）——条件序列里补的黑帧
  不必全部过 VAE，只编码前若干帧、潜变量末帧重复填充到 chunk 对齐长度。
- **条件张量布局**：`[mask(temporal_ratio 通道) | latent(z_dim 通道)]`
  共 20 通道，时间维对齐到块大小；只有首潜帧是真实图像内容，其余 mask
  置 0（对齐上游 `lingbot_fast_server._prepare_latents_causal`）。

## 测试

`test/unit/realtime/test_lingbot_causal_denoising.py` 与 realtime 套件
覆盖去噪阶段与会话通路（注意：干净树上有 4 个既有失败——lingbot
cache-config ×2、realtime adapter、webui presets——先于本仓库的全部
稀疏注意力工作存在）。

## 注意力结构测量

cache 调度、sink 使用两极分化、register 绑定内容等量化结论见
[`../experiments/lingbot-world-2.md`](../experiments/lingbot-world-2.md)。
稀疏注意力后端尚未接入其注意力通路。
