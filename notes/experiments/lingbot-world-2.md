# LingBot-World v2（14B）：注意力结构测量与稀疏化含义

日期：2026-07-22 / 07-23
模型：LingBot-World v2 causal-fast（14B，接入见 [`../models/lingbot-world-v2.md`](../models/lingbot-world-v2.md)；块因果世界模型：3 帧块、9 帧固定
sink、18 帧滑窗）。稀疏注意力后端**尚未接入**本模型的注意力通路
（`--sparse-attention` 目前只对 `CausalWanSelfAttention` 家族生效）；
本笔记是为其设计 cache/稀疏策略的测量依据。

## 测量命令

一次性 CLI 请求只生成一个 chunk，块因果结构不会出现——**必须用 realtime
websocket 会话**驱动多 chunk 生成；探针在客户端断开时把整个会话落进一个
运行目录（`RealtimeCausalDiTState.on_dispose` 钩子）：

```bash
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_token_10s/lingbot \
SGLANG_DIFFUSION_ATTENTION_MAP_TOKEN_SCORES=true \
sglang serve --model-path <LingBot-World-v2-causal-fast>   # + websocket 客户端循环
# （init payload → 若干帧批次 → close；14 chunk 会话 126 s）

python -m sglang.multimodal_gen.tools.plot_token_attention_maps <run_dir>
python -m sglang.multimodal_gen.tools.plot_token_attention_bars <run_dir> \
  --layers 0,6,11,17,22,28,33,39
```

原始转储（2026-07 的 165 帧与 9 chunk 会话）未保留；下述结论来自那两次
运行的分析，按上述方式重新采集可复现（websocket 驱动脚本当时在会话
scratchpad，未入库——需要时重写一个 init → N 帧批次 → close 的循环即可）。

## 主要发现

- **cache 调度直接可见**：chunk 8 的帧面板里，帧 0–8（固定 sink）明亮
  （0.45–0.58×均匀，帧 0 独占 ~0.95），帧 9–17 从未入 cache（灰），帧
  18–26 向自身上升（自 chunk 2.36–2.75×）。帧 0 在 sink 内部又获得 ~2×
  邻帧的质量——sink 里再套一个涌现 sink。
- 稳态时间分布（10 s 扫描中位数）：自块 0.41 / 前一块 0.17 / sink 0.089。
- **sink 使用两极分化**：17% 的 (layer, head) 位点对 sink 质量 <2%，
  212/1600 个位点 >95% 的质量全在滑窗内完全无视 sink；层 22 内部 sink
  份额从 0.9%（h30）到 66.4%（h33）。逐头丢 sink 是零风险的显存/带宽
  收益。
- 注意力集中但**头间差异是主轴**：top-2048 token（7.3% 序列）的份额
  中位 46.9%、跨头范围 8%–100%，同层头间差中位 67 个百分点——头平均后的
  "弥散"是异质头相互抵消的假象。
- **register 部分绑定内容**：跨 prompt 单元 Jaccard 仅 0.23（Wan-1.3B
  家族为 ~0.7），主体上方有成簇 register——须在每 chunk 第一去噪步运行时
  收集（跨步不变性 r=0.92–0.98 使"第一步收集、其余步复用"成立），不能
  离线固定。

## 含义

逐头 sink 策略 + 帧级近窗 mask 的收益结构与 1.3B 家族相同，但 register
列必须运行时收集。把稀疏后端接到 LingBot 的注意力通路（及其 realtime
会话状态）是先决工程项。
