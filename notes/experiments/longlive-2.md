# LongLive-2.0（5B）：注意力结构测量与稀疏化含义

日期：2026-07-23
模型：`Rabinovich/LongLive-2.0-5B-Diffusers`（NVIDIA，Wan2.2-TI2V-5B 上的
少步蒸馏 TI2V，接入见 [`../models/longlive-2.md`](../models/longlive-2.md)）。稀疏注意力后端尚未接入其通路；本笔记回答"它是否
chunk 因果、结构长什么样"。

## 测量命令

```bash
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_token_10s/longlive2 \
SGLANG_DIFFUSION_ATTENTION_MAP_TOKEN_SCORES=true \
sglang generate --model-path Rabinovich/LongLive-2.0-5B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 253 --seed 1234 --save-output

python -m sglang.multimodal_gen.tools.plot_chunk_attention_maps <run_dir>
python -m sglang.multimodal_gen.tools.plot_token_attention_maps <run_dir>
```

原始转储（2026-07-23，253 帧 = 64 潜帧 = 8 chunk，23 s 生成）未保留；
seed 固定，按上述命令重新采集即可复现。Wan2.2 的 16× VAE 使网格只有
15×26，探针开销极小。

## 主要发现

- **确证块因果**：8 潜帧块、8 帧固定 sink、32 帧局部窗
  （`configs/models/dits/longlive2.py`），chunk 矩阵严格下三角。
- chunk 矩阵（层+头+步均值）：自块 0.71–0.76（六模型中最高；94% 集中在
  三个行块内），前一块 0.10–0.13，再前 0.04–0.06——窗内近因衰减锐利。
- **sink 列在出窗后恒定 ~0.11**——固定但远比 LingBot 的 sink 轻
  （0.45–0.58）；窗+sink 之外为精确 0（真淘汰）。
- register 稀疏而散乱、跨 prompt Jaccard 仅 0.34——与 LingBot 同属
  "需运行时收集"一侧，与 Wan-1.3B 的静态栅格相反。

## 探针暴露的两个接入 bug（已修）

1. `LongLive2Transformer3DModel.__init__` 在 `super().__init__` 盖过
   `attn1.layer_index` 之后**重建 `self.blocks`**，所有层保持 -1，探针把
   `token_mass[chunk, -1]` 写进 0 层缓冲——重建后需重新盖章
   （`runtime/models/dits/longlive2.py`）。
2. `LongLive2CausalDenoisingStage._forward_one_shot_common` 覆写了
   `forward` 却不经过基类的 flush 路径，探针永不落盘——返回前补
   `_flush_attention_maps`（与 LingBot 会话 bug 同型）。

## 含义

自块质量极高 + 窗口小，意味着帧级稀疏空间不大（窗口本身已是强稀疏）；
若要做，逐头 sink 削减与窗内近因 mask 是仅有的两个杠杆，register 须
运行时收集。
