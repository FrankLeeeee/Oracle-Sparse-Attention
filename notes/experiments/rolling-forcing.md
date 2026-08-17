# Rolling Forcing：注意力结构测量与稀疏化含义

日期：2026-07-21 / 07-23
模型：`/data/projects/vision-gen/models/RollingForcing-Wan2.1-T2V-1.3B-Diffusers`
（接入见 [`../models/rolling-forcing.md`](../models/rolling-forcing.md)）。
与 Self-Forcing 同一注意力通路，OSA 等方法可直接开启；但其滚动窗口结构
使 mask 设计有一条本模型特有的约束（见下）。

## 测量命令

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_maps \
sglang generate \
  --model-path /data/projects/vision-gen/models/RollingForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 81 --seed 42 --save-output

python -m sglang.multimodal_gen.tools.plot_chunk_attention_maps <run_dir>
```

原始转储（2026-07 的 `attn_maps` / `attn_token_10s/rf`）未保留；seed
固定，按上述命令重新采集即可复现。把 `--num-frames` 提到 201（17 chunk）
可覆盖 cache 淘汰：chunk 16 对被淘汰的 chunk 1–10 记录精确 0，对 sink
（chunk 0）保持 0.10，5 个在窗 chunk 上呈上升坡至自身 0.48——
`summary.png` 的"亮列 0 + 黑色淘汰区 + 5 chunk 对角带"是该机制最清晰的
单图。

## 主要发现

- **因果性只在窗口间成立**：窗内 5 块联合去噪、相互可见，早期 chunk 会把
  质量放在"更晚生成"的 chunk 上（chunk 0 对 chunk 1 有 0.20）。因此
  **mask 必须是窗口相对的**——它的质量不集中在最新帧上，"最近 N 帧"式
  选择在本模型上错得最狠（StreamingLLM 30% 预算只回收 0.31，CF 上是
  0.67）。
- 固定 `sink_size=3` 的 sink 列对每个 chunk 稳定保持 ~0.14，比相邻旧
  chunk 高一个数量级——与 CF 的涌现 sink 不同，这是设计出来的常驻锚点。
- 等预算下静态逐 (layer, head) 帧 mask 依旧接近 oracle（20% 预算 0.566
  对 0.591），跨步不变性 0.923。
- **强局部头集合是 CF 的严格子集**（78/360 对 137/360 个 ≥95% 局部位点）
  ——在 RF 上标定的 mask 可安全用于 CF，反之不行。
- register 单元格栅格跨 prompt Jaccard **0.72**（三个 1.3B 模型中最高），
  离线标定 register 列成立。
- 三个 1.3B 模型中 RF 的长视频保色最好（10 s 处 SF 漂灰、CF 过饱和）。

## 含义

OSA 现行的"参考 chunk 校准"假设 query 是最新 chunk——对 RF 的联合去噪
窗口，校准与策略都应改为窗口相对坐标后再启用；帧级静态 mask 与逐头
sink/register 处理照搬 CF 的结论即可。
