# QK 注意力矩阵的采集与可视化

日期：2026-08-16
范围：从块因果（block-causal）视频 DiT 的真实前向中导出 **query × key 注意力概率矩阵**（每层、每头），并绘制成图。这是本项目所有稀疏注意力分析的底层数据来源。

## 为什么需要运行时探针

模型使用融合注意力核（FA / SDPA），概率矩阵从不落地；且 key 轴是 KV cache 的滚动视图，"第 i 行 key" 对应视频哪一帧只有模型内部知道。因此探针实现在运行时内（`runtime/utils/attention_map_probe.py`），由环境变量开关，关闭时每次注意力调用只多一次 `None` 检查。

## 采集命令

`SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS` 指定要转储的 chunk 编号；对每个指定 chunk，探针在其**第一个去噪步**记录真实 softmax 概率（query 按 stride 8 采样、key 列按 stride 16 采样，softmax 在完整 key 轴上进行，因此每个保留列都是精确概率）：

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_qk_stationarity \
SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS=3,5,7,9,11,13 \
sglang generate \
  --model-path /data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 165 --seed 42
```

输出目录 `<DIR>/<ModelTag>-<时间戳>/` 中，每个指定 chunk 一个
`qk_chunk_<c>.npz`：`scores [layers, heads, queries, keys]`（float16）加上
`query_positions` / `key_positions`（全局 token 坐标）与 `layer_ids`。
可选 `SGLANG_DIFFUSION_ATTENTION_MAP_QK_KEY_STRIDE`（默认 16）控制 key 采样密度。

已有数据：`/data/projects/vision-gen/attn_qk_stationarity/CausalWanTransformer3DModel-20260816-114341/`
（上述命令的产物，chunk 3–13，共 4.4 GB）。

## 绘图命令

```bash
python -m sglang.multimodal_gen.tools.plot_qk_attention_maps <run_dir> \
    [--chunks 7,13] [--layers 0,10,20,29] [--heads 0-11] [--out-dir ...]
# -> <run_dir>/qk_plots/chunk_XXX_layer_YY_head_ZZ.png
```

每张图是一个 (chunk, layer, head) 的 query × key 矩阵（对数色标）：
key 轴上画出 chunk 边界细线，query 自身所在 chunk 用青色括出。sink 列、
近因带（recency band）、对角线、register token 都能直接读出。

## 从图上能读到什么（Self-Forcing 一例）

- 多数头呈**逐帧对角**结构：每个 query 帧对齐到自己附近的 key 帧，且该
  结构以 dt = query 帧 − key 帧 表示时跨 chunk 稳定 —— 这是 OSA dt 粒度
  的实验依据（见 `../experiments/self-forcing.md`）。
- 少数头呈整 chunk 块状（chunk 对齐）、周期性斜线或竖直条纹。
- 窗口最老几帧常有一条竖直亮带（滑窗内的"边缘 sink"）。

## 成本与限制

- 仅记录被指定 chunk 的第一个去噪步、batch 0、world rank 0（TP 下只覆盖
  rank 0 的头）。
- 480p 稳态一个 chunk 的转储约 820 MB（30 层 × 12 头 × 585 query ×
  2048 key，fp16），刻意**不压缩**存储（zlib 对这种数据几乎无效却极慢）。
- 探针会拖慢生成（重算 softmax），只用于分析运行，不要在计时运行中开启。
