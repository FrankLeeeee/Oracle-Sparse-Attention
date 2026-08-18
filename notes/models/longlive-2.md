# LongLive-2.0（5B）模型接入

模型：**LongLive-2.0-5B**（NVIDIA，上游 <https://github.com/NVlabs/LongLive>）
——Wan2.2-TI2V-5B 骨干上的少步蒸馏 TI2V，块因果自回归生成，24 fps。

- 权重：官方发布为 `Efficient-Large-Model/LongLive-2.0-5B`（无 diffusers
  布局）；推理直接用现成的 diffusers 转档
  `Rabinovich/LongLive-2.0-5B-Diffusers`，无需任何 checkpoint 预处理，
  注册表对两个仓库名都能解析。
- 因果几何（`configs/models/dits/longlive2.py`）：**8 潜帧块**、
  8 帧固定 sink、32 帧局部注意力窗（`local_attn_size: 32`）。
- 管线配置（`configs/pipeline_configs/longlive2.py`，继承
  `Wan2_2_TI2V_5B_Config`）：TI2V 任务、4 步 DMD `[1000, 750, 500, 250]`、
  `flow_shift 5.0`、bf16 VAE；`adjust_num_frames` 会把请求帧数取整到
  潜帧数能被 8 整除。
- 代码：DiT `runtime/models/dits/longlive2.py`、管线
  `runtime/pipelines/longlive2_pipeline.py`、去噪阶段
  `runtime/pipelines_core/stages/model_specific_stages/longlive2.py`。

## 生成命令

直接用 HF 仓库路径，权重在首次运行时自动下载：

```bash
sglang generate --model-path Rabinovich/LongLive-2.0-5B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 253 --seed 1234 --save-output
```

253 像素帧 = 64 潜帧 = 8 chunk（10.5 s @ 24 fps），单 H200 约 23 s。
Wan2.2 的 16× 空间 VAE 使 480p 网格只有 15×26，逐 token 分析开销很小。

## 接入期间修的两个 bug（注意力探针暴露）

1. `LongLive2Transformer3DModel.__init__` 在 `super().__init__` 盖过
   `attn1.layer_index` 之后重建 `self.blocks`，所有层的 `layer_index`
   保持 -1——重建后需重新盖章。
2. `LongLive2CausalDenoisingStage._forward_one_shot_common` 覆写了
   `forward` 却绕过基类的探针 flush 路径——返回前补
   `_flush_attention_maps`。

## 注意力结构测量

块因果结构（严格下三角 chunk 矩阵、恒定 ~0.11 的轻量 sink 列、锐利的
窗内近因衰减）的量化见
[`../experiments/longlive-2.md`](../experiments/longlive-2.md)。
稀疏注意力后端（`--sparse-attention`）尚未接入其注意力通路。
