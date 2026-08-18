# Self-Forcing：OSA 的实现与全部对比实验

日期：2026-08-02（首版 OSA）/ 08-13（720p 后端对比）/ 08-16（dt 粒度 + 模式库）
模型：`/data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers`
（接入见 [`../models/self-forcing.md`](../models/self-forcing.md)）。
Self-Forcing 是全部稀疏注意力实验的主要试验台。

## 1. OSA（Oracle Sparse Attention）设计

代码：`runtime/layers/attention/sparse/`，开关 `--sparse-attention osa`
（同一软件包还实现了 X-Attention、SVG1/2、Radial、FAST-AR、LightForcing
六个对照方法，共用同一个 range-block-sparse Triton kernel）。

设计的经验前提：一个头的时间足迹在**相对坐标下是平稳的**——在参考 chunk
上观测一次即可冻结复用（"oracle" 的含义）。三代粒度：

- `granularity=chunk`（首版）：每头策略 = `(keep_sink, num_recent)`，按整
  chunk 保留；
- `granularity=frame`：同样的策略改以潜帧为单位，去掉 3 帧量化；
- `granularity=dt`（2026-08-16，现推荐）：坐标细化到 **dt = query 帧 −
  key 帧**。校准时保留 query 帧轴（`measure_query_frame_mass`），每头在
  一个**模式库**上做模型选择（`choose_pattern_head_policies`）：
  对角带 band、chunk 对齐 block、周期斜线 dt_comb、绝对周期竖纹 v_comb，
  外加可组合的"窗口最老 m 帧"edge 尾巴；取满足保留率的最便宜实例并冻结。
  每个 query block 编译为至多两段连续 key 区间，走共享 kernel，计划按
  (chunk, layer) 记忆化。`refresh_at_full_window`（默认开）在滑窗首次发生
  淘汰的 chunk 上重新观测一次后永久冻结——修复部分窗口校准的保留率衰减。

正确性：`test/unit/realtime/test_sparse_attention.py`（45 个测试）。
三类断言分开：kernel 对每个方法的真实计划与 masked SDPA 逐位对齐；每个
方法的选择逻辑符合其声称的模式（含模式库的植入恢复测试）；任何方法都不得
跨去噪步缓存当前 chunk 的 key。

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_sparse_attention.py -q
```

## 2. dt 平稳性验证（OSA dt 粒度的实验依据）

观测：同一 (layer, head) 的帧级注意力模式与当前生成到第几个 chunk 无关
——对角头对每一**帧**都对角，而非对每个 chunk。验证：

```bash
# 采集（见 ../exploration/qk-attention-maps.md）
CUDA_VISIBLE_DEVICES=0 \
SGLANG_DIFFUSION_ATTENTION_MAP_DIR=/data/projects/vision-gen/attn_qk_stationarity \
SGLANG_DIFFUSION_ATTENTION_MAP_QK_CHUNKS=3,5,7,9,11,13 \
sglang generate --model-path /data/projects/vision-gen/models/SelfForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 165 --seed 42

# 分析
python results/sparse_eval/osa_dt_analysis/analyze_qk_stationarity.py <run_dir>
python results/sparse_eval/osa_dt_analysis/compare_policy_cost.py <run_dir>
python results/sparse_eval/osa_dt_analysis/frozen_band_test.py <run_dir>
python results/sparse_eval/osa_dt_analysis/pattern_taxonomy.py <run_dir>
```

结论（数据：`attn_qk_stationarity/CausalWanTransformer3DModel-20260816-114341`）：

1. **chunk 内**：3 个 query 帧的分布按相同 dt 对齐时中位相关 r≈0.89–0.92；
   按 chunk 相对坐标（平移对齐）只有 0.36–0.74。约 52% 的头强 dt 对齐，
   35% 真为 chunk 对齐。
2. **跨 chunk**：dt 分布在 chunk 3→13 间相关 0.87–0.98（稳态 7 对 13：
   0.935）——观测一次、复用终身成立。
3. **等保留率成本**（稳态 chunk 13，保留率 0.9）：chunk 粒度密度 0.571
   （恰为 kernel 盈亏线）；dt 任意集 0.458；band+edge（≤2 区间）0.498。
4. **部分窗口校准是失效模式**：冻在 chunk 3 的策略到 chunk 13 保留率跌至
   ~0.75；冻在首个满窗 chunk 7 则整程保持 0.90——`refresh_at_full_window`
   的依据。
5. **模式分类**（保留率 0.9，`pattern_taxonomy.py`）：band 90.3%、
   block 6.4%、dt_comb 2.5%（周期 2/3/6）、v_comb 0.8%；全库密度 0.502 对
   仅 band 0.509——21 帧窗口上周期模式真实但稀少，其收益随窗口长度增长。
   kernel 形状无关（等保留 token 下：1 段连续区间 375 TFLOP/s，7 段散布
   comb 区间 359–361，见 `bench_pattern_kernel.py`）——**无需专用模式 kernel**，
   每 chunk 一次的"模式→区间"编译即是特化。

## 3. 480p 端到端（狐狸 prompt，seed 42，单张空闲 H200，顺序运行）

dense / 首版 chunk-OSA / dt-OSA 三方对比（本会话重测，2026-08-16）：

```bash
sglang generate --model-path .../SelfForcing-Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A red fox trotting across a snowy field, camera follows" \
  --num-frames 321 --seed 42 --save-output \
  --sparse-attention osa --sparse-attention-config granularity=dt   # dt-OSA
# chunk 基线：--sparse-attention-config granularity=chunk,refresh_at_full_window=false
# 质量分析：python results/sparse_eval/osa_dt_analysis/video_quality.py <输出目录>
```

| 20 s / 321 帧 | 去噪 s | 加速 | 全程密度 | PSNR f0-48/48-96/96-321 | 平均绝对差 |
|---|---|---|---|---|---|
| dense | 13.69 | 1.00 | 1.000 | — | — |
| osa chunk（冻在 chunk 3） | 11.59 | 1.18× | 0.470 | 37.8 / 19.1 / 14.2 | 27.6 |
| osa dt（chunk 7 刷新） | 13.18 | 1.04× | 0.581 | 37.9 / 18.2 / 14.9 | 26.0 |

两行处于**不同的实际保留率**：chunk 行的低密度来自冻结策略的隐性欠保留
（稳态约 0.85，也解释了其更差的长段保真与更强漂移）；dt 行整程守住 0.9，
稳态密度 ~0.50 与离线预测一致。诚实前沿上的对比是：chunk 粒度要守住 0.9
需要满窗校准、密度 0.659、比 dense 还慢（见下方 ablation），而 dt 以 ~0.50
做到。质量（帧图 `quality_frame_sheet.png`，各结果目录内）：20 s 处 dense
自身粉色偏色坍塌（chroma 1.68），chunk-OSA 同样漂移且 ~256 帧出现块状损坏；
**dt-OSA 是三者中最健康的轨迹**（chroma 1.22，无块状伪影）。

10 s / 165 帧：dense 8.80 s；chunk 7.98 s（1.10×，密度 0.558）；dt 8.82 s
（1.00×，0.616）；dt+模式库 9.27 s（0.614，多出的 0.45 s 是两次模式库校准
的 Python 开销）。校准分类：band 96% / block 1% / dt_comb 1% / v_comb 2%。

OSA 超参消融（20 s，四卡并行——只读密度列）：`reference_chunk` 2/3/6 →
密度 0.378/0.470/0.659（满窗校准贵过盈亏线，这正是 dt 粒度的价值）；
`retention` 0.8/0.9/0.95 → 0.431/0.470/0.488，单调换保真。
数据：`results/sparse_eval/osa_ablation_20s/`。

早期（2026-08-02）与六个对照方法的 480p 全表位于
`results/sparse_eval/osa_comparison_{10s,20s}/results.json`；注意其中
Radial/SVG1/X-Attention 三行测于 parity 修复（§5）之前，**不可引用**；
OSA 与 dense 行不受影响。有效的跨方法对比见 §4 的 720p 表。

## 4. 720p 端到端（帆船 prompt，seed 42，20 s = 321 帧，1280×720）

720p 下注意力占去噪 72%，是真正 attention-bound 的场景。六后端对比
（2026-08-13，parity 修复后；复现脚本
`results/sparse_eval/backend_sweep_720p/run_sweep.sh`）：

| 后端 | 配置 | 全程密度 | 去噪 s | 对 dense | 200 帧画质 |
|---|---|---|---|---|---|
| dense (FA3) | — | 1.0 | 39.7 | 1.00× | 干净 |
| lightforcing | sparsity 0.90 | 0.147 | 20.4 | 1.95× | 干净 |
| osa（frame 粒度） | retention 0.3, sink 1f | 0.305 | 21.7 | 1.83× | 干净 |
| xattention | threshold 0.15 | 0.061 | 22.4 | 1.78× | **崩成噪声** |
| svg1 | band_frames 0.125 | 0.181 | 23.5 | 1.69× | 顶部条带损坏 |
| osa（chunk 粒度） | retention 0.3 | 0.327 | 23.9 | 1.66× | 干净 |
| radial | decay_factor 0.25 | 0.295 | 24.0 | 1.66× | 干净 |
| svg2 | top_p 0.02 | 0.218 | 35.8 | 1.11× | 干净（最好看） |

dt-OSA 保留率扫描（2026-08-16；逐调用注意力加速 = FA3 7.4 ms 对该稳态
密度下的 range kernel，`bench_720p_attention.py`；质量帧图
`results/sparse_eval/osa_dt_720p_20s/quality_frame_sheet.png`）：

| retention | 去噪 s | 对 dense | 稳态密度 | 逐调用注意力加速 | 画质 |
|---|---|---|---|---|---|
| dense | 39.46 | 1.00× | — | 1.00× | 干净（720p 为分布外，后段泛绿） |
| 0.9（默认） | 38.07 | 1.04× | 0.53 | 1.20× | 干净，最贴近 dense |
| 0.75 | 29.49 | 1.34× | 0.32 | 1.87× | 干净 |
| 0.6 | 25.06 | 1.57× | 0.21 | 2.59× | 干净（构图漂移但无伪影） |
| 0.3 | 20.48 | 1.93× | 0.10 | 6.29× | **~190 帧起崩成噪声——无效** |

要记住的发现：dt 策略能到达模型撑不住的密度（0.10 ≈ 每帧只留 2 帧对角带
——chunk/frame 粒度因"整 own chunk"下限 3/21 反而到不了那么低）。质量把关
之后，本模型的干净前沿是 retention 0.6–0.75。

## 5. 对照方法的 parity（为什么 §4 可信而 08-02 的基线行不可信）

2026-08-06 对照上游源码审计：Radial（缺长程帧抽稀、带宽 ~5× 过宽）、
SVG1（缺首帧 sink）、X-Attention（softmax 次序错误）三者实现有误并已修复；
SVG2 本来就对。上游依赖不可安装的 CUDA 扩展，故将定义各方法的纯 PyTorch
函数**逐字 vendored** 到 `test/unit/realtime/reference/` 并以测试对齐：

```bash
python -m pytest python/sglang/multimodal_gen/test/unit/realtime/test_radial_parity.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_xattention_parity.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_svg_parity.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_lightforcing_parity.py -q
```

parity 声明的准确表述：在上游能运行的几何上与上游相等；在 Wan 480p 真实
几何（1560 token 帧，上游会直接报错）上按同一规则外推。各家自有 kernel
未比较（装不上），只比选择器与 mask。

## 6. Kernel 与微基准

```bash
python -m sglang.multimodal_gen.tools.benchmark_sparse_attention
```

共享 Triton range-sparse kernel 在 Self-Forcing 480p 形状（q=4680,
kv=32760, 12 头×128）达 ~376 TFLOP/s = FA3 的 57%——因此**密度 < 0.57 才
开始省钱**，这是全部方法共同的约束。密度上的路线（FA4 SM90 块稀疏、或把
OSA 的 ≤2 连续区间打包成 varlen FA3 调用：实测 0.86 ms 对 Triton 1.07 /
FA3-dense 1.42，即 1.65× 对 1.30×）是把 dt 的密度优势变成时钟的下一步。
微基准中数据依赖方法的密度在随机激活上无意义，诚实读数是 planning 成本列。

## 结果索引（`results/sparse_eval/`）

- `osa_dt_analysis/`：本笔记 §2–§4 的分析/基准脚本与运行日志
- `osa_dt_480p_{10s,20s}/`、`osa_dt_720p_20s/`：dense/chunk/dt（720p 含
  retention 扫描）的视频与质量帧图
- `osa_comparison_{10s,20s}/`、`osa_ablation_20s/`：08-02 的 480p 对比与
  消融原始数据（基线行已过时，见 §5）
- `backend_sweep_720p/`：§4 六后端对比的完整日志、trace、视频、
  逐帧画面与 `figures/`（去噪耗时分解图）
- qk 转储：`/data/projects/vision-gen/attn_qk_stationarity/`（库外，4.4 GB）
