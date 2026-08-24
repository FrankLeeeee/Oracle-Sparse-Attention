# SPDX-License-Identifier: Apache-2.0
"""Hand-written per-model prose for the Feishu sections.

Everything numeric in the docs comes from the results files; what lives here is
the part a script cannot produce — the visual read of the frame sheets, the
geometry description, and the per-model caveats. Keep each entry factual and
tied to something visible in the figures the same section publishes.
"""

PROMPT_LABELS = {
    "p0_tokyo": "p0 · 东京夜街",
    "p1_forest": "p1 · 雨林逃亡",
    "p2_plating": "p2 · 主厨摆盘",
    "p3_raccoon": "p3 · 浣熊吉他",
    "p4_teacup": "p4 · 茶杯倒水",
    "p5_tsunami": "p5 · 巷道海啸",
}

MODEL_NOTES: dict[str, dict] = {
    "self_forcing": {
        "model_line": "Self-Forcing 1.3B 全上下文（fullctx 转档；注意力占稠密去噪约 74%）",
        "geometry_line": "无窗口上限，末 chunk 上下文约 28.1 万 token",
        "time_metric": "去噪耗时",
        "walltime_bullets": [
            "<b>同实际密度下 OSA 全档最快</b>；Radial 次之（连续帧带对 kernel 友好），"
            "LightForcing 紧随其后且可达更低密度。",
            "<b>STA 的密度—耗时曲线最陡</b>：0.55 档几乎不省（1.09×），"
            "但 0.10 档 21.7 s（3.38×）是全场最快——其保留集合是"
            "“每帧同一批空间 tile”，tile 越窄越接近连续读取。",
            "XAttention 逐步 antidiagonal 估计的开销限制了高密度档的收益"
            "（0.45 档仅 1.07×），低密度档才拉开（0.10 档 2.50×）。",
            "SVG2 的逐步聚类开销仍然最重：≥0.26 档不快于稠密，只有最低档略快。",
            "注：稠密参考与上一轮（73.1 s）一致，全部数字为独占 GPU 下重测。",
        ],
        "quality_bullets": [
            "dense 自身在后段出现背景漂移（东京夜街 13 s 后街面反光发蓝、洗白，"
            "人物尚在）——模型本身的长程退化，PSNR 的参考轨迹并不干净。",
            "<b>LightForcing 全程最忠实</b>跟随 dense 轨迹（含其退化），PSNR 也最高；"
            "Radial 结构保持完整，后段街面反光偏品红。",
            "<b>OSA 全程保持人物与街景结构</b>，后段招牌与霓虹饱和度渐高"
            "（内容漂移而非崩坏）；其较低 PSNR 来自轨迹分歧而非画质损失。",
            "<b>STA 同样全程保住人物与街道结构</b>（0.32 档），只是霓虹饱和度"
            "在后段升高；它的 PSNR（11.4 / 13.9）介于 OSA 与 SVG1 之间。",
            "真实质量缺陷在别处：SVG1 从约 10 s 起洗白成品红色块；"
            "XAttention 从 7 s 起场景被 LED 墙面占满、人物退成剪影；"
            "SVG2 到 16 s 后完全丢失场景（红紫几何色面）。",
            "<b>范围</b>：单 seed；PSNR 只作轨迹分歧的量度，视觉检查为准。",
        ],
        "prompt_bullets": [
            "<b>密度与画面内容无关</b>：OSA（0.295）、LightForcing（0.300）、"
            "Radial（0.258）、STA（0.316）在 5 个 prompt 上给出完全相同的密度——"
            "它们的保留集合由几何决定；SVG1 / SVG2 只有 ±0.01 的浮动；"
            "只有 <b>XAttention 随场景大幅变化（0.272–0.380）</b>，"
            "因为它逐步按内容估计保留块。",
            "跨 prompt 的质量结论与单 prompt 一致：LightForcing 最忠实，"
            "OSA / STA 结构保持完好，XAttention 在多个 prompt 上后段丢结构"
            "（如 p5 巷道海啸的两侧墙体在 7 s 后消失），"
            "Radial / SVG1 在 p5 后段出现品红色辉光。",
        ],
    },
    "causal_forcing": {
        "model_line": "Causal Forcing 1.3B（thu-ml/Causal-Forcing chunk-wise，"
        "转档为 frankleeeee/CausalForcing-Wan2.1-T2V-1.3B-Diffusers）",
        "geometry_line": "3 帧 chunk、21 潜帧滑动窗口、1 帧 sink",
        "time_metric": "去噪耗时",
        "walltime_bullets": [],
        "quality_bullets": [],
        "prompt_bullets": [],
    },
    "rolling_forcing": {
        "model_line": "Rolling Forcing 1.3B（TencentARC/RollingForcing，滚动窗口联合去噪）",
        "geometry_line": "3 帧 chunk、5 块滚动窗口、21 潜帧注意力上限、3 帧 sink",
        "time_metric": "去噪耗时",
        "setup_extra": [
            "<b>与 Self-Forcing 的关键差异</b>：注意力窗口封顶在 21 潜帧，"
            "因此注意力只占稠密去噪的一部分，整体加速比天然低于全上下文模型；"
            "另外去噪窗口一次覆盖 5 个 chunk（15 帧查询），且 sink 块在稳态去噪时"
            "被重新 RoPE 到工作缓存之前——距离型方法必须按<b>位置帧号</b>而非"
            "视频帧号计算时间距离（本轮为此新增了 positional_frame_ids）。",
        ],
        "walltime_bullets": [
            "<b>封顶窗口把“整帧保留型”方法的密度下限抬高了</b>：OSA 下限 0.375、"
            "Radial 0.443、SVG2 0.619——它们的固定保留（自身 chunk + sink + 近期帧）"
            "在 21 帧的可见集合里就已占掉大半，于是最高只能到 1.46× / 1.29× / 0.83×。",
            "<b>能同时削减空间的方法才走得下去</b>：STA 0.22 档 19.7 s（2.46×）为全场最快，"
            "LightForcing 0.22 档 21.7 s（2.23×）、XAttention 0.20 档 24.7 s（1.96×）次之。"
            "这与 Self-Forcing 的排序相反，原因就是窗口封顶。",
            "SVG2 在本模型上始终慢于稠密（0.62 档 0.83×）：它的可见历史只有 6 帧左右，"
            "逐步聚类的开销无法被摊薄。",
        ],
        "quality_bullets": [
            "<b>LightForcing 最忠实</b>（0.29 档 PSNR 11.4 / 前 5 s 19.4），"
            "全程与 dense 几乎同框；Radial（0.44）与 SVG1（0.36）同样干净，"
            "但两者密度都高、加速有限。",
            "<b>OSA（0.38）保持人物与街道结构</b>，后段店面内容有替换，属于内容漂移。",
            "<b>XAttention 是本模型上唯一的明确崩坏</b>：0.28 档从 4 s 起画面裂成"
            "多条水平霓虹招牌带，场景完全丢失（PSNR 7.3 也是最低）。",
            "<b>STA（0.32）主体与街道结构全程完好</b>，但场景逐渐替换成更密集、"
            "更饱和的霓虹街景（反光被放大）——不是崩坏，是轨迹分歧；"
            "其 PSNR 偏低（8.3）主要来自这一分歧。",
        ],
        "prompt_bullets": [],
    },
    "longlive2": {
        "model_line": "LongLive-2.0 5B（Efficient-Large-Model/LongLive-2.0-5B）",
        "geometry_line": "8 帧 chunk、32 潜帧可见窗口（8 帧 sink + 窗口）",
        "time_metric": "去噪耗时",
        "walltime_bullets": [],
        "quality_bullets": [],
        "prompt_bullets": [],
    },
    "lingbot_world_v2": {
        "model_line": "LingBot-World v2 14B causal fast（realtime I2V，WebSocket 会话）",
        "geometry_line": "3 帧 chunk、9 帧 sink + 9 帧近期窗口",
        "time_metric": "逐 chunk forward 累计耗时",
        "walltime_bullets": [],
        "quality_bullets": [],
        "prompt_bullets": [],
    },
}
