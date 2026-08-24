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
            "<b>密度与画面内容无关</b>：OSA / LightForcing / Radial / STA 的实际密度"
            "在 5 个 prompt 上完全一致（浮动 &lt; 0.01），"
            "只有 XAttention 这种逐步按内容估计的方法随场景浮动（0.27–0.38）。",
            "跨 prompt 的质量结论与单 prompt 一致：LightForcing 最忠实、"
            "OSA 结构保持最好、SVG2 与 XAttention 在复杂场景后段最先崩坏。",
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
        "walltime_bullets": [],
        "quality_bullets": [],
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
