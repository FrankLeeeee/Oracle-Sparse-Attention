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
        "本轮转档并上传为 frankleeeee/CausalForcing-Wan2.1-T2V-1.3B-Diffusers）",
        "geometry_line": "3 帧 chunk、21 潜帧滑动窗口、无 sink（sink_size=0）",
        "time_metric": "去噪耗时",
        "setup_extra": [
            "<b>重要前提：本模型只训练到 81 帧（5 s）。</b>上游 README 明确写着"
            "“Causal Forcing 不原生支持超过 81 帧的视频……直接把 5 秒训练的 "
            "Causal Forcing 当作长视频基线是极不公平的”，长视频应改用其 "
            "longvideo 检查点（本轮也已转档为 "
            "frankleeeee/CausalForcing-Long-Wan2.1-T2V-1.3B-Diffusers，"
            "它其实是基于 Rolling Forcing 框架重训的）。",
            "因此本页给出<b>两组实验</b>：20 s 一组只为与其余四个模型口径一致地"
            "比较<b>耗时</b>——该时长下<b>稠密输出自身已经崩坏</b>"
            "（约 7 s 后画面被霓虹涂抹吞没，16 s 后主体只剩剪影），"
            "PSNR 与视觉对比都失去意义；"
            "<b>5 s 一组（81 帧，模型的训练分布内）才是质量结论的依据</b>。",
        ],
        # Quality is read off the in-distribution 5 s sweep, not the 20 s one.
        "quality_results_file": "results_5s.json",
        "quality_sheets": [
            (
                "quality_sheet_target0.3_5s.png",
                "帧对比（5 s 原生时长，模型分布内；行：方法；列：帧号 / 时间，共 7 帧）",
            ),
            (
                "quality_sheet_target0.3.png",
                "对照：同样的方法在 20 s 下——注意 Dense 一行自身已崩坏，"
                "该时长的质量比较不成立",
            ),
        ],
        "quality_lead_extra": (
            "<p><b>本节的数字与图都来自 5 s（81 帧）这一组</b>，即模型的训练分布内；"
            "20 s 那一组的 dense 参考自身已崩坏，任何“相对 dense”的指标都失去意义。"
            "另需注意：5 s 只有 7 个 chunk，累计密度被前几个稠密 chunk 主导，"
            "因此各档实际密度都落在 0.27–0.55，<b>这一组回答的是“会不会坏”，"
            "而不是“哪个密度更快”</b>。另外视频本身就只有 5 s，"
            "所以下表“整段”与“前 5 s”两列按定义相同，保留只为与其他四页同构。</p>"
        ),
        "walltime_bullets": [
            "<b>本表仅用于与其余四个模型口径一致地比较耗时</b>；质量结论见第 3 节的 5 s 组。",
            "封顶窗口（21 潜帧）下，各方法的加速幅度介于 Self-Forcing 与 "
            "Rolling Forcing 之间：<b>OSA 最快</b>（0.29 档 22.0 s，1.94×），"
            "Radial 次之（0.29 档 24.2 s，1.76×），STA（0.30 档 26.2 s，1.63×）"
            "与 LightForcing（1.43×）接近。",
            "SVG2 在 0.42 档仍略慢于稠密（0.96×），到 0.22 档才到 1.21×——"
            "与其余模型上的结论一致：它的逐步聚类开销最难摊薄。",
        ],
        "quality_bullets": [
            "<b>5 s 原生时长下，全部方法都可用</b>：dense、OSA、LightForcing、"
            "Radial、SVG1、SVG2、XAttention 都保持人物、红裙与街景结构，"
            "没有任何一个出现崩坏。",
            "<b>唯一的可见伪影来自 STA</b>（0.50 档）：从第 2 s 起画面中出现"
            "<b>第二个主体</b>（同一位女性的复制），此后一直存在。"
            "这与它在 Rolling Forcing 上的失效同源——tile 窗口切断了空间上"
            "相距较远的区域之间的关联，模型于是各自成像。",
            "各方法在 5 s 下的实际密度都在 0.27–0.55，差异不大，"
            "因此 PSNR 高低更多反映轨迹分歧速度而非画质：XAttention 密度最低（0.27）"
            "而 PSNR 也最低，属预期。",
        ],
        "prompt_bullets": [
            "<b>密度与画面内容无关</b>：OSA（0.294）、Radial（0.295）、STA（0.300）"
            "在 5 个 prompt 上完全一致，LightForcing / SVG1 浮动 ≤ 0.002，"
            "SVG2 0.275–0.288，仍是逐步按内容估计的 XAttention 最不稳定"
            "（0.245–0.274）。",
            "<b>本节的帧对比图仍是 20 s 的，因此同样带着前述前提</b>："
            "该时长下 dense 自身已崩坏，图中各行的画面退化主要来自模型而非方法，"
            "可用于确认“某方法有没有额外引入崩坏”，不宜据此比较画质高低；"
            "分布内的质量结论请看第 3 节的 5 s 组。",
        ],
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
            "<b>注意各方法的实际密度差异很大</b>（下表括号内）：Radial 0.44、"
            "SVG2 0.62 是它们的下限，而 XAttention 0.28、STA 0.27 已接近各自最稀疏档，"
            "因此 PSNR 不能横向直接比大小。",
            "<b>LightForcing 最忠实</b>（0.29 档 PSNR 11.4 / 前 5 s 19.4），"
            "全程与 dense 几乎同框；Radial、SVG1、SVG2 同样干净，但密度都高、加速有限。",
            "<b>OSA（0.38）保持人物与街道结构</b>，后段店面内容有替换，属内容漂移。",
            "<b>XAttention 与 STA 在本模型上都会崩坏</b>：从约 4 s 起画面下半部裂成"
            "多条水平霓虹色带，此后不再恢复（PSNR 7.3 / 8.6，为全场最低两名）。",
            "<b>崩坏的机制：这两个方法会稀疏掉“正在联合去噪的窗口”本身。</b>"
            "Rolling Forcing 每次前向同时对 <b>5 个 chunk（15 潜帧）</b>以交错噪声"
            "水平联合去噪，这 15 帧不是历史、而是正在被生成的内容，必须互相看到。"
            "STA 校准出的 kernel_t 为 13 / 9 / 7（均小于 15），"
            "kernel_t=9 时最旧的查询帧只能看到 ±4 帧，永远看不到最新的查询帧；"
            "XAttention 则按估计分数在窗口内任意丢块。"
            "反观干净的几个方法：SVG2 把 <code>kv_len - q_len</code> 之后的整段"
            "（即整个窗口）恒定保留，OSA 对每个查询 chunk 保留其自身整块再对其余帧"
            "复制 tile 图案，LightForcing 把窗口内各帧全部置为可选——"
            "它们都没有切断窗口内部的连接。",
            "<b>可以修，但代价是失去意义</b>：若强制保留整个去噪窗口，"
            "STA 的密度下限会被抬到 15/21 ≈ 0.71，本模型上也就不再有加速空间。"
            "所以这里按上游语义如实报告：<b>STA 与 XAttention 不适配滚动窗口联合去噪</b>，"
            "而不是把它们改造成必然不稀疏的版本。",
        ],
        "prompt_bullets": [
            "<b>密度完全与内容无关</b>：OSA（0.375）、LightForcing（0.286）、"
            "Radial（0.443）、SVG2（0.619）、STA（0.274）在 5 个 prompt 上给出"
            "逐位相同的密度，SVG1 浮动 0.001，XAttention 也只有 0.276–0.289"
            "（比 Self-Forcing 上的 0.27–0.38 稳定得多，因为窗口封顶限制了它的选择空间）。",
            "<b>崩坏结论跨 prompt 成立</b>：STA 在全部 5 个 prompt 上都从约 4 s 起"
            "出现色带噪声（雨林一档尤其明显，画面几乎被竖直彩条吞没）；"
            "XAttention 表现为强烈拖影与色块。二者都不是个别场景的偶发问题。",
            "<b>干净的方法同样跨 prompt 稳定</b>：OSA / LightForcing / Radial / SVG2 "
            "在 5 个 prompt 上都保持主体与场景结构；SVG1 会增殖次要物体"
            "（雨林里鹿越来越多、出现悬空的鹿），属于内容漂移而非崩坏。",
        ],
    },
    "longlive2": {
        "model_line": "LongLive-2.0 5B（Efficient-Large-Model/LongLive-2.0-5B，24 头 ×128）",
        "geometry_line": "8 帧 chunk、32 潜帧可见窗口（8 帧 sink + 窗口）、"
        "720p post-patch 网格 22×40（880 token/帧）",
        "time_metric": "去噪耗时",
        "setup_extra": [
            "<b>本模型的注意力规模远小于其他四个</b>：720p 每帧只有 880 token，"
            "窗口封顶 32 帧 ⇒ 每次注意力调用的 KV 只有 <b>28k token</b>"
            "（Self-Forcing 全上下文末 chunk 为 292k，相差 10 倍）。"
            "这是下面所有结论的成因。",
        ],
        "walltime_bullets": [
            "<b>结论：在 LongLive-2 上稀疏注意力基本不划算</b>——除 OSA 外，"
            "所有方法都在 1.0× 上下（部分低于稠密）。",
            "<b>原因是被优化的对象本身太小</b>。在本模型的形状"
            "（q=7040、kv=28160、24 头 ×128）上直接测量单次调用："
            "稠密注意力仅 <b>3.2 ms</b>，而各方法每次调用的<b>规划开销</b>为 "
            "0.24 ms（STA）/ 0.89 ms（LightForcing）/ 1.14 ms（XAttention）/ "
            "1.50 ms（SVG2）——规划本身就占到被优化对象的 8%–47%。"
            "逐步做估计或聚类的方法（XAttention、SVG2）因此整体只有 0.5×。",
            "<b>OSA 是唯一稳定为正的方法</b>（0.40 档 22.4 s，1.27×）："
            "它的计划按 (层, chunk) 缓存，去噪步之间零重复规划，"
            "所以省下的读取能真正兑现。",
            "<b>端到端角度更悲观</b>：稠密端到端 59.8 s 里去噪只占 28.5 s，"
            "即使注意力完全免费，端到端也只能到 1.9×。",
            "注：本模型单次注意力只有毫秒级，逐行 ±10% 的运行间波动属正常，"
            "表中个别非单调点（如 STA 0.39 档反而慢于 0.48 档）应按此理解。",
        ],
        "quality_bullets": [
            "<b>质量上没有问题：全部七个方法都干净</b>。dense、OSA、LightForcing、"
            "Radial、SVG1、SVG2、XAttention、STA 在整段 20 s 里都保持人物、红裙"
            "与商业街结构，没有任何一个出现崩坏或明显伪影。",
            "<b>LightForcing 最忠实</b>（0.30 档 PSNR 12.9 / 前 5 s 19.4），"
            "Radial 与 XAttention 的前 5 s PSNR 也在 20 dB 上下；"
            "OSA 的 PSNR 偏低（0.40 档 11.6 / 14.2）来自内容轨迹分歧"
            "——它较早换掉了两侧店面内容，但结构始终完好。",
            "<b>STA 这一行是本轮修复的直接证据</b>：修复前（tile 时间窗把 8 帧 sink "
            "赶出可见集合）画面从约 4 s 起碎成水平色带；给 sink 加保护后，"
            "同一模型同一档位全程干净（见第 1 节说明）。",
            "换句话说，<b>在 LongLive-2 上限制稀疏注意力的是收益而不是风险</b>："
            "方法都能用，只是省不下时间。",
        ],
        "prompt_bullets": [
            "<b>密度与画面内容无关</b>：OSA（0.398）、LightForcing（0.303）、"
            "Radial（0.539）、STA（0.385）在 5 个 prompt 上完全一致，"
            "SVG1 / SVG2 浮动 ≤ 0.002，仍然只有 XAttention 随内容变化"
            "（0.236–0.266）。",
            "<b>质量结论跨 prompt 一致</b>：5 个 prompt 上没有任何方法出现崩坏，"
            "与单 prompt 的结论相同——这也再次说明本模型的瓶颈不在质量。",
        ],
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
