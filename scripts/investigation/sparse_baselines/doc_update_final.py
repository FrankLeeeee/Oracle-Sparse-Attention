# SPDX-License-Identifier: Apache-2.0
"""Final 2026-08-27 doc round: Self-Forcing and Causal Forcing.

Both docs keep exactly five sections — 1. 实验设置 / 2. Walltime / 3. 生成质量
/ 4. 多 prompt 验证与原始视频 / 复现命令. OSA appears as its optimal
configuration only (OSA + demand schedule: no whole-frame exemptions,
FLOPs-matched 1/sqrt(kv) chunk schedule, demand-weighted per-frame allocation,
flat-row dense fallback). The Self-Forcing doc is edited surgically (its
whiteboards survive); the Causal Forcing doc body is rewritten wholesale and
additionally carries the 30-second long-video study on the checkpoint the
upstream repository ships for long videos.

    python doc_update_final.py --doc self_forcing  --stage text|figures|videos|all
    python doc_update_final.py --doc causal_forcing --stage ...
"""

import argparse
import json
import pathlib
import shutil
import sys

from common import METHOD_LABELS, MODELS, PROMPTS, ROOT, newest_video
from doc_update import DOCS, rewrite_managed, section_ids
from doc_update_self_forcing import find_block, replace_span, top_blocks
from notes import PROMPT_LABELS
from sections import bullets, esc, table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import cli, insert_after, last_block_id, published_media  # noqa: E402

# Display order and the tier columns of the main tables. OSA's optimal
# configuration runs 0.3 / 0.2 / 0.15 (0.15 is its clean floor at 5 s); the
# baselines were calibrated to 0.3 / 0.2 / 0.1.
FINAL_METHODS = ["osasched", "lightforcing", "radial", "svg1", "svg2", "xattention", "sta"]
TIERS = ["0.3", "0.2", "0.15", "0.1"]
TRIO_TAGS = [("osasched_0.2", "osasched"), ("lightforcing_0.2", "lightforcing")]
PROMPT_KEYS = ["p1_forest", "p2_plating", "p3_raccoon", "p4_teacup", "p5_tsunami"]
FILE_LABELS = {
    "dense": "dense",
    "osasched": "osa_demand_schedule",
    "lightforcing": "lightforcing",
    "radial": "radial_attention",
    "svg1": "sparse_videogen_1",
    "svg2": "sparse_videogen_2",
    "xattention": "xattention",
    "sta": "sliding_tile_attention",
}

# Hand-written visual reads, filled after inspecting the final sheets.
QUALITY_BULLETS: dict[str, list[str]] = {
    "self_forcing": [
        "OSA 最优配置在目标密度 0.3 / 0.2 档（实际 0.43 / 0.34）画面干净、紧随 "
        "dense；0.15 档（实际 0.29）是它的干净下限，主体与街景保持完好。",
        "LightForcing 全档干净；Radial 干净但密度下限高（实际 ~0.50）；SVG1 在 "
        "0.3 档干净但速度接近 dense；SVG2 场景漂移（街景变亮变宽，主体尚在）；"
        "XAttention 最低档下半幅出现彩色噪声，STA 最低档大面积噪声。",
        "各方法的失效形式仍是「漂移而非损坏」：PSNR 度量的是偏离 dense 轨迹的"
        "速度；5 秒在训练分布内，dense 参考轨迹本身干净。",
    ],
    "causal_forcing": [
        "OSA 最优配置在 0.3 / 0.2 / 0.15 三档（实际 0.43 / 0.34 / 0.29）均干净，"
        "金色招牌街景与人物紧随 dense；LightForcing 同样全档干净。",
        "SVG2 与 XAttention 出现场景漂移（招牌与街道重构，主体尚在）；STA 在实际"
        "密度 0.29 档从第 2 秒起大面积彩色噪声；Radial 干净但实际密度停在 0.50。",
    ],
}
PROMPT_BULLETS: dict[str, list[str]] = {
    "self_forcing": [
        "密度与画面内容完全无关：OSA 在全部五个 prompt 的实际密度恰为 0.336，"
        "LightForcing 为 0.357–0.358——两者的保留集合都不依赖具体内容。",
        "整段 PSNR 上 OSA 与 LightForcing 的差距为 0.2–1.9 dB（且 OSA 的实际密度"
        "更低），没有任何 prompt 出现伪影或损坏；两者的差异表现为内容细节的轨迹"
        "分歧（如摆盘食材数量、人群构成），不是画质问题。",
    ],
    "causal_forcing": [
        "密度同样与内容无关（OSA 全部 prompt 恰为 0.336，LightForcing "
        "0.357–0.361）。",
        "五个 prompt 中 OSA 与 LightForcing 的 PSNR 差距 -1.4 至 +1.1 dB（p4 · "
        "茶杯倒水上 OSA 反超 1.4 dB）；p5 · 巷道海啸两者都低（~9.5 dB）——湍流"
        "内容下与 dense 的轨迹分歧最快，但两者画面本身均连贯无伪影。",
    ],
}
LONG_BULLETS: list[str] = [
    "<b>30 秒长视频中 OSA 是两个稀疏方法里唯一存活的</b>：OSA（density 参数 "
    "0.2，实际 0.31）在全部六个 prompt 上保持场景完整到 29 秒，前 5 秒 PSNR "
    "20.9–25.2 dB、整段 9.3–14.0 dB，走势紧随 dense 自身的长程演化；"
    "LightForcing（实际 0.16）在全部 prompt 上从约第 6 秒起崩坏为水平涂抹并"
    "持续到结尾（前 5 秒 PSNR 仅 12.3–15.9 dB）——耗时上的领先建立在崩坏的"
    "输出上，不具参考意义。",
    "LightForcing 的控制参数借自 Rolling Forcing 的校准（两者窗口几何一致，"
    "见第 1 节），其崩坏说明该参数不能跨检查点迁移；OSA 的 density 参数按构造"
    "就是平均逐调用密度，无需任何校准即可迁移到新检查点——这是冻结模式 + 解析"
    "调度的直接好处。",
    "OSA 在滚动窗口管线上的累计密度下限约 0.25（density 参数 0.1 时实际 "
    "0.25）：ramp-up 阶段的各次前向兼任在线校准、必须稠密执行。已识别的后续"
    "改进：模式冻结后立即把剩余 ramp-up 窗口切换为稀疏执行，可显著压低长视频"
    "的累计密度下限。",
]


def load(model: str, name: str) -> dict:
    path = ROOT / model / name
    return json.loads(path.read_text()) if path.exists() else {}


def entries_of(results: dict, method: str) -> dict[str, dict]:
    out = {}
    for tag, entry in results.items():
        if tag.startswith(f"{method}_") and entry.get("returncode") == 0:
            out[tag.rsplit("_", 1)[1]] = entry
    return out


def walltime_table(results: dict, *, tiers: list[str] = None) -> str:
    tiers = tiers or TIERS
    dense = results["dense"]["denoise_s"]
    rows = [["Dense", f"{dense:.1f} s（1.00）1.00×"] + ["—"] * (len(tiers) - 1)]
    for method in FINAL_METHODS:
        entries = entries_of(results, method)
        if not entries:
            continue
        cells = [METHOD_LABELS[method]]
        for tier in tiers:
            entry = entries.get(tier)
            if entry is None:
                cells.append("—")
                continue
            cells.append(
                f"{entry['denoise_s']:.1f} s（{entry.get('density', 1.0):.2f}）"
                f"{dense / entry['denoise_s']:.2f}×"
            )
        rows.append(cells)
    return table(["方法"] + [f"目标密度 {t}" for t in tiers], rows)


def psnr_table(results: dict, *, tiers: list[str] = None) -> str:
    tiers = tiers or TIERS
    rows = []
    for method in FINAL_METHODS:
        entries = entries_of(results, method)
        cells, seen = [METHOD_LABELS[method]], False
        for tier in tiers:
            entry = entries.get(tier)
            if entry is None or "psnr_overall_db" not in entry:
                cells.append("—")
                continue
            seen = True
            cells.append(
                f"{entry['psnr_overall_db']:.1f} dB（{entry.get('density', 0):.2f}）"
            )
        if seen:
            rows.append(cells)
    return table(["方法"] + [f"目标密度 {t}" for t in tiers], rows)


def prompt_tables(prompts_results: dict) -> str:
    """Density stability + PSNR per prompt for the dense/OSA/LightForcing trio."""
    header = ["Prompt"] + [
        f"{METHOD_LABELS[m]}：实际密度 / PSNR (dB)" for _tag, m in TRIO_TAGS
    ]
    rows = []
    for prompt_key in PROMPT_KEYS:
        cells = [PROMPT_LABELS[prompt_key]]
        for tag, _method in TRIO_TAGS:
            entry = prompts_results.get(f"{prompt_key}_{tag}", {})
            if entry.get("returncode") != 0:
                cells.append("—")
                continue
            cells.append(
                f"{entry.get('density', 0):.3f} / "
                f"{entry.get('psnr_overall_db', float('nan')):.1f}"
            )
        rows.append(cells)
    return table(header, rows)


def osa_description() -> str:
    return (
        "<p><b>OSA 最优配置（本文档所有 OSA 结果均指此配置）</b>：冻结的 2-D "
        "帧对帧 tile 模式（chunk 0 稠密校准、末步覆盖、按 (层, chunk) 缓存计划）"
        "之上叠加三项 2026-08-26 的分配改进：<b>一</b>，chunk 级密度调度——各 "
        "chunk 的逐调用密度按 1/√(kv 帧数) 前置递减，斜率解至全程 FLOPs 加权平均"
        "恰等于 density 参数（因此 density 参数无需割线校准，本身就是平均逐调用"
        "密度）；<b>二</b>，需求加权逐帧分配——不再整帧豁免任何帧，每帧的 tile 数"
        "按实测注意力需求权重分配（自身 chunk 帧 8×、最近帧 4×、sink 2.5×、历史帧"
        "随年龄按 1/age 衰减；权重来自逐组注意力质量探针，跨内容与密度稳定）；"
        "<b>三</b>，平坦行稠密回退——质量分布平坦、冻结 top-k 无峰可选的 query 行"
        "（按该 chunk 的实际预算判定）回退为对全部可见 KV 的精确注意力，预算充足时"
        "自动失效。执行仍为等长 block-sparse Triton kernel（~480–505 TFLOP/s），"
        "计划缓存使去噪步间零重复规划。代码：<code>sparse/osa.py</code> / "
        "<code>sparse/block_kernel.py</code>。</p>"
    )


# ---------------------------------------------------------------------------
# Self-Forcing (surgical edits)
# ---------------------------------------------------------------------------


def sf_setup() -> str:
    return bullets(
        [
            "<b>模型</b>：Self-Forcing 1.3B 全上下文（fullctx 转档），单卡 H200。",
            "<b>配置</b>：720p（1280×720）/ 5 秒（81 像素帧 = 21 潜帧 = 7 个 "
            "chunk）——5 秒是 Self-Forcing 的训练时长，全部结论都在分布内。"
            "seed 42；主实验 prompt 为 p0 · 东京夜街，多 prompt 验证覆盖文档内"
            "全部六个 prompt（见第 4 节）。",
            "<b>方法</b>：dense、<b>OSA（最优配置，见下）</b>目标密度 0.3 / 0.2 / "
            "0.15，与六个基线（LightForcing、Radial、SVG1、SVG2、XAttention、"
            "STA）目标密度 0.3 / 0.2 / 0.1。",
            "<b>匹配口径</b>：整段生成的实际累计读取密度（回退 dense 的调用按 "
            "1.0 计），表中括号内为实际值；5 秒只有 7 个 chunk，各方法的稠密"
            "校准段 / ramp-up 在累计中占比高于长视频，实际值普遍高于目标档。",
            "<b>校准</b>：基线的稀疏控制参数沿用 480p / 20 秒割线校准（STA 在 "
            "720p 上按密度阶梯校准）；OSA 最优配置的 density 参数即平均逐调用"
            "密度，无需校准。",
            "<b>计时口径</b>：表中的去噪耗时均为独占整机的串行测量；多 prompt "
            "验证运行在并行工位上，只报告密度与 PSNR。",
        ]
    )


def sf_push_text() -> None:
    results = load("self_forcing", "results_5s.json")
    prompts_results = load("self_forcing", "results_prompts_5s.json")
    doc = DOCS["self_forcing"]

    replace_span(doc, after="1. 实验设置", before="OSA 实现", xml=sf_setup())
    # The OSA-implementation description: the paragraph+bullets between the
    # "OSA 实现" marker and the whiteboard, plus the structure-update note.
    blocks = top_blocks(doc)
    start = find_block(blocks, text="OSA 实现")
    end = find_block(blocks, tag="whiteboard", start=start + 1)
    doomed = [bid for _, ids, _ in blocks[start:end] for bid in ids]
    cli("docs", "+update", "--doc", doc, "--command", "block_insert_after",
        "--block-id", blocks[start - 1][1][-1], "--content", osa_description())
    for i in range(0, len(doomed), 20):
        cli("docs", "+update", "--doc", doc, "--command", "block_delete",
            "--block-id", ",".join(doomed[i : i + 20]))
    print("replaced the OSA-implementation description")

    lead = (
        f"<p>720p / 5 秒稠密参考：去噪耗时 {results['dense']['denoise_s']:.1f} 秒，"
        f"端到端 {results['dense']['e2e_s']:.1f} 秒。下表为各方法在各目标密度档的"
        "<b>去噪耗时（加速比）</b>，括号内为实际累计密度：</p>"
    )
    replace_span(
        doc,
        after="2. Walltime",
        before="walltime_vs_density",
        xml=lead + walltime_table(results) + bullets(
            [
                "OSA 最优配置在每个匹配密度档都快于 LightForcing（0.2 档 8.5 对 "
                "8.9 秒；0.15 档与 LightForcing 最快干净档持平），且是唯一无需"
                "逐步估计、计划零重复规划的方法。",
                "估计类基线（SVG1 / SVG2）在 5 秒形状上接近或低于收支平衡；STA "
                "的 tile 窗在短上下文中夹不动，实际密度停在 0.46–0.48。",
                "5 秒的稠密去噪只有约 10.7 秒，注意力占比远低于长视频，一切方法"
                "的加速上限约 1.4×；方法间对比看密度—耗时曲线更有意义。",
            ]
        ),
    )
    quality_lead = (
        "<p><b>PSNR（相对 dense 输出，dB）：</b>5 秒在训练分布内，dense 参考轨迹"
        "干净，PSNR 直接可比。括号内为实际累计密度：</p>"
    )
    inspect = (
        "<p><b>视觉检查</b>（帧对比图见下，目标密度 0.3 / 0.2 / 0.1 三档各一张；"
        "行：方法；列：帧号 / 时间）：</p>"
    )
    replace_span(
        doc,
        after="3. 生成质量",
        before="quality_sheet",
        xml=quality_lead
        + psnr_table(results)
        + inspect
        + (bullets(QUALITY_BULLETS["self_forcing"]) if QUALITY_BULLETS["self_forcing"] else ""),
    )
    # Remove the 3.5 shake subsection wholesale (its heading and everything
    # up to section 4), per the final structure.
    blocks = top_blocks(doc)
    try:
        start = find_block(blocks, tag="h3", text="3.5")
        end = find_block(blocks, tag="h2", text="4. 多 prompt", start=start + 1)
        doomed = [bid for _, ids, _ in blocks[start:end] for bid in ids]
        for i in range(0, len(doomed), 20):
            cli("docs", "+update", "--doc", doc, "--command", "block_delete",
                "--block-id", ",".join(doomed[i : i + 20]))
        print(f"removed the 3.5 subsection ({len(doomed)} blocks)")
    except LookupError:
        pass
    # Section 4: everything between its heading and 复现命令 is replaced.
    blocks = top_blocks(doc)
    start = find_block(blocks, tag="h2", text="4. 多 prompt")
    end = find_block(blocks, tag="h2", text="复现命令", start=start + 1)
    doomed = [bid for _, ids, _ in blocks[start + 1 : end] for bid in ids]
    cli("docs", "+update", "--doc", doc, "--command", "block_insert_after",
        "--block-id", blocks[start][1][-1],
        "--content", sf_prompt_section(prompts_results))
    for i in range(0, len(doomed), 20):
        cli("docs", "+update", "--doc", doc, "--command", "block_delete",
            "--block-id", ",".join(doomed[i : i + 20]))
    print("replaced section 4")
    # 复现命令 (last section).
    blocks = top_blocks(doc)
    start = find_block(blocks, tag="h2", text="复现命令")
    doomed = [bid for _, ids, _ in blocks[start + 1 :] for bid in ids]
    cli("docs", "+update", "--doc", doc, "--command", "block_insert_after",
        "--block-id", blocks[start][1][-1], "--content", commands_section("self_forcing"))
    for i in range(0, len(doomed), 20):
        cli("docs", "+update", "--doc", doc, "--command", "block_delete",
            "--block-id", ",".join(doomed[i : i + 20]))
    print("replaced 复现命令")


def sf_prompt_section(prompts_results: dict) -> str:
    xml = (
        "<p>文档内全部六个 prompt 在 720p / 5 秒下验证 dense、OSA（最优配置，"
        "目标密度 0.2）与 LightForcing（目标密度 0.2）。下表为每个 prompt 的"
        "实际累计密度与整段 PSNR（相对该 prompt 自己的 dense 输出）：</p>"
        + prompt_tables(prompts_results)
    )
    if PROMPT_BULLETS["self_forcing"]:
        xml += bullets(PROMPT_BULLETS["self_forcing"])
    for prompt_key in ["p0_tokyo"] + PROMPT_KEYS:
        xml += (
            f"<p><b>{PROMPT_LABELS[prompt_key]}</b>："
            + esc(PROMPTS[prompt_key])
            + "</p>"
        )
    return xml


# ---------------------------------------------------------------------------
# Causal Forcing (full-body rewrite)
# ---------------------------------------------------------------------------


def cf_body() -> str:
    results = load("causal_forcing", "results_5s.json")
    prompts_results = load("causal_forcing", "results_prompts_5s.json")
    long_results = load("causal_forcing_long", "results_30_seconds.json")

    setup = bullets(
        [
            "<b>模型</b>：Causal Forcing 1.3B（chunkwise 蒸馏，21 潜帧注意力窗口"
            "），单卡 H200。5 秒（81 像素帧）是该模型的原生时长。",
            "<b>长视频检查点</b>：30 秒实验使用上游仓库专为长视频发布的检查点"
            "（github.com/thu-ml/Causal-Forcing 的 long_video · "
            "longvideo.pt generator_ema，本仓库转档为 "
            "CausalForcing-Long-Wan2.1-T2V-1.3B-Diffusers），其推理管线为 "
            "Rolling Forcing 式滚动窗口——按上游说明，用普通检查点跑长视频是不"
            "公平的比较。",
            "<b>配置</b>：720p；5 秒（7 个 chunk）与 30 秒（477 像素帧 = 120 潜帧"
            "）；seed 42；全部六个 prompt（见第 4 节）。",
            "<b>方法</b>：dense、<b>OSA（最优配置，与 Self-Forcing 文档同一配置"
            "）</b>目标密度 0.3 / 0.2 / 0.15，六个基线目标密度 0.3 / 0.2 / 0.1；"
            "30 秒长视频部分运行 dense / OSA / LightForcing 三方法（其余基线在"
            "长视频检查点上没有公平的校准，未列入）。",
            "<b>匹配口径</b>：实际累计读取密度（回退 dense 按 1.0 计）；"
            "LightForcing 在长视频检查点上的控制参数沿用 Rolling Forcing 的校准"
            "（两者窗口几何一致），表中报告实际达到值。",
            "<b>计时口径</b>：Walltime 表为独占整机的串行测量；多 prompt 与 30 秒"
            "逐 prompt 运行在并行工位上，只报告密度与 PSNR。",
        ]
    )
    walltime = (
        f"<p>720p / 5 秒稠密参考：去噪耗时 {results['dense']['denoise_s']:.1f} 秒。"
        "各方法在各目标密度档的<b>去噪耗时（加速比）</b>，括号内为实际累计密度："
        "</p>"
        + walltime_table(results)
    )
    if long_results.get("p0_tokyo_dense"):
        dense30 = long_results["p0_tokyo_dense"]["denoise_s"]
        rows = [["Dense", f"{dense30:.1f} s（1.00）1.00×"]]
        for tag, method in (
            ("osasched_0.2", "osasched"),
            ("osasched_0.1", "osasched"),
            ("lightforcing_0.2", "lightforcing"),
        ):
            entry = long_results.get(f"p0_tokyo_{tag}")
            if entry and entry.get("returncode") == 0:
                label = METHOD_LABELS[method]
                if method == "osasched":
                    label += f"（density 参数 {tag.rsplit('_', 1)[1]}）"
                rows.append(
                    [
                        label,
                        f"{entry['denoise_s']:.1f} s"
                        f"（{entry.get('density', 0):.2f}）"
                        f"{dense30 / entry['denoise_s']:.2f}×",
                    ]
                )
        walltime += (
            "<p><b>30 秒长视频（长视频检查点，p0 · 东京夜街）</b>——注意力上下文"
            "达到窗口上限后滚动，去噪耗时（加速比）（实际累计密度）：</p>"
            + table(["方法", "720p / 30 秒"], rows)
        )
    quality = (
        "<p><b>PSNR（相对 dense 输出，dB，720p / 5 秒）：</b>括号内为实际累计"
        "密度：</p>"
        + psnr_table(results)
        + "<p><b>视觉检查</b>（帧对比图见下）：</p>"
        + (bullets(QUALITY_BULLETS["causal_forcing"]) if QUALITY_BULLETS["causal_forcing"] else "")
    )
    prompts_xml = (
        "<p>全部六个 prompt 在 720p / 5 秒验证 dense、OSA（最优配置，目标密度 "
        "0.2）与 LightForcing（目标密度 0.2）；30 秒长视频部分对同样的三个方法"
        "重复全部 prompt。5 秒逐 prompt 的实际密度与整段 PSNR：</p>"
        + prompt_tables(prompts_results)
    )
    long_prompt_rows = []
    for prompt_key in ["p0_tokyo"] + PROMPT_KEYS:
        cells = [PROMPT_LABELS[prompt_key]]
        for tag, _m in TRIO_TAGS:
            entry = long_results.get(f"{prompt_key}_{tag}", {})
            if entry.get("returncode") != 0:
                cells.append("—")
            else:
                cells.append(
                    f"{entry.get('density', 0):.3f} / "
                    f"{entry.get('psnr_overall_db', float('nan')):.1f}"
                )
        long_prompt_rows.append(cells)
    prompts_xml += (
        "<p><b>30 秒长视频逐 prompt</b>（实际密度 / 整段 PSNR，相对各 prompt 的 "
        "30 秒 dense）：</p>"
        + table(
            ["Prompt"]
            + [f"{METHOD_LABELS[m]}：实际密度 / PSNR (dB)" for _t, m in TRIO_TAGS],
            long_prompt_rows,
        )
    )
    if LONG_BULLETS:
        prompts_xml += bullets(LONG_BULLETS)
    for prompt_key in ["p0_tokyo"] + PROMPT_KEYS:
        prompts_xml += (
            f"<p><b>{PROMPT_LABELS[prompt_key]}</b>："
            + esc(PROMPTS[prompt_key])
            + "</p>"
        )
    return (
        "<h2>1. 实验设置</h2>" + setup + osa_description()
        + "<h2>2. Walltime</h2>" + walltime
        + "<h2>3. 生成质量</h2>" + quality
        + "<h2>4. 多 prompt 验证与原始视频</h2>" + prompts_xml
        + "<h2>复现命令</h2>" + commands_section("causal_forcing")
    )


def commands_section(model: str) -> str:
    extra = ""
    if model == "causal_forcing":
        extra = (
            "python final_round.py --model causal_forcing_long --duration 30 \\\n"
            "  --prompts p0_tokyo p1_forest p2_plating p3_raccoon p4_teacup "
            "p5_tsunami \\\n  --methods osasched:0.2 lightforcing:0.2 \\\n"
            "  --out results_30_seconds.json --runs-dir runs_30_seconds \\\n"
            "  --sheet-prefix prompt_sheet_30_seconds_\n"
        )
    return (
        "<p>全部命令位于 <code>scripts/investigation/sparse_baselines/</code>；"
        "OSA 最优配置的变体键为 <code>osasched</code>（density 参数即平均逐调用"
        "密度，无需校准）：</p>"
        f"<pre lang=\"bash\"><code># 单元测试\nPYTHONPATH=python python -m pytest "
        "python/sglang/multimodal_gen/test/unit/realtime/test_sparse_attention.py\n"
        f"# 主表：720p / 5 秒 sweep（dense + 各方法各档）\n"
        f"python run_sweep.py --model {model} --duration 5 --res 720p \\\n"
        "  --out results_5s.json --runs-dir runs_5s\n"
        f"# 质量：PSNR + 帧对比图\npython quality.py --model {model} --duration 5 \\\n"
        "  --out results_5s.json --runs-dir runs_5s --sheet-suffix _5s\n"
        "# 多 prompt（dense + OSA 最优 + LightForcing）\n"
        f"python final_round.py --model {model} --duration 5 \\\n"
        "  --prompts p1_forest p2_plating p3_raccoon p4_teacup p5_tsunami \\\n"
        "  --methods osasched:0.2 lightforcing:0.2 \\\n"
        "  --out results_prompts_5s.json --runs-dir runs_prompts_5s \\\n"
        "  --sheet-prefix prompt_sheet_5_seconds_\n"
        f"{extra}"
        "# Walltime—密度图\n"
        f"python plot.py --model {model} --results results_5s.json --suffix _5s \\\n"
        "  --duration 5 --methods osasched lightforcing radial svg1 svg2 "
        "xattention sta\n"
        f"# 文档发布\npython doc_update_final.py --doc {model}</code></pre>"
    )


# ---------------------------------------------------------------------------
# Figures and videos
# ---------------------------------------------------------------------------


def push_figures(model: str) -> None:
    doc = DOCS[model]
    model_root = ROOT / model
    ids = section_ids(doc)
    walltime_h2 = next(k for k in ids if "Walltime" in k)
    quality_h2 = next(k for k in ids if "生成质量" in k)
    published = published_media(doc, ids[walltime_h2])
    anchor = last_block_id(doc, ids[walltime_h2])
    figure = model_root / "walltime_vs_density_5s.png"
    if figure.exists():
        from doc_update import upsert_media

        upsert_media(
            doc, published, anchor, figure,
            caption="各方法去噪耗时 vs 实际累计读取密度（720p / 5 秒；虚线为 "
            "dense 参考）",
        )
    published = published_media(doc, ids[quality_h2])
    anchor = last_block_id(doc, ids[quality_h2])
    from doc_update import upsert_media

    for tier in ("0.3", "0.2", "0.1"):
        sheet = model_root / f"quality_sheet_target{tier}_5s.png"
        if sheet.exists():
            anchor = upsert_media(
                doc, published, anchor, sheet,
                caption=f"帧对比（p0 · 东京夜街，目标密度 {tier} 档；行：方法；"
                "列：帧号 / 时间）",
            )
    print("figures pushed")


def stage_final_videos(model: str) -> dict[str, list[pathlib.Path]]:
    """{prompt: [staged files]} with fully descriptive names."""
    staged_root = ROOT / model / "upload_videos_final"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir()
    out: dict[str, list[pathlib.Path]] = {}

    def stage(prompt: str, name: str, run_dir: pathlib.Path) -> None:
        source = newest_video(run_dir)
        if source is None:
            print(f"missing video: {run_dir}")
            return
        target = staged_root / name
        shutil.copy(source, target)
        out.setdefault(prompt, []).append(target)

    results = load(model, "results_5s.json")
    # p0 gallery: dense + every tabled run.
    stage("p0_tokyo", f"{model}_720p_5_seconds_p0_tokyo_dense.mp4",
          ROOT / model / "runs_5s" / "dense")
    for method in FINAL_METHODS:
        for tier, entry in sorted(entries_of(results, method).items(), reverse=True):
            if tier not in TIERS:
                continue
            stage(
                "p0_tokyo",
                f"{model}_720p_5_seconds_p0_tokyo_"
                f"{FILE_LABELS[method]}_target_density_{tier}.mp4",
                ROOT / model / "runs_5s" / f"{method}_{tier}",
            )
    # Per-prompt trio.
    for prompt in PROMPT_KEYS:
        stage(prompt, f"{model}_720p_5_seconds_{prompt}_dense.mp4",
              ROOT / model / "runs_prompts_5s" / f"{prompt}_dense")
        for tag, method in TRIO_TAGS:
            stage(
                prompt,
                f"{model}_720p_5_seconds_{prompt}_"
                f"{FILE_LABELS[method]}_target_density_0.2.mp4",
                ROOT / model / "runs_prompts_5s" / f"{prompt}_{tag}",
            )
    # 30-second long-video runs live under causal_forcing_long.
    if model == "causal_forcing":
        long_root = ROOT / "causal_forcing_long"
        for prompt in ["p0_tokyo"] + PROMPT_KEYS:
            stage(
                prompt,
                f"causal_forcing_long_video_720p_30_seconds_{prompt}_dense.mp4",
                long_root / "runs_30_seconds" / f"{prompt}_dense",
            )
            for tag, method in TRIO_TAGS:
                stage(
                    prompt,
                    f"causal_forcing_long_video_720p_30_seconds_{prompt}_"
                    f"{FILE_LABELS[method]}_target_density_0.2.mp4",
                    long_root / "runs_30_seconds" / f"{prompt}_{tag}",
                )
    return out


def push_videos(model: str) -> None:
    doc = DOCS[model]
    ids = section_ids(doc)
    prompts_h2 = next(k for k in ids if "prompt" in k)
    staged = stage_final_videos(model)
    # Per-prompt sheets + videos are appended after each prompt's paragraph.
    import re as _re

    for prompt_key in ["p0_tokyo"] + PROMPT_KEYS:
        data = cli(
            "docs", "+fetch", "--doc", doc, "--scope", "section",
            "--start-block-id", ids[prompts_h2], "--detail", "with-ids",
        )
        match = _re.search(
            rf'<p id="([^"]+)"><b>{_re.escape(PROMPT_LABELS[prompt_key])}</b>',
            data["document"]["content"],
        )
        if match is None:
            print(f"no anchor paragraph for {prompt_key}")
            continue
        anchor = match.group(1)
        for name in (
            ROOT / model / f"prompt_sheet_5_seconds_{prompt_key}.png",
            ROOT / "causal_forcing_long" / f"prompt_sheet_30_seconds_{prompt_key}.png"
            if model == "causal_forcing" else pathlib.Path("/nonexistent"),
        ):
            if name.exists():
                anchor = insert_after(
                    doc, anchor, str(name),
                    caption=f"{PROMPT_LABELS[prompt_key]} 帧对比"
                    + ("（30 秒）" if "30_seconds" in name.name else "（5 秒）"),
                    width=760,
                )
        for path in staged.get(prompt_key, []):
            anchor = insert_after(doc, anchor, str(path), media_type="file",
                                  file_view="preview")
            print(f"inserted {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, choices=["self_forcing", "causal_forcing"])
    parser.add_argument("--stage", default="all",
                        choices=["all", "text", "figures", "videos"])
    args = parser.parse_args()
    if args.stage in ("all", "text"):
        if args.doc == "self_forcing":
            sf_push_text()
        else:
            rewrite_managed(DOCS["causal_forcing"], cf_body())
    if args.stage in ("all", "figures"):
        push_figures(args.doc)
    if args.stage in ("all", "videos"):
        push_videos(args.doc)
    print("done")


if __name__ == "__main__":
    main()
