# SPDX-License-Identifier: Apache-2.0
"""Push the Self-Forcing 5 s study into the doc (tables, figures, videos).

The 2026-08-26 rerun at the model's trained duration: 720p / 5 s, dense + all
baselines + the OSA anchor progression (osa2 -> osa2s -> osa2a -> osa),
tiers 0.1/0.2/0.3 only. Sections 1-3 and the p0 video gallery become the 5 s
study; the kernel-optimization history, the 3.5 shake analysis and the
multi-prompt validation stay as clearly-marked 20 s material.

    python doc_update_sf5s.py [--stage text|figures|videos]
"""

import argparse
import json
import pathlib
import shutil
import sys

from common import METHOD_LABELS, ROOT, newest_video
from doc_update_self_forcing import (
    DOC,
    find_block,
    replace_span,
    top_blocks,
)
from sections import bullets, esc, table

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import cli, insert_after, replace_media  # noqa: E402

MODEL = "self_forcing"
MODEL_ROOT = ROOT / MODEL
TIERS = ["0.3", "0.2", "0.1"]  # display order, denser first
# OSA anchor progression first (additive labels), then the baselines.
ORDER = [
    "osa2",
    "osa2s",
    "osa2a",
    "osa",
    "osasched",
    "lightforcing",
    "radial",
    "svg1",
    "svg2",
    "xattention",
    "sta",
]

# Hand-written visual reads from the three 5 s sheets (2026-08-26).
QUALITY_BULLETS: list[str] = [
    "实际密度 ≥0.37（knob ≥0.27）：裸 OSA 与 OSA + sink full 干净且贴近 "
    "dense；OSA 全锚点（0.62）与 dense 几乎不可区分。",
    "0.28–0.29 档：裸 OSA 主体与街景完好，但画面中部出现一条位置固定的彩色噪声"
    "带；+ sink full 前 1–2 s 涂抹、之后恢复出主体。",
    "最低档（knob 0.05–0.06，实际 0.19–0.24）：裸 OSA 下半幅彩色噪声，+ sink "
    "full 整段涂抹。注意校准口径：knob 是按 20 s 累计密度标定的，5 s 下同一 "
    "knob 的<b>逐调用</b>密度只有 5–6%，远低于表中的实际累计值。",
    "<b>OSA + sink full + recent full 在 5 s 全档最差</b>（整段涂抹，PSNR 低于"
    "裸 OSA）：早期 chunk 只有 4–6 帧可见，sink + recent 两帧整帧保留把密度预算"
    "吃光，自身 chunk（3 帧）的 pattern tile 被挤到饥饿。20 s 轮「锚点修晃动」的"
    "结论在短窗前段不适用；own chunk full 全开的经典几何不受影响。",
    "基线：LightForcing 全档干净（chunk-aware 调度让早期 chunk 天然更稠），是 "
    "0.22 实际密度下唯一干净的行，也是本轮<b>最快的画质完好配置</b>（8.1 s，"
    "1.32×）；Radial / SVG1 干净但下限高（0.31–0.50）；SVG2 内容漂移（街景变亮"
    "变宽，主体尚在）；XAttention 0.21 档下半幅噪声，STA 0.25 档大面积噪声。",
    "口径提醒：5 s 稠密去噪只有 10.7 s（20 s 为 73.4 s），注意力占比大幅缩水，"
    "所有方法的加速上限被压到 ~1.4×；密度—速度曲线的对比意义大于绝对加速比。",
    "<b>OSA + demand schedule</b>（无整帧豁免 + 1/√kv 前置调度 + 需求加权逐帧"
    "分配 + <b>平坦行稠密回退</b>：质量分布平坦、frozen top-k 无峰可选的 query "
    "行按该 chunk 的实际预算判定后回退为全量注意力，预算充足时自动失效）："
    "0.43 / 0.34 档干净且快于 LightForcing 同档（0.34 档 8.5 vs 8.9 s）；"
    "0.28 档（knob 0.1 + 回退）主体与街景保持、地面区域仍有残余噪声——"
    "per-call 0.1 低于冻结模式的可用下限，回退只救回最差 ~4% 行，阈值再高"
    "则密度失控（0.65 阈值 → 0.50 密度）。<b>5 s 干净下限 ≈ knob 0.15</b>："
    "8.1 s @ 0.29，与 LightForcing 最快干净档（8.1 s @ 0.22）持平；更低密度"
    "属于逐步重估计方法（LF）的结构性优势区。20 s 下 osasched 在匹配密度处"
    "全面更快（0.29 档 32.9 s vs LF 0.30 档 37.4 s；knob 0.1 达 2.97×），"
    "注意力占比越大优势越大。",
]


def load() -> dict:
    return json.loads((MODEL_ROOT / "results_5s.json").read_text())


def method_entries(results: dict, method: str) -> dict[str, dict]:
    """{tier: entry} for one method's 5 s runs."""
    out = {}
    for tag, entry in results.items():
        if not tag.startswith(f"{method}_") or entry.get("returncode") != 0:
            continue
        out[tag.rsplit("_", 1)[1]] = entry
    return out


def setup_section() -> str:
    return bullets(
        [
            "<b>模型</b>：Self-Forcing 1.3B 全上下文（fullctx 转档），单卡 H200。",
            "<b>配置</b>：720p（1280×720）/ <b>5 s</b>（81 像素帧 = 21 潜帧 = 7 "
            "chunk，末 chunk 上下文约 7.6 万 token）——5 s 是 Self-Forcing 的"
            "训练时长，本轮所有结论都在分布内。统一 prompt（p0 · 东京夜街，见第 4 "
            "节）、seed 42。",
            "<b>方法</b>：dense、<b>OSA 锚点递进</b>（裸 OSA → + sink full → "
            "+ sink full + recent full → + own chunk full + sink full + recent "
            "full，即整帧豁免开关从全关逐个打开）、LightForcing、Radial、SVG1、"
            "SVG2、XAttention、STA；目标读取密度 <b>0.1 / 0.2 / 0.3</b>。",
            "<b>匹配口径</b>：整段生成的<b>实际累计读取密度</b>（回退 dense 的调用"
            "按 1.0 计）。5 s 只有 7 个 chunk：OSA 的 chunk-0 稠密校准与各基线的 "
            "ramp-up 稠密段在累计中占比远高于 20 s，因此实际累计密度明显高于目标"
            "档，表中以实际值为准。",
            "<b>校准</b>：各方法 knob 沿用 480p / 20 s 割线校准（同一套 "
            "configs.json；STA 在 720p 上按密度阶梯校准），本轮只取 0.1 / 0.2 / "
            "0.3 三档配置在 5 s 下重跑。",
            "<b>上一轮</b>：720p / 20 s 全量结果（0.1–0.5 档 + 多 prompt）见本文档"
            "历史（2026-08-24 版）；第 4 节的多 prompt 表与 3.5 晃动分析仍来自该轮。",
        ]
    )


def walltime_section(results: dict) -> str:
    dense = results["dense"]
    dense_denoise = dense["denoise_s"]

    def ref_time(method: str) -> float:
        entries = method_entries(results, method)
        if not entries:
            return 1e9
        entry = entries.get("0.3") or min(
            entries.values(), key=lambda e: abs(e.get("density", 1.0) - 0.30)
        )
        return entry["denoise_s"]

    rows = [["Dense", f"{dense_denoise:.1f} s（1.00）1.00×", "—", "—"]]
    for method in sorted(ORDER, key=ref_time):
        entries = method_entries(results, method)
        if not entries:
            continue
        cells = [METHOD_LABELS[method]]
        for tier in TIERS:
            entry = entries.get(tier)
            if entry is None:
                cells.append("—")
                continue
            speedup = dense_denoise / entry["denoise_s"]
            cells.append(
                f"{entry['denoise_s']:.1f} s（{entry.get('density', 1.0):.2f}）"
                f"{speedup:.2f}×"
            )
        rows.append(cells)

    lead = (
        f"<p>720p / 5 s 稠密参考：去噪耗时 {dense_denoise:.1f} s"
        + (f"，端到端 {dense['e2e_s']:.1f} s" if "e2e_s" in dense else "")
        + "。下表为各方法在 0.3 / 0.2 / 0.1 目标档的<b>去噪耗时（加速比）</b>，"
        "括号内为实际累计密度；同一方法多档收敛到同一实际密度（校准下限高于目标）"
        "时只保留一行有数：</p>"
    )
    walltime_bullets = [
        "同实际密度下 OSA 递进行最快（7.6–8.8 s）；裸 OSA 与 + sink full 几乎"
        "重合——sink 一帧的整帧成本可忽略；+ recent full 把 5 s 下限抬到 ~0.32。",
        "OSA 全锚点（own chunk + sink + recent 全开）在 5 s 只有一个有效档：7 个 "
        "chunk 里稠密校准与整帧豁免占比过高，实际密度钉在 0.62（1.11×）。",
        "估计类基线在 5 s 形状上接近或低于收支平衡：SVG1 0.3 档 0.93×、SVG2 全档 "
        "0.94–1.07×（每步估计开销摊不薄）；STA 的 tile 窗在短上下文里夹不动，"
        "0.3/0.2 档实际密度停在 0.46–0.48。",
    ]
    return (
        lead
        + table(["方法"] + [f"~{t} 档" for t in TIERS], rows)
        + bullets(walltime_bullets)
    )


def quality_section(results: dict) -> str:
    lead = (
        "<p><b>PSNR（相对 dense 输出，dB）：</b>5 s 在训练分布内，dense 参考轨迹"
        "本身干净，PSNR 直接可比（20 s 轮中 dense 自身后段漂移的免责不再需要）。"
        "括号内为实际累计密度：</p>"
    )
    rows = []
    for method in ORDER:
        entries = method_entries(results, method)
        cells = [METHOD_LABELS[method]]
        seen = False
        for tier in TIERS:
            entry = entries.get(tier)
            if entry is None or "psnr_overall_db" not in entry:
                cells.append("—")
                continue
            seen = True
            cells.append(
                f"{entry['psnr_overall_db']:.1f}（{entry.get('density', 0):.2f}）"
            )
        if seen:
            rows.append(cells)
    body = table(["方法"] + [f"~{t} 档" for t in TIERS], rows)
    inspect = (
        "<p><b>视觉检查</b>（帧对比图见下，0.3 / 0.2 / 0.1 三档各一张；"
        "行：方法；列：帧号 / 时间）：</p>"
    )
    return lead + body + inspect + (bullets(QUALITY_BULLETS) if QUALITY_BULLETS else "")


def commands_section() -> str:
    return (
        "<p>本轮（720p / 5 s，0.1–0.3 档）一键链（校准 osa2s → replay 校验 → "
        "sweep → 质量 + 三张 sheet）：</p>"
        "<pre lang=\"bash\"><code>cd scripts/investigation/sparse_baselines\n"
        "GPUS=6 bash sf_5s_study.sh</code></pre>"
        "<p>逐步等价命令：</p>"
        "<pre lang=\"bash\"><code>python calibrate.py --model self_forcing "
        "--methods osa2s --gpus 6 --workers 1\n"
        "python run_sweep.py --model self_forcing \\\n"
        "  --methods osa osa2 osa2s osa2a lightforcing radial svg1 svg2 "
        "xattention sta \\\n"
        "  --tiers 0.1 0.2 0.3 --duration 5 --res 720p \\\n"
        "  --out results_5s.json --runs-dir runs_5s --gpus 6 --workers 1\n"
        "python quality.py --model self_forcing --duration 5 "
        "--out results_5s.json --runs-dir runs_5s --tier 0.3 --sheet-suffix _5s\n"
        "python plot.py --model self_forcing --results results_5s.json "
        "--suffix _5s --duration 5\n"
        "python doc_update_sf5s.py</code></pre>"
        "<p>20 s 轮的复现命令见文档历史（2026-08-24 版）。</p>"
    )


def push_text() -> None:
    results = load()
    replace_span(
        DOC,
        after="1. 实验设置",
        before="OSA 实现",
        xml=setup_section(),
    )
    replace_span(
        DOC,
        after="2. Walltime",
        before="优化过程",
        xml=walltime_section(results),
    )
    replace_span(
        DOC,
        after="3. 生成质量",
        before="quality_sheet",
        xml=quality_section(results),
    )
    # 复现命令 is the doc's last section: replace everything after its h2.
    blocks = top_blocks(DOC)
    start = next(
        i for i, (t, _, x) in enumerate(blocks) if t == "h2" and "复现命令" in x
    )
    doomed = [bid for _, ids, _ in blocks[start + 1 :] for bid in ids]
    cli(
        "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
        "--block-id", blocks[start][1][-1], "--content", commands_section(),
    )
    for i in range(0, len(doomed), 20):
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_delete",
            "--block-id", ",".join(doomed[i : i + 20]),
        )


def push_figures() -> None:
    """Replace each published figure with the current file of the same role."""
    targets = [
        (
            ("walltime_vs_density_5s.png", "walltime_vs_density.png"),
            MODEL_ROOT / "walltime_vs_density_5s.png",
            "各方法去噪耗时 vs 实际累计读取密度（720p / 5 s；虚线为 dense 参考）",
        ),
    ] + [
        (
            (f"quality_sheet_target{tier}_5s.png", f"quality_sheet_target{tier}.png"),
            MODEL_ROOT / f"quality_sheet_target{tier}_5s.png",
            f"帧对比（p0 · 东京夜街，~{tier} 档；行：方法；列：帧号 / 时间）",
        )
        for tier in ("0.3", "0.2", "0.1")
    ]
    for names, path, caption in targets:
        blocks = top_blocks(DOC)
        old = None
        for name in names:
            try:
                old = blocks[find_block(blocks, text=name, tag="img")][1][0]
                break
            except LookupError:
                continue
        if old is None:
            print(f"no published figure matching {names}; skipping")
            continue
        replace_media(DOC, old, str(path), caption=caption, width=760)
        print(f"replaced {path.name}")


# Variant keys are internal shorthand; files carry the additive switch names.
FILE_LABELS = {
    "osa2": "osa",
    "osa2s": "osa_sink_full",
    "osa2a": "osa_sink_full_recent_full",
    "osa": "osa_own_chunk_full_sink_full_recent_full",
    "osasched": "osa_demand_schedule",
}


def stage_videos() -> list[tuple[str, pathlib.Path]]:
    """[(tag, staged path)] for dense + every 5 s run, in gallery order."""
    staged_dir = MODEL_ROOT / "upload_videos_5s"
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir()
    results = load()
    tags = ["dense"] + [
        f"{method}_{tier}"
        for method in ORDER
        for tier in TIERS
        if f"{method}_{tier}" in results
        and results[f"{method}_{tier}"].get("returncode") == 0
    ]
    out = []
    for tag in tags:
        source = newest_video(MODEL_ROOT / "runs_5s" / tag)
        if source is None:
            print(f"missing video: {tag}")
            continue
        if tag == "dense":
            label = "dense"
        else:
            method, tier = tag.rsplit("_", 1)
            label = f"{FILE_LABELS.get(method, method)}_{tier}"
        target = staged_dir / f"self_forcing_5s_{label}.mp4"
        shutil.copy(source, target)
        out.append((tag, target))
    return out


def push_videos() -> None:
    """Replace the old 20 s p0 gallery with the 5 s per-run gallery."""
    blocks = top_blocks(DOC)
    # Old p0 videos are the figure blocks named self_forcing_p0_tokyo_*.
    doomed = [
        ids[0]
        for tag, ids, text in blocks
        if tag == "figure"
        and ("self_forcing_p0_tokyo_" in text or "self_forcing_5s_" in text)
    ]
    for block_id in doomed:
        cli("docs", "+update", "--doc", DOC, "--command", "block_delete",
            "--block-id", block_id)
    print(f"dropped {len(doomed)} old p0 videos")

    blocks = top_blocks(DOC)
    anchor_index = find_block(blocks, text="p0 · 东京夜街")
    anchor = blocks[anchor_index][1][0]
    # Refresh the p0 lead paragraph to describe the 5 s gallery.
    cli(
        "docs", "+update", "--doc", DOC, "--command", "block_replace",
        "--block-id", anchor,
        "--content",
        "<p><b>p0 · 东京夜街</b>（主实验 prompt，帧对比图见第 3 节）——以下为本轮 "
        "720p / 5 s 的全部原始视频：dense 与每个方法在 0.3 / 0.2 / 0.1 档的输出"
        "（文件名 = 方法_目标档；多档收敛到同一配置的只出现一次）：<br/>"
        + esc(
            "A stylish woman walks down a Tokyo street filled with warm "
            "glowing neon and animated city signage. She wears a black "
            "leather jacket, a long red dress, and black boots, and carries a "
            "black purse. She wears sunglasses and red lipstick. She walks "
            "confidently and casually. The street is damp and reflective, "
            "creating a mirror effect of the colorful lights. Many "
            "pedestrians walk about."
        )
        + "</p>",
    )
    blocks = top_blocks(DOC)
    anchor = blocks[find_block(blocks, text="p0 · 东京夜街")][1][0]
    for tag, path in stage_videos():
        anchor = insert_after(DOC, anchor, str(path), media_type="file",
                              file_view="preview")
        print(f"inserted {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", default="all", choices=["all", "text", "figures", "videos"]
    )
    args = parser.parse_args()
    if args.stage in ("all", "text"):
        push_text()
    if args.stage in ("all", "figures"):
        push_figures()
    if args.stage in ("all", "videos"):
        push_videos()
    print("done")


if __name__ == "__main__":
    main()
