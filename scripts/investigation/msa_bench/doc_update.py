# SPDX-License-Identifier: Apache-2.0
"""Publish the MSA implementation + benchmark section into the OSA Properties doc.

    python doc_update.py
"""

import argparse
import json
import pathlib
import re
import shutil
import statistics
import sys
from xml.sax.saxutils import escape

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from doc_media import cli, replace_media  # noqa: E402
from paths import results_dir  # noqa: E402

DOC = "Rs3sdTCinoc6kqxdiGxcUDIQnfd"
ROOT = results_dir("msa_bench")
PROMPTS = json.loads((HERE / "bench_prompts.json").read_text())
METHOD_LABELS = {
    "dense": "dense",
    "msa": "MSA（content 0.20）",
    "msa25": "MSA（content 0.25）",
    "msa10": "MSA（content 0.10）",
    "msasched": "MSA-sched（均值 0.20）",
    "msasched15": "MSA-sched（均值 0.15）",
    "lightforcing": "LightForcing（0.2 档）",
    "lf10": "LightForcing（0.1 档）",
}
# The video gallery, per prompt, in presentation order.
VIDEO_METHODS = ("dense", "msa", "msa10", "lightforcing", "lf10")


def bench_5s_table() -> str:
    results = json.loads((ROOT / "results_5s.json").read_text())
    header = (
        "<th>去噪耗时（5 prompt 均值）</th><th>对 dense 加速</th>"
        "<th>实际累计密度</th><th>PSNR vs dense（均值）</th><th>前 2 秒 PSNR</th>"
    )
    dense_mean = statistics.mean(
        results[f"b{i}_dense_5s"]["denoise_s"] for i in range(1, 6)
    )
    rows = []
    for method in ("dense", "msa", "msa25", "lightforcing"):
        entries = [results[f"b{i}_{method}_5s"] for i in range(1, 6)]
        denoise = statistics.mean(e["denoise_s"] for e in entries)
        cells = [f"<td>{denoise:.2f} s</td><td>{dense_mean / denoise:.2f}×</td>"]
        if method == "dense":
            cells.append("<td>1.0</td><td>—</td><td>—</td>")
        else:
            psnr = statistics.mean(e["psnr"] for e in entries)
            first = statistics.mean(e["psnr_first"] for e in entries)
            cells.append(
                f"<td>{entries[1]['density']:.3f}</td>"
                f"<td>{psnr:.2f}</td><td>{first:.2f}</td>"
            )
        rows.append(
            f"<tr><td><p>{METHOD_LABELS[method]}</p></td>{''.join(cells)}</tr>"
        )
    return (
        '<table><colgroup><col width="190"/><col span="5" width="128"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def per_prompt_table() -> str:
    results = json.loads((ROOT / "results_5s.json").read_text())
    header = "".join(
        f"<th>{pid} · {escape(PROMPTS[pid]['label'])}</th>" for pid in PROMPTS
    )
    rows = []
    for method in ("msa", "msa25", "lightforcing"):
        cells = "".join(
            f"<td>{results[f'{pid}_{method}_5s']['psnr']:.1f}</td>" for pid in PROMPTS
        )
        rows.append(f"<tr><td><p>{METHOD_LABELS[method]}</p></td>{cells}</tr>")
    return (
        '<table><colgroup><col width="190"/><col span="5" width="105"/></colgroup>'
        f"<thead><tr><th>PSNR vs dense</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def bench_20s_table() -> str:
    results = json.loads((ROOT / "results_20s.json").read_text())
    rows = []
    for method in ("dense", "msa", "lightforcing"):
        entry = results[f"b1_{method}_20s"]
        density = entry.get("density")
        rows.append(
            f"<tr><td><p>{METHOD_LABELS[method]}</p></td>"
            f"<td>{entry['denoise_s']:.2f} s</td>"
            f"<td>{density if density else 1.0}</td></tr>"
        )
    return (
        '<table><colgroup><col width="190"/><col span="2" width="150"/></colgroup>'
        "<thead><tr><th></th><th>去噪耗时（b1）</th><th>实际累计密度</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def section_xml() -> str:
    prompt_items = "".join(
        f"<p><b>{pid} · {escape(entry['label'])}</b>：{escape(entry['prompt'])}</p>"
        for pid, entry in PROMPTS.items()
    )
    return (
        "<h2>MSA：混合稀疏注意力的实现与基准</h2>"
        "<p>把上文的策略落成了可运行的 backend（<code>--sparse-attention msa</code>，"
        "commit 7ad05e42e 起）：每个头按离线画像的家族执行——<b>局部头</b>"
        "（r≤2 的静态行窗口，逐帧复制）与<b>短窗头</b>（只读最新 m 帧）零运行时"
        "规划，由 <code>msa_kernel.py</code> 的帧复制 Triton kernel 执行"
        "（帧复制写在循环结构里，静态计划只有几百字节、按 (层, 布局) 缓存）；"
        "<b>内容依赖头</b>走 LightForcing 语义的两阶段 pooled top-k 选择，由"
        "带头索引的 range kernel 执行，且<b>计划每 chunk 只做一次</b>"
        "（首个去噪步测、其余步与 cache 刷新复用——LightForcing 每次调用都要"
        "重新规划）。画像用给定的 5 个 prompt（p1–p5）标定（τ=0.85 需 5 个 "
        "prompt 同时满足），并加执行感知的门槛：行量化后执行密度超过内容头"
        "预算的 local 头（r≥4 时为每帧 22–44%）与统计上不稳的弥散头一律改走"
        "运行时选择——最终 87/360 头静态（34 局部 + 53 短窗）、273 头内容。"
        "校准文件 <code>qk_map_similarity/msa_taxonomy_self_forcing.json</code>，"
        "单测 59 项全过（kernel 对掩码参考逐位校验）。</p>"
        "<h3>基准设置与新 prompt</h3>"
        "<p>画像标定用了 p1–p5，因此基准换用 5 个<b>新写的</b> prompt（标定外），"
        "720p / 5 秒、seed 42、独占 GPU 串行计时；质量为对同 prompt dense 输出的 "
        "PSNR。LightForcing 用 sparse_baselines 标定的 0.2 档配置"
        "（即已发表 5 秒数字背后的设置）。</p>"
        f"{prompt_items}"
        "<h3>结果（720p / 5 秒）</h3>"
        f"{bench_5s_table()}"
        f"{per_prompt_table()}"
        "<p><b>结论：</b>MSA（content 0.25）在质量与 LightForcing 相当"
        "（PSNR 均值 17.72 vs 17.86，逐 prompt 差 ≤0.7 dB）的同时略快"
        "（8.94 vs 9.00 s）；MSA（content 0.20）再快 3%（8.73 s，比 dense 快 "
        "1.22×），代价是均值低 ~0.85 dB。两档读取的 key 都少于 LightForcing"
        "（0.337 / 0.369 vs 0.357）。</p>"
        "<h3>20 秒计时与已知短板</h3>"
        f"{bench_20s_table()}"
        "<p>20 秒档 MSA（33.2 s）仍慢于 LightForcing（29.0 s）：静态头的帧复制"
        "图案在 81 个可见帧下退化为大量短 key 走查（局部头每帧 4–6 个 tile），"
        "kernel 吞吐 ~250–320 TFLOP/s，低于合并长 range 的 ~500；LightForcing "
        "的两阶段资格约束天然把保留块聚拢成长 range。修复方向（未实现）："
        "静态头改 gather-复制执行（OSA 曾验证 gather + FA varlen ~607 TFLOP/s）、"
        "或 query 置换后做真正的 2-D 窗口。开发过程中被基准推翻的两个预设也值得"
        "记录：纯全局 top-k 的内容选择比两阶段版本执行慢 ~2×（散块不合并）；"
        "弥散头的帧抽样在短视频上有统计偏差（b5 PSNR 掉 2.8 dB），最终并入"
        "运行时选择。</p>"
        '<pre lang="bash" caption="复现命令"><code>'
        + escape(
            "# 画像标定（需 p1-p5 的 sweep 捕获，见上文）+ 导出 backend 校准文件\n"
            "cd scripts/investigation/qk_map_similarity\n"
            "CUDA_VISIBLE_DEVICES=<idle> python taxonomy_sweep.py \\\n"
            "  --runs p1_sweep,p2_sweep,p3_sweep,p4_sweep,p5_sweep --export msa_taxonomy_self_forcing.json\n"
            "cd ../msa_bench\n"
            "python run_bench.py                                  # dense/msa/lightforcing x b1-b5, 5s\n"
            "python run_bench.py --methods msa25                  # content 0.25 档\n"
            "python run_bench.py --seconds 20 --prompts b1        # 20s 计时\n"
            "python doc_update.py\n"
            "# 单测\n"
            "PYTHONPATH=python python -m pytest \\\n"
            "  python/sglang/multimodal_gen/test/unit/realtime/test_sparse_attention.py -q"
        )
        + "</code></pre>"
    )


def tier_table(methods: tuple[str, ...]) -> str:
    results = json.loads((ROOT / "results_5s.json").read_text())
    dense_mean = statistics.mean(
        results[f"b{i}_dense_5s"]["denoise_s"] for i in range(1, 6)
    )
    header = (
        "<th>去噪耗时（均值）</th><th>对 dense 加速</th><th>实际累计密度</th>"
        "<th>PSNR vs dense（均值）</th><th>前 2 秒 PSNR</th>"
    )
    rows = []
    for method in methods:
        entries = [results[f"b{i}_{method}_5s"] for i in range(1, 6)]
        denoise = statistics.mean(e["denoise_s"] for e in entries)
        psnr = statistics.mean(e["psnr"] for e in entries)
        first = statistics.mean(e["psnr_first"] for e in entries)
        rows.append(
            f"<tr><td><p>{METHOD_LABELS[method]}</p></td>"
            f"<td>{denoise:.2f} s</td><td>{dense_mean / denoise:.2f}×</td>"
            f"<td>{entries[1]['density']:.3f}</td>"
            f"<td>{psnr:.2f}</td><td>{first:.2f}</td></tr>"
        )
    return (
        '<table><colgroup><col width="190"/><col span="5" width="128"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def videos_xml() -> str:
    parts = ["<h3>0.1 档对比与原始视频</h3>"]
    parts.append(
        "<p>补充 0.1 档（MSA content 0.10、LightForcing 0.1 档标定配置）："
        "下表为 5 个基准 prompt 的均值，之后按 prompt 给出全部原始视频"
        "（顺序：dense、MSA 0.2、MSA 0.1、LightForcing 0.2、LightForcing 0.1）。</p>"
    )
    parts.append(tier_table(("msa", "msa10", "lightforcing", "lf10")))
    results = json.loads((ROOT / "results_5s.json").read_text())
    for pid, entry in PROMPTS.items():
        parts.append(f"<p><b>{pid} · {escape(entry['label'])}</b></p>")
        for method in VIDEO_METHODS:
            record = results.get(f"{pid}_{method}_5s", {})
            note = ""
            if record.get("psnr") is not None:
                note = f"（PSNR {record['psnr']:.1f}）"
            parts.append(
                f"<p>{METHOD_LABELS[method]}{note}</p>"
                f"<p>[[video:{pid}_{method}]]</p>"
            )
    return "".join(parts)


def newest_mp4(tag: str) -> pathlib.Path:
    candidates = sorted(
        (ROOT / "runs" / tag / "outputs").glob("*.mp4"), key=lambda p: p.stat().st_mtime
    )
    assert candidates, f"no video for {tag}"
    return candidates[-1]


def stage_videos() -> None:
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "0.1 档对比与原始视频" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "append",
            "--content", videos_xml(),
        )
        data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
        content = data["document"]["content"]
    placeholders = dict(
        re.findall(r'<p id="([^"]+)"[^>]*>\[\[video:([\w.]+)\]\]</p>', content)
    )
    placeholders = {name: block for block, name in placeholders.items()}
    for pid in PROMPTS:
        for method in VIDEO_METHODS:
            key = f"{pid}_{method}"
            if key not in placeholders:
                continue
            source = newest_mp4(f"{pid}_{method}_5s")
            named = source.parent / f"self_forcing_720p_5s_{key}.mp4"
            shutil.copyfile(source, named)
            replace_media(DOC, placeholders[key], str(named), media_type="file")
    print("[msa-doc] tier table + videos published")


def profiling_xml() -> str:
    summary = json.loads((ROOT / "profiling" / "summary.json").read_text())["runs"]

    def row(method: str, seconds: int) -> str:
        record = summary[f"{method}_{seconds}s"]
        return (
            f"<tr><td><p>{METHOD_LABELS[method]}</p></td><td>{seconds} s</td>"
            f"<td>{record['self_attn_denoise_total_ms'] / 1000:.2f} s</td>"
            f"<td>{record['self_attn_cache_update_total_ms'] / 1000:.2f} s</td>"
            f"<td>{record['forward_denoise_total_ms'] / 1000:.2f} s</td></tr>"
        )

    table = (
        '<table><colgroup><col width="190"/><col width="70"/>'
        '<col span="3" width="150"/></colgroup>'
        "<thead><tr><th></th><th>时长</th><th>自注意力（去噪步合计）</th>"
        "<th>自注意力（cache 刷新）</th><th>DiT forward 合计</th></tr></thead>"
        "<tbody>"
        + "".join(row(m, s) for s in (5, 20) for m in ("dense", "msa", "lightforcing"))
        + "</tbody></table>"
    )
    micro = json.loads((ROOT / "profiling" / "micro.json").read_text())
    m21, m81 = micro["21"], micro["81"]
    return (
        "<h3>运行时剖析：MSA vs LightForcing</h3>"
        "<p>两个视角。<b>真实运行</b>（chunk-timing 探针，CUDA event 包住每个注意力"
        "模块，b1、独占 GPU）：下图为每 chunk 的自注意力耗时曲线，下表为合计。"
        "<b>逐调用 microbenchmark</b>（第 14 层：2 静态 + 10 内容头，合成形状、"
        "独占 GPU）：按组件拆分单次 attend() 的去向。</p>"
        "<p>[[map:profile_curves]]</p>"
        f"{table}"
        "<p>[[map:profile_components]]</p>"
        "<p><b>发现一：5 秒档两者的注意力时间几乎相同，MSA 的端到端优势来自"
        "规划的摊销。</b>去噪步自注意力合计 MSA 3.24 s vs LF 3.17 s、DiT forward "
        "合计 7.29 vs 7.26 s——差距不在 GPU 上；e2e 去噪 8.73 vs 9.00 s 的差主要是"
        " forward 之外的主机侧开销（MSA 1.44 s vs LF 1.74 s）：LF 每层每步都要"
        "发起 pool + 打分 + top-k 的一串小算子（实测每调用 "
        f"{m21['lf_plan_ms']:.2f} ms），MSA 每 chunk 只做一次"
        f"（计划命中时整调用 {m21['msa_attend_hit_ms']:.2f} ms vs 未命中 "
        f"{m21['msa_attend_miss_ms']:.2f} ms），cache 刷新 forward 也复用计划"
        "（其注意力 0.39 vs 0.50 s）。</p>"
        "<p><b>发现二：20 秒档的差距不是 kernel 质量，而是密度调度策略。</b>"
        "microbenchmark 里 MSA 的静态头成本随上下文<b>恒定</b>"
        f"（{m21['msa_static_kernel_ms']:.2f} ms，从 6 帧到 81 帧不变——短窗头"
        "只读最新 m 帧的设计兑现了）；内容头 kernel 则随可见帧数线性增长"
        f"（21 帧 {m21['msa_content_kernel_ms']:.2f} ms → 81 帧 "
        f"{m81['msa_content_kernel_ms']:.2f} ms），因为 MSA 的 content_density "
        "是每调用恒定的 0.2；而 LightForcing 的 chunk 感知 sparsity 调度让晚期 "
        "chunk 越来越稀（其累计密度日志从 0.50 一路降到 0.20），81 帧时它每调用"
        "读的 key 少得多（exec kernel "
        f"{m81['lf_exec_kernel_ms']:.2f} ms vs MSA 内容头 "
        f"{m81['msa_content_kernel_ms']:.2f} ms）。真实运行曲线一致：20 秒档 LF "
        "每 chunk 注意力近乎平坦（300→740 ms），MSA 线性上升（400→1130 ms）。"
        "≤21 帧（整个 5 秒档）MSA 每调用反而更快。</p>"
        "<p><b>含义：</b>把 MSA 内容头的密度做成 chunk 级调度（如 OSA 曾用的 "
        "1/√kv 前置递减，或 LF 式随进度衰减）即可在不动 kernel 的情况下抹平 "
        "20 秒档的大部分差距——静态头已经天然平坦。</p>"
        '<pre lang="bash" caption="复现命令"><code>'
        + escape(
            "cd scripts/investigation/msa_bench\n"
            "python profile_runtime.py --stage runs   # b1 x {dense,msa,lf} x {5s,20s}, chunk-timing 探针\n"
            "python profile_runtime.py --stage micro  # 独占 GPU 逐调用组件拆分\n"
            "python profile_runtime.py --stage report # 曲线/组件图 + summary.json\n"
            "python doc_update.py --stage profiling"
        )
        + "</code></pre>"
    )


def schedule_xml() -> str:
    results_20s = json.loads((ROOT / "results_20s.json").read_text())

    def cell_20s(method: str) -> str:
        record = results_20s[f"b1_{method}_20s"]
        return f"{record['denoise_s']:.2f} s（密度 {record.get('density', 1.0)}）"

    return (
        "<h3>调度修复的验证：content 头的 chunk 级密度调度</h3>"
        "<p>按剖析的结论落地：<code>content_schedule=\"flops_matched\"</code> 给 "
        "content 头按 OSA 的 <latex>\\mathrm{floor} + \\beta/\\sqrt{kv}</latex> "
        "前置递减调度（β 解至 kv 加权均值恰等于 content_density，静态头本就"
        "平坦不参与），计划缓存按 (层, chunk) 不变。两个预期都得到验证，"
        "外加一个当初没想到的教训。</p>"
        f"{tier_table(('msa', 'msasched', 'msasched15', 'lightforcing'))}"
        "<p><b>一，同 FLOPs 下调度只买质量不买速度</b>——flops_matched 保持 kv "
        "加权均值不变，20 秒耗时纹丝不动（33.27 vs 33.18 s）；这修正了剖析节"
        "结论的表述：调度本身不省时间，省时间靠“调度买到的质量余量换更低的"
        "均值”。<b>二，质量增益实打实</b>：均值 0.20 不变，PSNR 17.01 → 17.72，"
        "追平 LightForcing（17.86）且更快（8.81 vs 9.00 s）。<b>三，兑现为"
        "速度</b>：均值降到 0.15 后质量仍高于未调度的 0.20（17.15 vs 17.01），"
        f"5 秒 8.48 s（比 LF 快 6%），20 秒 {cell_20s('msasched15')}——与 "
        f"LightForcing 的 {cell_20s('lightforcing')} 差距从 14% 收敛到 2%。"
        "MSA-sched 0.15 现在是推荐配置：全时长不慢于 LightForcing，5 秒档"
        "更快，质量介于 LF 0.2 档与 0.1 档之间、显著优于同速的 LF 档位。</p>"
        '<pre lang="bash" caption="复现命令"><code>'
        + escape(
            "cd scripts/investigation/msa_bench\n"
            "python run_bench.py --methods msasched            # 均值 0.2，质量验证\n"
            "python run_bench.py --methods msasched15          # 均值 0.15，速度验证\n"
            "python run_bench.py --methods msasched15 --seconds 20 --prompts b1\n"
            "python doc_update.py --stage schedule"
        )
        + "</code></pre>"
    )


def stage_schedule() -> None:
    data = cli("docs", "+fetch", "--doc", DOC)
    if "调度修复的验证" in data["document"]["content"]:
        print("[msa-doc] schedule subsection already present")
        return
    cli(
        "docs", "+update", "--doc", DOC, "--command", "append",
        "--content", schedule_xml(),
    )
    print("[msa-doc] schedule subsection appended")


def stage_profiling() -> None:
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "运行时剖析：MSA vs LightForcing" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "append",
            "--content", profiling_xml(),
        )
        data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
        content = data["document"]["content"]
    placeholders = dict(
        re.findall(r'<p id="([^"]+)"[^>]*>\[\[map:([\w.]+)\]\]</p>', content)
    )
    placeholders = {name: block for block, name in placeholders.items()}
    for name in ("profile_curves", "profile_components"):
        if name in placeholders:
            replace_media(
                DOC, placeholders[name], str(ROOT / "profiling" / f"{name}.png")
            )
    print("[msa-doc] profiling subsection published")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="section",
        choices=["section", "videos", "profiling", "schedule"],
    )
    args = parser.parse_args()
    if args.stage == "videos":
        stage_videos()
        return
    if args.stage == "profiling":
        stage_profiling()
        return
    if args.stage == "schedule":
        stage_schedule()
        return
    data = cli("docs", "+fetch", "--doc", DOC)
    if "MSA：混合稀疏注意力" in data["document"]["content"]:
        print("[msa-doc] section already present")
        return
    cli("docs", "+update", "--doc", DOC, "--command", "append", "--content", section_xml())
    print("[msa-doc] section appended")


if __name__ == "__main__":
    main()
