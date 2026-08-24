# SPDX-License-Identifier: Apache-2.0
"""Build one model's Feishu section XML from its results.

The five model docs share a structure (实验设置 / Walltime / 生成质量 /
多 prompt 验证与原始视频 / 复现命令); everything numeric is generated from
``results.json`` / ``results_prompts.json`` / ``configs.json`` so the prose and
the tables cannot drift apart. The only hand-written parts are the per-model
notes in :mod:`notes` — the visual read of the frame sheets, which no script
can produce.
"""

import html
import json
import pathlib

from common import METHOD_LABELS, METHODS, MODELS, ROOT

from notes import MODEL_NOTES, PROMPT_LABELS

TIERS = ["0.5", "0.4", "0.3", "0.2", "0.1"]


def esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def load(model: str) -> tuple[dict, dict, dict]:
    root = ROOT / model
    results = json.loads((root / "results.json").read_text())
    prompts_path = root / "results_prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}
    configs = json.loads((ROOT / "configs.json").read_text())[model]
    return results, prompts, configs


def method_rows(results: dict, method: str) -> dict[str, dict]:
    """{tier: entry} of a method's successful sweep runs."""
    out = {}
    for tag, entry in results.items():
        if not tag.startswith(f"{method}_") or entry.get("returncode") != 0:
            continue
        out[tag.split("_", 1)[1]] = entry
    return out


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th><p>{esc(h)}</p></th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td><p>{cell}</p></td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def setup_section(model: str, results: dict, configs: dict) -> str:
    spec = MODELS[model]
    notes = MODEL_NOTES[model]
    tiers = [t for t in TIERS if any(t in configs[m] for m in METHODS)]
    width, height = spec["resolutions"]["720p"]
    duration = 20
    frames = spec["frames"][duration]
    latents = spec["latents_20s"]
    per_chunk = 8 if model == "longlive2" else 3
    floors = []
    for method in METHODS:
        achieved = [
            entry["density"]
            for entry in method_rows(results, method).values()
            if "density" in entry
        ]
        if achieved:
            floors.append((METHOD_LABELS[method], min(achieved)))
    floors.sort(key=lambda item: item[1])
    floor_text = "、".join(f"{name} ≈ {value:.2f}" for name, value in floors)

    items = [
        f"<b>模型</b>：{notes['model_line']}，单卡 H200。",
        (
            f"<b>配置</b>：720p（{width}×{height}）/ {duration} s"
            f"（{frames} 像素帧 = {latents} 潜帧 = {latents // per_chunk} chunk，"
            f"{notes['geometry_line']}），统一 prompt（p0 · 东京夜街，见第 4 节）、seed 42。"
        ),
        (
            "<b>方法</b>：dense、OSA、LightForcing、Radial、SVG1、SVG2、"
            "XAttention、STA，目标读取密度 " + " / ".join(tiers) + "。"
        ),
        (
            "<b>匹配口径</b>：整段生成的<b>实际累计读取密度</b>"
            "（后端采样统计上报，回退 dense 的调用按 1.0 计）。"
        ),
        (
            "<b>校准</b>：各基线的稀疏度控制参数（threshold / band_frames / "
            "top_p / decay_factor / sparsity）与 OSA 的 density 参数，均用 "
            "480p / 20 s（同 chunk 轨迹）割线搜索校准到各档目标；"
            "<b>STA 的控制参数是离散的 (kernel_t, kernel_h, kernel_w) tile 核</b>，"
            "空间占比由 tile 网格决定、随分辨率改变，因此按密度阶梯直接在 "
            "720p 上校准。"
        ),
        f"<b>实际密度下限</b>：{floor_text}。",
    ]
    if notes.get("setup_extra"):
        items += notes["setup_extra"]
    return "<h2>1. 实验设置</h2>" + bullets(items)


def walltime_section(model: str, results: dict) -> str:
    notes = MODEL_NOTES[model]
    dense = results["dense"]
    dense_denoise = dense["denoise_s"]
    metric = notes["time_metric"]
    tiers = [t for t in TIERS if any(t in method_rows(results, m) for m in METHODS)]

    rows = []
    ordered = sorted(
        METHODS,
        key=lambda m: min(
            (e["denoise_s"] for e in method_rows(results, m).values()), default=1e9
        ),
    )
    for method in ordered:
        entries = method_rows(results, method)
        if not entries:
            continue
        cells = [METHOD_LABELS[method]]
        for tier in tiers:
            entry = entries.get(tier)
            if entry is None:
                cells.append("—")
                continue
            seconds = entry["denoise_s"]
            density = entry.get("density", 1.0)
            speedup = dense_denoise / seconds
            cells.append(f"{seconds:.1f} s（{density:.2f}）{speedup:.2f}×")
        rows.append(cells)

    header = ["方法"] + [f"~{float(t):.2f} 档" for t in tiers]
    lead = (
        f"<p>720p / 20 s 稠密参考：{metric} {dense_denoise:.1f} s"
        + (f"，端到端 {dense['e2e_s']:.1f} s" if "e2e_s" in dense else "")
        + f"。下表为各方法在各密度档的<b>{metric}（加速比）</b>，括号内为实际达到的密度：</p>"
    )
    return (
        "<h2>2. Walltime</h2>"
        + lead
        + table(header, rows)
        + bullets(notes["walltime_bullets"])
    )


def quality_section(model: str, results: dict) -> str:
    notes = MODEL_NOTES[model]
    lead = (
        "<p><b>PSNR（相对 dense 输出，dB）：</b>自回归 rollout 中任何扰动都会"
        "复合成内容轨迹分歧，因此该指标衡量的是“偏离 dense 轨迹的速度”而非画质"
        "本身；前 5 s（轨迹尚未分开）更有参考价值。下表为各方法 ~0.30 档"
        "（括号内为实际密度）：</p>"
    )
    rows = []
    for method in METHODS:
        entries = method_rows(results, method)
        candidates = [
            (abs(e.get("density", 1.0) - 0.30), tier, e)
            for tier, e in entries.items()
            if "psnr_overall_db" in e
        ]
        if not candidates:
            continue
        _, _, entry = min(candidates)
        rows.append(
            [
                f"{METHOD_LABELS[method]}（{entry.get('density', 0):.2f}）",
                f"{entry['psnr_overall_db']:.1f}",
                f"{entry['psnr_first5s_db']:.1f}",
            ]
        )
    return "<h2>3. 生成质量</h2>" + lead + table(
        ["方法（~0.30 档）", "整段 PSNR", "前 5 s PSNR"], rows
    ) + "<p><b>视觉检查</b>（帧对比图见下：各方法 ~0.30 档，7 帧，" "t = 1–19 s，方法名与帧号已标注在图内）：</p>" + bullets(
        notes["quality_bullets"]
    )


def prompts_section(model: str, prompts: dict) -> str:
    notes = MODEL_NOTES[model]
    rows = []
    for method in METHODS:
        densities = [
            entry["density"]
            for key, entry in prompts.items()
            if key.endswith(f"_{method}_0.3")
            and entry.get("returncode") == 0
            and "density" in entry
        ]
        if not densities:
            continue
        rows.append(
            [
                METHOD_LABELS[method],
                f"{min(densities):.3f} – {max(densities):.3f}",
                f"{sum(densities) / len(densities):.3f}",
            ]
        )
    lead = (
        "<p>新增 5 个 prompt（p1–p5）在 720p / 20 s 下重跑各方法的 ~0.30 档配置，"
        "检查密度是否与画面内容无关、以及单 prompt 的质量结论是否成立。"
        "各方法跨 prompt 的实际密度：</p>"
    )
    body = (
        "<h2>4. 多 prompt 验证与原始视频</h2>"
        + lead
        + table(["方法", "跨 prompt 密度区间", "均值"], rows)
        + bullets(notes["prompt_bullets"])
    )
    for key, label in PROMPT_LABELS.items():
        body += f"<p><b>{esc(label)}</b></p>"
    return body


def commands_section(model: str) -> str:
    spec = MODELS[model]
    calib_extra = (
        "\n# STA 的 tile 核随分辨率变化，单独在 720p 上校准\n"
        f"python calibrate.py --model {model} --methods sta --res 720p"
    )
    blocks = [
        (
            "# 稀疏注意力单元测试\n"
            "cd /data/projects/vision-gen/sglang\n"
            "PYTHONPATH=python python -m pytest "
            "python/sglang/multimodal_gen/test/unit/realtime/test_sparse_attention.py "
            "python/sglang/multimodal_gen/test/unit/realtime/test_sta_parity.py "
            "python/sglang/multimodal_gen/test/unit/realtime/test_svg_parity.py "
            "python/sglang/multimodal_gen/test/unit/realtime/test_radial_parity.py "
            "python/sglang/multimodal_gen/test/unit/realtime/test_lightforcing_parity.py "
            "python/sglang/multimodal_gen/test/unit/realtime/test_xattention_parity.py"
        ),
        (
            "# 密度校准（480p / 20 s 割线搜索）-> configs.json\n"
            "cd scripts/investigation/sparse_baselines\n"
            f"python calibrate.py --model {model} --workers 2" + calib_extra
        ),
        (
            "# 主实验：720p / 20 s，dense + 各方法各密度档\n"
            f"python run_sweep.py --model {model} --workers 2"
        ),
        ("# 质量：PSNR + 帧对比图\n" f"python quality.py --model {model}"),
        (
            "# 多 prompt 验证（p1–p5，各方法 ~0.30 档）\n"
            f"python multi_prompt.py --model {model} --workers 2"
        ),
        ("# Walltime-密度图\n" f"python plot.py --model {model}"),
    ]
    body = "<h2>复现命令</h2>"
    for block in blocks:
        body += f'<code language="Bash">{esc(block)}</code>'
    return body


def build(model: str) -> str:
    results, prompts, configs = load(model)
    return (
        setup_section(model, results, configs)
        + walltime_section(model, results)
        + quality_section(model, results)
        + prompts_section(model, prompts)
        + commands_section(model)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    xml = build(args.model)
    if args.out:
        args.out.write_text(xml)
        print(f"wrote {args.out} ({len(xml)} chars)")
    else:
        print(xml)
