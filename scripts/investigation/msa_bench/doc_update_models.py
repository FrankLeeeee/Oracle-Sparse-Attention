# SPDX-License-Identifier: Apache-2.0
"""Publish the CF / RF / CF-long MSA benchmark section into the doc.

    python doc_update_models.py
"""

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from doc_media import cli  # noqa: E402
from paths import results_dir  # noqa: E402

DOC = "Rs3sdTCinoc6kqxdiGxcUDIQnfd"
ROOT = results_dir("msa_bench")
MODELS = (
    ("causal_forcing", 5, "Causal Forcing（chunkwise，窗口 21）"),
    ("rolling_forcing", 5, "Rolling Forcing（滚动窗口，联合去噪 15 帧 query）"),
    ("causal_forcing_long", 20, "Causal Forcing long_video 权重（20 秒，滚动管线）"),
)
METHOD_LABELS = {
    "dense": "dense",
    "msasched": "MSA-sched（content 0.2）",
    "lightforcing": "LightForcing（0.2 档标定）",
}


def model_table(model: str, seconds: int) -> str:
    results = json.loads(
        (ROOT / f"results_{model}_{seconds}s.json").read_text()
    )
    dense_mean = statistics.mean(
        results[f"{model}_b{i}_dense_{seconds}s"]["denoise_s"] for i in range(1, 6)
    )
    rows = []
    for method in ("dense", "msasched", "lightforcing"):
        entries = [
            results[f"{model}_b{i}_{method}_{seconds}s"] for i in range(1, 6)
        ]
        denoise = statistics.mean(e["denoise_s"] for e in entries)
        cells = [f"<td>{denoise:.2f} s</td><td>{dense_mean / denoise:.2f}×</td>"]
        if method == "dense":
            cells.append("<td>1.0</td><td>—</td>")
        else:
            psnr = statistics.mean(e["psnr"] for e in entries)
            cells.append(
                f"<td>{entries[1]['density']:.3f}</td><td>{psnr:.2f}</td>"
            )
        rows.append(
            f"<tr><td><p>{METHOD_LABELS[method]}</p></td>{''.join(cells)}</tr>"
        )
    return (
        '<table><colgroup><col width="210"/><col span="4" width="130"/></colgroup>'
        "<thead><tr><th></th><th>去噪耗时（5 prompt 均值）</th><th>对 dense 加速</th>"
        "<th>实际累计密度</th><th>PSNR vs dense（均值）</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def section_xml() -> str:
    parts = ["<h2>MSA 扩展到 Causal Forcing 与 Rolling Forcing</h2>"]
    parts.append(
        "<p>MSA 移植到另外两类几何：<b>chunkwise 的 Causal Forcing</b>（与 SF 同"
        "捕获路径）与<b>滚动窗口管线</b>（Rolling Forcing 与 Causal Forcing 的 "
        "long_video 权重——上游 thu-ml/Causal-Forcing 的 longvideo 检查点，经 "
        "RollingForcing 管线运行）。backend 改动：<b>一</b>，窗口安全——一次 "
        "forward 携带多 chunk 联合去噪 query 时，shortwin 头保留 max(m, "
        "query_frames) 帧，任何 query chunk 不会失去自身帧（tile 窗口类基线在 "
        "RF 上损坏的根因）；<b>二</b>，调度求解按窗口截断 kv 加权"
        "（schedule_window_frames）；<b>三</b>，导出门槛补上 shortwin——窗口模型"
        "的 shortwin 以 m≥query 宽度（15+/21 帧）通过分类，其“静态”密度 "
        "0.71–0.90 严格劣于运行时选择（未加门槛时 MSA 在 RF 上比 dense 还慢，"
        "density 0.85——已由基准数据揭示并修复）。每模型独立标定画像"
        "（p1+p4 dense 捕获、τ=0.85）：CF = 116/360 静态；RF/CF-long 的窗口几何"
        "把静态家族压缩到仅 33–37 个 local 头，其余全部内容依赖——<b>窗口模型"
        "在结构上更接近纯运行时选择</b>。content 阶段的 keep_sink/near/frames "
        "取各模型 LF 标定值（含 RF 系的 3 帧钉扎 sink）。</p>"
    )
    for model, seconds, label in MODELS:
        parts.append(f"<h3>{label}</h3>")
        parts.append(model_table(model, seconds))
    parts.append(
        "<p><b>读数。</b>Causal Forcing 复现 SF 的结论：更低密度（0.347 vs "
        "0.360）下更快（8.92 vs 9.10 s），质量相当（13.38 vs 13.74，两者绝对值"
        "都低——CF 对自身 dense 的轨迹分歧本来就快）。滚动窗口模型上结论变为"
        "<b>质量-速度分治</b>：RF 5 秒 MSA 保真度远高（PSNR 23.68 vs 13.56，"
        "密度相近 0.676 vs 0.646）但比 LF 慢（11.59 vs 9.53 s，仍比 dense 快 "
        "21%）；CF-long 20 秒 MSA 在匹配密度（0.369 vs 0.356）下 +1.5 dB"
        "（11.64 vs 10.14）、1.59× 于 dense，但 LF 更快（23.25 vs 30.09 s）。"
        "机理：滚动管线稳态每窗口只有一次 forward，MSA 的「每 chunk 规划一次、"
        "各步复用」优势无物可摊；静态家族又近乎缺席——MSA 退化为「LF mask + "
        "更保守的 ramp 处理」，于是买到保真、付出速度。要在窗口模型上同时超越，"
        "已识别的下一步是把 content 调度从 kv 截断的近平坦解改为按全局 chunk "
        "序数前置递减（对齐 LF 的 front-loaded sparsity），并为 33–37 个 local "
        "头保留静态执行。</p>"
        '<pre lang="bash" caption="复现命令"><code>'
        "cd scripts/investigation/qk_map_similarity\n"
        "python run.py --model causal_forcing --spec sweep --chunks 0,3,6 --steps 3 --prompts p1,p4\n"
        "python run.py --model rolling_forcing --spec sweep --chunks 0,1,2 --steps 0,1,2,3 --prompts p1,p4\n"
        "python run.py --model causal_forcing_long --seconds 20 --spec sweep --chunks 0,11,22 --steps 0,1,2,3 --prompts p1,p4\n"
        "python taxonomy_sweep.py --runs p1_causal_forcing_sweep,p4_causal_forcing_sweep \\\n"
        "  --ref-chunk 0 --deploy-chunk 6 --step 3 --out-name taxonomy_causal_forcing \\\n"
        "  --export msa_taxonomy_causal_forcing.json   # RF: ref 1 deploy 2 step 0; CF-long: 11/22/0\n"
        "cd ../msa_bench\n"
        "python run_bench.py --model causal_forcing --methods dense,msasched,lightforcing\n"
        "python run_bench.py --model rolling_forcing --methods dense,msasched,lightforcing\n"
        "python run_bench.py --model causal_forcing_long --seconds 20 --methods dense,msasched,lightforcing"
        "</code></pre>"
    )
    return "".join(parts)


def main() -> None:
    data = cli("docs", "+fetch", "--doc", DOC)
    if "MSA 扩展到 Causal Forcing" in data["document"]["content"]:
        print("[models-doc] section already present")
        return
    cli("docs", "+update", "--doc", DOC, "--command", "append", "--content", section_xml())
    print("[models-doc] section appended")


if __name__ == "__main__":
    main()
