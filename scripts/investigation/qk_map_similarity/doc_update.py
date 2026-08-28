# SPDX-License-Identifier: Apache-2.0
"""Publish the QK-map study into the Feishu doc (OSA Properties).

    python doc_update.py --stage sections   # text + tables with placeholders
    python doc_update.py --stage media      # swap placeholders for videos/plots
    python doc_update.py --stage verify     # no placeholder left, media counts

Doc layout (per the task spec):
  1. Sample Videos      — the five prompts + their dense 720p/5s videos
  2. Attention Map      — four (layer, head) picks x (chunk percentile x step)
                          tables, one plot per cell
  3. Pattern Similarity — per-pick cosine tables of the frame-to-frame pattern

The sections stage writes every media position as a ``[[kind:name]]``
placeholder paragraph; the media stage uploads each file to the doc end
(``media-insert`` can only append) and move-replaces its placeholder, which is
how a plot ends up inside its table cell.
"""

import argparse
import json
import pathlib
import re
import shutil
import sys
from xml.sax.saxutils import escape

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from doc_media import cli, replace_media  # noqa: E402
from paths import results_dir  # noqa: E402

from run import (  # noqa: E402
    CHUNK_IDS,
    EXTRA_HEAD_SPECS,
    HEAD_SPECS,
    NUM_CHUNKS,
    STEP_IDS,
)

DOC = "Rs3sdTCinoc6kqxdiGxcUDIQnfd"
ROOT = results_dir("qk_map_similarity")
PROMPTS = json.loads((HERE.parent / "prompts.json").read_text())
PLOT_RUN = "p1"  # the run whose attention maps / similarity go into the doc
SIM_STEP = 3  # published similarity tables: last denoising step
SIM_CHUNK = CHUNK_IDS[-1]  # ... at the 100th-percentile chunk


SUMMARY_H3 = "<h3>各 chunk 汇总（step 3，每行自身列除外的均值）</h3>"

REPRO_VIDEOS = """\
cd scripts/investigation/qk_map_similarity
# 5 个 prompt 的 dense 视频 + 四组 (层, 头) 的 Q/K 捕获（独占 GPU，自动等空闲卡）
python run.py
# 发布：文本骨架 + 视频/图占位替换
python doc_update.py --stage sections
python doc_update.py --stage media"""

REPRO_MAPS = """\
cd scripts/investigation/qk_map_similarity
# 由 runs/p1/qk/ 的原始 Q/K 重算全 key 轴 softmax，渲染 4 组 x 4 chunk x 4 步的图
CUDA_VISIBLE_DEVICES=<idle-gpu> python plot_maps.py --run p1
python doc_update.py --stage media"""

REPRO_SIMILARITY = """\
cd scripts/investigation/qk_map_similarity
# 深度验证的 5 个额外 (层, 头) 需补一次捕获（确定性重生成 p1，dump 共享到 runs/p1/qk/）
python run.py --spec extra --prompts p1
# 自参考余弦表 + 以 chunk 0 自身图为参考的时序一致性表（temporal_*.json）
CUDA_VISIBLE_DEVICES=<idle-gpu> python similarity.py --run p1 --spec main
CUDA_VISIBLE_DEVICES=<idle-gpu> python similarity.py --run p1 --spec extra
python plot_temporal.py --run p1
# 发布：--stage sections 已含主表与汇总；验证小节 / 时序小节各自追加
python doc_update.py --stage extra
python doc_update.py --stage temporal"""


def repro_pre(code: str) -> str:
    return f'<pre lang="bash" caption="复现命令"><code>{escape(code)}</code></pre>'


def chunk_label(chunk: int) -> str:
    return f"chunk {chunk}（{round(chunk / (NUM_CHUNKS - 1) * 100)}%）"


def spec_title(spec: dict) -> str:
    return f"{spec['task']} Layer {spec['layer']} · Head {spec['head']}"


def map_table(spec: dict) -> str:
    header = "".join(f"<th>step {s}</th>" for s in STEP_IDS)
    rows = []
    for chunk in CHUNK_IDS:
        cells = "".join(
            f"<td><p>[[map:L{spec['layer']:02d}_h{spec['head']}_c{chunk}_s{s}]]</p></td>"
            for s in STEP_IDS
        )
        rows.append(f"<tr><td><p>{chunk_label(chunk)}</p></td>{cells}</tr>")
    return (
        '<table><colgroup><col width="120"/><col span="4" width="340"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def load_similarity(spec: dict, chunk: int, step: int) -> dict:
    name = f"sim_L{spec['layer']:02d}_h{spec['head']}_c{chunk}_s{step}.json"
    return json.loads((ROOT / "similarity" / PLOT_RUN / name).read_text())


def similarity_table(spec: dict) -> str:
    record = load_similarity(spec, SIM_CHUNK, SIM_STEP)
    key_frames = record["key_frames"]
    header = "".join(f"<th>kf {j}</th>" for j in range(key_frames))
    rows = []
    for i, row in enumerate(record["cosine"]):
        cells = "".join(f"<td>{value:.3f}</td>" for value in row)
        rows.append(f"<tr><td><p>q frame {i}</p></td>{cells}</tr>")
    cols = f'<colgroup><col width="90"/><col span="{key_frames}" width="58"/></colgroup>'
    return (
        f"<table>{cols}<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def similarity_summary_table(specs=HEAD_SPECS) -> str:
    """Mean cosine over all (q frame, key frame>0) pairs, per pick and chunk."""
    header = "".join(f"<th>{chunk_label(c)}</th>" for c in CHUNK_IDS)
    rows = []
    for spec in specs:
        cells = []
        for chunk in CHUNK_IDS:
            cells.append(f"<td>{mean_cosine(spec, chunk, SIM_STEP):.3f}</td>")
        rows.append(
            f"<tr><td><p>L{spec['layer']} · h{spec['head']}</p></td>{''.join(cells)}</tr>"
        )
    return (
        '<table><colgroup><col width="110"/><col span="4" width="130"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def similarity_intro_xml() -> str:
    return (
        "<p>帧对帧 pattern 相似度：一次注意力调用有 <latex>S_q</latex> 个 query "
        "token、<latex>S_k</latex> 个 key token，每个潜帧 <latex>T=3600</latex> 个 "
        "token，即 <latex>S_q/T</latex> 个 query 帧 × <latex>S_k/T</latex> 个 key 帧"
        "的帧对。对 query 帧 <latex>i</latex> 与 key 帧 <latex>j</latex>，取 "
        r"<latex>A_{i,j}=\mathrm{softmax}\!\left(Q_i K_j^{\top}/\sqrt{d}\right)"
        "</latex>（softmax 仅在该 key 帧的 <latex>T</latex> 个 token 上进行，即该"
        "帧对的独立注意力图）。参考图取该 query 帧<b>对自身</b>的注意力图 "
        r"<latex>A_{i,\,S_k/T-3+i}</latex>（chunk 自身的 3 帧是可见 cache 中最新的 "
        "3 帧，query 帧 <latex>i</latex> 即倒数第 <latex>3-i</latex> 个 key 帧），"
        "表中数值为展平后的余弦相似度 "
        r"<latex>\cos\!\left(A_{i,\,S_k/T-3+i},\,A_{i,j}\right)</latex>——第 "
        "<latex>j</latex> 列衡量 query 帧 <latex>i</latex> 对 key 帧 <latex>j</latex> "
        "的图案与它对自己的图案有多相似，每行的自身列恒为 1。"
        f"下表取自 p1 的 {chunk_label(SIM_CHUNK)}、去噪步 {SIM_STEP}"
        "（末步，OSA 校准所用的步）；其余 chunk / 步的完整表见 "
        "results/investigation/qk_map_similarity/similarity/。</p>"
    )


def sections_xml() -> str:
    parts = ["<h2>1. Sample Videos</h2>"]
    parts.append(
        "<p>五个 prompt 均以 dense Self-Forcing 1.3B（全上下文）生成 720p / 5 秒视频"
        "（81 像素帧 = 21 潜帧 = 7 个 chunk，seed 42），全程稠密注意力，"
        "后续两节的注意力图与相似度全部取自这些 dense 运行。</p>"
    )
    for pid, entry in PROMPTS.items():
        parts.append(f"<p><b>{pid} · {escape(entry['label'])}</b></p>")
        parts.append(f"<p>{escape(entry['prompt'])}</p>")
        parts.append(f"<p>[[video:{pid}]]</p>")
    parts.append(repro_pre(REPRO_VIDEOS))

    parts.append("<h2>2. Attention Map</h2>")
    parts.append(
        "<p>以下注意力图取自 p1 · 雨林逃亡 的 dense 运行：对每个选定的 (层, 头)，"
        "在第 0 / 33 / 66 / 100 百分位 chunk（7 个 chunk 中的 0 / 2 / 4 / 6）"
        "的全部 4 个去噪步上，重算完整 softmax(QKᵀ/√d) 概率矩阵。"
        "纵轴为 query token 序号（当前 chunk 的 3 个潜帧），横轴为 key token 序号"
        "（全部可见潜帧，自旧到新）；颜色为 softmax 后的注意力值（对数色标，"
        "均值池化到显示分辨率）。绿线为潜帧边界（每帧 3600 token），"
        "白色竖线为当前 chunk 的 key 起点。</p>"
    )
    for spec in HEAD_SPECS:
        parts.append(f"<h3>{spec_title(spec)}</h3>")
        parts.append(map_table(spec))
    parts.append(repro_pre(REPRO_MAPS))

    parts.append("<h2>3. Pattern Similarity</h2>")
    parts.append(similarity_intro_xml())
    for spec in HEAD_SPECS:
        parts.append(f"<h3>{spec_title(spec)}</h3>")
        parts.append(similarity_table(spec))
    parts.append(SUMMARY_H3)
    parts.append(similarity_summary_table())
    parts.append(repro_pre(REPRO_SIMILARITY))
    return "".join(parts)


def fetch_placeholders() -> dict[str, str]:
    """``{kind:name -> block id}`` of every placeholder paragraph in the doc."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    found = {}
    for block_id, kind, name in re.findall(
        r'<p id="([^"]+)"[^>]*>\[\[(\w+):([\w.]+)\]\]</p>', content
    ):
        found[f"{kind}:{name}"] = block_id
    return found


def stage_sections() -> None:
    data = cli("docs", "+fetch", "--doc", DOC)
    body = re.sub(
        r"<title>[^<]*</title>|<p\s*/>|<p></p>", "", data["document"]["content"]
    ).strip()
    if body:
        raise SystemExit(
            "doc is not empty — refusing to append a second copy; "
            "clear it or edit doc_update.py deliberately"
        )
    cli("docs", "+update", "--doc", DOC, "--command", "append", "--content", sections_xml())
    print(f"[sections] appended, {len(fetch_placeholders())} placeholders")


def mean_cosine(spec: dict, chunk: int, step: int) -> float:
    """Mean over all (i, j) with each row's trivially-1 self column excluded."""
    record = load_similarity(spec, chunk, step)
    self_columns = record["self_columns"]
    values = [
        value
        for i, row in enumerate(record["cosine"])
        for j, value in enumerate(row)
        if j != self_columns[i]
    ]
    return sum(values) / len(values)


def extra_intro_xml() -> str:
    main_means = "、".join(
        f"L{spec['layer']}·h{spec['head']} {mean_cosine(spec, SIM_CHUNK, SIM_STEP):.2f}"
        for spec in HEAD_SPECS
    )
    extra_means = "、".join(
        f"L{spec['layer']}·h{spec['head']} {mean_cosine(spec, SIM_CHUNK, SIM_STEP):.2f}"
        for spec in EXTRA_HEAD_SPECS
    )
    return (
        f"<p>上面四组显示 pattern 相似度在层 0 很高而其余层明显更低（{main_means}）。"
        "为验证这一深度趋势，再取 5 个此前未用过的头，均匀铺开在不同层："
        f"{extra_means}（均为 {chunk_label(SIM_CHUNK)}、step {SIM_STEP}、"
        "每行自身列除外的均值）。结果一致：帧对帧图案的高度可复制性基本只属于"
        "层 0，浅层（L5）居中，中间与深层（L10–L29）大幅下降，其中 L10 / L25 "
        "接近 0。</p>"
    )


def extra_xml() -> str:
    """The depth-verification subsection: intro + one table per extra pick."""
    parts = ["<h3>深度验证：另取 5 个不同层的头</h3>", extra_intro_xml()]
    for spec in sorted(EXTRA_HEAD_SPECS, key=lambda s: s["layer"]):
        parts.append(f"<h4>Layer {spec['layer']} · Head {spec['head']}</h4>")
        parts.append(similarity_table(spec))
    return "".join(parts)


def stage_extra() -> None:
    """Insert the verification tables and widen the summary to all nine picks."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    summary_h3 = re.search(r'<h3 id="([^"]+)">各 chunk 汇总', content)
    assert summary_h3, "summary h3 not found"
    anchor = re.findall(r'<table id="([^"]+)"', content[: summary_h3.start()])[-1]
    if "深度验证" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
            "--block-id", anchor, "--content", extra_xml(),
        )
    summary_table = re.search(
        r'<table id="([^"]+)"', content[summary_h3.start() :]
    ).group(1)
    all_specs = sorted(
        [*HEAD_SPECS, *EXTRA_HEAD_SPECS], key=lambda s: (s["layer"], s["head"])
    )
    cli(
        "docs", "+update", "--doc", DOC, "--command", "block_replace",
        "--block-id", summary_table, "--content", similarity_summary_table(all_specs),
    )
    print("[extra] verification subsection + 9-row summary published")


def all_specs_sorted() -> list[dict]:
    return sorted([*HEAD_SPECS, *EXTRA_HEAD_SPECS], key=lambda s: (s["layer"], s["head"]))


def load_temporal(spec: dict) -> dict:
    name = f"temporal_L{spec['layer']:02d}_h{spec['head']}_ref0_s{SIM_STEP}.json"
    return json.loads((ROOT / "similarity" / PLOT_RUN / name).read_text())


def temporal_xml() -> str:
    header = "".join(f"<th>{chunk_label(c)}</th>" for c in CHUNK_IDS)
    rows = []
    for spec in all_specs_sorted():
        chunks = load_temporal(spec)["chunks"]
        cells = []
        for chunk in CHUNK_IDS:
            values = [v for row in chunks[str(chunk)] for v in row]
            cells.append(f"<td>{sum(values) / len(values):.3f}</td>")
        rows.append(
            f"<tr><td><p>L{spec['layer']} · h{spec['head']}</p></td>{''.join(cells)}</tr>"
        )
    table = (
        '<table><colgroup><col width="110"/><col span="4" width="130"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return (
        "<h3>时序一致性：chunk 0 的自身图案随视频推进的保持度</h3>"
        "<p>时序一致性衡量在参考 chunk <latex>C</latex> 测得的注意力图案在后续 "
        "chunk 是否仍然成立：取 chunk <latex>C</latex> 中 query 帧 <latex>i</latex> "
        r"对自身的注意力图 <latex>A_{C,i,\,-3+i}</latex> 为参考（此处 "
        "<latex>C=0</latex>，即 OSA 校准所用的 chunk），与生成 chunk "
        "<latex>c</latex> 时的各帧对图比较，报告 "
        r"<latex>\cos\!\left(A_{C,i,\,-3+i},\,A_{c,i,j}\right)</latex>"
        "（<latex>c</latex> 取被捕获的 0 / 2 / 4 / 6，step 3）。下图每个子图为一个 "
        "(层, 头)，横轴为 key 帧 <latex>j</latex>（全局编号），纵轴为对 3 个 query "
        "帧取均值的余弦相似度，每条线为一个生成 chunk；下表为对全部 "
        "<latex>(i, j)</latex> 的均值。层 0 的两个头全程平坦地保持在 0.92–0.99——"
        "chunk 0 测一次、全程可复用；其余层的相似度随 chunk 推进快速衰减"
        "（如 L14·h2 从 0.40 落到 0.06），仅各 chunk 自己的 3 帧出现小幅回升。</p>"
        "<p>[[map:temporal_ref0_s3]]</p>"
        f"{table}"
    )


def stage_resim() -> None:
    """Republish everything the self-referenced recomputation changed."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]

    def block_replace(block_id: str, new_content: str) -> None:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_replace",
            "--block-id", block_id, "--content", new_content,
        )

    intro = re.search(r'<p id="([^"]+)">帧对帧 pattern 相似度', content)
    block_replace(intro.group(1), similarity_intro_xml())
    verification = re.search(r'<p id="([^"]+)">上面四组显示', content)
    block_replace(verification.group(1), extra_intro_xml())
    for spec in all_specs_sorted():
        title = (
            spec_title(spec)
            if spec in HEAD_SPECS
            else f"Layer {spec['layer']} · Head {spec['head']}"
        )
        # The similarity table is the first table after the pick's heading in
        # section 3 — search from the *last* occurrence of the title, since
        # the main picks' titles also head their section-2 map tables.
        position = content.rindex(f">{title}</h")
        table_id = re.search(r'<table id="([^"]+)"', content[position:]).group(1)
        block_replace(table_id, similarity_table(spec))
    summary_h3 = re.search(r'<h3 id="([^"]+)">各 chunk 汇总[^<]*</h3>', content)
    block_replace(summary_h3.group(1), SUMMARY_H3)
    summary_table = re.search(
        r'<table id="([^"]+)"', content[summary_h3.start() :]
    ).group(1)
    block_replace(summary_table, similarity_summary_table(all_specs_sorted()))
    print("[resim] intro + 9 tables + verification note + summary republished")


def stage_temporal() -> None:
    """Append the temporal-consistency subsection at the end of section 3."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "时序一致性" not in content:
        summary_h3 = re.search(r'<h3 id="([^"]+)">各 chunk 汇总', content)
        anchor = re.search(
            r'<table id="([^"]+)"', content[summary_h3.start() :]
        ).group(1)
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
            "--block-id", anchor, "--content", temporal_xml(),
        )
    placeholders = fetch_placeholders()
    figure = ROOT / "plots" / PLOT_RUN / "temporal_ref0_s3.png"
    key = "map:temporal_ref0_s3"
    if key in placeholders:
        replace_media(DOC, placeholders[key], str(figure))
    print("[temporal] subsection + figure published")


REPRO_DEEPDIVE = """\
cd scripts/investigation/qk_map_similarity
# 全 7 个 chunk x 9 组 (层, 头) 的 Q/K 捕获（确定性重生成 p1）
python run.py --spec all9 --chunks 0,1,2,3,4,5,6 --prompts p1
# 参考 chunk 矩阵 / oracle 质量召回 / 局部窗口 / 帧级分布 / 步间一致性
CUDA_VISIBLE_DEVICES=<idle-gpu> python deep_dive.py --run p1
python doc_update.py --stage deepdive"""


def deep_dive_results() -> dict:
    root = ROOT / "deep_dive" / PLOT_RUN
    return {
        name: json.loads((root / f"{name}.json").read_text())
        for name in (
            "ref_matrix",
            "mass_transfer",
            "local_window",
            "frame_mass",
            "step_consistency",
        )
    }


def pick_key(spec: dict) -> str:
    return f"L{spec['layer']:02d}_h{spec['head']}"


def pick_label(spec: dict) -> str:
    return f"L{spec['layer']} · h{spec['head']}"


def ref_sweep_xml() -> str:
    return (
        "<h3>参考 chunk 扫描：换任何参考都救不了中间层的整图复制</h3>"
        "<p>把参考 chunk <latex>C</latex> 与生成 chunk <latex>c</latex> 全部扫一遍"
        "（cos 对全部 <latex>(i, j)</latex> 取均值，step 3，全部 7 个 chunk 均已"
        "捕获）。结论：换参考几乎不改变图景。层 0 两个头对任意 <latex>C</latex> "
        "都是 0.92–0.99；中间层即使用<b>上一个 chunk</b>（<latex>C=c-1</latex>，"
        "最新可能的冻结参考）也只有 0.04–0.2（L10 / L14 / L25），与 "
        "<latex>C=0</latex> 相差无几；L20·h7 用 <latex>C\\le 4</latex> 保持 "
        "0.4–0.6，但 <latex>C=5,6</latex> 反而更差（其图案后期自身在漂移）。"
        "整图级的 pattern 复制在中间层不成立，不是"
        "“校准得太早”的问题，而是这些头的帧对图本身逐 chunk 变化。</p>"
        "<p>[[map:ref_matrix]]</p>"
    )


def mass_recall_table() -> str:
    transfer = deep_dive_results()["mass_transfer"]
    header = (
        "<th>frozen@10% c0</th><th>frozen@10% c3</th><th>frozen@10% c6</th>"
        "<th>prev@10% c6</th><th>refreshed@10% c6</th><th>frozen@20% c6</th>"
    )
    rows = []
    for spec in all_specs_sorted():
        record = transfer[pick_key(spec)]
        cells = [
            record["frozen@0.10"][0],
            record["frozen@0.10"][3],
            record["frozen@0.10"][6],
            record["prev@0.10"][6],
            record["refreshed@0.10"][6],
            record["frozen@0.20"][6],
        ]
        body = "".join(f"<td>{value:.3f}</td>" for value in cells)
        rows.append(f"<tr><td><p>{pick_label(spec)}</p></td>{body}</tr>")
    return (
        '<table><colgroup><col width="100"/><col span="6" width="118"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def mass_recall_xml() -> str:
    return (
        "<p>实验定义：取 chunk 0 的自身图 <latex>A_{0,i,i}</latex>，对其中每个 "
        "query token 取帧内 top-p% 的 key 位置（<b>帧相对</b>索引）。生成 chunk "
        "<latex>c</latex> 时对全部可见 KV 做完整 softmax，把这些位置复制到每个"
        "可见帧上并把注意力值求和，得到该 query 的被捕获质量——质量接近 1 说明"
        "冻结的逐 query 位置足以支撑稀疏注意力。两条对照：<b>refreshed</b> 用"
        "当前 chunk 自身图取位置（逐 chunk 重校准的上限）；<b>prev</b> 用上一 "
        "chunk 的自身图（可在上一 chunk 的 KV cache 刷新 forward 里免费测得，"
        "部署上最现实）。下图与下表均为对 query 取平均、再对 3 个 query 帧取"
        "平均，step 3。</p>"
        "<p>[[map:mass_transfer]]</p>"
        f"{mass_recall_table()}"
        "<p><b>发现一：top-k 质量与整图余弦是两回事。</b>L20·h7 整图余弦只有 "
        "~0.4 却有 0.97–1.00 的 top-10% 质量——它的 top 集合稳定，变的只是低质量"
        "部分；L0·h0 余弦 0.92+ 但 top-10% 只收 0.16——它的行近乎均匀，"
        "“图相似”只是因为都均匀，top-k 无意义。第 3 节的余弦低不直接否定稀疏"
        "可行性，本节的质量召回才是决定性指标。"
        "<b>发现二：九个头分成三个家族。</b>几何 / 局部头（L0·h1、L20·h7）："
        "冻结即 ~1.0；弥散头（L0·h0、L5·h4）：任何 10% 都只收 0.16–0.31，"
        "质量随密度线性走（行近均匀）；内容依赖头（L10 / L14 / L15 / L25 / "
        "L29）：冻结 0.25–0.63 且随 chunk 衰减，逐 chunk 重校准 +0.05–0.24，"
        "prev 版本恢复其中大部分（如 L10：0.44 → prev 0.62 → refreshed 0.68）。"
        "<b>发现三：弥散头的低召回并不致命。</b>行近均匀时注意力输出是大范围"
        "均值，等步长 / 池化子采样即可低方差近似——这类头适合无选择的结构化"
        "降采样而不是 top-k。</p>"
        f"{repro_pre(REPRO_DEEPDIVE)}"
    )


def structure_table() -> str:
    results = deep_dive_results()
    header = (
        "<th>90% 质量需要的帧数 / 21</th><th>own 3 帧质量</th>"
        "<th>recent 帧质量</th><th>sink 帧质量</th>"
        "<th>步间 cos s0↔s3</th><th>步间 cos s2↔s3</th>"
    )
    rows = []
    for spec in all_specs_sorted():
        key = pick_key(spec)
        frame = results["frame_mass"][key]
        steps = results["step_consistency"][key]
        own = sum(frame["per_frame"][18:21])
        cells = (
            f"<td>{frame['frames_for_90pct']}</td><td>{own:.2f}</td>"
            f"<td>{frame['per_frame'][17]:.2f}</td>"
            f"<td>{frame['per_frame'][0]:.2f}</td>"
            f"<td>{steps['s0_vs_s3']:.2f}</td><td>{steps['s2_vs_s3']:.2f}</td>"
        )
        rows.append(f"<tr><td><p>{pick_label(spec)}</p></td>{cells}</tr>")
    return (
        '<table><colgroup><col width="100"/><col span="6" width="118"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def strategy_xml() -> str:
    return (
        "<h2>稀疏结构画像与策略提议</h2>"
        "<p>目标：<b>比 LightForcing 更快、视频质量相当</b>。LightForcing 的成本"
        "结构是三笔钱：每个注意力调用（每层 × 每去噪步）都要 mean-pool Q/K、"
        "算 block 分数、取 top-k 生成 mask（逐调用规划）；所有层 / 头共用同一个 "
        "sparsity 参数；所有头都走同一种内容依赖选择。本节的测量说明这三笔都有"
        "可省的空间。</p>"
        "<h3>结构画像</h3>"
        "<p><b>空间局部性</b>（下图：以 query 自身网格位置为中心、切比雪夫半径 "
        "<latex>r</latex> 的窗口复制到所有帧后捕获的质量，chunk 6、step 3）："
        "L0·h1 半径 1（密度 0.25%）即 0.82、半径 2（0.7%）0.91——纯几何图案，"
        "零校准零规划；L20·h7 半径 4（2.3%）0.76；L25·h8 半径 9（10%）0.76。"
        "其余头不局部，局部窗口对它们只是按密度线性收质量。</p>"
        "<p>[[map:local_window]]</p>"
        "<p><b>帧级分布与步间一致性</b>（下表，chunk 6）：L20·h7 的 own 3 帧就有 "
        "0.99 的质量、90% 质量只需 3 帧——它根本不看历史，直接跳过历史 KV；"
        "L0·h1 / L5·h4 / L25·h8 需要 4–6 帧（own + 少量 recent）；"
        "L10 / L14 / L15 / L29 需要 11–17 帧，是真正的全局头。步间一致性上，"
        "稳定头（L0 / L5 / L20 / L29）s0↔s3 已有 0.65–0.98，计划一次全 chunk "
        "复用是安全的；中间层 s0↔s3 只有 0.11–0.27 但随去噪单调上升"
        "（s2↔s3 0.38–0.56）——它们的图案在去噪过程中才收紧成形，规划宜取"
        "晚步测量，或直接用上一 chunk 的 cache 刷新 forward（输入是干净潜变量，"
        "天然等价于“最末步之后”）。</p>"
        f"{structure_table()}"
        "<h3>策略提议</h3>"
        "<p><b>一、离线逐头画像，分家族执行。</b>用本文的三个指标（top-k 质量"
        "召回、局部窗口质量、90% 质量帧数）把每个 (层, 头) 一次性归入四类——"
        "局部头：静态 frame-relative 窗口（逐头半径），连续 block、零运行时"
        "规划；own-chunk / 短窗头：只读自身 chunk 加逐头 w 个 recent 帧，零规划；"
        "弥散头：等步长 / 池化 KV 降采样（输出≈均值，低方差近似），零规划；"
        "内容依赖头（本样本中主要在 L10–L15 与部分深层）：保留 LightForcing 式"
        "逐 query-block 的运行时选择。画像的家族划分在保守划界下跨内容稳定"
        "（静态家族严格内容无关；内容依赖头的指标本身随内容波动，因而必须走"
        "运行时选择——直接证据与限定见下文「画像的内容无关性验证」节），"
        "一个模型只需标定一次家族归属。</p>"
        f"{family_examples_xml()}"
        "<p><b>二、把规划移出逐步循环。</b>内容依赖头的 top-k 位置在上一 chunk "
        "的 KV cache 刷新 forward 里顺带测量（该 forward 本来就要跑，且输入干净），"
        "本 chunk 的 4 个去噪步全部复用。上表 prev@10% 已证明这一近似只比逐步"
        "重校准低 0.03–0.07（L10 0.62 vs 0.68），而规划调用数从 每步×每层 降到 "
        "每 chunk×内容依赖头子集。这是对 LightForcing 最直接的加速点：它的逐"
        "调用规划被整体摊销掉。</p>"
        "<p><b>三、逐头预算分配代替全局单一 knob。</b>局部 / own-chunk 头跑在 "
        "0.25%–2.3% 的密度，释放的预算给内容依赖头加密（它们从 10% 提到 20% "
        "收益明显：L10 0.44→0.61）；同一平均密度下总召回高于所有头共用一个 "
        "sparsity 的方案，这就是在不掉质量的前提下把平均密度压得比 LightForcing "
        "低的空间。</p>"
        "<p><b>四、待验证的下一步。</b>(a) 全 30×12=360 头的画像扫描（只需 "
        "c0 / c3 / c6 末步的全头 Q/K，约 7 GB dump）确认各家族占比，占比直接"
        "决定可摊销的规划比例；(b) 用 osa_recall 的 LightForcing hook 在匹配密度"
        "下对拍逐头质量召回，把“质量相当”落到可比数字；(c) 原型混合 backend："
        "静态三家族 + prev-chunk 规划的内容依赖头，端到端对 LightForcing 计时。"
        "</p>"
        f"{repro_pre(REPRO_DEEPDIVE)}"
    )


REPRO_STABILITY = """\
cd scripts/investigation/qk_map_similarity
# p2-p5 的全 chunk Q/K 捕获（p1 已有；同 seed、不同内容）
python run.py --spec all9 --chunks 0,1,2,3,4,5,6 --prompts p2,p3,p4,p5
# 跨 prompt 稳定性：own/cross@10%、top-k 位置重合率、局部窗口、90% 帧数
CUDA_VISIBLE_DEVICES=<idle-gpu> python content_stability.py
python doc_update.py --stage stability"""


def stability_results() -> dict:
    return json.loads(
        (ROOT / "deep_dive" / "content_stability.json").read_text()
    )


def stability_table(record: dict) -> str:
    runs = record["runs"]
    header = (
        "<th>own@10%（5 prompt 范围）</th><th>cross@10%（范围）</th>"
        "<th>top-10% 位置重合率</th><th>local r=9（范围）</th>"
        "<th>90% 帧数（范围）</th>"
    )
    rows = []
    for spec in all_specs_sorted():
        pick = record["picks"][pick_key(spec)]
        own = [pick[run]["own@10%"] for run in runs]
        cross = [pick[run]["cross@10%"] for run in runs]
        overlap = [pick[run]["overlap"] for run in runs]
        local = [pick[run]["local_r9"] for run in runs]
        frames = [pick[run]["frames_90"] for run in runs]
        cells = (
            f"<td>{min(own):.2f}–{max(own):.2f}</td>"
            f"<td>{min(cross):.2f}–{max(cross):.2f}</td>"
            f"<td>{sum(overlap) / len(overlap):.2f}</td>"
            f"<td>{min(local):.2f}–{max(local):.2f}</td>"
            f"<td>{min(frames)}–{max(frames)}</td>"
        )
        rows.append(f"<tr><td><p>{pick_label(spec)}</p></td>{cells}</tr>")
    return (
        '<table><colgroup><col width="100"/><col span="5" width="140"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def stability_xml() -> str:
    record = stability_results()
    runs = record["runs"]

    def stats(key: str, metric: str) -> tuple[float, float, list[float]]:
        values = [record["picks"][key][run][metric] for run in runs]
        return min(values), max(values), values

    static_lines = []
    for key, family in (
        ("L00_h1", "几何"), ("L20_h7", "own-chunk"),
        ("L00_h0", "弥散"), ("L05_h4", "弥散"),
    ):
        low, high, _ = stats(key, "own@10%")
        overlap = sum(record["picks"][key][run]["overlap"] for run in runs) / len(runs)
        label = key.replace("_", "·").replace("L0", "L", 1)
        static_lines.append(
            f"{label}（{family}）own@10% {low:.2f}–{high:.2f}、位置重合率 {overlap:.2f}"
        )
    volatile_lines = []
    for key in ("L10_h5", "L14_h2", "L25_h8", "L29_h3"):
        low, high, _ = stats(key, "own@10%")
        r_low, r_high, _ = stats(key, "local_r9")
        label = key.replace("_", "·")
        volatile_lines.append(
            f"{label} own@10% {low:.2f}–{high:.2f}、local r9 {r_low:.2f}–{r_high:.2f}"
        )
    return (
        "<h2>画像的内容无关性验证（跨 prompt）</h2>"
        "<p>上文策略引用了“画像内容无关，一个模型只需标定一次”。此前只有间接"
        "证据（多 prompt 实验里各稀疏方法的实际读取密度跨 prompt 一致）。本节"
        "直接检验：对全部 5 个 prompt（同 seed、不同内容）重复全 chunk Q/K "
        "捕获，在 chunk 6、step 3 上比较画像的三个决定性指标，并做最直接的"
        "测试——用 prompt A 的 chunk 0 校准出的逐 query top-10% 位置，部署到 "
        "prompt B 的生成上收集质量（cross@10%）。</p>"
        "<p>[[map:content_stability]]</p>"
        f"{stability_table(record)}"
        "<p><b>结论一：静态家族的画像严格内容无关。</b>"
        f"{'；'.join(static_lines)}——四个静态家族头在 5 个 prompt 上数值几乎不"
        "动，几何 / own-chunk 头的位置集合本身跨内容高度重合，cross ≈ own，"
        "“标定一次、任意内容部署”对它们在位置层面成立。"
        "<b>结论二：内容依赖头的指标本身随内容明显波动，原句需要限定。</b>"
        f"{'；'.join(volatile_lines)}——同一个头在静态场景（茶杯、浣熊）上"
        "接近可冻结，在高动态场景（p1 雨林逃亡在每个波动头上都处于最低端）上"
        "显著变难，90% 帧数的极差可达 11 帧（L25·h8）。对它们，"
        "cross@10% 也系统性低于 own@10%。<b>对策略的含义：</b>家族划分本身"
        "仍可一次标定，但必须<b>保守划界</b>——(a) 只有在全部标定内容上都处于"
        "安全区的头才进静态家族，边界头（如 L25·h8）一律归入运行时选择；"
        "(b) 内容依赖头恰好就是策略里保留逐 chunk 运行时选择的那部分，其内容"
        "波动正是运行时选择存在的理由；(c) 它们的密度预算要按最难内容（高动态"
        "场景）标定，而不是按均值。内容无关的是<b>家族归属（在保守边界下）与"
        "静态头的具体图案</b>，不是每个头的逐 token 位置。</p>"
        f"{repro_pre(REPRO_STABILITY)}"
    )


def family_examples_xml() -> str:
    """One exemplar head per family, with its measured evidence."""
    deep = deep_dive_results()
    stability = stability_results()
    runs = stability["runs"]

    def transfer(key: str, name: str, index: int) -> float:
        return deep["mass_transfer"][key][name][index]

    local = deep["local_window"]
    own_share = lambda key: sum(deep["frame_mass"][key]["per_frame"][18:21])  # noqa: E731
    frames90 = lambda key: deep["frame_mass"][key]["frames_for_90pct"]  # noqa: E731
    own_range = lambda key: [  # noqa: E731
        stability["picks"][key][run]["own@10%"] for run in runs
    ]
    overlap = sum(stability["picks"]["L00_h1"][run]["overlap"] for run in runs) / len(
        runs
    )
    content_own = own_range("L14_h2")
    rows = [
        (
            "局部头",
            "L0 · h1",
            f"半径 1 窗口（密度 0.25%）即 {local['L00_h1']['1']:.2f} 质量、"
            f"半径 2（0.7%）{local['L00_h1']['2']:.2f}；top-10% 位置跨 prompt "
            f"重合率 {overlap:.2f}，own@10% 在 5 个 prompt 上均为 1.00",
            "静态 frame-relative 窗口（逐头半径），零规划",
        ),
        (
            "短窗头",
            "L20 · h7",
            f"own 3 帧质量 {own_share('L20_h7'):.2f}、90% 质量只需 "
            f"{frames90('L20_h7')} 帧；冻结 top-10% 质量 "
            f"{transfer('L20_h7', 'frozen@0.10', 6):.2f}（c6），历史帧可整体不读",
            "只读自身 chunk（+ 逐头 w 个 recent 帧），零规划",
        ),
        (
            "弥散头",
            "L0 · h0",
            f"top-10% 只收 {transfer('L00_h0', 'frozen@0.10', 6):.2f}、top-20% "
            f"{transfer('L00_h0', 'frozen@0.20', 6):.2f}（质量≈密度，行近均匀）；"
            f"r=9 几何窗口也只有 {local['L00_h0']['9']:.2f}——不是选择问题，"
            "是没有可选的峰",
            "等步长 / 池化 KV 降采样（输出≈均值），零规划",
        ),
        (
            "内容依赖头",
            "L14 · h2",
            f"冻结 top-10% 从 {transfer('L14_h2', 'frozen@0.10', 0):.2f}（c0）衰减到 "
            f"{transfer('L14_h2', 'frozen@0.10', 6):.2f}（c6），prev-chunk 规划回到 "
            f"{transfer('L14_h2', 'prev@0.10', 6):.2f}；90% 质量需 "
            f"{frames90('L14_h2')} 帧；own@10% 跨 prompt 波动 "
            f"{min(content_own):.2f}–{max(content_own):.2f}",
            "LightForcing 式运行时选择，prev-chunk 规划、全步复用",
        ),
    ]
    body = "".join(
        f"<tr><td><p><b>{family}</b></p></td><td><p>{example}</p></td>"
        f"<td><p>{evidence}</p></td><td><p>{policy}</p></td></tr>"
        for family, example, evidence, policy in rows
    )
    l5_own = own_range("L05_h4")
    return (
        '<table><colgroup><col width="90"/><col width="80"/>'
        '<col width="330"/><col width="220"/></colgroup>'
        "<thead><tr><th>家族</th><th>代表头</th>"
        "<th>证据（chunk 6、step 3，另注明处除外）</th><th>执行方式</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        "<p>家族在帧级与帧内两个维度上可组合：L5·h4 帧级上是短窗头"
        f"（90% 质量只需 {frames90('L05_h4')} 帧、own 3 帧 "
        f"{own_share('L05_h4'):.2f}），但帧内弥散（top-10% 只收 "
        f"{min(l5_own):.2f}–{max(l5_own):.2f}），执行上应为"
        "“短窗 × 帧内降采样”。</p>"
    )


def stage_examples() -> None:
    """Insert the per-family exemplar table after the taxonomy paragraph."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "代表头" in content:
        print("[examples] table already present, nothing to do")
        return
    anchor = re.search(
        r'<p id="([^"]+)"[^>]*><b>一、离线逐头画像', content
    ).group(1)
    cli(
        "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
        "--block-id", anchor, "--content", family_examples_xml(),
    )
    print("[examples] family exemplar table inserted")


REPRO_NEXTSTEPS = """\
cd scripts/investigation/qk_map_similarity
# (a) 全 360 头画像扫描：p1/p4 全头捕获（c0/c3/c6、末步）+ 分类
python run.py --spec sweep --chunks 0,3,6 --steps 3 --prompts p1,p4
CUDA_VISIBLE_DEVICES=<idle-gpu> python taxonomy_sweep.py
# (b) LightForcing 逐头召回（真实 mask，osa_recall 的 LF hook）+ 对拍
cd ../osa_recall
python run.py --model self_forcing --method lightforcing --density 0.2 \\
  --seconds 5 --res 720p --prompt p1_forest --gpu <idle-gpu>
python run.py --model self_forcing --method lightforcing --density 0.3 \\
  --seconds 5 --res 720p --prompt p1_forest --gpu <idle-gpu>
cd ../qk_map_similarity
CUDA_VISIBLE_DEVICES=<idle-gpu> python lf_compare.py
# (c) LF 规划成本 microbenchmark
CUDA_VISIBLE_DEVICES=<idle-gpu> python bench_lf_plan.py
python doc_update.py --stage nextsteps"""


def nextsteps_xml() -> str:
    taxonomy = json.loads((ROOT / "deep_dive" / "taxonomy.json").read_text())
    compare = json.loads((ROOT / "deep_dive" / "lf_compare.json").read_text())
    bench = json.loads((ROOT / "deep_dive" / "bench_lf_plan.json").read_text())
    counts = taxonomy["summary"]["family_counts"]
    total = sum(counts.values())
    mean_density = taxonomy["summary"]["mean_density"]

    family_rows = "".join(
        f"<tr><td><p>{name}</p></td><td>{counts[key]}</td>"
        f"<td>{counts[key] / total:.1%}</td></tr>"
        for key, name in (
            ("local", "局部头（静态窗口）"),
            ("shortwin", "短窗头（own+recent 帧）"),
            ("frozen", "冻结头（chunk-0 top-10% 位置）"),
            ("diffuse", "弥散头（降采样）"),
            ("content", "内容依赖头（运行时选择）"),
        )
    )
    family_table = (
        '<table><colgroup><col width="240"/><col span="2" width="110"/></colgroup>'
        "<thead><tr><th>家族</th><th>头数</th><th>占比</th></tr></thead>"
        f"<tbody>{family_rows}</tbody></table>"
    )

    def compare_row(label: str, record: dict, density_key: str) -> str:
        return (
            f"<tr><td><p>{label}</p></td><td>{record[density_key]:.3f}</td>"
            f"<td>{record['mean_recall']:.3f}</td><td>{record['p10_recall']:.3f}</td>"
            f"<td>{record['share_below_0.5']:.1%}</td></tr>"
        )

    compare_table = (
        '<table><colgroup><col width="250"/><col span="4" width="120"/></colgroup>'
        "<thead><tr><th>方案</th><th>平均密度</th><th>平均召回</th>"
        "<th>p10 召回</th><th>召回&lt;0.5 头占比</th></tr></thead><tbody>"
        + compare_row("hybrid（内容头用 top-20% 复制代理）", compare["hybrid"], "mean_density")
        + compare_row(
            "hybrid（内容头用 LF 自身选择）", compare["hybrid_lf_content"], "mean_density"
        )
        + compare_row("LightForcing d0.2", compare["lf_d0.2"], "per_call_density")
        + compare_row("LightForcing d0.3", compare["lf_d0.3"], "per_call_density")
        + "</tbody></table>"
    )

    plan_c6 = bench["c6"]["plan_ms"]
    dense_c6 = bench["c6"]["dense_attention_ms"]
    plan_total = bench["per_video"]["estimated_total_plan_s"]
    return (
        "<h2>「待验证的下一步」执行结果</h2>"
        "<p>三项均在 720p / 5 秒、seed 42 上完成；(a)(b) 为真实测量，(c) 为 "
        "microbenchmark + 由实测数字外推的预估。</p>"
        "<h3>(a) 全 360 头画像扫描</h3>"
        "<p>对全部 30 层 × 12 头，在 p1（最难）与 p4（最易）两个内容极端上捕获 "
        "c0 / c3 / c6 的末步 Q/K，按保守规则分类（阈值 τ=0.85 须在两个 prompt "
        "上同时满足，family 按执行密度从便宜到贵依次尝试）。结果："
        f"<b>{total - counts['content']} / {total}（"
        f"{taxonomy['summary']['planning_free_share']:.1%}）的头零规划成本</b>，"
        f"全队平均密度 <b>{mean_density:.3f}</b>（内容头预算按 20% 计），"
        "平均策略召回 p1 0.769 / p4 0.831。静态家族集中在网络两端（层 0 有 "
        "11/12 静态，层 27–29 有 6–8 个），中间层（9–26）几乎全部内容依赖——"
        "与九头抽样的深度结论一致。</p>"
        "<p>[[map:taxonomy_map]]</p>"
        f"{family_table}"
        "<h3>(b) 与 LightForcing 的逐头质量召回对拍</h3>"
        "<p>LightForcing 侧用 osa_recall 的 LF hook 在真实运行中测其 mask 的逐头"
        "捕获质量（p1、720p/5s、chunk 6、step 3——与画像同一口径）。"
        "组合系统（静态家族按画像执行 + 内容依赖头沿用 LF 自身的选择）在 "
        f"<b>{compare['hybrid_lf_content']['mean_density']:.3f}</b> 的平均密度"
        f"（LF d0.2 的 84%）下达到平均召回 "
        f"<b>{compare['hybrid_lf_content']['mean_recall']:.3f}</b>"
        f"（LF d0.2 为 {compare['lf_d0.2']['mean_recall']:.3f}），p10 与低召回头"
        "占比同样接近——<b>质量代理指标与 LF 基本持平，key 读取量少 16%</b>。"
        "纯冻结代理（内容头用 top-20% 帧复制位置）则落后 LF 明显，再次确认内容头"
        "必须保留运行时选择。</p>"
        "<p>[[map:lf_compare]]</p>"
        f"{compare_table}"
        "<h3>(c) 规划成本 microbenchmark 与端到端预估</h3>"
        f"<p>用 LightForcing 真实的规划流水线（pool + block mask + 计划构建）在 "
        f"c6 形状（12 头、10800 q × 75600 kv）上实测：<b>每调用 {plan_c6:.2f} "
        f"ms</b>，同形状稠密注意力 {dense_c6:.1f} ms——规划只占注意力调用的 "
        f"~2–4%，整段 5 秒视频的规划总量约 <b>{plan_total:.2f} s</b>"
        "（840 次调用）。这推翻了我们此前的一个预期：在 LongLive-2 上决定性的"
        "“摊销规划”杠杆，在 Self-Forcing 720p 的大注意力调用下只值 ~3%；"
        "<b>真正的杠杆是密度</b>。把 LF 的实测密度-耗时阶梯（dense 10.67 s，"
        "d0.50/0.36/0.22 → 9.86/8.85/8.11 s）线性拟合（t ≈ 7.76 + 3.10·d），"
        "组合系统在密度 0.166、规划移出去噪路径后预估去噪 <b>≈ 8.0 s：比 dense "
        "快 1.34×，比 LF d0.2（实测 8.85 s）快 ~1.1×</b>。预估未计入的上行空间："
        "静态头的连续 pattern（滑窗 / 帧区间）kernel 效率高于散点 block；"
        "5 秒时注意力占比低，时长越长密度优势放大。<b>结论与下一步：</b>在 5 秒"
        "档，混合方案对 LF 的优势是实在但有限的（~10%），瓶颈是 63.6% 的内容"
        "依赖头占比——要拿到显著优势需要 (1) 压缩内容头份额（更细的静态图案，"
        "如逐头帧集合、对角带 mask），(2) 更强 / 更便宜的运行时选择器，"
        "(3) 真实 backend 原型做端到端计时验证（本节预估的最终检验）。</p>"
        f"{repro_pre(REPRO_NEXTSTEPS)}"
    )


def stage_nextsteps() -> None:
    """Publish the next-step experiment results as a final section."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "「待验证的下一步」执行结果" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "append",
            "--content", nextsteps_xml(),
        )
    if "（执行结果见文末" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "str_replace",
            "--pattern", "四、待验证的下一步。",
            "--content", "四、待验证的下一步（执行结果见文末的结果节）。",
        )
    placeholders = fetch_placeholders()
    for name in ("taxonomy_map", "lf_compare"):
        key = f"map:{name}"
        if key in placeholders:
            replace_media(DOC, placeholders[key], str(ROOT / "plots" / f"{name}.png"))
    print("[nextsteps] results section published")


def stage_stability() -> None:
    """Publish the cross-prompt content-independence section."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "内容无关性验证" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "append",
            "--content", stability_xml(),
        )
    marker = "画像是内容无关的（多 prompt 轮已证密度与图案跨内容稳定），一个模型只需标定一次"
    if marker in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "str_replace",
            "--pattern", marker,
            "--content",
            "画像的家族划分在保守划界下跨内容稳定（静态家族严格内容无关；"
            "内容依赖头的指标本身随内容波动，因而必须走运行时选择——直接证据与"
            "限定见下文「画像的内容无关性验证」节），一个模型只需标定一次家族归属",
        )
    placeholders = fetch_placeholders()
    key = "map:content_stability"
    if key in placeholders:
        replace_media(
            DOC, placeholders[key], str(ROOT / "plots" / "content_stability.png")
        )
    print("[stability] content-independence section published")


def stage_deepdive() -> None:
    """Publish the deep-dive: ref sweep (sec 3), mass recall, strategy section."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "参考 chunk 扫描" not in content:
        temporal = content.index("时序一致性")
        anchor = re.search(r'<table id="([^"]+)"', content[temporal:]).group(1)
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
            "--block-id", anchor, "--content", ref_sweep_xml(),
        )
    if "实验定义" not in content:
        oracle = re.search(
            r'<h2 id="([^"]+)"[^>]*>Oracle Attention Mass Recall</h2>'
            r'(?:<p id="([^"]+)"></p>)?',
            content,
        )
        anchor = oracle.group(2) or oracle.group(1)
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
            "--block-id", anchor, "--content", mass_recall_xml(),
        )
    if "稀疏结构画像" not in content:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "append",
            "--content", strategy_xml(),
        )
    placeholders = fetch_placeholders()
    for name in ("ref_matrix", "mass_transfer", "local_window"):
        key = f"map:{name}"
        if key in placeholders:
            replace_media(
                DOC, placeholders[key], str(ROOT / "plots" / PLOT_RUN / f"{name}.png")
            )
    print("[deepdive] ref sweep + mass recall + strategy sections published")


def stage_repro() -> None:
    """Insert one reproduction-command code block at the end of each section."""
    data = cli("docs", "+fetch", "--doc", DOC, "--detail", "with-ids")
    content = data["document"]["content"]
    if "复现命令" in content:
        print("[repro] blocks already present, nothing to do")
        return
    section2 = content.index(">2. Attention Map</h2>")
    section3 = content.index(">3. Pattern Similarity</h2>")
    temporal = content.index("时序一致性", section3)
    anchors = [
        # end of section 1: the last video figure before the section-2 h2
        (re.findall(r'<figure id="([^"]+)"', content[:section2])[-1], REPRO_VIDEOS),
        # end of section 2: the last map table before the section-3 h2
        (re.findall(r'<table id="([^"]+)"', content[:section3])[-1], REPRO_MAPS),
        # end of section 3: the temporal-consistency mean table
        (re.search(r'<table id="([^"]+)"', content[temporal:]).group(1), REPRO_SIMILARITY),
    ]
    for anchor, code in anchors:
        cli(
            "docs", "+update", "--doc", DOC, "--command", "block_insert_after",
            "--block-id", anchor, "--content", repro_pre(code),
        )
    print("[repro] three command blocks inserted")


def stage_media() -> None:
    placeholders = fetch_placeholders()
    print(f"[media] {len(placeholders)} placeholders to fill")
    for pid in PROMPTS:
        key = f"video:{pid}"
        if key not in placeholders:
            continue
        videos = sorted((ROOT / "runs" / pid / "outputs").glob("*.mp4"))
        assert len(videos) == 1, f"{pid}: expected one mp4, found {videos}"
        # Upload under a stable, readable name (the generator's file name is a
        # prompt slug + timestamp).
        named = ROOT / "runs" / pid / f"self_forcing_720p_5s_{pid}_dense.mp4"
        shutil.copyfile(videos[0], named)
        replace_media(DOC, placeholders[key], str(named), media_type="file")
    plot_dir = ROOT / "plots" / PLOT_RUN
    for key, block_id in placeholders.items():
        if not key.startswith("map:"):
            continue
        replace_media(DOC, block_id, str(plot_dir / f"{key[4:]}.png"))
    print("[media] done")


def stage_verify() -> None:
    data = cli("docs", "+fetch", "--doc", DOC)
    content = data["document"]["content"]
    leftovers = re.findall(r"\[\[\w+:[\w.]+\]\]", content)
    videos = content.count("<figure")
    images = content.count("<img")
    print(f"[verify] placeholders left: {leftovers or 'none'}")
    print(f"[verify] video blocks: {videos} (want {len(PROMPTS)})")
    maps = len(HEAD_SPECS) * len(CHUNK_IDS) * len(STEP_IDS)
    print(f"[verify] image blocks: {images} (want {maps} maps + 1 temporal figure)")
    if leftovers:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "sections", "media", "extra", "resim", "temporal",
            "repro", "deepdive", "stability", "examples", "nextsteps", "verify",
        ],
    )
    args = parser.parse_args()
    {
        "sections": stage_sections,
        "media": stage_media,
        "extra": stage_extra,
        "resim": stage_resim,
        "temporal": stage_temporal,
        "repro": stage_repro,
        "deepdive": stage_deepdive,
        "stability": stage_stability,
        "examples": stage_examples,
        "nextsteps": stage_nextsteps,
        "verify": stage_verify,
    }[args.stage]()


if __name__ == "__main__":
    main()
