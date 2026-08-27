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
        choices=["sections", "media", "extra", "resim", "temporal", "repro", "verify"],
    )
    args = parser.parse_args()
    {
        "sections": stage_sections,
        "media": stage_media,
        "extra": stage_extra,
        "resim": stage_resim,
        "temporal": stage_temporal,
        "repro": stage_repro,
        "verify": stage_verify,
    }[args.stage]()


if __name__ == "__main__":
    main()
