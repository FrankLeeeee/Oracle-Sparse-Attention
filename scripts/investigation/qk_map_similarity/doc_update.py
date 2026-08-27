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

from run import CHUNK_IDS, HEAD_SPECS, NUM_CHUNKS, STEP_IDS  # noqa: E402

DOC = "Rs3sdTCinoc6kqxdiGxcUDIQnfd"
ROOT = results_dir("qk_map_similarity")
PROMPTS = json.loads((HERE.parent / "prompts.json").read_text())
PLOT_RUN = "p1"  # the run whose attention maps / similarity go into the doc
SIM_STEP = 3  # published similarity tables: last denoising step
SIM_CHUNK = CHUNK_IDS[-1]  # ... at the 100th-percentile chunk


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


def similarity_summary_table() -> str:
    """Mean cosine over all (q frame, key frame>0) pairs, per pick and chunk."""
    header = "".join(f"<th>{chunk_label(c)}</th>" for c in CHUNK_IDS)
    rows = []
    for spec in HEAD_SPECS:
        cells = []
        for chunk in CHUNK_IDS:
            table = load_similarity(spec, chunk, SIM_STEP)["cosine"]
            values = [v for row in table for v in row[1:]]
            cells.append(f"<td>{sum(values) / len(values):.3f}</td>")
        rows.append(
            f"<tr><td><p>L{spec['layer']} · h{spec['head']}</p></td>{''.join(cells)}</tr>"
        )
    return (
        '<table><colgroup><col width="110"/><col span="4" width="130"/></colgroup>'
        f"<thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
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

    parts.append("<h2>3. Pattern Similarity</h2>")
    parts.append(
        "<p>帧对帧 pattern 相似度：一次注意力调用有 Sq 个 query token、Sk 个 key "
        "token，每个潜帧 T=3600 token，即 Sq/T 个 query 帧 × Sk/T 个 key 帧的帧对。"
        "对 query 帧 i 与 key 帧 j，取 A_ij = softmax(Q_i K_jᵀ/√d)（softmax 仅在该 "
        "key 帧的 T 个 token 上进行，即该帧对的独立注意力图），"
        "表中数值为 A_i0 与 A_ij 展平后的余弦相似度——第 j 列衡量 query 帧 i "
        "对 key 帧 j 的图案与它对 key 帧 0 的图案有多相似，第 0 列恒为 1。"
        f"下表取自 p1 的 {chunk_label(SIM_CHUNK)}、去噪步 {SIM_STEP}"
        "（末步，OSA 校准所用的步）；其余 chunk / 步的完整表见 "
        "results/investigation/qk_map_similarity/similarity/。</p>"
    )
    for spec in HEAD_SPECS:
        parts.append(f"<h3>{spec_title(spec)}</h3>")
        parts.append(similarity_table(spec))
    parts.append("<h3>各 chunk 汇总（step 3，key frame 0 列除外的均值）</h3>")
    parts.append(similarity_summary_table())
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
    print(f"[verify] image blocks: {images} (want {len(HEAD_SPECS) * len(CHUNK_IDS) * len(STEP_IDS)})")
    if leftovers:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["sections", "media", "verify"])
    args = parser.parse_args()
    {"sections": stage_sections, "media": stage_media, "verify": stage_verify}[
        args.stage
    ]()


if __name__ == "__main__":
    main()
