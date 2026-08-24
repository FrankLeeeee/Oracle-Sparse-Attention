# SPDX-License-Identifier: Apache-2.0
"""Update the Self-Forcing doc in place, preserving what this study does not own.

Unlike the other four model docs, this one already carries material that is
still current and cannot be regenerated: the OSA implementation subsection and
its whiteboard, the OSA kernel-optimization history, and the whole granularity
study (section 5). So instead of rewriting the body, each managed span is
replaced between two markers and everything else is left untouched.

    python doc_update_self_forcing.py [--stage text|figures|videos]
"""

import argparse
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import sections
from common import ROOT
from doc_update import DOCS, push_videos, stage_videos

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import cli, insert_after, published_media, replace_media  # noqa: E402

DOC = DOCS["self_forcing"]
MODEL = "self_forcing"


def top_blocks(doc: str) -> list[tuple[str, list[str], str]]:
    """[(tag, block ids, leading text)] of the doc's top-level blocks.

    A list is one top-level element whose *items* are the blocks Feishu knows,
    so a ``ul`` reports its ``li`` ids.
    """
    content = cli("docs", "+fetch", "--doc", doc, "--detail", "with-ids")["document"][
        "content"
    ]
    root = ET.fromstring(f"<root>{content}</root>")
    blocks = []
    for child in root:
        if child.get("id"):
            ids = [child.get("id")]
        else:
            ids = [item.get("id") for item in child if item.get("id")]
        text = " ".join("".join(child.itertext()).split())[:100]
        blocks.append((child.tag, ids, text))
    return blocks


def find_block(
    blocks, *, text: str | None = None, tag: str | None = None, name: str | None = None
) -> int:
    for index, (block_tag, _, block_text) in enumerate(blocks):
        if text is not None and text not in block_text:
            continue
        if tag is not None and block_tag != tag:
            continue
        if name is not None and name not in block_text:
            continue
        return index
    raise LookupError(f"no block matching text={text!r} tag={tag!r}")


def replace_span(doc: str, *, after: str, before: str, xml: str) -> None:
    """Replace the blocks strictly between the ``after`` and ``before`` markers.

    The new content goes in first and the old blocks are deleted afterwards, so
    an interrupted run leaves a duplicated span rather than a hole.
    """
    blocks = top_blocks(doc)
    start = find_block(blocks, text=after)
    end = find_block(blocks, text=before)
    if end <= start:
        raise LookupError(f"{before!r} does not follow {after!r}")
    doomed = [block_id for _, ids, _ in blocks[start + 1 : end] for block_id in ids]
    anchor = blocks[start][1][-1]
    cli(
        "docs",
        "+update",
        "--doc",
        doc,
        "--command",
        "block_insert_after",
        "--block-id",
        anchor,
        "--content",
        xml,
    )
    for chunk in range(0, len(doomed), 20):
        cli(
            "docs",
            "+update",
            "--doc",
            doc,
            "--command",
            "block_delete",
            "--block-id",
            ",".join(doomed[chunk : chunk + 20]),
        )
    print(f"replaced {len(doomed)} blocks between {after!r} and {before!r}")


def strip_heading(xml: str) -> str:
    """Drop the leading ``<h2>…</h2>`` — the doc's own heading stays."""
    return re.sub(r"^<h2>.*?</h2>", "", xml, count=1)


AUDIT_PARAGRAPH = (
    "<p><b>基线实现审计</b>：每个基线都对照其原始代码库逐一核验，选择语义在可对齐"
    "几何上逐位一致（x-attention <code>e379887</code> / LightForcing "
    "<code>d1e6333</code> / Sparse-VideoGen <code>f89aeda</code> / "
    "radial-attention <code>72788d4</code> / FastVideo <code>98f761e</code>）。"
    "2026-08-20 一轮修复了 XAttention 的 padding 行 softmax 泄漏与估计器融合、"
    "SVG1 的时间头 spatial-major 查询置换与帧对齐 key tile、SVG2 的末块 "
    "padded-mean 与跨 chunk warm-start、LightForcing 的跨视频 pooled-history "
    "缓存重置。本轮新增 STA（Sliding Tile Attention）并修复其两处问题："
    "<b>tile 核不随分辨率迁移</b>（空间占比由 tile 网格决定，因此改为直接在 720p "
    "校准）与<b>查询块过小</b>（64 行对齐 tile，使 kernel 的 <code>tl.dot</code> "
    "行数不足：同密度下 1.07× → 1.85×，改为 128 行）；另外为 Rolling Forcing 的"
    "重新 RoPE sink 增加了位置帧号（Radial 据此计算时间距离），并让 SVG1 的 "
    "dense_sink_frames 覆盖多帧 sink。"
)


def push_text() -> None:
    results, prompts, configs = sections.load(MODEL)
    replace_span(
        DOC,
        after="1. 实验设置",
        before="OSA 实现",
        xml=strip_heading(sections.setup_section(MODEL, results, configs)),
    )
    replace_span(
        DOC,
        after="2. Walltime",
        before="优化过程",
        xml=strip_heading(sections.walltime_section(MODEL, results)),
    )
    replace_span(
        DOC,
        after="3. 生成质量",
        before="quality_sheet",
        xml=strip_heading(sections.quality_section(MODEL, results)),
    )
    replace_span(
        DOC,
        after="4. 多 prompt",
        before="p0 · 东京夜街",
        # The per-prompt anchor paragraphs already exist in this doc (with the
        # prompt text and their video groups under them), so only the lead-in
        # and the density table are replaced.
        xml=re.sub(
            r"<p><b>p\d · .*$",
            "",
            strip_heading(sections.prompts_section(MODEL, prompts)),
        ),
    )
    replace_span(
        DOC,
        after="复现命令",
        before="5. OSA 粒度对比",
        xml=strip_heading(sections.commands_section(MODEL)),
    )
    # The audit paragraph is one block; rewrite it rather than the whole span,
    # because the OSA implementation bullets and whiteboard sit right above it.
    blocks = top_blocks(DOC)
    audit = blocks[find_block(blocks, text="基线实现审计")][1][0]
    cli(
        "docs",
        "+update",
        "--doc",
        DOC,
        "--command",
        "block_replace",
        "--block-id",
        audit,
        "--content",
        AUDIT_PARAGRAPH,
    )
    print("rewrote the baseline-audit paragraph")


def push_figures() -> None:
    model_root = ROOT / MODEL
    blocks = top_blocks(DOC)
    # The stale per-target bar chart is no longer produced; the density plot
    # carries the same comparison.
    try:
        stale = blocks[find_block(blocks, text="speedup_bars")][1][0]
        cli(
            "docs",
            "+update",
            "--doc",
            DOC,
            "--command",
            "block_delete",
            "--block-id",
            stale,
        )
        print("dropped the stale speedup_bars figure")
    except LookupError:
        pass

    for section_marker, filename, caption in (
        (
            "2. Walltime",
            "walltime_vs_density.png",
            "各方法去噪耗时 vs 实际累计读取密度（虚线为 dense 参考）",
        ),
        (
            "3. 生成质量",
            "quality_sheet_target0.3.png",
            "帧对比（p0 · 东京夜街，各方法 ~0.30 档；行：方法；列：帧号 / 时间，共 7 帧）",
        ),
    ):
        blocks = top_blocks(DOC)
        heading = find_block(blocks, text=section_marker)
        existing = None
        for _, ids, text in blocks[heading + 1 :]:
            if filename in text:
                existing = ids[0]
                break
            if text.startswith(("2. ", "3. ", "4. ", "5. ", "复现命令")):
                break
        path = model_root / filename
        if existing:
            replace_media(DOC, existing, str(path), caption=caption, width=760)
            print(f"replaced {filename}")
        else:
            anchor = blocks[heading][1][-1]
            insert_after(DOC, anchor, str(path), caption=caption, width=760)


def clear_prompt_media() -> None:
    """Drop the previous round's sheets and videos from section 4.

    Every video in that section is regenerated by this rerun, so replacing the
    files in place is not enough — the old naming scheme differs and would
    leave both copies behind.
    """
    blocks = top_blocks(DOC)
    start = find_block(blocks, text="4. 多 prompt")
    end = find_block(blocks, text="复现命令")
    doomed = [
        block_id
        for tag, ids, _ in blocks[start + 1 : end]
        if tag in ("img", "figure")
        for block_id in ids
    ]
    for chunk in range(0, len(doomed), 20):
        cli(
            "docs",
            "+update",
            "--doc",
            DOC,
            "--command",
            "block_delete",
            "--block-id",
            ",".join(doomed[chunk : chunk + 20]),
        )
    print(f"cleared {len(doomed)} old media blocks from section 4")


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
        clear_prompt_media()
        push_videos(DOC, MODEL)
    print("done")


if __name__ == "__main__":
    main()
