# SPDX-License-Identifier: Apache-2.0
"""Append the frame-similarity section to the investigation doc.

Text comes from section.xml, figures from this topic's plot output. The
section is a new top-level one appended at the end of the document, so a
figure inserted right after the previous block lands in reading order without
any moving around.

The videos are deliberately not uploaded: the probe only reads q/k, so this
sweep's 24 videos are byte-identical to the ones already published in the
Runtime Breakdown section.

    python doc_update.py [--stage all|text|figures]
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import (  # noqa: E402
    insert_after,
    insert_blocks,
    last_block_id,
    published_media,
    replace_media,
)
from paths import results_dir  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = results_dir("attention_bidirectional")
DOC = "GT64dNBSKo5KSqxdpzIcxgeMnyg"

FIGURES = [
    ("concentration.png", "注意力集中度随去噪步的变化（横轴去噪步，一条线一层）"),
    ("sheets/480p_5s_layer00.png", "480p/5s · 第 0 层（窄对角带 = 局部算子）"),
    ("sheets/480p_5s_layer07.png", "480p/5s · 第 7 层（25% 深度，最稠密）"),
    ("sheets/480p_5s_layer29.png", "480p/5s · 第 29 层（最深）"),
    ("sheets/480p_20s_layer00.png", "480p/20s · 第 0 层"),
    ("sheets/480p_20s_layer07.png", "480p/20s · 第 7 层"),
    ("sheets/720p_20s_layer07.png", "720p/20s · 第 7 层（超出训练分布，见正文保留）"),
]


def section_start(title: str) -> str:
    """Block id of the h2 this section is written under."""
    from doc_media import cli

    data = cli("docs", "+fetch", "--doc", DOC, "--scope", "outline")
    content = data["document"]["content"]
    import re

    for block_id, heading in re.findall(r'<h2 id="([^"]+)">([^<]*)</h2>', content):
        if heading.strip() == title:
            return block_id
    raise RuntimeError(f"section {title!r} not found in the doc outline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", "text", "figures"])
    args = parser.parse_args()

    if args.stage in ("all", "text"):
        # -1 appends at the end of the document, where this new section belongs.
        insert_blocks(DOC, "-1", (HERE / "section.xml").read_text())

    start = section_start("双向模型对照：Wan2.1-T2V-1.3B")
    published = published_media(DOC, start)
    anchor = last_block_id(DOC, start)
    if args.stage in ("all", "figures"):
        for name, caption in FIGURES:
            path = ROOT / name
            if path.name in published:
                replace_media(
                    DOC, published[path.name], str(path), caption=caption, width=880
                )
            else:
                anchor = insert_after(
                    DOC, anchor, str(path), caption=caption, width=880
                )
    print(f"done (stage={args.stage})")


if __name__ == "__main__":
    main()
