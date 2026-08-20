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
ROOT = results_dir("attention_layers")
DOC = "GT64dNBSKo5KSqxdpzIcxgeMnyg"

FIGURES = [
    (
        "concentration_720p_20s.png",
        "注意力集中度（720p / 20s，横轴 chunk，一条线一层）",
    ),
    ("concentration_480p_20s.png", "注意力集中度（480p / 20s）"),
    (
        "sheets/self_forcing_720p_20s_layer00.png",
        "3.2 · Self-Forcing · 第 0 层（可见 key 随 chunk 增长）",
    ),
    ("sheets/rolling_forcing_720p_20s_layer00.png", "3.2 · Rolling Forcing · 第 0 层"),
    ("sheets/longlive2_720p_20s_layer00.png", "3.2 · LongLive-2 · 第 0 层"),
    (
        "sheets/lingbot_world_v2_720p_20s_layer00.png",
        "3.2 · LingBot-World v2 · 第 0 层",
    ),
    (
        "sheets/self_forcing_720p_20s_layer07.png",
        "3.3 · Self-Forcing · 第 7 层（25% 深度，最稀疏）",
    ),
    (
        "sheets/self_forcing_720p_20s_layer14.png",
        "3.3 · Self-Forcing · 第 14 层（50% 深度）",
    ),
    (
        "sheets/self_forcing_720p_20s_layer29.png",
        "3.3 · Self-Forcing · 第 29 层（最深）",
    ),
    ("sheets/rolling_forcing_720p_20s_layer29.png", "3.3 · Rolling Forcing · 第 29 层"),
    ("sheets/longlive2_720p_20s_layer29.png", "3.3 · LongLive-2 · 第 29 层"),
    (
        "sheets/lingbot_world_v2_720p_20s_layer39.png",
        "3.3 · LingBot-World v2 · 第 39 层",
    ),
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

    start = section_start("注意力随视频推进与网络深度的变化")
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
