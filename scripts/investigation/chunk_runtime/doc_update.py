# SPDX-License-Identifier: Apache-2.0
"""Append the per-chunk runtime subsection to the investigation doc.

Text comes from section.xml, figures and the 24 sample videos from this
directory's sweep output. Everything is inserted after the anchor block (the
last figure of the existing "Runtime Breakdown" section), so the subsection
lands inside that section rather than at the end of the document.

    python doc_update.py [--anchor <block_id>] [--skip-videos]
"""

import argparse
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import insert_after, insert_blocks, last_block_id  # noqa: E402
from paths import results_dir  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = results_dir("chunk_runtime")
DOC = "GT64dNBSKo5KSqxdpzIcxgeMnyg"
# Last figure of the existing "Runtime Breakdown" h2 section.
DEFAULT_ANCHOR = "doxcnkbQVuCcoJZV7U4qZF7MNLe"
# The "Runtime Breakdown" h2 the subsection belongs to; re-fetched to find the
# id of the last block after each text insert.
SECTION_START = "L53IdtFczofbMvxdPLyc6752n2c"

MODELS = ["self_forcing", "rolling_forcing", "longlive2", "lingbot_world_v2"]
MODEL_LABELS = {
    "self_forcing": "Self-Forcing 1.3B",
    "rolling_forcing": "Rolling Forcing 1.3B",
    "longlive2": "LongLive-2.0 5B",
    "lingbot_world_v2": "LingBot-World v2 14B",
}
RESOLUTIONS = ["480p", "720p"]
DURATIONS = [5, 10, 20]

FIGURES = [
    ("chunk_walltime_720p.png", "逐 chunk 耗时（720p，4 模型 × 3 时长）"),
    ("chunk_walltime_480p.png", "逐 chunk 耗时（480p，4 模型 × 3 时长）"),
    ("attention_share.png", "attention 占 forward 的比例（20s）"),
]


def stage_videos() -> list[tuple[str, pathlib.Path]]:
    """Copy each config's mp4 to a self-describing name for upload."""
    staged = ROOT / "upload_videos"
    staged.mkdir(exist_ok=True)
    files = []
    for model in MODELS:
        for res in RESOLUTIONS:
            for duration in DURATIONS:
                source = ROOT / "runs" / model / f"{res}_{duration}s" / "video.mp4"
                if not source.exists():
                    print(f"missing video: {source}")
                    continue
                target = staged / f"{model}_{res}_{duration}s.mp4"
                shutil.copy(source, target)
                files.append((f"{MODEL_LABELS[model]} · {res} · {duration}s", target))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "text", "figures", "videos"],
        help="resume a partially applied update instead of redoing the text",
    )
    args = parser.parse_args()

    if args.stage in ("all", "text"):
        insert_blocks(DOC, args.anchor, (HERE / "section.xml").read_text())
    anchor = last_block_id(DOC, SECTION_START)
    if args.stage in ("all", "text", "figures"):
        for name, caption in FIGURES:
            anchor = insert_after(
                DOC, anchor, str(ROOT / name), caption=caption, width=760
            )
    if args.stage in ("text", "figures"):
        print(f"done (stage={args.stage})")
        return

    videos = stage_videos()
    insert_blocks(
        DOC,
        anchor,
        "<p><b>本轮 24 个配置生成的视频</b>（同一 prompt、seed 42，"
        "文件名为 <code>模型_分辨率_时长</code>）：</p>",
    )
    anchor = last_block_id(DOC, SECTION_START)
    for _label, path in videos:
        anchor = insert_after(
            DOC, anchor, str(path), media_type="file", file_view="preview"
        )
    print(f"done: {len(videos)} videos")


if __name__ == "__main__":
    main()
