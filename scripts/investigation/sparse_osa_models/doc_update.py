# SPDX-License-Identifier: Apache-2.0
"""Push one model's dense-vs-OSA sections into its Feishu doc.

Text comes from section_<model>.xml (authored from results.json /
results_prompts.json), figures and videos from the sweep output. The docs
start empty, so the text is appended; media is inserted at the end and moved
behind its anchor; a re-run replaces already-published media in place.

    python doc_update.py --model rolling_forcing [--stage text|figures|videos]
"""

import argparse
import pathlib
import re
import shutil
import sys

from common import PROMPTS, ROOT

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import (  # noqa: E402
    cli,
    insert_after,
    last_block_id,
    published_media,
    replace_media,
)

HERE = pathlib.Path(__file__).resolve().parent

DOCS = {
    "rolling_forcing": "KdAcdfJfooUUEjxnYsNcHxACn6d",
    "longlive2": "WUkLdNsJMoT1WHxwCvmc355DnLh",
    "lingbot_world_v2": "C5KfdLRIYoEunKxM3GlcgG5unGe",
}

# Doc order of the per-prompt media groups in the multi-prompt section; the
# anchor for each group is the paragraph whose bold lead-in carries the label.
PROMPT_LABELS = {
    "p0_tokyo": "p0 · 东京夜街",
    "p1_forest": "p1 · 雨林逃亡",
    "p2_plating": "p2 · 主厨摆盘",
    "p3_raccoon": "p3 · 浣熊吉他",
    "p4_teacup": "p4 · 茶杯倒水",
    "p5_tsunami": "p5 · 巷道海啸",
}


def section_ids(doc: str) -> dict[str, str]:
    """Map h2 heading text -> block id from the doc outline."""
    data = cli(
        "docs", "+fetch", "--doc", doc, "--scope", "outline", "--max-depth", "2"
    )
    content = data["document"]["content"]
    return {
        text: block_id
        for block_id, text in re.findall(r'<h2 id="([^"]+)">([^<]+)</h2>', content)
    }


def newest_mp4(run_dir: pathlib.Path) -> pathlib.Path | None:
    found = sorted(run_dir.glob("outputs/*.mp4")) or sorted(run_dir.glob("*.mp4"))
    return found[-1] if found else None


def stage_videos(model: str) -> dict[str, dict[str, pathlib.Path]]:
    """Copy each run's mp4 to a self-describing name; {prompt: {suffix: path}}."""
    model_root = ROOT / model
    staged = model_root / "upload_videos"
    staged.mkdir(exist_ok=True)
    files: dict[str, dict[str, pathlib.Path]] = {}

    def stage(prompt_key: str, suffix: str, run_dir: pathlib.Path) -> None:
        source = newest_mp4(run_dir)
        if source is None:
            print(f"missing video: {run_dir}")
            return
        target = staged / f"{prompt_key}_{suffix}.mp4"
        shutil.copy(source, target)
        files.setdefault(prompt_key, {})[suffix] = target

    stage("p0_tokyo", "dense", ROOT / model / "runs" / "dense")
    stage("p0_tokyo", "osa", ROOT / model / "runs" / "osa_0.3")
    for prompt_key in PROMPTS:
        if prompt_key == "p0_tokyo":
            continue
        for suffix in ("dense", "osa"):
            stage(
                prompt_key,
                suffix,
                model_root / "runs_prompts" / f"{prompt_key}_{suffix}",
            )
    return files


def upsert_media(
    doc: str,
    published: dict[str, str],
    anchor: str,
    path: pathlib.Path,
    *,
    caption: str | None = None,
    media_type: str = "image",
    keep_existing: bool = False,
) -> str:
    """Insert or in-place replace one figure/video; returns the next anchor.

    ``keep_existing`` leaves an already-published copy untouched (videos never
    change between doc pushes, and re-uploading them is expensive).
    """
    if not path.exists():
        print(f"missing media: {path}")
        return anchor
    kwargs: dict = (
        {"media_type": "file", "file_view": "preview"}
        if media_type == "file"
        else {"caption": caption, "width": 760}
    )
    if path.name in published:
        if keep_existing:
            return published[path.name]
        return replace_media(doc, published[path.name], str(path), **kwargs)
    return insert_after(doc, anchor, str(path), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(DOCS))
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "text", "figures", "videos"],
        help="resume a partially applied update instead of redoing the text",
    )
    args = parser.parse_args()
    doc = DOCS[args.model]
    model_root = ROOT / args.model

    if args.stage in ("all", "text"):
        section_xml = (HERE / f"section_{args.model}.xml").read_text()
        cli(
            "docs",
            "+update",
            "--doc",
            doc,
            "--command",
            "append",
            "--content",
            section_xml,
        )
        print("text appended")

    sections = section_ids(doc)
    walltime_h2 = next(k for k in sections if "Walltime" in k)
    quality_h2 = next(k for k in sections if "生成质量" in k)
    prompts_h2 = next(k for k in sections if "prompt" in k)

    if args.stage in ("all", "figures"):
        published = published_media(doc, sections[walltime_h2])
        upsert_media(
            doc,
            published,
            last_block_id(doc, sections[walltime_h2]),
            model_root / "walltime_vs_density.png",
            caption="去噪 / 端到端 walltime vs 实际读取密度（虚线为 dense 参考）",
        )
        published = published_media(doc, sections[quality_h2])
        upsert_media(
            doc,
            published,
            last_block_id(doc, sections[quality_h2]),
            model_root / "quality_sheet.png",
            caption="帧对比（p0 · 东京夜街；行：Dense 与各密度档 OSA；列：帧号 / 时间）",
        )

    if args.stage in ("all", "videos"):
        published = published_media(doc, sections[prompts_h2])
        videos = stage_videos(args.model)
        for prompt_key, label in PROMPT_LABELS.items():
            data = cli(
                "docs",
                "+fetch",
                "--doc",
                doc,
                "--scope",
                "section",
                "--start-block-id",
                sections[prompts_h2],
                "--detail",
                "with-ids",
            )
            match = re.search(
                rf'<p id="([^"]+)"><b>{re.escape(label)}</b>',
                data["document"]["content"],
            )
            if match is None:
                print(f"no anchor paragraph for {label}; skipping its media")
                continue
            anchor = match.group(1)
            if prompt_key != "p0_tokyo":
                anchor = upsert_media(
                    doc,
                    published,
                    anchor,
                    model_root / f"prompt_sheet_{prompt_key}.png",
                    caption=f"{label} 帧对比（行：Dense / OSA；列：帧号 / 时间）",
                    keep_existing=True,
                )
            for suffix in ("dense", "osa"):
                path = videos.get(prompt_key, {}).get(suffix)
                if path is not None:
                    anchor = upsert_media(
                        doc,
                        published,
                        anchor,
                        path,
                        media_type="file",
                        keep_existing=True,
                    )
    print("done")


if __name__ == "__main__":
    main()
