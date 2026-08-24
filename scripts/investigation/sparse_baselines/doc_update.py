# SPDX-License-Identifier: Apache-2.0
"""Push one model's sections, figures and videos into its Feishu doc.

Text is generated from the results files by :mod:`sections`; this module only
places it. Figures go at the end of their section, videos under the per-prompt
anchor paragraphs of section 4. Stages are separate so a partially applied
push can resume, and already-published media is replaced in place rather than
appended twice.

    python doc_update.py --model causal_forcing [--stage text|figures|videos]
"""

import argparse
import pathlib
import re
import shutil
import sys

import sections
from common import METHODS, MODELS, ROOT, newest_video

from notes import PROMPT_LABELS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from doc_media import (  # noqa: E402
    cli,
    insert_after,
    last_block_id,
    published_media,
    replace_media,
)

DOCS = {
    "self_forcing": "Hx3OdnyKNomq9JxV9jfcYkf8n9d",
    "causal_forcing": "UiNcdabgcoVmJDxIGHicpoMYn5e",
    "rolling_forcing": "KdAcdfJfooUUEjxnYsNcHxACn6d",
    "longlive2": "WUkLdNsJMoT1WHxwCvmc355DnLh",
    "lingbot_world_v2": "C5KfdLRIYoEunKxM3GlcgG5unGe",
}

# Sections this study owns in each doc. The Self-Forcing doc additionally
# carries an OSA-implementation subsection and a granularity study that the
# user asked to leave alone, so it is pushed by its own script.
MANAGED_HEADINGS = ("实验设置", "Walltime", "生成质量", "prompt", "复现命令")


def top_level_blocks(doc: str) -> list[tuple[str, str]]:
    """[(block id, tag)] of the document's top-level blocks, in order."""
    data = cli("docs", "+fetch", "--doc", doc, "--detail", "with-ids")
    content = data["document"]["content"]
    return re.findall(r"<(\w+)[^>]*? id=\"([^\"]+)\"", content) and [
        (block_id, tag)
        for tag, block_id in re.findall(r"<(\w+)[^>]*? id=\"([^\"]+)\"", content)
    ]


def section_ids(doc: str) -> dict[str, str]:
    """{h2 text: block id} from the doc outline."""
    data = cli("docs", "+fetch", "--doc", doc, "--scope", "outline", "--max-depth", "2")
    return {
        text: block_id
        for block_id, text in re.findall(
            r'<h2 id="([^"]+)">([^<]+)</h2>', data["document"]["content"]
        )
    }


def rewrite_managed(doc: str, xml: str) -> None:
    """Replace the document body with ``xml``.

    Used for the four docs whose entire content is this study. The old blocks
    are deleted only after the new ones are in, so a failure mid-way leaves
    the old content intact rather than an empty document.
    """
    before = [block_id for block_id, _ in top_level_blocks(doc)]
    cli("docs", "+update", "--doc", doc, "--command", "append", "--content", xml)
    if before:
        for chunk_start in range(0, len(before), 20):
            cli(
                "docs",
                "+update",
                "--doc",
                doc,
                "--command",
                "block_delete",
                "--block-id",
                ",".join(before[chunk_start : chunk_start + 20]),
            )
    print(f"rewrote body ({len(xml)} chars, dropped {len(before)} old blocks)")


def stage_videos(model: str) -> dict[str, dict[str, pathlib.Path]]:
    """Copy each run's mp4 to a self-describing name; {prompt: {method: path}}."""
    model_root = ROOT / model
    staged = model_root / "upload_videos"
    staged.mkdir(exist_ok=True)
    files: dict[str, dict[str, pathlib.Path]] = {}

    def stage(prompt_key: str, label: str, run_dir: pathlib.Path) -> None:
        source = newest_video(run_dir)
        if source is None:
            print(f"missing video: {run_dir}")
            return
        target = staged / f"{model}_{prompt_key}_{label}.mp4"
        shutil.copy(source, target)
        files.setdefault(prompt_key, {})[label] = target

    tags = ["dense"] + [f"{method}_0.3" for method in METHODS]
    for tag in tags:
        label = tag.rsplit("_", 1)[0] if tag != "dense" else "dense"
        stage("p0_tokyo", label, model_root / "runs" / tag)
        for prompt_key in PROMPT_LABELS:
            if prompt_key == "p0_tokyo":
                continue
            stage(
                prompt_key,
                label,
                model_root / "runs_prompts" / f"{prompt_key}_{tag}",
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
    """Insert or in-place replace one figure/video; returns the next anchor."""
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


def push_figures(doc: str, model: str) -> None:
    model_root = ROOT / model
    ids = section_ids(doc)
    walltime = next(k for k in ids if "Walltime" in k)
    quality = next(k for k in ids if "生成质量" in k)

    published = published_media(doc, ids[walltime])
    upsert_media(
        doc,
        published,
        last_block_id(doc, ids[walltime]),
        model_root / "walltime_vs_density.png",
        caption="各方法去噪耗时 vs 实际累计读取密度（虚线为 dense 参考）",
    )
    published = published_media(doc, ids[quality])
    upsert_media(
        doc,
        published,
        last_block_id(doc, ids[quality]),
        model_root / "quality_sheet_target0.3.png",
        caption="帧对比（p0 · 东京夜街，各方法 ~0.30 档；行：方法；列：帧号 / 时间，共 7 帧）",
    )


def push_videos(doc: str, model: str) -> None:
    model_root = ROOT / model
    ids = section_ids(doc)
    prompts_h2 = next(k for k in ids if "prompt" in k)
    published = published_media(doc, ids[prompts_h2])
    videos = stage_videos(model)

    for prompt_key, label in PROMPT_LABELS.items():
        data = cli(
            "docs",
            "+fetch",
            "--doc",
            doc,
            "--scope",
            "section",
            "--start-block-id",
            ids[prompts_h2],
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
                caption=f"{label} 帧对比（行：Dense 与各方法 ~0.30 档；列：帧号 / 时间）",
                keep_existing=True,
            )
        for method_label in ["dense"] + list(METHODS):
            path = videos.get(prompt_key, {}).get(method_label)
            if path is not None:
                anchor = upsert_media(
                    doc,
                    published,
                    anchor,
                    path,
                    media_type="file",
                    keep_existing=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--stage", default="all", choices=["all", "text", "figures", "videos"]
    )
    args = parser.parse_args()
    doc = DOCS[args.model]
    if args.model == "self_forcing" and args.stage in ("all", "text"):
        raise SystemExit(
            "the Self-Forcing doc carries sections this study does not own; "
            "push its text with doc_update_self_forcing.py"
        )

    if args.stage in ("all", "text"):
        rewrite_managed(doc, sections.build(args.model))
    if args.stage in ("all", "figures"):
        push_figures(doc, args.model)
    if args.stage in ("all", "videos"):
        push_videos(doc, args.model)
    print("done")


if __name__ == "__main__":
    main()
