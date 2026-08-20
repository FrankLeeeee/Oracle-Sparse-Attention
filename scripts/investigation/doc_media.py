# SPDX-License-Identifier: Apache-2.0
"""Insert local images / videos into a Feishu doc at a chosen anchor block.

``docs +media-insert`` can only append to the end of a document, so putting a
figure in the middle is a two-step dance: insert, then ``block_move_after`` it
behind the anchor. Every helper here returns the new block id, which is the
anchor for the next insert — chaining them keeps a run of figures in order.

    from doc_media import insert_after, insert_blocks, last_block_id
    insert_blocks(DOC, anchor, "<p>text</p>")
    anchor = last_block_id(DOC, SECTION_H2)
    anchor = insert_after(DOC, anchor, "plot.png", caption="…")
"""

import json
import os
import pathlib
import re
import subprocess

ENV = {
    **os.environ,
    "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
    "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
}


def cli(*args, cwd: str | None = None) -> dict:
    proc = subprocess.run(
        ["lark-cli", *args, "--format", "json", "--as", "user"],
        capture_output=True,
        text=True,
        env=ENV,
        cwd=cwd,
    )
    raw = proc.stdout.strip() or proc.stderr.strip()
    payload = json.loads(raw)
    if not payload.get("ok"):
        raise RuntimeError(f"{' '.join(args[:2])} failed: {raw[:600]}")
    return payload["data"]


def insert_blocks(doc: str, anchor_id: str, content: str) -> None:
    """Insert XML blocks right after ``anchor_id``.

    ``block_insert_after`` does not report the ids of the plain text blocks it
    creates, so callers that need to keep chaining follow this with
    :func:`last_block_id`.
    """
    cli(
        "docs",
        "+update",
        "--doc",
        doc,
        "--command",
        "block_insert_after",
        "--block-id",
        anchor_id,
        "--content",
        content,
    )


def last_block_id(doc: str, section_start_id: str) -> str:
    """Id of the last block of the section headed by ``section_start_id``."""
    data = cli(
        "docs",
        "+fetch",
        "--doc",
        doc,
        "--scope",
        "section",
        "--start-block-id",
        section_start_id,
        "--detail",
        "with-ids",
    )
    ids = re.findall(r'id="([^"]+)"', data["document"]["content"])
    if not ids:
        raise RuntimeError("section fetch returned no block ids")
    return ids[-1]


def insert_after(
    doc: str,
    anchor_id: str,
    path: str,
    *,
    caption: str | None = None,
    media_type: str = "image",
    file_view: str | None = None,
    width: int | None = None,
) -> str:
    # media-insert refuses absolute paths, so run it from the file's directory.
    source = pathlib.Path(path).resolve()
    args = ["docs", "+media-insert", "--doc", doc, "--file", f"./{source.name}"]
    if media_type == "file":
        args += ["--type", "file"]
        if file_view:
            args += ["--file-view", file_view]
    else:
        if caption:
            args += ["--caption", caption]
        if width:
            args += ["--width", str(width)]
    new_id = cli(*args, cwd=str(source.parent))["block_id"]
    cli(
        "docs",
        "+update",
        "--doc",
        doc,
        "--command",
        "block_move_after",
        "--block-id",
        anchor_id,
        "--src-block-ids",
        new_id,
    )
    print(f"inserted {path} -> {new_id}", flush=True)
    return new_id
