import json
import os
import subprocess
import sys

DOC = "Hx3OdnyKNomq9JxV9jfcYkf8n9d"
ENV = {
    **os.environ,
    "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
    "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
}


def cli(*args):
    p = subprocess.run(
        ["lark-cli", *args, "--format", "json"], capture_output=True, text=True, env=ENV
    )
    out = p.stdout.strip() or p.stderr.strip()
    d = json.loads(out)
    if not d.get("ok"):
        raise RuntimeError(f"{args[1]} failed: {out[:400]}")
    return d["data"]


def replace(old_id, path, *, media_type="image", caption=None):
    ins = ["docs", "+media-insert", "--doc", DOC, "--file", path]
    if media_type == "file":
        ins += ["--type", "file"]
    elif caption:
        ins += ["--caption", caption]
    new_id = cli(*ins)["block_id"]
    cli(
        "docs",
        "+update",
        "--doc",
        DOC,
        "--command",
        "block_move_after",
        "--block-id",
        old_id,
        "--src-block-ids",
        new_id,
    )
    cli(
        "docs",
        "+update",
        "--doc",
        DOC,
        "--command",
        "block_delete",
        "--block-id",
        old_id,
    )
    print("OK", path, flush=True)


if __name__ == "__main__":
    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        old_id, path, mtype = parts[0], parts[1], parts[2]
        caption = parts[3] if len(parts) > 3 else None
        replace(old_id, path, media_type=mtype, caption=caption)
