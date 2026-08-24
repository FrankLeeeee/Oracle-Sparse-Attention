#!/bin/bash
# Re-do one model's STA rows after an STA config change: recalibrate at 720p,
# drop the stale rows and their videos, re-run the sweep and multi-prompt
# slices, then regenerate the figures that show them.
#
#   setsid nohup bash redo_sta.sh <model> > .../redo_sta_<model>.log 2>&1 & disown
set -uo pipefail
MODEL="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GPUS="${GPUS:-3,4,5,6,7}"

echo "=== redo_sta $MODEL: calibrate $(date -u +%H:%M:%S)"
python calibrate.py --model "$MODEL" --methods sta --res 720p --workers 1 \
  --gpus "$GPUS" --port-base "${PORT_BASE:-33000}"

echo "=== redo_sta $MODEL: drop stale rows $(date -u +%H:%M:%S)"
python - "$MODEL" <<'PY'
import json, pathlib, shutil, sys
sys.path.insert(0, ".")
from common import ROOT

model = sys.argv[1]
root = ROOT / model
for name in ("results.json", "results_prompts.json"):
    path = root / name
    if not path.exists():
        continue
    rows = json.loads(path.read_text())
    stale = [key for key in rows if "sta_" in key]
    for key in stale:
        rows.pop(key)
    path.write_text(json.dumps(rows, indent=2))
    print(f"{name}: dropped {len(stale)}")
for sub in ("runs", "runs_prompts"):
    directory = root / sub
    if not directory.exists():
        continue
    for child in list(directory.iterdir()):
        if "sta_" in child.name:
            shutil.rmtree(child)
PY

echo "=== redo_sta $MODEL: sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model "$MODEL" --methods sta --skip-dense --workers 2 \
  --gpus "$GPUS" --port-base "${PORT_BASE:-33000}"
echo "=== redo_sta $MODEL: quality $(date -u +%H:%M:%S)"
python quality.py --model "$MODEL"
echo "=== redo_sta $MODEL: multi_prompt $(date -u +%H:%M:%S)"
python multi_prompt.py --model "$MODEL" --methods sta --workers 2 \
  --gpus "$GPUS" --port-base "${PORT_BASE2:-34000}"
echo "=== redo_sta $MODEL: sheets + plot $(date -u +%H:%M:%S)"
python multi_prompt.py --model "$MODEL" --sheets-only
python plot.py --model "$MODEL"
echo "=== redo_sta $MODEL: DONE $(date -u +%H:%M:%S)"
