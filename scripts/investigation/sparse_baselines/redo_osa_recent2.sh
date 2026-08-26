#!/bin/bash
# Rerun the OSA rows of one model's study with the optimal config
# (num_recent_frames=2, the precision campaign's adopted setting):
# recalibrate every tier, redo the 720p/20 s sweep runs, quality PSNR +
# sheets, and the p1-p5 multi-prompt validation. Dense and the other
# methods' runs are untouched.
#
#   GPUS=0,2,6 PORT_BASE=51000 bash redo_osa_recent2.sh self_forcing
set -uo pipefail
MODEL="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GPUS="${GPUS:-0,1,2,3,6,7}"
PORT_BASE="${PORT_BASE:-51000}"

echo "=== redo_osa $MODEL: calibrate $(date -u +%H:%M:%S)"
python calibrate.py --model "$MODEL" --methods osa --workers 1 \
  --gpus "$GPUS" --port-base "$PORT_BASE"

echo "=== redo_osa $MODEL: purge stale osa rows $(date -u +%H:%M:%S)"
python - "$MODEL" <<'EOF'
import json, pathlib, sys
from common import ROOT
model = sys.argv[1]
root = ROOT / model
targets = ["results.json", "results_prompts.json"]
if model == "causal_forcing":
    targets.append("results_5s.json")
for name in targets:
    path = root / name
    if not path.exists():
        continue
    rows = json.loads(path.read_text())
    stale = [
        key
        for key in rows
        if key.startswith("osa_") or ("_osa_" in key and "osa2" not in key)
    ]
    for key in stale:
        del rows[key]
    path.write_text(json.dumps(rows, indent=2))
    print(f"{name}: purged {len(stale)} osa rows")
EOF

echo "=== redo_osa $MODEL: sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model "$MODEL" --methods osa --skip-dense --workers 2 \
  --gpus "$GPUS" --port-base "$((PORT_BASE + 100))"

if [ "$MODEL" = "causal_forcing" ]; then
  echo "=== redo_osa $MODEL: 5s sweep $(date -u +%H:%M:%S)"
  python run_sweep.py --model "$MODEL" --methods osa --skip-dense --workers 2 \
    --duration 5 --out results_5s.json --runs-dir runs_5s \
    --gpus "$GPUS" --port-base "$((PORT_BASE + 200))"
fi

echo "=== redo_osa $MODEL: quality $(date -u +%H:%M:%S)"
python quality.py --model "$MODEL"
if [ "$MODEL" = "causal_forcing" ]; then
  python quality.py --model "$MODEL" --duration 5 --out results_5s.json \
    --runs-dir runs_5s --sheet-suffix _5s
fi

echo "=== redo_osa $MODEL: multi-prompt $(date -u +%H:%M:%S)"
python multi_prompt.py --model "$MODEL" --methods osa --workers 2 \
  --gpus "$GPUS" --port-base "$((PORT_BASE + 300))"
# Sheets must carry the full method list; a --methods osa run would
# otherwise overwrite them with single-method sheets.
python multi_prompt.py --model "$MODEL" --sheets-only

echo "=== redo_osa $MODEL: plot $(date -u +%H:%M:%S)"
python plot.py --model "$MODEL"
echo "=== redo_osa $MODEL: DONE $(date -u +%H:%M:%S)"
