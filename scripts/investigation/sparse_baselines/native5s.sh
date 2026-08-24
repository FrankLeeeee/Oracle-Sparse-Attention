#!/bin/bash
# A second sweep at the model's native training length. Causal Forcing is a
# 5-second (81-frame) model and upstream explicitly warns against judging it
# on long video; at 20 s even its dense output collapses, so the 20 s figures
# measure sparse attention against a broken reference. This run keeps the same
# calibrated configs but generates 81 frames, where the model is in
# distribution and the comparison means something.
set -uo pipefail
MODEL="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GPUS="${GPUS:-3,4,5,6,7}"
echo "=== native5s $MODEL: sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model "$MODEL" --duration 5 --workers 2 --gpus "$GPUS" \
  --out results_5s.json --runs-dir runs_5s --port-base "${PORT_BASE:-47000}"
echo "=== native5s $MODEL: quality $(date -u +%H:%M:%S)"
python quality.py --model "$MODEL" --duration 5 --out results_5s.json \
  --runs-dir runs_5s --sheet-suffix _5s
echo "=== native5s $MODEL: DONE $(date -u +%H:%M:%S)"
