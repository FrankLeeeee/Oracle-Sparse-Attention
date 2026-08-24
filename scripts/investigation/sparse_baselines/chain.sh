#!/bin/bash
# One model's full campaign after calibration: sweep -> quality -> multi-prompt
# -> plot. Launch detached:
#   setsid nohup bash chain.sh <model> > .../chain_<model>.log 2>&1 & disown
set -uo pipefail
MODEL="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=== chain $MODEL: run_sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model "$MODEL" --workers "${WORKERS:-2}" --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --port-base "${PORT_BASE:-36000}"
echo "=== chain $MODEL: quality $(date -u +%H:%M:%S)"
python quality.py --model "$MODEL"
echo "=== chain $MODEL: multi_prompt $(date -u +%H:%M:%S)"
python multi_prompt.py --model "$MODEL" --workers "${WORKERS:-2}" --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --port-base "${PORT_BASE2:-37000}"
echo "=== chain $MODEL: plot $(date -u +%H:%M:%S)"
python plot.py --model "$MODEL"
echo "=== chain $MODEL: ALL STAGES DONE $(date -u +%H:%M:%S)"
