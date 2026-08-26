#!/bin/bash
# Calibrate + 5 s-sweep + quality for the demand-scheduled OSA variant.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GPUS="${GPUS:-0,5,6}"
echo "=== osasched: calibrate $(date -u +%H:%M:%S)"
python calibrate.py --model self_forcing --methods osasched --workers 2 \
  --gpus "$GPUS" --port-base 54000
echo "=== osasched: 5s sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model self_forcing --methods osasched --skip-dense \
  --tiers 0.1 0.2 0.3 --duration 5 --res 720p \
  --out results_5s.json --runs-dir runs_5s --workers 2 \
  --gpus "$GPUS" --port-base 54200
echo "=== osasched: quality $(date -u +%H:%M:%S)"
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.3 --sheet-suffix _5s
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s --sheet-only
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.1 --sheet-suffix _5s --sheet-only
echo "=== osasched: CHAIN_DONE $(date -u +%H:%M:%S)"
