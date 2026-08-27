#!/bin/bash
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,4,5,6}"
echo "=== flat: 5s sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model self_forcing --methods osasched --skip-dense \
  --tiers 0.1 0.2 0.3 --duration 5 --res 720p \
  --out results_5s.json --runs-dir runs_5s --workers 3 \
  --gpus "$GPUS" --port-base 55000
echo "=== flat: 5s quality $(date -u +%H:%M:%S)"
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.1 --sheet-suffix _5s
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s --sheet-only
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.3 --sheet-suffix _5s --sheet-only
echo "=== flat: 20s sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model self_forcing --methods osasched --skip-dense \
  --tiers 0.1 0.2 0.3 --duration 20 --res 720p --workers 3 \
  --gpus "$GPUS" --port-base 55200
echo "=== flat: FLAT_DONE $(date -u +%H:%M:%S)"
