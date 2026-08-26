#!/bin/bash
# Self-Forcing 5 s study (the model's trained duration): dense + all baselines
# + the OSA anchor progression (osa2 none -> osa2s +sink -> osa2a +sink+recent
# -> osa all), tiers 0.1/0.2/0.3 only, 720p.
#
#   GPUS=6 bash sf_5s_study.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GPUS="${GPUS:-6}"
PORT_BASE="${PORT_BASE:-52000}"
METHODS="osa osa2 osa2s osa2a lightforcing radial svg1 svg2 xattention sta"

echo "=== sf5s: calibrate osa2s $(date -u +%H:%M:%S)"
python calibrate.py --model self_forcing --methods osa2s --workers 1 \
  --gpus "$GPUS" --port-base "$PORT_BASE"

echo "=== sf5s: replay-only configs guard $(date -u +%H:%M:%S)"
python calibrate.py --model self_forcing --replay-only

echo "=== sf5s: 720p/5s sweep $(date -u +%H:%M:%S)"
python run_sweep.py --model self_forcing --methods $METHODS \
  --tiers 0.1 0.2 0.3 --duration 5 --res 720p \
  --out results_5s.json --runs-dir runs_5s --workers 1 \
  --gpus "$GPUS" --port-base "$((PORT_BASE + 100))"

echo "=== sf5s: quality + sheets $(date -u +%H:%M:%S)"
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.3 --sheet-suffix _5s
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s --sheet-only
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.1 --sheet-suffix _5s --sheet-only

echo "=== sf5s: CHAIN_DONE $(date -u +%H:%M:%S)"
