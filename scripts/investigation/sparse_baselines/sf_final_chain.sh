#!/bin/bash
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
GPUS="0,4,5,6"
PROMPTS="p1_forest p2_plating p3_raccoon p4_teacup p5_tsunami"
echo "=== sf-final: p0 osasched 0.15 $(date -u +%H:%M:%S)"
python run_sweep.py --model self_forcing --methods osasched --skip-dense \
  --tiers 0.15 --duration 5 --res 720p --out results_5s.json --runs-dir runs_5s \
  --workers 1 --gpus "$GPUS" --port-base 57000
echo "=== sf-final: multi-prompt trio $(date -u +%H:%M:%S)"
python final_round.py --model self_forcing --duration 5 --prompts $PROMPTS \
  --methods osasched:0.2 lightforcing:0.2 \
  --out results_prompts_5s.json --runs-dir runs_prompts_5s \
  --sheet-prefix prompt_sheet_5_seconds_ --gpus "$GPUS" --workers 2 --port-base 57100
echo "=== sf-final: p0 quality refresh $(date -u +%H:%M:%S)"
python quality.py --model self_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s
echo "=== sf-final: SF_FINAL_DONE $(date -u +%H:%M:%S)"
