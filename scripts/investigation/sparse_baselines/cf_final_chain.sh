#!/bin/bash
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
GPUS="0,4,5,6"
PROMPTS="p1_forest p2_plating p3_raccoon p4_teacup p5_tsunami"
echo "=== cf-final: p0 osasched tiers $(date -u +%H:%M:%S)"
python run_sweep.py --model causal_forcing --methods osasched --skip-dense \
  --tiers 0.15 0.2 0.3 --duration 5 --res 720p --out results_5s.json --runs-dir runs_5s \
  --workers 2 --gpus "$GPUS" --port-base 57300
echo "=== cf-final: multi-prompt trio $(date -u +%H:%M:%S)"
python final_round.py --model causal_forcing --duration 5 --prompts $PROMPTS \
  --methods osasched:0.2 lightforcing:0.2 \
  --out results_prompts_5s.json --runs-dir runs_prompts_5s \
  --sheet-prefix prompt_sheet_5_seconds_ --gpus "$GPUS" --workers 2 --port-base 57400
echo "=== cf-final: p0 quality refresh $(date -u +%H:%M:%S)"
python quality.py --model causal_forcing --duration 5 \
  --out results_5s.json --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s
echo "=== cf-final: 30-second long-video trio $(date -u +%H:%M:%S)"
python final_round.py --model causal_forcing_long --duration 30 --prompts p0_tokyo $PROMPTS \
  --methods osasched:0.2 lightforcing:0.2 \
  --out results_30_seconds.json --runs-dir runs_30_seconds \
  --sheet-prefix prompt_sheet_30_seconds_ --gpus "$GPUS" --workers 2 --port-base 57500
echo "=== cf-final: CF_FINAL_DONE $(date -u +%H:%M:%S)"
