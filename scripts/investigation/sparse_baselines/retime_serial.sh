#!/bin/bash
# Serial exclusive-box retimes for every row that lands in a Walltime table.
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
python - <<'PY'
import json
from common import ROOT
for model, name, prefixes in (
    ("self_forcing", "results_5s.json", ["osasched_0.15"]),
    ("causal_forcing", "results_5s.json", ["osasched_0.15", "osasched_0.2", "osasched_0.3"]),
    ("causal_forcing_long", "results_30_seconds.json",
     ["p0_tokyo_dense", "p0_tokyo_osasched_0.2", "p0_tokyo_osasched_0.1",
      "p0_tokyo_lightforcing_0.2"]),
):
    path = ROOT / model / name
    if not path.exists():
        continue
    rows = json.loads(path.read_text())
    for key in [k for k in rows if any(k == p for p in prefixes)]:
        del rows[key]
    path.write_text(json.dumps(rows, indent=2))
    print(model, "purged for retime:", prefixes)
PY
python run_sweep.py --model self_forcing --methods osasched --skip-dense \
  --tiers 0.15 --duration 5 --res 720p --out results_5s.json --runs-dir runs_5s \
  --workers 1 --gpus 0,4,5,6 --port-base 58000
python run_sweep.py --model causal_forcing --methods osasched --skip-dense \
  --tiers 0.15 0.2 0.3 --duration 5 --res 720p --out results_5s.json \
  --runs-dir runs_5s --workers 1 --gpus 0,4,5,6 --port-base 58100
python final_round.py --model causal_forcing_long --duration 30 --prompts p0_tokyo \
  --methods osasched:0.2 osasched:0.1 lightforcing:0.2 \
  --out results_30_seconds.json --runs-dir runs_30_seconds \
  --sheet-prefix prompt_sheet_30_seconds_ --workers 1 --gpus 0,4,5,6 --port-base 58200
python quality.py --model self_forcing --duration 5 --out results_5s.json \
  --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s
python quality.py --model causal_forcing --duration 5 --out results_5s.json \
  --runs-dir runs_5s --tier 0.2 --sheet-suffix _5s
echo RETIME_DONE
