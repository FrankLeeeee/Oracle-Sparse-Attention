# One-off recovery chain from the 2026-08-19 OSA round; run from this directory.
set -e
cd "$(dirname "$0")"
RESULTS="$(cd ../../.. && pwd)/results/investigation/sparse_osa"
python3 - "$RESULTS" <<'PY'
import json, pathlib, sys
from common import run_generate
ROOT = pathlib.Path(sys.argv[1])
configs = json.loads((ROOT / "configs.json").read_text())
results = json.loads((ROOT / "results.json").read_text())
r = run_generate(out_dir=ROOT / "runs" / "osa_0.2", log_name="timing.log", gpu=4,
                 port_base=39500, width=1280, height=720, num_frames=321,
                 method="osa", method_config=configs["osa"]["0.2"]["config"],
                 save_output=True, timeout_s=2400)
r["config"] = configs["osa"]["0.2"]["config"]
results["osa_0.2"] = r
(ROOT / "results.json").write_text(json.dumps(results, indent=2))
print("OSA02 DONE", r.get("denoise_s"), r.get("e2e_s"), r.get("density"), flush=True)
PY
python quality.py --sheet-target 0.3
python multi_prompt.py --gpus 4,7
python plot.py
echo "CHAIN DONE"
