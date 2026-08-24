#!/bin/bash
# Wait for the targeted STA redo to finish, then run the full chain so the
# remaining multi-prompt rows (the six non-STA methods) are filled in. The two
# must not overlap: they write the same results file.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
while ps -eo cmd | grep -q "[r]edo_sta.sh longlive2"; do sleep 60; done
echo "=== STA redo finished, running the full chain $(date -u +%H:%M:%S)"
exec env WORKERS=2 GPUS=3,4,5,6,7 PORT_BASE=43000 PORT_BASE2=44000 bash chain.sh longlive2
