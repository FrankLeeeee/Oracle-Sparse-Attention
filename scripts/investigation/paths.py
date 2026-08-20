# SPDX-License-Identifier: Apache-2.0
"""Where investigation scripts live and where their output goes.

Scripts sit in ``scripts/investigation/<topic>/`` and are version controlled;
everything they produce — run directories, figures, json, videos — goes to
``results/investigation/<topic>/``, which is gitignored. A script reads its own
assets (a sibling helper, a doc fragment) relative to its own directory and
writes everything else under :func:`results_dir`.

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from paths import REPO, results_dir

    HERE = pathlib.Path(__file__).resolve().parent   # scripts
    ROOT = results_dir("chunk_runtime")              # outputs
"""

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO / "results" / "investigation"


def results_dir(topic: str) -> pathlib.Path:
    """The output directory for one investigation topic, created on demand."""
    path = RESULTS_ROOT / topic
    path.mkdir(parents=True, exist_ok=True)
    return path
