"""Run a script under candidate_select_cluster/scripts/."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_CLUSTER_ROOT = Path(__file__).resolve().parent
_SCRIPTS_DIR = _CLUSTER_ROOT / "scripts"


def run_script(name: str) -> None:
    script = _SCRIPTS_DIR / name
    if not script.is_file():
        raise FileNotFoundError(script)
    sys.path.insert(0, str(_CLUSTER_ROOT))
    runpy.run_path(str(script), run_name="__main__")
