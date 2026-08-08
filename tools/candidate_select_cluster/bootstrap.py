"""Path bootstrap for candidate_select_cluster scripts."""

from __future__ import annotations

import sys
from pathlib import Path

CLUSTER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLUSTER_ROOT.parent


def setup() -> None:
    """Ensure repo root and cluster root are importable."""
    for path in (REPO_ROOT, CLUSTER_ROOT):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
