#!/usr/bin/env python3
"""Global-cluster random dedupe (2592 → 1813 / adjusted_2)."""

from __future__ import annotations

import sys
from pathlib import Path

_CLUSTER_ROOT = Path(__file__).resolve().parents[1]
if str(_CLUSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_ROOT))

from bootstrap import setup

setup()

from src.global_random_dedupe.run_dedupe import main

if __name__ == "__main__":
    raise SystemExit(main())
