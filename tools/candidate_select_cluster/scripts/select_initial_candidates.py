#!/usr/bin/env python3
"""Select 8 initial candidates per qualifying Batch02 question."""

from __future__ import annotations

import sys
from pathlib import Path

_CLUSTER_ROOT = Path(__file__).resolve().parents[1]
if str(_CLUSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_ROOT))

from bootstrap import setup

setup()

from src.initial_candidate_select.run_select import main

if __name__ == "__main__":
    raise SystemExit(main())
