#!/usr/bin/env python3
"""Equal-weight fusion + spherical K-means clustering."""

from __future__ import annotations

import sys
from pathlib import Path

_CLUSTER_ROOT = Path(__file__).resolve().parents[1]
if str(_CLUSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_ROOT))

from bootstrap import setup

setup()

from src.equal_fusion_cluster.run import main

if __name__ == "__main__":
    raise SystemExit(main())
