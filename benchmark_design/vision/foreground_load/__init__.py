"""Effective-region foreground load metrics."""

from __future__ import annotations

from benchmark_design.vision.foreground_load.models import (
    BlockForegroundLoadResult,
    ForegroundLoadThresholds,
    PageForegroundLoadResult,
)
from benchmark_design.vision.foreground_load.pipeline import compute_foreground_load_results

__all__ = [
    "BlockForegroundLoadResult",
    "ForegroundLoadThresholds",
    "PageForegroundLoadResult",
    "compute_foreground_load_results",
]
