"""Page-level pure image analysis configuration."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.page_level.models import AspectRatioBin

DEFAULT_PAGE_LEVEL_OUTPUT = Path("page_level")

DEFAULT_ASPECT_RATIO_BINS: tuple[AspectRatioBin, ...] = (
    AspectRatioBin(name="portrait", min_ratio=0.0, max_ratio=0.90),
    AspectRatioBin(name="near_square", min_ratio=0.90, max_ratio=1.20),
    AspectRatioBin(name="landscape", min_ratio=1.20, max_ratio=999.0),
)
