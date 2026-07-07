"""Fast polygon rasterization for foreground load masks (re-exports unified masks)."""

from __future__ import annotations

from benchmark_design.vision.masks import (
    PageMaskBundle,
    build_page_masks,
    build_unified_page_masks,
    polygon_out_of_bounds,
    rasterize_polygon,
    union_effective_masks,
)

__all__ = [
    "PageMaskBundle",
    "build_page_masks",
    "build_unified_page_masks",
    "polygon_out_of_bounds",
    "rasterize_polygon",
    "union_effective_masks",
]
