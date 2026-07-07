"""Deleted-Block Scale metric package."""

from benchmark_design.vision.deleted_block_scale.models import (
    BlockDeletedScaleGeometryRecord,
    PageDeletedBlockScaleResult,
)
from benchmark_design.vision.deleted_block_scale.pipeline import compute_deleted_block_scale_results

__all__ = [
    "BlockDeletedScaleGeometryRecord",
    "PageDeletedBlockScaleResult",
    "compute_deleted_block_scale_results",
]
