"""Block-level domain — flow structure classification and dataset scaffolding."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure import (
    BlockGeometryRecord,
    PageFlowStructureResult,
    classify_page_flow_structure,
    load_page_annotations,
)
from benchmark_design.block_level.flow_structure.pipeline import compute_flow_structure_results
from benchmark_design.block_level.page_metrics import (
    BlockLevelBenchmarkResults,
    compute_block_level_benchmark_results,
)
from benchmark_design.block_level.processing_options import BlockLevelProcessingOptions
from benchmark_design.block_level.sample_record import ImageSampleRecord

__all__ = [
    "BlockGeometryRecord",
    "BlockLevelBenchmarkResults",
    "BlockLevelProcessingOptions",
    "ImageSampleRecord",
    "PageFlowStructureResult",
    "classify_page_flow_structure",
    "compute_block_level_benchmark_results",
    "compute_flow_structure_results",
    "load_page_annotations",
]

# Backward-compatible alias (deprecated).
VisionProcessingOptions = BlockLevelProcessingOptions
