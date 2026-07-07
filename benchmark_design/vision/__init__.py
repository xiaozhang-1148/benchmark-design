"""Vision domain — image-side benchmark statistics and export."""

from __future__ import annotations

from benchmark_design.vision.flow_structure import (
    BlockGeometryRecord,
    PageFlowStructureResult,
    classify_page_flow_structure,
    load_page_annotations,
)
from benchmark_design.vision.flow_structure.pipeline import compute_flow_structure_results
from benchmark_design.vision.foreground_load.pipeline import compute_foreground_load_results
from benchmark_design.vision.processing_options import VisionProcessingOptions
from benchmark_design.vision.sample_record import ImageSampleRecord

__all__ = [
    "BlockGeometryRecord",
    "ImageSampleRecord",
    "PageFlowStructureResult",
    "VisionProcessingOptions",
    "classify_page_flow_structure",
    "compute_flow_structure_results",
    "compute_foreground_load_results",
    "load_page_annotations",
]
