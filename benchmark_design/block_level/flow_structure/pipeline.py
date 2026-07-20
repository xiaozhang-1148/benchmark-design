"""Run Answer-Block Flow Structure analysis over a benchmark directory."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path

from benchmark_design.progress import parallel_map
from benchmark_design.block_level.flow_structure.classifier import classify_page_flow_structure
from benchmark_design.block_level.flow_structure.models import PageAnnotation, PageFlowStructureResult
from benchmark_design.block_level.flow_structure.page_loader import load_page_annotations
from benchmark_design.block_level.processing_options import VisionProcessingOptions


def compute_flow_structure_results(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: str = "ours",
    input_dir_for_images: Path | None = None,
    pages: Sequence[PageAnnotation] | None = None,
) -> list[PageFlowStructureResult]:
    processing = processing or VisionProcessingOptions()
    image_dir = input_dir_for_images or input_dir
    if pages is None:
        pages = load_page_annotations(input_dir, dataset=dataset, processing=processing)
    classify_page = partial(classify_page_flow_structure, input_dir=image_dir)
    return parallel_map(
        classify_page,
        pages,
        description="Classifying flow structure",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
