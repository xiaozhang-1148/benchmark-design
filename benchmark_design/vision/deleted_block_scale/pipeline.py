"""Batch Deleted-Block Scale analysis."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path

from benchmark_design.progress import parallel_map
from benchmark_design.vision.deleted_block_scale.compute import compute_page_deleted_block_scale
from benchmark_design.vision.deleted_block_scale.models import PageDeletedBlockScaleResult
from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.flow_structure.page_loader import load_page_annotations
from benchmark_design.vision.processing_options import VisionProcessingOptions


def compute_deleted_block_scale_results(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: str = "ours",
    input_dir_for_images: Path | None = None,
    pages: Sequence[PageAnnotation] | None = None,
) -> list[PageDeletedBlockScaleResult]:
    processing = processing or VisionProcessingOptions()
    image_dir = input_dir_for_images or input_dir
    if pages is None:
        pages = load_page_annotations(input_dir, dataset=dataset, processing=processing)
    compute_page = partial(compute_page_deleted_block_scale, input_dir=image_dir)
    return parallel_map(
        compute_page,
        pages,
        description="Computing deleted-block scale",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
