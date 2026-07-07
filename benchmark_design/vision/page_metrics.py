"""Single-pass per-page vision metric computation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from benchmark_design.progress import parallel_map
from benchmark_design.vision.deleted_block_scale.compute import compute_page_deleted_block_scale
from benchmark_design.vision.deleted_block_scale.models import PageDeletedBlockScaleResult
from benchmark_design.vision.flow_structure.classifier import classify_page_flow_structure
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageFlowStructureResult
from benchmark_design.vision.flow_structure.page_loader import load_page_annotations
from benchmark_design.vision.foreground_load.compute import compute_page_foreground_load
from benchmark_design.vision.foreground_load.models import (
    ForegroundLoadThresholds,
    GlobalForegroundLoadConfig,
    PageForegroundLoadResult,
)
from benchmark_design.vision.foreground_load.pipeline import (
    assign_foreground_density_diagnostics,
    collect_darkness_histograms,
    compute_foreground_load_thresholds,
    estimate_global_foreground_config,
)
from benchmark_design.vision.masks import build_unified_page_masks
from benchmark_design.vision.processing_options import VisionProcessingOptions


@dataclass(frozen=True, slots=True)
class PageVisionMetricsResult:
    flow_structure: PageFlowStructureResult
    foreground_load: PageForegroundLoadResult
    deleted_block_scale: PageDeletedBlockScaleResult


@dataclass(frozen=True, slots=True)
class VisionBenchmarkResults:
    flow_structure: list[PageFlowStructureResult]
    foreground_load: list[PageForegroundLoadResult]
    deleted_block_scale: list[PageDeletedBlockScaleResult]
    foreground_load_thresholds: ForegroundLoadThresholds
    global_foreground_config: GlobalForegroundLoadConfig


def compute_page_vision_metrics(
    page: PageAnnotation,
    *,
    input_dir: Path | None = None,
    global_config: GlobalForegroundLoadConfig | None = None,
) -> PageVisionMetricsResult:
    if global_config is None:
        raise ValueError("global_config is required")
    image_dir = input_dir or Path(page.source_file).parent
    unified_masks = build_unified_page_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )
    flow_structure = classify_page_flow_structure(page, input_dir=image_dir)
    foreground_load = compute_page_foreground_load(
        page,
        input_dir=image_dir,
        unified_masks=unified_masks,
        global_config=global_config,
    )
    deleted_block_scale = compute_page_deleted_block_scale(
        page,
        input_dir=image_dir,
        unified_masks=unified_masks,
    )
    return PageVisionMetricsResult(
        flow_structure=flow_structure,
        foreground_load=foreground_load,
        deleted_block_scale=deleted_block_scale,
    )


def compute_vision_benchmark_results(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: str = "ours",
    input_dir_for_images: Path | None = None,
    pages: Sequence[PageAnnotation] | None = None,
) -> VisionBenchmarkResults:
    processing = processing or VisionProcessingOptions()
    image_dir = input_dir_for_images or input_dir
    if pages is None:
        pages = load_page_annotations(input_dir, dataset=dataset, processing=processing)

    provisional_config = GlobalForegroundLoadConfig(
        tau_D=0.5,
        threshold_method="pooled_otsu",
    )
    histograms = collect_darkness_histograms(
        pages,
        input_dir=image_dir,
        processing=processing,
        global_config=provisional_config,
    )
    global_config = estimate_global_foreground_config(histograms)

    compute_page = partial(
        compute_page_vision_metrics,
        input_dir=image_dir,
        global_config=global_config,
    )
    page_results = parallel_map(
        compute_page,
        pages,
        description="Computing vision metrics",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
    flow_results = [result.flow_structure for result in page_results]
    fg_raw = [result.foreground_load for result in page_results]
    dbs_results = [result.deleted_block_scale for result in page_results]
    fg_thresholds = compute_foreground_load_thresholds(fg_raw, global_config=global_config)
    fg_results = assign_foreground_density_diagnostics(fg_raw, fg_thresholds)
    return VisionBenchmarkResults(
        flow_structure=flow_results,
        foreground_load=fg_results,
        deleted_block_scale=dbs_results,
        foreground_load_thresholds=fg_thresholds,
        global_foreground_config=global_config,
    )
