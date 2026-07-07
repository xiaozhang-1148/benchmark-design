"""Unified HMER + Vision benchmark export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.config.shared import DEFAULT_OUTPUT_ROOT, DEFAULT_UNIFIED_OUTPUT_ROOT
from benchmark_design.config.vision import DEFAULT_VISION_OUTPUT_ROOT
from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus_cached
from benchmark_design.progress import default_worker_count, run_parallel_tasks
from benchmark_design.report.dataset_overview import run_dataset_overview_export
from benchmark_design.report.export_pipeline import run_benchmark_export
from benchmark_design.report.vision.export_pipeline import run_vision_benchmark_export
from benchmark_design.vision.dataset import load_vision_benchmark_dataset_cached
from benchmark_design.vision.processing_options import VisionProcessingOptions


@dataclass(frozen=True, slots=True)
class UnifiedExportResult:
    output_root: Path
    hmer_output: Path
    vision_output: Path
    hmer_manifest: dict[str, Path]
    vision_manifest: dict[str, str]
    dataset_overview: Path


def run_unified_benchmark_export(
    input_dir: Path,
    output_root: Path,
    *,
    hmer_output: Path | None = None,
    vision_output: Path | None = None,
    processing: ProcessingOptions | None = None,
    vision_processing: VisionProcessingOptions | None = None,
    skip_hmer_figures: bool = False,
    skip_cross_benchmark: bool = False,
    cross_benchmark_datasets: list[str] | None = None,
    skip_flow_figures: bool = False,
    skip_foreground_load_figures: bool = False,
    skip_deleted_block_scale_figures: bool = False,
) -> UnifiedExportResult:
    """Load shared inputs once, then export HMER, Vision, and overview in parallel."""
    processing = processing or ProcessingOptions()
    vision_processing = vision_processing or VisionProcessingOptions()
    output_root.mkdir(parents=True, exist_ok=True)

    hmer_dir = hmer_output or (output_root / DEFAULT_OUTPUT_ROOT.name)
    vision_dir = vision_output or (output_root / DEFAULT_VISION_OUTPUT_ROOT.name)

    load_workers = min(2, default_worker_count())
    with ThreadPoolExecutor(max_workers=load_workers) as load_pool:
        enriched_future = load_pool.submit(
            build_enriched_corpus_cached,
            "ours",
            input_dir,
            processing,
        )
        vision_dataset_future = load_pool.submit(
            load_vision_benchmark_dataset_cached,
            input_dir,
            processing=vision_processing,
        )
        enriched = enriched_future.result()
        vision_dataset = vision_dataset_future.result()

    export_workers = min(3, default_worker_count())
    export_results = run_parallel_tasks(
        {
            "hmer": lambda: run_benchmark_export(
                input_dir,
                hmer_dir,
                processing=processing,
                enriched=enriched,
                skip_figures=skip_hmer_figures,
                skip_cross_benchmark=skip_cross_benchmark,
                cross_benchmark_datasets=cross_benchmark_datasets,
                skip_dataset_overview=True,
            ),
            "vision": lambda: run_vision_benchmark_export(
                input_dir,
                vision_dir,
                processing=vision_processing,
                dataset=vision_dataset,
                skip_flow_figures=skip_flow_figures,
                skip_foreground_load_figures=skip_foreground_load_figures,
                skip_deleted_block_scale_figures=skip_deleted_block_scale_figures,
                skip_dataset_overview=True,
            ),
            "overview": lambda: run_dataset_overview_export(
                input_dir,
                output_root,
                processing=processing,
                vision_processing=vision_processing,
                enriched=enriched,
                vision_samples=vision_dataset.samples,
                vision_pages=vision_dataset.pages,
                skip_domain_overviews=True,
            ),
        },
        description="Running unified benchmark export",
        show_progress=processing.show_progress or vision_processing.show_progress,
        workers=export_workers,
    )

    hmer_manifest = export_results["hmer"]
    vision_manifest = export_results["vision"]
    overview_path = export_results["overview"]

    return UnifiedExportResult(
        output_root=output_root,
        hmer_output=hmer_dir,
        vision_output=vision_dir,
        hmer_manifest=hmer_manifest,
        vision_manifest=vision_manifest,
        dataset_overview=overview_path,
    )
