"""Unified HMER + page-level + line-level + split project export."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from benchmark_design.config.page_level_latex_split import DEFAULT_PAGE_LEVEL_LATEX_SPLIT_CONFIG
from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus_cached
from benchmark_design.page_level_latex.pipeline import run_page_level_latex_export
from benchmark_design.foreground.analysis import export_foreground_analysis
from benchmark_design.foreground.calibration import calibration_to_threshold_config
from benchmark_design.page_level_latex.split_inputs import prepare_split_inputs
from benchmark_design.page_level_latex_split.audit import SplitAcceptanceError
from benchmark_design.page_level_latex_split.config import load_split_config
from benchmark_design.page_level_latex_split.pipeline import run_page_level_latex_split
from benchmark_design.progress import default_worker_count, partition_workers, run_parallel_tasks
from benchmark_design.export_layout import (
    BenchmarkExportLayout,
    HYBRID_LAYOUT_FLOW_STRUCTURES,
    STRUCTURE_LAYOUT_FLOW_STRUCTURES,
    prune_empty_directories,
    write_export_pipeline_doc,
)
from benchmark_design.block_level.page_metrics import compute_block_level_benchmark_results
from benchmark_design.block_level.block_foreground_density import compute_block_foreground_densities
from benchmark_design.project.models import ProjectExportResult
from benchmark_design.project.pipeline_manifest import build_pipeline_manifest, write_pipeline_manifest
from benchmark_design.project.summary import build_project_summary, write_project_summary
from benchmark_design.report.block_level.export_pipeline import (
    run_block_level_export,
    write_block_density_exports,
)
from benchmark_design.report.dataset_overview import compute_dataset_overview, run_dataset_overview_export
from benchmark_design.report.export_pipeline import run_benchmark_export
from benchmark_design.report.line_level.export_pipeline import run_line_level_export
from benchmark_design.report.page_level.export_pipeline import run_page_level_export
from benchmark_design.block_level.dataset import load_block_level_benchmark_dataset_cached
from benchmark_design.block_level.processing_options import BlockLevelProcessingOptions


def run_project_export(
    input_dir: Path,
    output_root: Path,
    *,
    hmer_output: Path | None = None,
    block_level_output: Path | None = None,
    page_level_output: Path | None = None,
    line_level_output: Path | None = None,
    page_level_hmer_output: Path | None = None,
    page_level_latex_split_output: Path | None = None,
    processing: ProcessingOptions | None = None,
    block_level_processing: BlockLevelProcessingOptions | None = None,
    skip_hmer_figures: bool = False,
    skip_cross_benchmark: bool = False,
    cross_benchmark_datasets: list[str] | None = None,
    skip_flow_figures: bool = False,
    skip_page_level: bool = False,
    skip_page_level_figures: bool = False,
    page_level_config: Path | None = None,
    skip_line_level: bool = False,
    skip_line_level_figures: bool = False,
    line_level_config: Path | None = None,
    skip_page_level_hmer: bool = False,
    skip_page_level_hmer_figures: bool = False,
    skip_page_level_latex_split: bool = True,
    page_level_latex_split_config: Path | None = None,
    skip_page_level_latex_split_figures: bool = False,
    allow_split_acceptance_failure: bool = False,
    # Backward-compatible aliases (deprecated).
    vision_output: Path | None = None,
    vision_processing: BlockLevelProcessingOptions | None = None,
) -> ProjectExportResult:
    """Load shared inputs once, export all benchmark layers, and write summary.json."""
    processing = processing or ProcessingOptions()
    block_level_processing = (
        block_level_processing
        or vision_processing
        or BlockLevelProcessingOptions()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    layout = BenchmarkExportLayout(output_root)

    hmer_dir = hmer_output or layout.hmer
    structure_layout_dir = (
        block_level_output
        or vision_output
        or layout.structure_layout
    )
    hybrid_layout_dir = layout.hybrid_layout
    page_level_dir = page_level_output or layout.page_level
    line_level_dir = line_level_output or layout.line_level
    page_level_hmer_dir = page_level_hmer_output or layout.page_level_hmer
    split_dir = page_level_latex_split_output or layout.page_level_latex_split
    split_config_path = page_level_latex_split_config or DEFAULT_PAGE_LEVEL_LATEX_SPLIT_CONFIG

    load_workers = min(2, default_worker_count())
    with ThreadPoolExecutor(max_workers=load_workers) as load_pool:
        enriched_future = load_pool.submit(
            build_enriched_corpus_cached,
            "ours",
            input_dir,
            processing,
        )
        block_dataset_future = load_pool.submit(
            load_block_level_benchmark_dataset_cached,
            input_dir,
            processing=block_level_processing,
        )
        enriched = enriched_future.result()
        block_dataset = block_dataset_future.result()

    block_benchmark_results = compute_block_level_benchmark_results(
        input_dir,
        processing=block_level_processing,
        input_dir_for_images=input_dir,
        pages=block_dataset.pages,
    )
    all_flow_results = block_benchmark_results.flow_structure

    phase1_task_count = 3  # HMER + structure_layout + hybrid_layout
    if not skip_page_level:
        phase1_task_count += 1
    total_workers = processing.workers
    per_task_workers = partition_workers(total_workers, phase1_task_count)
    hmer_processing = (
        replace(processing, workers=per_task_workers)
        if total_workers is not None
        else processing
    )
    block_level_processing_effective = (
        replace(block_level_processing, workers=per_task_workers)
        if total_workers is not None
        else block_level_processing
    )
    export_workers = min(phase1_task_count, default_worker_count())
    page_gray_cache_root: Path | None = None
    if not skip_page_level:
        page_gray_cache_root = page_level_dir / ".cache" / "gray"
        page_gray_cache_root.mkdir(parents=True, exist_ok=True)
    phase1_tasks: dict[str, object] = {
        "hmer": lambda: run_benchmark_export(
            input_dir,
            hmer_dir,
            processing=hmer_processing,
            enriched=enriched,
            skip_figures=skip_hmer_figures,
            skip_cross_benchmark=skip_cross_benchmark,
            cross_benchmark_datasets=cross_benchmark_datasets,
            skip_dataset_overview=True,
        ),
        "structure_layout": lambda: run_block_level_export(
            input_dir,
            structure_layout_dir,
            processing=block_level_processing_effective,
            dataset=block_dataset,
            flow_results=all_flow_results,
            flow_structures=STRUCTURE_LAYOUT_FLOW_STRUCTURES,
            layout_kind="structure_layout",
            skip_flow_figures=skip_flow_figures,
            skip_dataset_overview=True,
        ),
        "hybrid_layout": lambda: run_block_level_export(
            input_dir,
            hybrid_layout_dir,
            processing=block_level_processing_effective,
            dataset=block_dataset,
            flow_results=all_flow_results,
            flow_structures=HYBRID_LAYOUT_FLOW_STRUCTURES,
            layout_kind="hybrid_layout",
            skip_flow_figures=skip_flow_figures,
            skip_dataset_overview=True,
        ),
    }
    if not skip_page_level:
        gray_cache_root = page_gray_cache_root
        phase1_tasks["page_level"] = lambda: run_page_level_export(
            input_dir,
            page_level_dir,
            config_path=page_level_config,
            workers=per_task_workers,
            show_progress=processing.show_progress or block_level_processing.show_progress,
            skip_figures=skip_page_level_figures,
            gray_cache_root=gray_cache_root,
            cleanup_gray_cache=False,
        )

    phase1_results = run_parallel_tasks(
        phase1_tasks,
        description="Running benchmark export (phase 1)",
        show_progress=processing.show_progress or block_level_processing.show_progress,
        workers=export_workers,
    )

    hmer_manifest = phase1_results["hmer"]
    structure_layout_manifest = phase1_results["structure_layout"]
    hybrid_layout_manifest = phase1_results["hybrid_layout"]
    page_level_manifest = phase1_results.get("page_level")

    calibration_result = None
    block_density_manifest: dict[str, str] | None = None
    block_density_rows = None
    try:
        if not skip_page_level and layout.density_calibration.is_file():
            from benchmark_design.line_level.bbox_ink import load_calibration_result

            calibration_result = load_calibration_result(layout.density_calibration)
            block_density_rows = compute_block_foreground_densities(
                list(block_dataset.pages),
                input_dir=input_dir,
                calibration=calibration_result,
                show_progress=processing.show_progress or block_level_processing.show_progress,
                workers=total_workers if total_workers is not None else block_level_processing.workers,
                gray_cache_root=page_gray_cache_root,
            )
            block_density_manifest = write_block_density_exports(
                pages=block_dataset.pages,
                flow_results=all_flow_results,
                input_dir=input_dir,
                calibration=calibration_result,
                structure_layout_dir=structure_layout_dir,
                hybrid_layout_dir=hybrid_layout_dir,
                block_level_dir=layout.block_level,
                processing=replace(
                    block_level_processing,
                    workers=total_workers if total_workers is not None else block_level_processing.workers,
                ),
                gray_cache_root=page_gray_cache_root,
                density_rows=block_density_rows,
            )
            if calibration_result is not None and page_level_manifest is not None:
                threshold_config = calibration_to_threshold_config(calibration_result)
                export_foreground_analysis(
                    output_dir=layout.foreground,
                    calibration=calibration_result,
                    threshold_config=threshold_config,
                    block_rows=block_density_rows,
                )
    finally:
        if page_gray_cache_root is not None:
            shutil.rmtree(page_gray_cache_root.parent, ignore_errors=True)

    phase2_task_count = 0
    if not skip_line_level:
        phase2_task_count += 1
    if not skip_page_level_hmer:
        phase2_task_count += 1
    phase2_workers = partition_workers(total_workers, phase2_task_count) if phase2_task_count else None
    phase2_tasks: dict[str, object] = {}
    if not skip_line_level:
        phase2_tasks["line_level"] = lambda: run_line_level_export(
            input_dir,
            line_level_dir,
            config_path=line_level_config,
            workers=phase2_workers,
            show_progress=processing.show_progress or block_level_processing.show_progress,
            skip_figures=skip_line_level_figures,
            calibration_path=layout.density_calibration if not skip_page_level else None,
            calibration=calibration_result,
        )
    if not skip_page_level_hmer:
        phase2_tasks["page_level_hmer"] = lambda: run_page_level_latex_export(
            input_dir,
            page_level_hmer_dir,
            workers=phase2_workers,
            show_progress=processing.show_progress or block_level_processing.show_progress,
            skip_figures=skip_page_level_hmer_figures,
            strict_consistency=False,
        )

    phase2_results: dict[str, object] = {}
    if phase2_tasks:
        phase2_results = run_parallel_tasks(
            phase2_tasks,
            description="Running benchmark export (phase 2)",
            show_progress=processing.show_progress or block_level_processing.show_progress,
            workers=min(len(phase2_tasks), default_worker_count()),
        )

    line_level_manifest = phase2_results.get("line_level")
    page_level_hmer_result = phase2_results.get("page_level_hmer")
    page_level_hmer_manifest = (
        getattr(page_level_hmer_result, "manifest", None)
        if page_level_hmer_result is not None
        else None
    )

    split_manifest: dict[str, str] | None = None
    split_selected_seed: int | None = None
    if not skip_page_level_latex_split:
        split_config = load_split_config(split_config_path)
        split_inputs_dir = layout.split_inputs
        prepare_split_inputs(
            input_dir,
            split_inputs_dir,
            dataset_version=split_config.dataset_version,
            workers=phase2_workers,
            show_progress=processing.show_progress or block_level_processing.show_progress,
            page_level_latex_output=None if skip_page_level_hmer else page_level_hmer_dir,
        )
        try:
            split_result = run_page_level_latex_split(
                split_inputs_dir,
                split_dir,
                config_path=split_config_path,
                config=split_config,
                skip_figures=skip_page_level_latex_split_figures,
                show_progress=processing.show_progress or block_level_processing.show_progress,
                workers=phase2_workers,
                allow_failed_acceptance=allow_split_acceptance_failure,
            )
        except SplitAcceptanceError:
            if not allow_split_acceptance_failure:
                raise
            split_manifest = None
        else:
            split_manifest = split_result.artifact_manifest
            split_selected_seed = split_result.selected_seed

    overview_metrics = compute_dataset_overview(
        input_dir,
        processing=processing,
        vision_processing=block_level_processing,
        enriched=enriched,
        vision_samples=block_dataset.samples,
        vision_pages=block_dataset.pages,
    )
    summary = build_project_summary(
        input_root=input_dir,
        output_root=output_root,
        hmer_output=hmer_dir,
        structure_layout_output=structure_layout_dir,
        hybrid_layout_output=hybrid_layout_dir,
        page_level_output=None if skip_page_level else page_level_dir,
        line_level_output=None if skip_line_level else line_level_dir,
        page_level_hmer_output=None if skip_page_level_hmer else page_level_hmer_dir,
        page_level_latex_split_output=None if skip_page_level_latex_split else split_dir,
        overview_metrics=overview_metrics,
        split_selected_seed=split_selected_seed,
    )
    summary_json = write_project_summary(summary, output_root)
    pipeline_doc = write_export_pipeline_doc(output_root)
    write_pipeline_manifest(
        build_pipeline_manifest(
            output_root=output_root,
            hmer_output=hmer_dir,
            structure_layout_output=structure_layout_dir,
            hybrid_layout_output=hybrid_layout_dir,
            page_level_output=None if skip_page_level else page_level_dir,
            line_level_output=None if skip_line_level else line_level_dir,
            page_level_hmer_output=None if skip_page_level_hmer else page_level_hmer_dir,
            page_level_latex_split_output=None if skip_page_level_latex_split else split_dir,
        ),
        output_root,
    )

    overview_path = run_dataset_overview_export(
        input_dir,
        output_root,
        processing=processing,
        vision_processing=block_level_processing,
        enriched=enriched,
        vision_samples=block_dataset.samples,
        vision_pages=block_dataset.pages,
        metrics=overview_metrics,
        skip_domain_overviews=True,
        page_level_detail_md=None if skip_page_level else layout.relative_density_report_md(),
        line_level_detail_md=None if skip_line_level else layout.relative_line_level_report_md(),
        structure_layout_detail_md=layout.relative_structure_layout_summary_md(),
        summary_json_path=summary_json,
        pipeline_doc_path=pipeline_doc,
    )
    prune_empty_directories(output_root)

    return ProjectExportResult(
        output_root=output_root,
        hmer_output=hmer_dir,
        structure_layout_output=structure_layout_dir,
        hybrid_layout_output=hybrid_layout_dir,
        page_level_output=None if skip_page_level else page_level_dir,
        line_level_output=None if skip_line_level else line_level_dir,
        page_level_hmer_output=None if skip_page_level_hmer else page_level_hmer_dir,
        page_level_latex_split_output=None if skip_page_level_latex_split else split_dir,
        hmer_manifest=hmer_manifest,
        structure_layout_manifest=structure_layout_manifest,
        hybrid_layout_manifest=hybrid_layout_manifest,
        page_level_manifest=page_level_manifest,
        block_density_manifest=block_density_manifest,
        line_level_manifest=line_level_manifest,
        page_level_hmer_manifest=page_level_hmer_manifest,
        page_level_latex_split_manifest=split_manifest,
        dataset_overview=overview_path,
        summary_json=summary_json,
        pipeline_doc=pipeline_doc,
    )
