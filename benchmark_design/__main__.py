"""CLI entry point for benchmark analysis."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from benchmark_design.config.line_level import DEFAULT_LINE_LEVEL_OUTPUT
from benchmark_design.config.page_level import DEFAULT_PAGE_LEVEL_OUTPUT
from benchmark_design.config import DEFAULT_BENCHMARK_INPUT, DEFAULT_OUTPUT_ROOT, DEFAULT_PROJECT_OUTPUT_ROOT
from benchmark_design.config.page_level_latex import DEFAULT_PAGE_LEVEL_LATEX_OUTPUT
from benchmark_design.config.page_level_latex_split import (
    DEFAULT_PAGE_LEVEL_LATEX_SPLIT_CONFIG,
    DEFAULT_PAGE_LEVEL_LATEX_SPLIT_OUTPUT,
)
from benchmark_design.config.block_level import (
    DEFAULT_BLOCK_LEVEL_INPUT,
    DEFAULT_BLOCK_LEVEL_OUTPUT_ROOT,
    DEFAULT_VISION_INPUT,
    DEFAULT_VISION_OUTPUT_ROOT,
)
from benchmark_design.ocr.processing import ProcessingOptions
from benchmark_design.ocr.consolidated import compute_ocr_consolidated_metrics
from benchmark_design.ocr.ast_statistics import compute_ocr_ast_statistics
from benchmark_design.ocr.length_bins import compute_ocr_length_bins
from benchmark_design.ocr.length_distribution import compute_ocr_length_distribution
from benchmark_design.ocr.scale import compute_ocr_scale
from benchmark_design.ocr.structure_complexity import compute_ocr_structure_complexity
from benchmark_design.ocr.structure_distribution import compute_ocr_structure_distribution
from benchmark_design.ocr.token_longtail import compute_ocr_token_longtail
from benchmark_design.ocr.token_taxonomy import compute_ocr_token_taxonomy
from benchmark_design.report.consolidated_table import write_consolidated_report
from benchmark_design.report.ast_statistics_table import write_ast_statistics_report
from benchmark_design.report.cross_benchmark_table import write_cross_benchmark_report
from benchmark_design.report.export_pipeline import run_benchmark_export
from benchmark_design.report.dataset_overview import run_dataset_overview_export
from benchmark_design.report.unified_export import run_unified_benchmark_export
from benchmark_design.project.export import run_project_export
from benchmark_design.project.config import load_project_config
from benchmark_design.report.line_level.export_pipeline import run_line_level_export
from benchmark_design.report.line_level.error_visualization import export_line_validation_error_figures
from benchmark_design.line_level.config import load_line_level_config
from benchmark_design.report.page_level.export_pipeline import run_page_level_export
from benchmark_design.page_level_latex.pipeline import run_page_level_latex_export
from benchmark_design.report.block_level.export_pipeline import run_block_level_export
from benchmark_design.report.block_level.flow_structure_export import (
    write_flow_structure_block_geometry_csv,
    write_flow_structure_page_metrics_csv,
)
from benchmark_design.block_level.flow_structure.pipeline import compute_flow_structure_results
from benchmark_design.report.length_bins_table import write_length_bins_report
from benchmark_design.report.length_table import write_length_report
from benchmark_design.report.output_layout import tables_dir
from benchmark_design.report.scale_table import write_scale_report
from benchmark_design.report.structure_complexity_table import write_structure_complexity_report
from benchmark_design.report.structure_distribution_table import write_structure_distribution_report
from benchmark_design.report.token_longtail_table import write_token_longtail_report
from benchmark_design.report.token_taxonomy_table import write_token_taxonomy_report
from benchmark_design.block_level.processing_options import BlockLevelProcessingOptions, VisionProcessingOptions


def _warn_vision_deprecated(command: str) -> None:
    warnings.warn(
        f"`{command}` is deprecated; use `block-level` or `project export` instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _add_unified_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory (shared by all pipelines)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROJECT_OUTPUT_ROOT,
        help="Output root (writes summary.json, dataset_overview.md, HMER/, page_level/, …)",
    )
    parser.add_argument(
        "--hmer-output",
        type=Path,
        default=None,
        help="HMER output directory (default: {output}/HMER)",
    )
    parser.add_argument(
        "--block-level-output",
        "--vision-output",
        type=Path,
        default=None,
        dest="block_level_output",
        help="Block-level / structure_layout output directory (default: {output}/block_level/structure_layout)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    parser.add_argument(
        "--skip-dimensions",
        action="store_true",
        help="Skip reading image width/height from files for block-level metrics",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip all figure generation (HMER matplotlib + block-level overlays)",
    )
    parser.add_argument(
        "--skip-cross-benchmark",
        action="store_true",
        help="Skip cross-benchmark comparison tables",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated cross-benchmark datasets (default: all configured)",
    )
    parser.add_argument(
        "--skip-flow-figures",
        action="store_true",
        help="Skip flow structure review overlay PNGs only",
    )
    parser.add_argument(
        "--page-level-output",
        type=Path,
        default=None,
        help="Page-level output directory (default: {output}/page_level)",
    )
    parser.add_argument(
        "--skip-page-level",
        action="store_true",
        help="Skip page-level pure image analysis export",
    )
    parser.add_argument(
        "--skip-page-level-figures",
        action="store_true",
        help="Skip page-level QA and paper figures only",
    )
    parser.add_argument(
        "--page-level-config",
        type=Path,
        default=None,
        help="Optional YAML config for page-level export",
    )
    parser.add_argument(
        "--line-level-output",
        type=Path,
        default=None,
        help="Line-level output directory (default: {output}/line_level)",
    )
    parser.add_argument(
        "--skip-line-level",
        action="store_true",
        help="Skip line-level feature analysis export",
    )
    parser.add_argument(
        "--skip-line-level-figures",
        action="store_true",
        help="Skip line-level plots and sample overlays only",
    )
    parser.add_argument(
        "--line-level-config",
        type=Path,
        default=None,
        help="Optional YAML config for line-level export",
    )
    parser.add_argument(
        "--skip-page-level-hmer",
        action="store_true",
        help="Skip page-level LaTeX / HMER bridge export (page_level_HMER/)",
    )
    parser.add_argument(
        "--skip-page-level-hmer-figures",
        action="store_true",
        help="Skip page_level_HMER figure generation only",
    )
    parser.add_argument(
        "--run-page-level-latex-split",
        action="store_true",
        help="Run stratified split after export (writes page_level_latex_split/)",
    )
    parser.add_argument(
        "--page-level-latex-split-config",
        type=Path,
        default=DEFAULT_PAGE_LEVEL_LATEX_SPLIT_CONFIG,
        help="YAML config for page_level_latex_split",
    )
    parser.add_argument(
        "--skip-page-level-latex-split-figures",
        action="store_true",
        help="Skip Chapter 7 split figures only",
    )
    parser.add_argument(
        "--allow-split-acceptance-failure",
        action="store_true",
        help="Do not fail project export when split acceptance checks fail",
    )


def _run_project_like_export(args: argparse.Namespace) -> int:
    processing = ProcessingOptions(
        show_progress=not args.no_progress,
        workers=args.workers,
    )
    block_level_processing = BlockLevelProcessingOptions(
        show_progress=not args.no_progress,
        workers=args.workers,
        read_image_dimensions=not args.skip_dimensions,
    )
    cross_datasets = None
    if args.datasets:
        cross_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    skip_all_figures = args.skip_figures
    result = run_project_export(
        args.input,
        args.output,
        hmer_output=args.hmer_output,
        block_level_output=args.block_level_output,
        page_level_output=args.page_level_output,
        processing=processing,
        block_level_processing=block_level_processing,
        skip_hmer_figures=skip_all_figures,
        skip_cross_benchmark=args.skip_cross_benchmark,
        cross_benchmark_datasets=cross_datasets,
        skip_flow_figures=skip_all_figures or args.skip_flow_figures,
        skip_page_level=args.skip_page_level,
        skip_page_level_figures=skip_all_figures or args.skip_page_level_figures,
        page_level_config=args.page_level_config,
        skip_line_level=args.skip_line_level,
        skip_line_level_figures=skip_all_figures or args.skip_line_level_figures,
        line_level_config=args.line_level_config,
        line_level_output=args.line_level_output,
        skip_page_level_hmer=args.skip_page_level_hmer,
        skip_page_level_hmer_figures=skip_all_figures or args.skip_page_level_hmer_figures,
        skip_page_level_latex_split=not args.run_page_level_latex_split,
        page_level_latex_split_config=args.page_level_latex_split_config,
        skip_page_level_latex_split_figures=skip_all_figures or args.skip_page_level_latex_split_figures,
        allow_split_acceptance_failure=args.allow_split_acceptance_failure,
    )
    print(f"wrote project export under {result.output_root.resolve()}")
    print(f"summary.json: {result.summary_json.resolve()}")
    print(f"PIPELINE.md: {result.pipeline_doc.resolve()}")
    print(f"HMER: {result.hmer_output.resolve()}")
    print(f"Structure layout: {result.structure_layout_output.resolve()}")
    if result.density_output is not None:
        print(f"Page density: {result.density_output.resolve()}")
    if result.line_level_output is not None:
        print(f"Line-level: {result.line_level_output.resolve()}")
    if result.page_level_hmer_output is not None:
        print(f"Page-level HMER: {result.page_level_hmer_output.resolve()}")
    if result.page_level_latex_split_output is not None:
        print(f"Split: {result.page_level_latex_split_output.resolve()}")
    print(f"dataset overview: {result.dataset_overview.resolve()}")
    print(f"HMER detail: {(result.hmer_output / 'summary.md').resolve()}")
    print(
        "Structure layout detail: "
        f"{(result.structure_layout_output / 'block_level_summary.md').resolve()}"
    )
    return 0


def _format_ast_value(metric: str, value: float | int) -> str:
    if metric == "Max max nested level":
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _add_processing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: CPU count, capped at 32)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )


def _processing_options(args: argparse.Namespace) -> ProcessingOptions:
    return ProcessingOptions(
        show_progress=not getattr(args, "no_progress", False),
        workers=getattr(args, "workers", None),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark_design", description="Benchmark dataset analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser("ocr", help="OCR text analysis")
    ocr_sub = ocr_parser.add_subparsers(dest="ocr_command", required=True)

    scale_parser = ocr_sub.add_parser("scale", help="Compute OCR data-scale statistics")
    scale_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    scale_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    length_parser = ocr_sub.add_parser("length", help="Compute OCR expression length distribution")
    length_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    length_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    bins_parser = ocr_sub.add_parser("bins", help="Compute fixed OCR length-bin counts")
    bins_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    bins_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    taxonomy_parser = ocr_sub.add_parser("taxonomy", help="Compute OCR token taxonomy composition")
    taxonomy_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    taxonomy_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    longtail_parser = ocr_sub.add_parser("longtail", help="Compute OCR token long-tail statistics")
    longtail_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    longtail_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    structure_parser = ocr_sub.add_parser("structure", help="Compute OCR structure-type distribution")
    structure_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    structure_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    complexity_parser = ocr_sub.add_parser(
        "structure-complexity",
        help="Compute OCR structure-type combination complexity",
    )
    complexity_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    complexity_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    ast_parser = ocr_sub.add_parser(
        "ast",
        help="Compute structure-forest AST depth statistics",
    )
    ast_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    ast_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    report_parser = ocr_sub.add_parser(
        "report",
        help="Compute and write consolidated OCR benchmark tables (1–7)",
    )
    report_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )

    export_parser = ocr_sub.add_parser(
        "export",
        help="Full benchmark export: tables, details, examples, figures, docs, summary",
    )
    export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (HMER layout)",
    )
    export_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip matplotlib figure generation",
    )
    export_parser.add_argument(
        "--skip-cross-benchmark",
        action="store_true",
        help="Skip cross-benchmark comparison tables",
    )
    export_parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated cross-benchmark datasets (default: all configured)",
    )

    cross_parser = ocr_sub.add_parser(
        "cross-benchmark",
        help="Cross-dataset comparison using unified tokenizer",
    )
    cross_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (tables written to output/tables/)",
    )
    cross_parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names (default: all configured)",
    )

    for subparser in (
        scale_parser,
        length_parser,
        bins_parser,
        taxonomy_parser,
        longtail_parser,
        structure_parser,
        complexity_parser,
        ast_parser,
        report_parser,
        export_parser,
        cross_parser,
    ):
        _add_processing_args(subparser)

    hmer_parser = subparsers.add_parser(
        "hmer",
        help="HMER LaTeX benchmark analysis (forwards to `ocr` subcommands)",
    )
    hmer_parser.add_argument(
        "forward_args",
        nargs=argparse.REMAINDER,
        help="OCR/HMER subcommand and options, e.g. `export --output ./HMER`",
    )

    overview_parser = subparsers.add_parser("overview", help="Dataset overview (总纲) export")
    overview_sub = overview_parser.add_subparsers(dest="overview_command", required=True)
    overview_export_parser = overview_sub.add_parser(
        "export",
        help="Write dataset overview to output root with HMER/ and block_level/ subdirectories",
    )
    overview_export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory",
    )
    overview_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root directory (writes dataset_overview.md, HMER/, block_level/)",
    )
    overview_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    overview_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    overview_export_parser.add_argument(
        "--skip-dimensions",
        action="store_true",
        help="Skip reading image width/height from files (infer from annotations)",
    )

    block_level_parser = subparsers.add_parser("block-level", help="Block-level image-side benchmark analysis")
    block_level_sub = block_level_parser.add_subparsers(dest="block_level_command", required=True)
    block_level_export_parser = block_level_sub.add_parser("export", help="Export block-level benchmark scaffold")
    block_level_export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BLOCK_LEVEL_INPUT,
        help="Benchmark image root (JSON page export directory for now)",
    )
    block_level_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BLOCK_LEVEL_OUTPUT_ROOT,
        help="Block-level output root",
    )
    block_level_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for JSON loading",
    )
    block_level_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    block_level_export_parser.add_argument(
        "--skip-dimensions",
        action="store_true",
        help="Skip reading image width/height (no Pillow required)",
    )
    block_level_export_parser.add_argument(
        "--skip-flow-figures",
        action="store_true",
        help="Skip flow structure review overlay PNG generation",
    )

    block_level_flow_parser = block_level_sub.add_parser(
        "flow-structure",
        help="Compute Answer-Block Flow Structure metrics only",
    )
    block_level_flow_parser.add_argument("--input", type=Path, default=DEFAULT_BLOCK_LEVEL_INPUT)
    block_level_flow_parser.add_argument("--output", type=Path, default=DEFAULT_BLOCK_LEVEL_OUTPUT_ROOT)
    block_level_flow_parser.add_argument("--workers", type=int, default=None)
    block_level_flow_parser.add_argument("--no-progress", action="store_true")
    block_level_flow_parser.add_argument("--skip-flow-figures", action="store_true")

    vision_parser = subparsers.add_parser("vision", help="[deprecated] Use block-level instead")
    vision_sub = vision_parser.add_subparsers(dest="vision_command", required=True)
    vision_export_parser = vision_sub.add_parser("export", help="Export vision benchmark scaffold")
    vision_export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VISION_INPUT,
        help="Benchmark image root (JSON page export directory for now)",
    )
    vision_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VISION_OUTPUT_ROOT,
        help="Vision output root (vision layout)",
    )
    vision_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for JSON loading",
    )
    vision_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    vision_export_parser.add_argument(
        "--skip-dimensions",
        action="store_true",
        help="Skip reading image width/height (no Pillow required)",
    )

    vision_export_parser.add_argument(
        "--skip-flow-figures",
        action="store_true",
        help="Skip flow structure review overlay PNG generation",
    )

    vision_flow_parser = vision_sub.add_parser(
        "flow-structure",
        help="Compute Answer-Block Flow Structure metrics only",
    )
    vision_flow_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VISION_INPUT,
        help="Benchmark JSON page export directory",
    )
    vision_flow_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VISION_OUTPUT_ROOT,
        help="Output root for flow structure CSV tables",
    )
    vision_flow_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for JSON loading",
    )
    vision_flow_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )

    vision_flow_parser.add_argument(
        "--skip-flow-figures",
        action="store_true",
        help="Skip flow structure review overlay PNG generation",
    )

    page_level_parser = subparsers.add_parser("page-level", help="Pure image-level page analysis")
    page_level_sub = page_level_parser.add_subparsers(dest="page_level_command", required=True)
    page_level_export_parser = page_level_sub.add_parser(
        "export",
        help="Export page-level image analysis (inventory, calibration, features, heatmaps, report)",
    )
    page_level_export_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/image_analysis.yaml"),
        help="YAML configuration file",
    )
    page_level_export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON/image root directory",
    )
    page_level_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PAGE_LEVEL_OUTPUT,
        help="Page-level output root directory",
    )
    page_level_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    page_level_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    page_level_export_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip QA and paper figure generation",
    )

    page_latex_parser = subparsers.add_parser(
        "page-level-latex",
        help="Page-level LaTeX / expression statistics (Chapter 6, HMER protocol)",
    )
    page_latex_sub = page_latex_parser.add_subparsers(dest="page_level_latex_command", required=True)
    page_latex_export_parser = page_latex_sub.add_parser(
        "export",
        help="Export expression_ and page-level LaTeX metrics, tables, and figures",
    )
    page_latex_export_parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark"),
        help="Benchmark JSON root (same source as HMER Chapter 5)",
    )
    page_latex_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PAGE_LEVEL_LATEX_OUTPUT,
        help="Output directory (default: ./page_level_latex)",
    )
    page_latex_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    page_latex_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    page_latex_export_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation",
    )
    page_latex_export_parser.add_argument(
        "--allow-consistency-failure",
        action="store_true",
        help="Write outputs even when Chapter-5 consistency checks fail",
    )

    page_latex_prepare_parser = page_latex_sub.add_parser(
        "prepare-split-inputs",
        help="Export dataset_manifest / page_hmer_features / page_token_counts for stratified split",
    )
    page_latex_prepare_parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark"),
        help="Benchmark JSON root",
    )
    page_latex_prepare_parser.add_argument(
        "--output",
        type=Path,
        default=Path("./page_level_latex_split/inputs"),
        help="Directory for frozen split input CSVs",
    )
    page_latex_prepare_parser.add_argument("--dataset-version", default="", help="Dataset version label")
    page_latex_prepare_parser.add_argument("--workers", type=int, default=None)
    page_latex_prepare_parser.add_argument("--no-progress", action="store_true")

    page_latex_split_parser = page_latex_sub.add_parser(
        "split",
        help="Multilabel stratified train/val/test split from frozen Chapter-6 inputs",
    )
    page_latex_split_parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("./page_level_latex_split/inputs"),
        help="Directory containing dataset_manifest / page_hmer_features / page_token_counts",
    )
    page_latex_split_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/page_level_latex_split.yaml"),
        help="Frozen split configuration YAML",
    )
    page_latex_split_parser.add_argument(
        "--output",
        type=Path,
        default=Path("./page_level_latex_split"),
        help="Split output root",
    )
    page_latex_split_parser.add_argument("--skip-figures", action="store_true")
    page_latex_split_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel candidate-seed workers (default min(3, cpu_count))",
    )
    page_latex_split_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )

    line_level_parser = subparsers.add_parser("line-level", help="Line-level feature analysis")
    line_level_sub = line_level_parser.add_subparsers(dest="line_level_command", required=True)
    line_level_export_parser = line_level_sub.add_parser(
        "export",
        help="Export line-level geometry statistics (size, orientation, position, nearest distance, IoU)",
    )
    line_level_export_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/line_analysis.yaml"),
        help="YAML configuration file",
    )
    line_level_export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON/image root directory",
    )
    line_level_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LINE_LEVEL_OUTPUT,
        help="Line-level output root directory",
    )
    line_level_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    line_level_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    line_level_export_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip plots and sample overlay generation",
    )
    line_level_errors_parser = line_level_sub.add_parser(
        "export-errors",
        help="Export visualizations for line polygon validation failures",
    )
    line_level_errors_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/line_analysis.yaml"),
        help="YAML configuration file",
    )
    line_level_errors_parser.add_argument(
        "--output",
        type=Path,
        default=Path("line_error"),
        help="Output directory for validation error figures",
    )
    line_level_errors_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    line_level_errors_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )

    project_parser = subparsers.add_parser("project", help="Unified benchmark project export")
    project_sub = project_parser.add_subparsers(dest="project_command", required=True)
    project_export_parser = project_sub.add_parser(
        "export",
        help="Run full HMER + page_level + line_level (+ optional split) export",
    )
    project_export_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (config/project.yaml)",
    )
    _add_unified_export_args(project_export_parser)

    unified_export_parser = subparsers.add_parser(
        "export",
        help="[deprecated] Use `project export` instead",
    )
    _add_unified_export_args(unified_export_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "hmer":
        forwarded = [token for token in (args.forward_args or []) if token != "--"]
        if not forwarded:
            parser.parse_args(["hmer", "export", "--help"])
        return main(["ocr", *forwarded])

    if args.command == "project" and args.project_command == "export":
        if args.config is not None:
            project_config = load_project_config(args.config)
            args.input = project_config.input_root
            args.output = project_config.output_root
            if project_config.workers is not None:
                args.workers = project_config.workers
            page_cfg = project_config.pipelines.page_level.get("config")
            if page_cfg and args.page_level_config is None:
                args.page_level_config = Path(page_cfg)
            line_cfg = project_config.pipelines.line_level.get("config")
            if line_cfg and args.line_level_config is None:
                args.line_level_config = Path(line_cfg)
            if project_config.pipelines.hmer.get("skip_cross_benchmark"):
                args.skip_cross_benchmark = True
            if project_config.pipelines.block_level.get("skip_figures"):
                args.skip_figures = True
            if project_config.pipelines.page_level_hmer.get("skip_figures"):
                args.skip_page_level_hmer_figures = True
            split_cfg = project_config.pipelines.page_level_latex_split
            if split_cfg.get("enabled"):
                args.run_page_level_latex_split = True
            split_config_path = split_cfg.get("config")
            if split_config_path and args.page_level_latex_split_config == DEFAULT_PAGE_LEVEL_LATEX_SPLIT_CONFIG:
                args.page_level_latex_split_config = Path(split_config_path)
        return _run_project_like_export(args)

    if args.command == "export":
        _warn_vision_deprecated("export")
        return _run_project_like_export(args)

    block_level_command = None
    if args.command == "block-level":
        block_level_command = args.block_level_command
    elif args.command == "vision":
        _warn_vision_deprecated("vision")
        block_level_command = args.vision_command

    if block_level_command == "export":
        block_level_processing = BlockLevelProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=not args.skip_dimensions,
        )
        manifest = run_block_level_export(
            args.input,
            args.output,
            processing=block_level_processing,
            skip_flow_figures=args.skip_flow_figures,
        )
        print(f"wrote {len(manifest)} block-level artifacts under {args.output.resolve()}")
        for key, rel_path in manifest.items():
            print(f"{key}: {rel_path}")
        return 0

    if args.command == "overview" and args.overview_command == "export":
        processing = ProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
        )
        block_level_processing = BlockLevelProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=not args.skip_dimensions,
        )
        overview_path = run_dataset_overview_export(
            args.input,
            args.output,
            processing=processing,
            vision_processing=block_level_processing,
        )
        print(f"wrote {overview_path.resolve()}")
        print(f"HMER overview: {(args.output / 'HMER' / 'overview.md').resolve()}")
        print(f"Block-level overview: {(args.output / 'block_level' / 'overview.md').resolve()}")
        return 0

    if block_level_command == "flow-structure":
        from benchmark_design.report.block_level.flow_structure_export import (
            write_flow_group_summary_csv,
            write_flow_structure_decisions_jsonl,
        )
        from benchmark_design.report.block_level.flow_structure_figures import export_flow_structure_figures
        from benchmark_design.report.block_level.flow_structure_summary import write_flow_structure_summary_md

        block_level_processing = BlockLevelProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=False,
        )
        results = compute_flow_structure_results(
            args.input,
            processing=block_level_processing,
            input_dir_for_images=args.input,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        tables_dir = args.output / "tables"
        details_dir = args.output / "details"
        tables_dir.mkdir(parents=True, exist_ok=True)
        details_dir.mkdir(parents=True, exist_ok=True)
        page_path = tables_dir / "flow_structure_page_metrics.csv"
        group_summary_path = tables_dir / "flow_group_summary.csv"
        block_path = tables_dir / "flow_structure_block_geometry.csv"
        decisions_path = details_dir / "flow_structure_decisions.jsonl"
        summary_path = args.output / "flow_structure_summary.md"
        write_flow_structure_page_metrics_csv(results, page_path)
        write_flow_group_summary_csv(results, group_summary_path)
        write_flow_structure_block_geometry_csv(
            [record for result in results for record in result.block_records],
            block_path,
        )
        write_flow_structure_decisions_jsonl(results, decisions_path)
        write_flow_structure_summary_md(results, summary_path)
        if not args.skip_flow_figures:
            export_flow_structure_figures(
                results,
                input_dir=args.input,
                figures_root=args.output / "figures" / "flow_structure",
                show_progress=not args.no_progress,
            )
        print(f"pages: {len(results):,}")
        print(f"wrote {page_path}")
        print(f"wrote {group_summary_path}")
        print(f"wrote {block_path}")
        print(f"wrote {decisions_path}")
        print(f"wrote {summary_path}")
        return 0

    processing = _processing_options(args) if args.command == "ocr" else ProcessingOptions()

    if args.command == "ocr" and args.ocr_command == "scale":
        metrics = compute_ocr_scale(args.input, processing=processing)
        paths = write_scale_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"expression count: {metrics.expression_count:,}")
        print(f"total token count: {metrics.total_token_count:,}")
        print(f"vocabulary size: {metrics.vocabulary_size:,}")
        print(f"unique normalized LaTeX count: {metrics.unique_normalized_latex_count:,}")
        print(f"duplicate rate: {metrics.duplicate_rate:.6f}")
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "length":
        metrics = compute_ocr_length_distribution(args.input, processing=processing)
        paths = write_length_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"expression count: {metrics.expression_count:,}")
        print(f"mean length: {metrics.mean_length:.6f}")
        print(f"std: {metrics.std:.6f}")
        print(f"cv: {metrics.cv:.6f}")
        print(f"p50: {metrics.p50:.6f}")
        print(f"p90: {metrics.p90:.6f}")
        print(f"max: {metrics.max_length}")
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "bins":
        metrics = compute_ocr_length_bins(args.input, processing=processing)
        paths = write_length_bins_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"expression count: {metrics.expression_count:,}")
        for item in metrics.bins:
            print(f"{item.label}: {item.count:,} ({item.share:.6f})")
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "taxonomy":
        metrics = compute_ocr_token_taxonomy(args.input, processing=processing)
        paths = write_token_taxonomy_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"total token count: {metrics.total_token_count:,}")
        for item in metrics.categories:
            print(f"{item.category.value}: {item.count:,} ({item.share:.6f})")
        print(f"other / unknown token ratio: {metrics.other_unknown_ratio:.6f}")
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "longtail":
        metrics = compute_ocr_token_longtail(args.input, processing=processing)
        paths = write_token_longtail_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"vocabulary size: {metrics.vocabulary_size:,}")
        print(f"total token count: {metrics.total_token_count:,}")
        print(f"gini: {metrics.gini:.6f}")
        for k, coverage in metrics.top_k_coverage:
            print(f"top-{k} coverage: {coverage:.6f}")
        for threshold, ratio in metrics.rare_vocab_ratio:
            print(f"rare_{threshold} vocab ratio: {ratio:.6f}")
        for threshold, ratio in metrics.rare_expression_ratio:
            print(f"rare_{threshold} expression ratio: {ratio:.6f}")
        print(f"wrote {paths['summary_csv']}")
        print(f"wrote {paths['summary_markdown']}")
        print(f"wrote {paths['frequency_csv']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "structure":
        metrics = compute_ocr_structure_distribution(args.input, processing=processing)
        paths = write_structure_distribution_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"expression count: {metrics.expression_count:,}")
        print(f"structural token count: {metrics.structural_token_count:,}")
        for row in metrics.rows:
            print(
                f"{row.structure_type}: expr_ratio={row.expression_ratio:.6f}, "
                f"occ_ratio={row.occurrence_ratio:.6f}, max_depth={row.max_depth}"
            )
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "structure-complexity":
        metrics = compute_ocr_structure_complexity(args.input, processing=processing)
        paths = write_structure_complexity_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"expression count: {metrics.expression_count:,}")
        for metric, _definition, value in metrics.as_rows():
            if isinstance(value, float) and metric != "Max structure type count":
                print(f"{metric}: {value:.6f}")
            else:
                print(f"{metric}: {value}")
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "ast":
        metrics = compute_ocr_ast_statistics(args.input, processing=processing)
        paths = write_ast_statistics_report(metrics, tables_dir(args.output), input_dir=args.input)
        print(f"expression count: {metrics.expression_count:,}")
        for metric, _definition, value in metrics.as_summary_rows():
            print(f"{metric}: {_format_ast_value(metric, value)}")
        for item in metrics.bins:
            print(f"bin {item.label}: {item.count:,} ({item.share:.6f})")
        print(f"wrote {paths['summary_csv']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "report":
        metrics = compute_ocr_consolidated_metrics(args.input, processing=processing)
        paths = write_consolidated_report(metrics, tables_dir(args.output))
        print(f"expression count: {metrics.scale.expression_count:,}")
        print(f"total token count: {metrics.scale.total_token_count:,}")
        print(f"wrote {paths['markdown']}")
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['metadata']}")
        return 0

    if args.command == "ocr" and args.ocr_command == "export":
        cross_datasets = None
        if args.datasets:
            cross_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
        manifest = run_benchmark_export(
            args.input,
            args.output,
            processing=processing,
            skip_figures=args.skip_figures,
            skip_cross_benchmark=args.skip_cross_benchmark,
            cross_benchmark_datasets=cross_datasets,
        )
        print(f"wrote {len(manifest)} artifacts under {args.output.resolve()}")
        print(f"summary: {manifest.get('summary')}")
        print(f"metadata: {manifest.get('metadata')}")
        return 0

    if args.command == "ocr" and args.ocr_command == "cross-benchmark":
        dataset_names = None
        if args.datasets:
            dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
        paths = write_cross_benchmark_report(
            args.output,
            dataset_names=dataset_names,
            processing=processing,
        )
        for key, path in paths.items():
            print(f"wrote {path}")
        return 0

    if args.command == "page-level" and args.page_level_command == "export":
        manifest = run_page_level_export(
            args.input,
            args.output,
            config_path=args.config,
            workers=args.workers,
            show_progress=not args.no_progress,
            skip_figures=args.skip_figures,
        )
        print(f"wrote {len(manifest)} page-level artifacts under {args.output.resolve()}")
        print(f"report: {(args.output / manifest['report']).resolve()}")
        print(f"features: {(args.output / manifest['image_features']).resolve()}")
        return 0

    if args.command == "page-level-latex" and args.page_level_latex_command == "export":
        result = run_page_level_latex_export(
            args.input,
            args.output,
            workers=args.workers,
            show_progress=not args.no_progress,
            skip_figures=args.skip_figures,
            strict_consistency=not args.allow_consistency_failure,
        )
        print(f"wrote page-level LaTeX artifacts under {args.output.resolve()}")
        print(f"consistency_passed={result.passed_consistency}")
        print(f"expression metrics: {(args.output / result.manifest['expression_latex_metrics']).resolve()}")
        print(f"page metrics: {(args.output / result.manifest['page_latex_metrics']).resolve()}")
        print(f"summary: {(args.output / result.manifest['dataset_summary']).resolve()}")
        return 0

    if args.command == "page-level-latex" and args.page_level_latex_command == "prepare-split-inputs":
        from benchmark_design.page_level_latex.split_inputs import prepare_split_inputs

        result = prepare_split_inputs(
            args.input,
            args.output,
            dataset_version=args.dataset_version,
            workers=args.workers,
            show_progress=not args.no_progress,
        )
        print(f"wrote {result.page_count} page split inputs under {result.output_dir.resolve()}")
        for key, rel in result.manifest.items():
            print(f"{key}: {(result.output_dir / rel).resolve()}")
        return 0

    if args.command == "page-level-latex" and args.page_level_latex_command == "split":
        from benchmark_design.page_level_latex_split import run_page_level_latex_split
        from benchmark_design.page_level_latex_split.audit import SplitAcceptanceError

        try:
            result = run_page_level_latex_split(
                args.inputs,
                args.output,
                config_path=args.config,
                skip_figures=args.skip_figures,
                show_progress=not args.no_progress,
                workers=args.workers,
            )
        except SplitAcceptanceError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"wrote stratified split under {result.output_root.resolve()}")
        print(f"selected_seed={result.selected_seed}")
        print(f"manifest_sha256={result.manifest_sha256}")
        print(f"acceptance_passed={result.acceptance.passed}")
        for key, rel in result.artifact_manifest.items():
            print(f"{key}: {rel}")
        return 0

    if args.command == "line-level" and args.line_level_command == "export":
        manifest = run_line_level_export(
            args.input,
            args.output,
            config_path=args.config,
            workers=args.workers,
            show_progress=not args.no_progress,
            skip_figures=args.skip_figures,
        )
        print(f"wrote {len(manifest)} line-level artifacts under {args.output.resolve()}")
        print(f"report: {(args.output / manifest['report']).resolve()}")
        print(f"line metrics: {(args.output / manifest['line_metrics']).resolve()}")
        return 0

    if args.command == "line-level" and args.line_level_command == "export-errors":
        config = load_line_level_config(
            args.config,
            output_root=args.output,
            workers=args.workers,
            show_progress=not args.no_progress,
        )
        summary = export_line_validation_error_figures(config, args.output)
        print(f"wrote {summary['total_error_count']} line validation error figures under {args.output.resolve()}")
        for reason, count in summary["counts_by_reason"].items():
            if count:
                label = summary["reason_labels"][reason]
                print(f"  {reason}: {count} ({label})")
        print(f"summary: {(args.output / 'summary.json').resolve()}")
        if (args.output / "error_index.csv").is_file():
            print(f"index: {(args.output / 'error_index.csv').resolve()}")
        return 0

    parser.error(f"Unhandled command: {args}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
