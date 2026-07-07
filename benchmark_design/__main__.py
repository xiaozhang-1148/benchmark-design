"""CLI entry point for benchmark analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_design.config import DEFAULT_BENCHMARK_INPUT, DEFAULT_OUTPUT_ROOT, DEFAULT_UNIFIED_OUTPUT_ROOT
from benchmark_design.config.vision import DEFAULT_VISION_INPUT, DEFAULT_VISION_OUTPUT_ROOT
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
from benchmark_design.report.vision.export_pipeline import run_vision_benchmark_export
from benchmark_design.report.vision.flow_structure_export import (
    write_flow_structure_block_geometry_csv,
    write_flow_structure_page_metrics_csv,
)
from benchmark_design.vision.flow_structure.pipeline import compute_flow_structure_results
from benchmark_design.report.length_bins_table import write_length_bins_report
from benchmark_design.report.length_table import write_length_report
from benchmark_design.report.output_layout import tables_dir
from benchmark_design.report.scale_table import write_scale_report
from benchmark_design.report.structure_complexity_table import write_structure_complexity_report
from benchmark_design.report.structure_distribution_table import write_structure_distribution_report
from benchmark_design.report.token_longtail_table import write_token_longtail_report
from benchmark_design.report.token_taxonomy_table import write_token_taxonomy_report
from benchmark_design.vision.processing_options import VisionProcessingOptions


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
        help="Compute PosFormer position-forest AST depth statistics",
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
        help="Write dataset overview to output root with HMER/ and vision/ subdirectories",
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
        help="Output root directory (writes dataset_overview.md, HMER/, vision/)",
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

    vision_parser = subparsers.add_parser("vision", help="Vision / image-side benchmark analysis")
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
    vision_export_parser.add_argument(
        "--skip-foreground-load-figures",
        action="store_true",
        help="Skip foreground load review overlay PNG generation",
    )
    vision_export_parser.add_argument(
        "--skip-deleted-block-scale-figures",
        action="store_true",
        help="Skip deleted-block scale review overlay PNG generation",
    )

    vision_fg_parser = vision_sub.add_parser(
        "foreground-load",
        help="Compute effective-region foreground load metrics only",
    )
    vision_fg_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VISION_INPUT,
        help="Benchmark JSON page export directory",
    )
    vision_fg_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VISION_OUTPUT_ROOT,
        help="Output root for foreground load tables",
    )
    vision_fg_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for JSON loading",
    )
    vision_fg_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    vision_fg_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip foreground load review overlay PNG generation",
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

    vision_dbs_parser = vision_sub.add_parser(
        "deleted-block-scale",
        help="Compute Deleted-Block Scale metrics only",
    )
    vision_dbs_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VISION_INPUT,
        help="Benchmark JSON page export directory",
    )
    vision_dbs_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VISION_OUTPUT_ROOT,
        help="Output root for deleted-block scale tables",
    )
    vision_dbs_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for JSON loading",
    )
    vision_dbs_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    vision_dbs_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip deleted-block scale review overlay PNG generation",
    )

    unified_export_parser = subparsers.add_parser(
        "export",
        help="Run full HMER + Vision export and dataset overview in one command",
    )
    unified_export_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BENCHMARK_INPUT,
        help="Benchmark JSON input directory (shared by HMER and Vision)",
    )
    unified_export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_UNIFIED_OUTPUT_ROOT,
        help="Output root (writes HMER/, vision/, dataset_overview.md)",
    )
    unified_export_parser.add_argument(
        "--hmer-output",
        type=Path,
        default=None,
        help="HMER output directory (default: {output}/HMER)",
    )
    unified_export_parser.add_argument(
        "--vision-output",
        type=Path,
        default=None,
        help="Vision output directory (default: {output}/vision)",
    )
    unified_export_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads",
    )
    unified_export_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars",
    )
    unified_export_parser.add_argument(
        "--skip-dimensions",
        action="store_true",
        help="Skip reading image width/height from files for Vision metrics",
    )
    unified_export_parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip all figure generation (HMER matplotlib + Vision overlays)",
    )
    unified_export_parser.add_argument(
        "--skip-cross-benchmark",
        action="store_true",
        help="Skip cross-benchmark comparison tables",
    )
    unified_export_parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated cross-benchmark datasets (default: all configured)",
    )
    unified_export_parser.add_argument(
        "--skip-flow-figures",
        action="store_true",
        help="Skip flow structure review overlay PNGs only",
    )
    unified_export_parser.add_argument(
        "--skip-foreground-load-figures",
        action="store_true",
        help="Skip foreground load review overlay PNGs only",
    )
    unified_export_parser.add_argument(
        "--skip-deleted-block-scale-figures",
        action="store_true",
        help="Skip deleted-block scale review overlay PNGs only",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "hmer":
        forwarded = [token for token in (args.forward_args or []) if token != "--"]
        if not forwarded:
            parser.parse_args(["hmer", "export", "--help"])
        return main(["ocr", *forwarded])

    if args.command == "export":
        processing = ProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
        )
        vision_processing = VisionProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=not args.skip_dimensions,
        )
        cross_datasets = None
        if args.datasets:
            cross_datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
        skip_all_figures = args.skip_figures
        result = run_unified_benchmark_export(
            args.input,
            args.output,
            hmer_output=args.hmer_output,
            vision_output=args.vision_output,
            processing=processing,
            vision_processing=vision_processing,
            skip_hmer_figures=skip_all_figures,
            skip_cross_benchmark=args.skip_cross_benchmark,
            cross_benchmark_datasets=cross_datasets,
            skip_flow_figures=skip_all_figures or args.skip_flow_figures,
            skip_foreground_load_figures=skip_all_figures or args.skip_foreground_load_figures,
            skip_deleted_block_scale_figures=skip_all_figures or args.skip_deleted_block_scale_figures,
        )
        print(f"wrote unified export under {result.output_root.resolve()}")
        print(f"HMER: {result.hmer_output.resolve()}")
        print(f"Vision: {result.vision_output.resolve()}")
        print(f"dataset overview: {result.dataset_overview.resolve()}")
        print(f"HMER detail: {(result.hmer_output / 'summary.md').resolve()}")
        print(f"Vision detail: {(result.vision_output / 'vision_benchmark_summary.md').resolve()}")
        return 0

    if args.command == "overview" and args.overview_command == "export":
        processing = ProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
        )
        vision_processing = VisionProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=not args.skip_dimensions,
        )
        overview_path = run_dataset_overview_export(
            args.input,
            args.output,
            processing=processing,
            vision_processing=vision_processing,
        )
        print(f"wrote {overview_path.resolve()}")
        print(f"HMER overview: {(args.output / 'HMER' / 'overview.md').resolve()}")
        print(f"Vision overview: {(args.output / 'vision' / 'overview.md').resolve()}")
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

    if args.command == "vision" and args.vision_command == "export":
        vision_processing = VisionProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=not args.skip_dimensions,
        )
        manifest = run_vision_benchmark_export(
            args.input,
            args.output,
            processing=vision_processing,
            skip_flow_figures=args.skip_flow_figures,
            skip_foreground_load_figures=args.skip_foreground_load_figures,
            skip_deleted_block_scale_figures=args.skip_deleted_block_scale_figures,
        )
        print(f"wrote {len(manifest)} vision artifacts under {args.output.resolve()}")
        for key, rel_path in manifest.items():
            print(f"{key}: {rel_path}")
        return 0

    if args.command == "vision" and args.vision_command == "flow-structure":
        from benchmark_design.report.vision.flow_structure_export import (
            write_flow_group_summary_csv,
            write_flow_structure_decisions_jsonl,
        )
        from benchmark_design.report.vision.flow_structure_figures import export_flow_structure_figures
        from benchmark_design.report.vision.flow_structure_summary import write_flow_structure_summary_md

        vision_processing = VisionProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=False,
        )
        results = compute_flow_structure_results(
            args.input,
            processing=vision_processing,
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

    if args.command == "vision" and args.vision_command == "foreground-load":
        from benchmark_design.report.vision.foreground_pixel_density_comparison_figures import (
            export_foreground_pixel_density_comparison_figures,
        )
        from benchmark_design.report.vision.foreground_pixel_density_export import (
            write_foreground_pixel_density_block_metrics_csv,
            write_foreground_pixel_density_diagnostics_json,
            write_foreground_pixel_density_overall_csv,
            write_foreground_pixel_density_page_metrics_csv,
            write_foreground_pixel_density_region_metrics_csv,
        )
        from benchmark_design.report.vision.foreground_pixel_density_figures import (
            export_foreground_pixel_density_figures,
        )
        from benchmark_design.report.vision.foreground_pixel_density_summary import (
            write_foreground_pixel_density_summary_md,
        )
        from benchmark_design.vision.flow_structure.page_loader import load_page_annotations
        from benchmark_design.vision.foreground_load.pipeline import compute_foreground_load_results

        vision_processing = VisionProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=False,
        )
        results, thresholds, global_config = compute_foreground_load_results(
            args.input,
            processing=vision_processing,
            input_dir_for_images=args.input,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        tables_dir = args.output / "tables"
        metadata_dir = args.output / "metadata"
        figures_dir = args.output / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        page_path = tables_dir / "foreground_pixel_density_page_metrics.csv"
        block_path = tables_dir / "foreground_pixel_density_block_metrics.csv"
        region_path = tables_dir / "foreground_pixel_density_region_metrics.csv"
        overall_path = tables_dir / "foreground_pixel_density_overall.csv"
        diagnostics_path = metadata_dir / "foreground_pixel_density_diagnostics.json"
        summary_path = args.output / "foreground_pixel_density_summary.md"
        page_stats, block_stats = write_foreground_pixel_density_overall_csv(results, overall_path)
        write_foreground_pixel_density_page_metrics_csv(results, page_path)
        write_foreground_pixel_density_block_metrics_csv(
            [block for result in results for block in result.block_results],
            block_path,
        )
        write_foreground_pixel_density_region_metrics_csv(results, region_path)
        write_foreground_pixel_density_diagnostics_json(
            results,
            thresholds,
            diagnostics_path,
            global_config=global_config,
        )
        write_foreground_pixel_density_summary_md(
            results,
            summary_path,
            page_stats=page_stats,
            block_stats=block_stats,
            flow_stats=[],
            thresholds=thresholds,
            diagnostics_json_path="metadata/foreground_pixel_density_diagnostics.json",
        )
        if not args.skip_figures:
            pages = load_page_annotations(args.input, processing=vision_processing)
            export_foreground_pixel_density_figures(
                results,
                pages,
                [],
                input_dir=args.input,
                figures_root=figures_dir,
            )
            export_foreground_pixel_density_comparison_figures(
                results,
                pages,
                input_dir=args.input,
                figures_root=figures_dir,
                global_config=global_config,
            )
        print(f"pages: {len(results):,}")
        print(f"wrote {page_path}")
        print(f"wrote {block_path}")
        print(f"wrote {region_path}")
        print(f"wrote {overall_path}")
        print(f"wrote {diagnostics_path}")
        print(f"wrote {summary_path}")
        return 0

    if args.command == "vision" and args.vision_command == "deleted-block-scale":
        from benchmark_design.report.vision.deleted_block_scale_export import (
            write_deleted_block_scale_block_geometry_csv,
            write_deleted_block_scale_diagnostics_json,
            write_deleted_block_scale_page_metrics_csv,
        )
        from benchmark_design.report.vision.deleted_block_scale_figures import (
            export_deleted_block_scale_figures,
        )
        from benchmark_design.report.vision.deleted_block_scale_stats import (
            compute_deleted_block_scale_summary_stats,
        )
        from benchmark_design.report.vision.deleted_block_scale_summary import (
            write_deleted_block_scale_summary_md,
        )
        from benchmark_design.vision.dataset import load_vision_benchmark_dataset
        from benchmark_design.vision.deleted_block_scale.pipeline import (
            compute_deleted_block_scale_results,
        )

        vision_processing = VisionProcessingOptions(
            show_progress=not args.no_progress,
            workers=args.workers,
            read_image_dimensions=False,
        )
        dataset = load_vision_benchmark_dataset(args.input, processing=vision_processing)
        pages = list(dataset.pages)
        results = compute_deleted_block_scale_results(
            args.input,
            processing=vision_processing,
            input_dir_for_images=args.input,
            pages=pages,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        tables_dir = args.output / "tables"
        metadata_dir = args.output / "metadata"
        tables_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        page_path = tables_dir / "deleted_block_scale_page_metrics.csv"
        block_path = tables_dir / "deleted_block_scale_block_geometry.csv"
        diagnostics_path = metadata_dir / "deleted_block_scale_diagnostics.json"
        summary_path = args.output / "deleted_block_scale_summary.md"
        dbs_stats = compute_deleted_block_scale_summary_stats(results)
        write_deleted_block_scale_page_metrics_csv(results, page_path)
        write_deleted_block_scale_block_geometry_csv(
            [record for result in results for record in result.block_records],
            block_path,
        )
        write_deleted_block_scale_diagnostics_json(dbs_stats, diagnostics_path)
        write_deleted_block_scale_summary_md(results, summary_path, stats=dbs_stats)
        if not args.skip_figures:
            export_deleted_block_scale_figures(
                results,
                pages,
                input_dir=args.input,
                figures_root=args.output / "figures" / "deleted_block_scale",
                show_progress=not args.no_progress,
            )
        print(f"pages: {len(results):,}")
        print(f"wrote {page_path}")
        print(f"wrote {block_path}")
        print(f"wrote {diagnostics_path}")
        print(f"wrote {summary_path}")
        return 0

    parser.error(f"Unhandled command: {args}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
