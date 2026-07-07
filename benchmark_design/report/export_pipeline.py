"""Unified benchmark export pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from pathlib import Path

from benchmark_design.ocr.ast_statistics import OcrAstStatisticsMetrics, compute_ocr_ast_statistics_from_features
from benchmark_design.ocr.consolidated import OcrConsolidatedMetrics, compute_ocr_consolidated_metrics_from_corpus
from benchmark_design.ocr.expression_features import corpus_token_counter, parse_success_rate
from benchmark_design.ocr.processing import EnrichedCorpus, ProcessingOptions, build_enriched_corpus_cached
from benchmark_design.ocr.tokenizer_docs import write_tokenizer_docs
from benchmark_design.progress import run_parallel_tasks
from benchmark_design.report.ast_statistics_table import write_ast_statistics_report
from benchmark_design.report.consolidated_table import write_consolidated_report
from benchmark_design.report.cross_benchmark_table import write_cross_benchmark_report
from benchmark_design.report.export_details import write_all_details
from benchmark_design.report.export_examples import write_all_examples
from benchmark_design.report.export_figures import write_all_figures
from benchmark_design.report.export_tables import write_all_tables
from benchmark_design.report.output_layout import BenchmarkOutputLayout, relative_input_path, relative_output_path
from benchmark_design.report.dataset_overview import run_dataset_overview_export
from benchmark_design.report.summary_md import write_benchmark_summary_md
from benchmark_design.vision.processing_options import VisionProcessingOptions


def _write_summary(
    layout: BenchmarkOutputLayout,
    enriched: EnrichedCorpus,
    *,
    consolidated: OcrConsolidatedMetrics,
    ast_metrics: OcrAstStatisticsMetrics,
) -> None:
    write_benchmark_summary_md(
        layout.summary_md,
        enriched,
        consolidated,
        ast_metrics,
    )


def _write_metadata(
    layout: BenchmarkOutputLayout,
    enriched: EnrichedCorpus,
    *,
    manifest: dict[str, str],
    processing: ProcessingOptions,
    input_anchor: Path,
) -> None:
    success_rate = parse_success_rate(list(enriched.features))
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": enriched.dataset,
        "input_dir": relative_input_path(enriched.input_dir, anchor=input_anchor),
        "json_file_count": enriched.json_file_count,
        "expression_count": len(enriched.features),
        "parse_success_rate": success_rate,
        "parse_failure_rate": 1.0 - success_rate,
        "processing": {
            "show_progress": processing.show_progress,
            "workers": processing.worker_count,
        },
        "manifest": manifest,
    }
    layout.metadata_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_benchmark_export(
    input_dir: Path,
    output_root: Path,
    *,
    dataset_name: str = "ours",
    processing: ProcessingOptions | None = None,
    enriched: EnrichedCorpus | None = None,
    skip_figures: bool = False,
    skip_cross_benchmark: bool = False,
    cross_benchmark_datasets: list[str] | None = None,
    skip_dataset_overview: bool = True,
) -> dict[str, Path]:
    processing = processing or ProcessingOptions()
    layout = BenchmarkOutputLayout(output_root)
    layout.ensure()

    enriched = build_enriched_corpus_cached(dataset_name, input_dir, processing, prebuilt=enriched)
    corpus_cache: dict[str, EnrichedCorpus] = {dataset_name: enriched}
    features = list(enriched.features)
    token_counter = corpus_token_counter(features)
    manifest: dict[str, str] = {}

    def _rel(path: Path) -> str:
        return relative_output_path(path, output_root)

    consolidated = compute_ocr_consolidated_metrics_from_corpus(enriched)
    ast_metrics = compute_ocr_ast_statistics_from_features(features)

    write_tasks: dict[str, Callable[[], Any]] = {
        "tables": lambda: write_all_tables(
            enriched,
            consolidated,
            layout.tables,
            appendix_dir=layout.tables_appendix,
            token_counter=token_counter,
        ),
        "consolidated": lambda: write_consolidated_report(
            consolidated,
            layout.tables,
            output_root=output_root,
            metadata_dir=layout.docs_metadata,
        ),
        "ast": lambda: write_ast_statistics_report(
            ast_metrics,
            layout.tables,
            input_dir=input_dir,
            output_root=output_root,
            metadata_dir=layout.docs_metadata,
        ),
        "docs": lambda: write_tokenizer_docs(
            features,
            layout.docs,
            layout.resources,
            token_counter=token_counter,
        ),
        "details": lambda: write_all_details(
            features,
            layout.details,
            layout.tables,
            input_dir=input_dir,
            token_counter=token_counter,
        ),
        "examples": lambda: write_all_examples(
            features,
            layout.examples,
            token_counter=token_counter,
        ),
    }
    if not skip_figures:
        write_tasks["figures"] = lambda: write_all_figures(
            features,
            layout.figures,
            token_counter=token_counter,
            input_dir=input_dir,
            expressions=list(enriched.expressions),
        )
    if not skip_cross_benchmark:
        write_tasks["cross"] = lambda: write_cross_benchmark_report(
            output_root,
            dataset_names=cross_benchmark_datasets,
            processing=processing,
            corpus_cache=corpus_cache,
        )

    write_results = run_parallel_tasks(
        write_tasks,
        description="Writing HMER export artifacts",
        show_progress=processing.show_progress,
        workers=processing.worker_count,
    )

    table_paths = write_results["tables"]
    manifest.update({f"table_{key}": _rel(path) for key, path in table_paths.items()})

    consolidated_paths = write_results["consolidated"]
    manifest.update({f"table_{key}": _rel(path) for key, path in consolidated_paths.items()})

    ast_paths = write_results["ast"]
    manifest.update({f"ast_{key}": _rel(path) for key, path in ast_paths.items()})

    doc_paths = write_results["docs"]
    manifest.update({f"doc_{key}": _rel(path) for key, path in doc_paths.items()})

    detail_paths = write_results["details"]
    manifest.update({f"detail_{key}": _rel(path) for key, path in detail_paths.items()})

    example_paths = write_results["examples"]
    manifest.update({f"example_{key}": _rel(path) for key, path in example_paths.items()})

    if "figures" in write_results:
        figure_paths = write_results["figures"]
        manifest.update({f"figure_{key}": _rel(path) for key, path in figure_paths.items()})

    if "cross" in write_results:
        cross_paths = write_results["cross"]
        manifest.update({f"cross_{key}": _rel(path) for key, path in cross_paths.items()})

    input_anchor = output_root.parent

    _write_summary(
        layout,
        enriched,
        consolidated=consolidated,
        ast_metrics=ast_metrics,
    )
    _write_metadata(
        layout,
        enriched,
        manifest=manifest,
        processing=processing,
        input_anchor=input_anchor,
    )
    manifest["summary"] = _rel(layout.summary_md)
    manifest["metadata"] = _rel(layout.metadata_json)

    if not skip_dataset_overview:
        overview_path = run_dataset_overview_export(
            input_dir,
            output_root,
            processing=processing,
            vision_processing=VisionProcessingOptions(
                show_progress=processing.show_progress,
                workers=processing.worker_count,
                read_image_dimensions=False,
            ),
            enriched=enriched,
        )
        manifest["dataset_overview"] = _rel(overview_path)

    return {key: Path(path) for key, path in manifest.items()}
