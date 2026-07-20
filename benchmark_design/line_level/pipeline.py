"""Line-level analysis pipeline with per-image parallel processing."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import replace

from benchmark_design.line_level.bbox_ink import load_calibration_result
from benchmark_design.line_level.config import default_line_level_workers
from benchmark_design.line_level.loader import discover_pages_from_benchmark
from benchmark_design.line_level.models import (
    LineLevelAnalysisResult,
    LineLevelConfig,
    PageProcessResult,
    ProcessingErrorRow,
)
from benchmark_design.line_level.page_worker import process_page
from benchmark_design.progress import _run_with_optional_progress


def _resolve_workers(config: LineLevelConfig) -> int:
    return config.workers if config.workers is not None else default_line_level_workers()


def run_line_level_analysis(config: LineLevelConfig) -> LineLevelAnalysisResult:
    effective_config = config
    if config.bbox_outside_ink_enabled and config.calibration is None:
        if config.calibration_path is not None and config.calibration_path.is_file():
            effective_config = replace(
                config,
                calibration=load_calibration_result(config.calibration_path),
            )
    pages = discover_pages_from_benchmark(effective_config)
    if not pages:
        raise ValueError(f"No benchmark pages discovered under {config.input_root}")

    workers = _resolve_workers(effective_config)
    max_inflight = effective_config.max_inflight_pages or (workers * 2)
    started = time.perf_counter()

    page_results: list[PageProcessResult] = []
    processing_errors: list[ProcessingErrorRow] = []
    discovered_page_count = len(pages)

    def _submit_batches(executor: ThreadPoolExecutor) -> dict:
        future_to_page = {}
        for page in pages:
            future = executor.submit(process_page, page, effective_config)
            future_to_page[future] = page
        return future_to_page

    with ThreadPoolExecutor(max_workers=min(workers, max_inflight)) as executor:
        future_to_page = _submit_batches(executor)

        def _collect(advance) -> None:
            for future in as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    page_results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    processing_errors.append(
                        ProcessingErrorRow(
                            image_id=page.image_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    )
                advance()

        _run_with_optional_progress(
            "Processing line-level pages",
            len(future_to_page),
            show_progress=effective_config.show_progress,
            runner=_collect,
        )

    page_results.sort(key=lambda item: item.page_metrics.image_id)
    page_metrics = tuple(result.page_metrics for result in page_results)
    line_metrics = tuple(
        line for result in page_results for line in sorted(result.line_metrics, key=lambda row: row.line_id)
    )
    invalid_rows = tuple(row for result in page_results for row in result.invalid_rows)
    pair_rows = tuple(
        row
        for result in page_results
        for row in sorted(result.pair_rows, key=lambda item: (item.line_id_a, item.line_id_b))
    )
    processing_errors.extend(
        error for result in page_results if result.error is not None for error in [result.error]
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return LineLevelAnalysisResult(
        config=config,
        page_metrics=page_metrics,
        line_metrics=line_metrics,
        invalid_rows=invalid_rows,
        processing_errors=tuple(processing_errors),
        pair_rows=pair_rows,
        processing_time_ms=elapsed_ms,
        discovered_page_count=discovered_page_count,
    )
