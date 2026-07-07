"""Batch foreground load analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from benchmark_design.progress import parallel_map
from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.flow_structure.page_loader import load_page_annotations
from benchmark_design.vision.foreground_load.classification import (
    diagnostic_level,
    diagnostic_tags,
    relative_load_tertile,
)
from benchmark_design.vision.foreground_load.compute import (
    collect_page_darkness_histogram,
    compute_page_foreground_load,
)
from benchmark_design.vision.foreground_load.models import (
    ForegroundLoadThresholds,
    GlobalForegroundLoadConfig,
    PageForegroundLoadResult,
)
from benchmark_design.vision.foreground_load.normalization import (
    DARKNESS_BINS,
    DEFAULT_Q_HIGH,
    DEFAULT_Q_LOW,
    merge_histograms,
)
from benchmark_design.vision.foreground_load.threshold_estimation import (
    SENSITIVITY_DELTA,
    estimate_dataset_threshold,
)
from benchmark_design.vision.foreground_load.thresholds import (
    D_REVIEW_HIGH,
    D_REVIEW_LOW,
    LEVEL_LOW_MAX,
    LEVEL_MEDIUM_MAX,
    TAG_VERY_HIGH_MIN,
)
from benchmark_design.vision.processing_options import VisionProcessingOptions


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), percentile))


def _rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def compute_sensitivity_analysis(
    results: list[PageForegroundLoadResult],
    *,
    delta: float = SENSITIVITY_DELTA,
) -> dict[str, object]:
    page_rows = [
        (
            result.page_id,
            result.D_page_eff,
            result.D_page_tau_minus,
            result.D_page_tau_plus,
        )
        for result in results
        if result.D_page_eff is not None
        and result.D_page_tau_minus is not None
        and result.D_page_tau_plus is not None
    ]
    block_rows = [
        (
            block.page_id,
            block.block_id,
            block.D_block_i,
            block.D_block_tau_minus,
            block.D_block_tau_plus,
        )
        for result in results
        for block in result.block_results
        if block.D_block_i is not None
        and block.D_block_tau_minus is not None
        and block.D_block_tau_plus is not None
    ]

    page_base = [row[1] for row in page_rows]
    page_minus = [row[2] for row in page_rows]
    page_plus = [row[3] for row in page_rows]
    block_base = [row[2] for row in block_rows]
    block_minus = [row[3] for row in block_rows]
    block_plus = [row[4] for row in block_rows]

    def _mean_abs_delta(base: list[float], shifted: list[float]) -> float | None:
        if not base:
            return None
        return float(np.mean(np.abs(np.array(base) - np.array(shifted))))

    return {
        "delta": delta,
        "offsets": [f"-{delta:.2f}", f"+0.00", f"+{delta:.2f}"],
        "page_count": len(page_rows),
        "block_count": len(block_rows),
        "page_rank_spearman": {
            "tau_minus": _spearman(page_base, page_minus),
            "tau_plus": _spearman(page_base, page_plus),
        },
        "block_rank_spearman": {
            "tau_minus": _spearman(block_base, block_minus),
            "tau_plus": _spearman(block_base, block_plus),
        },
        "page_mean_abs_delta": {
            "tau_minus": _mean_abs_delta(page_base, page_minus),
            "tau_plus": _mean_abs_delta(page_base, page_plus),
        },
        "block_mean_abs_delta": {
            "tau_minus": _mean_abs_delta(block_base, block_minus),
            "tau_plus": _mean_abs_delta(block_base, block_plus),
        },
    }


def estimate_global_foreground_config(
    histograms: Sequence[np.ndarray],
    *,
    q_low: float = DEFAULT_Q_LOW,
    q_high: float = DEFAULT_Q_HIGH,
    darkness_bins: int = DARKNESS_BINS,
    sensitivity_delta: float = SENSITIVITY_DELTA,
) -> GlobalForegroundLoadConfig:
    non_empty = [hist for hist in histograms if hist.sum() > 0]
    merged = merge_histograms(non_empty)
    tau_D, method = estimate_dataset_threshold(merged)
    return GlobalForegroundLoadConfig(
        tau_D=tau_D,
        threshold_method=str(method),
        q_low=q_low,
        q_high=q_high,
        darkness_bins=darkness_bins,
        sensitivity_delta=sensitivity_delta,
        calibration_histogram=tuple(int(value) for value in merged.tolist()),
    )


def collect_darkness_histograms(
    pages: Sequence[PageAnnotation],
    *,
    input_dir: Path,
    processing: VisionProcessingOptions,
    global_config: GlobalForegroundLoadConfig | None = None,
) -> list[np.ndarray]:
    collect = partial(
        collect_page_darkness_histogram,
        input_dir=input_dir,
        global_config=global_config,
    )
    page_hists = parallel_map(
        collect,
        pages,
        description="Collecting calibration darkness histograms",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
    return [item.histogram for item in page_hists if item is not None]


def compute_foreground_load_thresholds(
    results: list[PageForegroundLoadResult],
    *,
    global_config: GlobalForegroundLoadConfig | None = None,
) -> ForegroundLoadThresholds:
    page_densities = [result.D_page_eff for result in results if result.D_page_eff is not None]
    block_densities = [
        block.D_block_i
        for result in results
        for block in result.block_results
        if block.D_block_i is not None
    ]
    return ForegroundLoadThresholds(
        absolute_low_medium=LEVEL_LOW_MAX,
        absolute_medium_high=LEVEL_MEDIUM_MAX,
        absolute_very_high=TAG_VERY_HIGH_MIN,
        page_p33=_percentile(page_densities, 33),
        page_p66=_percentile(page_densities, 66),
        block_p33=_percentile(block_densities, 33),
        block_p66=_percentile(block_densities, 66),
        review_low=D_REVIEW_LOW,
        review_high=D_REVIEW_HIGH,
        tau_D=global_config.tau_D if global_config is not None else None,
        threshold_method=global_config.threshold_method if global_config is not None else None,
        q_low=global_config.q_low if global_config is not None else None,
        q_high=global_config.q_high if global_config is not None else None,
    )


def assign_foreground_density_diagnostics(
    results: list[PageForegroundLoadResult],
    thresholds: ForegroundLoadThresholds,
) -> list[PageForegroundLoadResult]:
    """Fill diagnostic-only fields; main reports use continuous density statistics."""
    updated: list[PageForegroundLoadResult] = []
    for result in results:
        page_level = diagnostic_level(result.D_page_eff)
        page_tertile = relative_load_tertile(
            result.D_page_eff,
            p33=thresholds.page_p33,
            p66=thresholds.page_p66,
        )
        page_tags = diagnostic_tags(result.D_page_eff, result.review_reason)
        block_results = tuple(
            replace(
                block,
                foreground_load_level=diagnostic_level(block.D_block_i),
                relative_load_tertile=relative_load_tertile(
                    block.D_block_i,
                    p33=thresholds.block_p33,
                    p66=thresholds.block_p66,
                ),
                foreground_load_tags=diagnostic_tags(block.D_block_i, block.review_reason),
            )
            for block in result.block_results
        )
        updated.append(
            replace(
                result,
                foreground_load_level=page_level,
                relative_load_tertile=page_tertile,
                foreground_load_tags=page_tags,
                block_results=block_results,
            )
        )
    return updated


def assign_foreground_load_labels(
    results: list[PageForegroundLoadResult],
    thresholds: ForegroundLoadThresholds,
) -> list[PageForegroundLoadResult]:
    """Backward-compatible alias."""
    return assign_foreground_density_diagnostics(results, thresholds)


def _run_two_pass_foreground_load(
    pages: Sequence[PageAnnotation],
    *,
    input_dir: Path,
    processing: VisionProcessingOptions,
) -> tuple[list[PageForegroundLoadResult], ForegroundLoadThresholds, GlobalForegroundLoadConfig]:
    provisional_config = GlobalForegroundLoadConfig(
        tau_D=0.5,
        threshold_method="pooled_otsu",
    )
    histograms = collect_darkness_histograms(
        pages,
        input_dir=input_dir,
        processing=processing,
        global_config=provisional_config,
    )
    global_config = estimate_global_foreground_config(histograms)

    compute_page = partial(
        compute_page_foreground_load,
        input_dir=input_dir,
        global_config=global_config,
    )
    raw_results = parallel_map(
        compute_page,
        pages,
        description="Computing foreground load",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
    thresholds = compute_foreground_load_thresholds(raw_results, global_config=global_config)
    results = assign_foreground_density_diagnostics(raw_results, thresholds)
    return results, thresholds, global_config


def compute_foreground_load_results(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: str = "ours",
    input_dir_for_images: Path | None = None,
    pages: Sequence[PageAnnotation] | None = None,
) -> tuple[list[PageForegroundLoadResult], ForegroundLoadThresholds, GlobalForegroundLoadConfig]:
    processing = processing or VisionProcessingOptions()
    image_dir = input_dir_for_images or input_dir
    if pages is None:
        pages = load_page_annotations(input_dir, dataset=dataset, processing=processing)
    return _run_two_pass_foreground_load(pages, input_dir=image_dir, processing=processing)


def compute_foreground_load_from_pages(
    pages: Sequence[PageAnnotation],
    *,
    input_dir: Path,
    processing: VisionProcessingOptions | None = None,
) -> tuple[list[PageForegroundLoadResult], ForegroundLoadThresholds, GlobalForegroundLoadConfig]:
    processing = processing or VisionProcessingOptions()
    return _run_two_pass_foreground_load(pages, input_dir=input_dir, processing=processing)
