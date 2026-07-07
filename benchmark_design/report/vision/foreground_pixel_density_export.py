"""CSV and JSON export for foreground pixel density metrics."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from benchmark_design.report.vision.foreground_pixel_density_stats import (
    DensityDistributionStats,
    FlowStructureDensityStats,
    compute_block_density_stats,
    compute_flow_structure_density_stats,
    compute_page_density_stats,
)
from benchmark_design.vision.flow_structure.models import PageFlowStructureResult
from benchmark_design.vision.foreground_load.classification import (
    FOREGROUND_LOAD_LEVELS,
    RELATIVE_LOAD_TERTILES,
)
from benchmark_design.vision.foreground_load.models import (
    BlockForegroundLoadResult,
    ForegroundLoadThresholds,
    GlobalForegroundLoadConfig,
    PageForegroundLoadResult,
)
from benchmark_design.vision.foreground_load.pipeline import compute_sensitivity_analysis

REGION_METRICS_COLUMNS: tuple[str, ...] = (
    "image_id",
    "region_id",
    "region_type",
    "region_pixels",
    "foreground_pixels",
    "foreground_density",
    "mean_darkness",
    "raw_otsu_density",
    "threshold_dataset",
    "raw_otsu_threshold",
    "diagnostic_tags",
    "needs_manual_review",
    "review_reason",
)

PAGE_METRICS_COLUMNS: tuple[str, ...] = (
    "page_id",
    "image_name",
    "image_width",
    "image_height",
    "page_area",
    "effective_region_area_ratio",
    "num_txtBlock",
    "num_figure",
    "num_chart",
    "num_deleted_text_block",
    "D_page",
    "mean_darkness",
    "raw_otsu_density",
    "D_page_tau_minus",
    "D_page_tau_plus",
    "diagnostic_tags",
    "threshold_dataset",
    "raw_otsu_threshold",
    "region_pixels_page",
    "foreground_pixels_page",
    "num_effective_blocks",
    "needs_manual_review",
    "review_reason",
    "review_image_path",
)

BLOCK_METRICS_COLUMNS: tuple[str, ...] = (
    "page_id",
    "block_id",
    "block_order",
    "block_type",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "D_block",
    "mean_darkness",
    "raw_otsu_density",
    "D_block_tau_minus",
    "D_block_tau_plus",
    "diagnostic_tags",
    "threshold_dataset",
    "raw_otsu_threshold",
    "region_pixels_block",
    "foreground_pixels_block",
    "needs_manual_review",
    "review_reason",
)

OVERALL_COLUMNS: tuple[str, ...] = (
    "metric",
    "unit",
    "n",
    "mean",
    "median",
    "p25",
    "p75",
    "p90",
    "area_weighted_density",
)

FLOW_STRUCTURE_COLUMNS: tuple[str, ...] = (
    "flow_structure",
    "pages",
    "median",
    "p25",
    "p75",
    "p90",
)


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _split_tags(tags: str) -> list[str]:
    return [tag for tag in tags.split(";") if tag]


def _region_row(
    *,
    image_id: str,
    region_id: str,
    region_type: str,
    region_pixels: int,
    foreground_pixels: int,
    foreground_density: float | None,
    mean_darkness: float | None,
    raw_otsu_density: float | None,
    threshold_dataset: float | None,
    raw_otsu_threshold: float | None,
    diagnostic_tags: str,
    needs_manual_review: bool,
    review_reason: str,
) -> list[str]:
    return [
        image_id,
        region_id,
        region_type,
        str(region_pixels),
        str(foreground_pixels),
        _fmt_float(foreground_density),
        _fmt_float(mean_darkness),
        _fmt_float(raw_otsu_density),
        _fmt_float(threshold_dataset),
        _fmt_float(raw_otsu_threshold),
        diagnostic_tags,
        str(needs_manual_review).lower(),
        review_reason,
    ]


def write_foreground_pixel_density_region_metrics_csv(
    results: list[PageForegroundLoadResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REGION_METRICS_COLUMNS)
        for result in results:
            writer.writerow(
                _region_row(
                    image_id=result.image_id,
                    region_id=result.region_id,
                    region_type=result.region_type,
                    region_pixels=result.mask_area,
                    foreground_pixels=result.foreground_pixels,
                    foreground_density=result.foreground_density,
                    mean_darkness=result.mean_darkness,
                    raw_otsu_density=result.raw_otsu_density,
                    threshold_dataset=result.threshold_dataset,
                    raw_otsu_threshold=result.raw_otsu_threshold,
                    diagnostic_tags=result.foreground_load_tags,
                    needs_manual_review=result.needs_manual_review,
                    review_reason=result.review_reason,
                )
            )
            for block in result.block_results:
                writer.writerow(
                    _region_row(
                        image_id=block.image_id,
                        region_id=block.region_id,
                        region_type=block.region_type,
                        region_pixels=block.mask_area,
                        foreground_pixels=block.foreground_pixels,
                        foreground_density=block.foreground_density,
                        mean_darkness=block.mean_darkness,
                        raw_otsu_density=block.raw_otsu_density,
                        threshold_dataset=block.threshold_dataset,
                        raw_otsu_threshold=block.raw_otsu_threshold,
                        diagnostic_tags=block.foreground_load_tags,
                        needs_manual_review=block.needs_manual_review,
                        review_reason=block.review_reason,
                    )
                )


def write_foreground_pixel_density_overall_csv(
    results: list[PageForegroundLoadResult],
    output_path: Path,
) -> tuple[DensityDistributionStats, DensityDistributionStats]:
    page_stats = compute_page_density_stats(results)
    block_stats = compute_block_density_stats(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OVERALL_COLUMNS)
        for stats in (page_stats, block_stats):
            writer.writerow(
                [
                    stats.metric,
                    stats.unit,
                    stats.n,
                    f"{stats.mean:.6f}" if stats.n else "",
                    f"{stats.median:.6f}" if stats.n else "",
                    f"{stats.p25:.6f}" if stats.n else "",
                    f"{stats.p75:.6f}" if stats.n else "",
                    f"{stats.p90:.6f}" if stats.n else "",
                    _fmt_float(stats.area_weighted_density),
                ]
            )
    return page_stats, block_stats


def write_foreground_pixel_density_by_flow_structure_csv(
    stats: list[FlowStructureDensityStats],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FLOW_STRUCTURE_COLUMNS)
        for row in stats:
            writer.writerow(
                [
                    row.flow_structure,
                    row.pages,
                    f"{row.median:.6f}" if row.pages else "",
                    f"{row.p25:.6f}" if row.pages else "",
                    f"{row.p75:.6f}" if row.pages else "",
                    f"{row.p90:.6f}" if row.pages else "",
                ]
            )


def write_foreground_pixel_density_page_metrics_csv(
    results: list[PageForegroundLoadResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAGE_METRICS_COLUMNS)
        for result in results:
            writer.writerow(
                [
                    result.page_id,
                    result.image_name,
                    result.image_width,
                    result.image_height,
                    result.page_area,
                    _fmt_float(result.effective_region_area_ratio),
                    result.num_txtBlock,
                    result.num_figure,
                    result.num_chart,
                    result.num_deleted_text_block,
                    _fmt_float(result.D_page_eff),
                    _fmt_float(result.mean_darkness),
                    _fmt_float(result.raw_otsu_density),
                    _fmt_float(result.D_page_tau_minus),
                    _fmt_float(result.D_page_tau_plus),
                    result.foreground_load_tags,
                    _fmt_float(result.threshold_dataset),
                    _fmt_float(result.page_otsu_threshold),
                    result.R_eff_area,
                    result.F_eff,
                    result.num_effective_blocks,
                    str(result.needs_manual_review).lower(),
                    result.review_reason,
                    result.review_image_path,
                ]
            )


def write_foreground_pixel_density_block_metrics_csv(
    block_results: list[BlockForegroundLoadResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BLOCK_METRICS_COLUMNS)
        for block in block_results:
            writer.writerow(
                [
                    block.page_id,
                    block.block_id,
                    block.block_order,
                    block.block_type,
                    f"{block.bbox_x1:.2f}",
                    f"{block.bbox_y1:.2f}",
                    f"{block.bbox_x2:.2f}",
                    f"{block.bbox_y2:.2f}",
                    _fmt_float(block.D_block_i),
                    _fmt_float(block.mean_darkness),
                    _fmt_float(block.raw_otsu_density),
                    _fmt_float(block.D_block_tau_minus),
                    _fmt_float(block.D_block_tau_plus),
                    block.foreground_load_tags,
                    _fmt_float(block.threshold_dataset),
                    _fmt_float(block.block_otsu_threshold),
                    block.block_mask_area,
                    block.F_i,
                    str(block.needs_manual_review).lower(),
                    block.review_reason,
                ]
            )


def write_foreground_pixel_density_diagnostics_json(
    results: list[PageForegroundLoadResult],
    thresholds: ForegroundLoadThresholds,
    output_path: Path,
    *,
    global_config: GlobalForegroundLoadConfig | None = None,
) -> None:
    tag_counts: Counter[str] = Counter()
    for result in results:
        for tag in _split_tags(result.foreground_load_tags):
            tag_counts[tag] += 1
        for block in result.block_results:
            for tag in _split_tags(block.foreground_load_tags):
                tag_counts[tag] += 1

    page_level_counts = Counter(
        result.foreground_load_level for result in results if result.foreground_load_level
    )
    block_level_counts = Counter(
        block.foreground_load_level
        for result in results
        for block in result.block_results
        if block.foreground_load_level
    )
    page_tertile_counts = Counter(
        result.relative_load_tertile for result in results if result.relative_load_tertile
    )
    block_tertile_counts = Counter(
        block.relative_load_tertile
        for result in results
        for block in result.block_results
        if block.relative_load_tertile
    )

    config = global_config
    if config is None and thresholds.tau_D is not None:
        config = GlobalForegroundLoadConfig(
            tau_D=thresholds.tau_D,
            threshold_method=thresholds.threshold_method or "pooled_otsu",
            q_low=thresholds.q_low or 1.0,
            q_high=thresholds.q_high or 99.0,
        )

    sensitivity = compute_sensitivity_analysis(results)

    payload = {
        "methodology": {
            "primary_metric": "Dataset-threshold foreground pixel density",
            "supplementary_metric": "Mean darkness (mean S over region)",
            "baseline_metric": "Raw Otsu Density",
            "effective_region": "txtBlock | deleted_text_block | chart | figure",
            "block_region": "txtBlock only",
            "pipeline": [
                "grayscale G in [0,255]",
                "robust percentile normalization G_tilde = clip((G-q_low)/(q_high-q_low),0,1)",
                "darkness S = 1 - G_tilde",
                "calibration set from pooled S within R_eff across dataset",
                "dataset threshold tau_D via bimodal valley, GMM intersection, or pooled Otsu",
                "foreground = {p | S(p) >= tau_D}",
            ],
            "tau_D": config.tau_D if config is not None else None,
            "threshold_method": config.threshold_method if config is not None else None,
            "q_low": config.q_low if config is not None else None,
            "q_high": config.q_high if config is not None else None,
            "sensitivity_delta": config.sensitivity_delta if config is not None else None,
        },
        "absolute_thresholds": {
            "low_medium": thresholds.absolute_low_medium,
            "medium_high": thresholds.absolute_medium_high,
            "extreme_candidate": thresholds.absolute_very_high,
        },
        "relative_tertile_thresholds": {
            "page_p33": thresholds.page_p33,
            "page_p66": thresholds.page_p66,
            "block_p33": thresholds.block_p33,
            "block_p66": thresholds.block_p66,
        },
        "review_thresholds": {
            "low": thresholds.review_low,
            "high": thresholds.review_high,
        },
        "diagnostic_tag_counts": dict(tag_counts),
        "diagnostic_level_counts": {
            "page": {label: page_level_counts.get(label, 0) for label in FOREGROUND_LOAD_LEVELS},
            "block": {label: block_level_counts.get(label, 0) for label in FOREGROUND_LOAD_LEVELS},
        },
        "diagnostic_tertile_counts": {
            "page": {label: page_tertile_counts.get(label, 0) for label in RELATIVE_LOAD_TERTILES},
            "block": {label: block_tertile_counts.get(label, 0) for label in RELATIVE_LOAD_TERTILES},
        },
        "sensitivity_analysis": sensitivity,
        "note": (
            "Thresholds and quantile bins are used only for internal quality diagnostics "
            "and are not used as benchmark density levels."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compute_and_write_flow_structure_density_csv(
    fg_results: list[PageForegroundLoadResult],
    flow_results: list[PageFlowStructureResult],
    output_path: Path,
) -> list[FlowStructureDensityStats]:
    stats = compute_flow_structure_density_stats(fg_results, flow_results)
    write_foreground_pixel_density_by_flow_structure_csv(stats, output_path)
    return stats
