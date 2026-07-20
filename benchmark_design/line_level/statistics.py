"""Dataset-level statistics for line-level geometry analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_design.line_level.models import (
    LineLevelAnalysisResult,
    LineMetricsRow,
    PageMetricsRow,
    TargetPairRow,
)
from benchmark_design.line_level.layout import (
    DEFAULT_HEIGHT_SIMILARITY_THRESHOLD,
    DEFAULT_HORIZONTAL_GAP_PX_THRESHOLD,
    DEFAULT_VERTICAL_OVERLAP_RATIO_THRESHOLD,
    horizontal_adjacent_scope_rows,
    summarize_target_pairs,
)
from benchmark_design.report.line_level.chapter_tables import (
    ink_natural_state_rows,
    orientation_validity_rows,
)

CONTINUOUS_LINE_METRICS: tuple[str, ...] = (
    "bbox_height_px",
    "bbox_width_px",
    "aspect_ratio",
    "orientation_deg",
    "bbox_outside_ink_ratio",
)

CONTINUOUS_PAGE_METRICS: tuple[str, ...] = (
    "line_count",
    "median_bbox_height_px",
)

PAGE_ORIENTATIONS: tuple[str, ...] = ("portrait", "landscape", "square")


def _continuous_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _valid_metric_values(rows: list[LineMetricsRow], metric: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if not row.is_valid:
            continue
        value = getattr(row, metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        values.append(float(value))
    return np.array(values, dtype=np.float64)


def validate_line_level_export(result: LineLevelAnalysisResult, summary: dict) -> list[str]:
    errors: list[str] = []
    if summary.get("page_count", 0) != len(result.page_metrics):
        errors.append("page_count mismatch")
    if summary.get("line_count", 0) != len(result.line_metrics):
        errors.append("line_count mismatch")
    valid = sum(1 for row in result.line_metrics if row.is_valid)
    if summary.get("valid_line_count", -1) != valid:
        errors.append("valid_line_count mismatch")
    return errors


def build_dataset_summary(result: LineLevelAnalysisResult) -> dict:
    line_rows = list(result.line_metrics)
    page_rows = list(result.page_metrics)
    valid_lines = [row for row in line_rows if row.is_valid]
    continuous = {metric: _continuous_stats(_valid_metric_values(line_rows, metric)) for metric in CONTINUOUS_LINE_METRICS}
    page_continuous = {
        metric: _continuous_stats(np.array([getattr(page, metric) for page in page_rows], dtype=np.float64))
        for metric in CONTINUOUS_PAGE_METRICS
    }
    orientation_counts = Counter(page.page_orientation for page in page_rows)
    page_count = len(page_rows)
    summary_counts = {
        "page_count": page_count,
        "line_count": len(line_rows),
        "valid_line_count": len(valid_lines),
    }
    pair_summary = summarize_target_pairs(list(result.pair_rows))
    pair_scope = horizontal_adjacent_scope_rows(
        list(result.pair_rows),
        valid_line_count=len(valid_lines),
        page_count=page_count,
    )
    orientation_validity = {
        row["item"]: int(row["count"]) for row in orientation_validity_rows(line_rows)
    }
    ink_states = ink_natural_state_rows(line_rows)
    return {
        "discovered_page_count": result.discovered_page_count,
        "page_count": page_count,
        "image_count": page_count,
        "line_count": len(line_rows),
        "valid_line_count": len(valid_lines),
        "invalid_line_count": len(result.invalid_rows),
        "processing_error_count": len(result.processing_errors),
        "processing_time_ms": result.processing_time_ms,
        "page_orientation_counts": {key: orientation_counts.get(key, 0) for key in PAGE_ORIENTATIONS},
        "continuous_line_metrics": continuous,
        "continuous_page_metrics": page_continuous,
        "orientation_validity": orientation_validity,
        "bbox_outside_ink_natural_states": ink_states,
        "target_pair_relations": {
            "scope": "same_page_horizontal_adjacent",
            "pair_count": pair_summary["pair_count"],
            "unique_line_count": pair_summary["unique_line_count"],
            "unique_page_count": pair_summary["unique_page_count"],
            "pair_scope": pair_scope,
            "ioa_positive_pair_count": pair_summary["ioa_positive_pair_count"],
            "horizontal_adjacent_pair_count": pair_summary["horizontal_adjacent_pair_count"],
            "thresholds": {
                "height_similarity": float(
                    getattr(result.config, "height_similarity_threshold", DEFAULT_HEIGHT_SIMILARITY_THRESHOLD)
                ),
                "vertical_overlap_ratio": float(
                    getattr(
                        result.config,
                        "vertical_overlap_ratio_threshold",
                        DEFAULT_VERTICAL_OVERLAP_RATIO_THRESHOLD,
                    )
                ),
                "horizontal_gap_px": float(
                    getattr(result.config, "horizontal_gap_px_threshold", DEFAULT_HORIZONTAL_GAP_PX_THRESHOLD)
                ),
            },
            "definitions": {
                "candidate_pairs": (
                    "AABB shares a vertical interval, and either x-ranges overlap "
                    "or horizontal gap <= horizontal_gap_px (disjoint bands / far x skipped)"
                ),
                "ioa": "intersection_area / min(area_a, area_b); positive when intersection_area > 0",
                "horizontal_adjacent": (
                    "IoA=0, horizontally separated AABBs, "
                    "S_h>=threshold, R_v>=threshold, G_x<=threshold_px"
                ),
            },
        },
        "export_validation_errors": validate_line_level_export(result, summary_counts),
        "config": {
            key: (
                str(value)
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in asdict(result.config).items()
        },
    }


def line_metrics_frame(rows: list[LineMetricsRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def page_metrics_frame(rows: list[PageMetricsRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def target_pair_frame(rows: list[TargetPairRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])
