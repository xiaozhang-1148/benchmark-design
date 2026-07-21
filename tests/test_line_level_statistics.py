"""Tests for line-level dataset summary fields."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.line_level.geometry import page_orientation
from benchmark_design.line_level.models import LineLevelAnalysisResult, LineLevelConfig, LineMetricsRow, PageMetricsRow
from benchmark_design.line_level.statistics import build_dataset_summary, validate_line_level_export


def _row(**kwargs: object) -> LineMetricsRow:
    defaults = {
        "image_id": "img",
        "line_id": "0:0",
        "bbox_width_px": 100.0,
        "bbox_height_px": 10.0,
        "aspect_ratio": 10.0,
        "orientation_deg": 1.0,
        "orientation_direction_valid": True,
        "is_ignore": False,
        "is_valid": True,
        "invalid_reason": "",
        "page_orientation": "portrait",
        "block_type": "Txtblock",
    }
    defaults.update(kwargs)
    return LineMetricsRow(**defaults)  # type: ignore[arg-type]


def test_page_orientation_classification() -> None:
    assert page_orientation(600, 800) == "portrait"
    assert page_orientation(800, 600) == "landscape"
    assert page_orientation(500, 500) == "square"


def test_build_dataset_summary_includes_geometry_statistics() -> None:
    config = LineLevelConfig(input_root=Path("."), output_root=Path("."))
    result = LineLevelAnalysisResult(
        config=config,
        page_metrics=(
            PageMetricsRow(
                image_id="img",
                width=600,
                height=800,
                total_pixels=480000,
                dpi=None,
                line_count=1,
                valid_line_count=1,
                ignore_line_count=0,
                ioa_positive_pair_count=0,
                horizontal_adjacent_pair_count=0,
                page_orientation="portrait",
                median_bbox_height_px=10.0,
                p05_bbox_height_px=10.0,
                p95_bbox_height_px=10.0,
                processing_time_ms=1.0,
                status="ok",
            ),
        ),
        line_metrics=(_row(),),
        invalid_rows=(),
        processing_errors=(),
        processing_time_ms=1.0,
        discovered_page_count=1,
    )
    summary = build_dataset_summary(result)
    assert "continuous_line_metrics" in summary
    assert "bbox_height_px" in summary["continuous_line_metrics"]
    assert "bbox_width_px" in summary["continuous_line_metrics"]
    assert "line_height_px" not in summary["continuous_line_metrics"]
    assert "long_side_px" not in summary["continuous_line_metrics"]
    assert "nearest_distance_px" not in summary["continuous_line_metrics"]
    assert "max_overlap_iou" not in summary["continuous_line_metrics"]
    assert "center_x_norm" not in summary["continuous_line_metrics"]
    assert "orientation_validity" in summary
    assert summary["orientation_validity"]["orientation_valid"] == 1
    assert "orientation_thresholds" not in summary
    assert "overlap_iou_tiers" not in summary
    assert "target_pair_relations" in summary
    assert summary["target_pair_relations"]["ioa_positive_pair_count"] == 0
    assert summary["target_pair_relations"]["horizontal_adjacent_pair_count"] == 0
    assert summary["target_pair_relations"]["thresholds"]["horizontal_gap_px"] == 50.0
    assert summary["export_validation_errors"] == []
    assert summary["image_count"] == 1


def test_validate_line_level_export_detects_count_mismatch() -> None:
    config = LineLevelConfig(input_root=Path("."), output_root=Path("."))
    result = LineLevelAnalysisResult(
        config=config,
        page_metrics=(),
        line_metrics=(_row(),),
        invalid_rows=(),
        processing_errors=(),
    )
    errors = validate_line_level_export(result, {"page_count": 0, "line_count": 99, "valid_line_count": 1})
    assert any("line_count mismatch" in error for error in errors)
