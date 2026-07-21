"""Tests for orientation validity summary used by chapter tables."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.line_level.models import LineLevelAnalysisResult, LineLevelConfig, LineMetricsRow
from benchmark_design.line_level.statistics import CONTINUOUS_LINE_METRICS, build_dataset_summary


def _config() -> LineLevelConfig:
    return LineLevelConfig(
        input_root=Path("."),
        output_root=Path("."),
        orientation_min_aspect_ratio=2.0,
    )


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


def test_continuous_line_metrics_are_geometry_only() -> None:
    assert "bbox_height_px" in CONTINUOUS_LINE_METRICS
    assert "bbox_width_px" in CONTINUOUS_LINE_METRICS
    assert "aspect_ratio" in CONTINUOUS_LINE_METRICS
    assert "orientation_deg" in CONTINUOUS_LINE_METRICS
    assert "line_height_px" not in CONTINUOUS_LINE_METRICS
    assert "long_side_px" not in CONTINUOUS_LINE_METRICS
    assert "nearest_distance_px" not in CONTINUOUS_LINE_METRICS
    assert "max_overlap_iou" not in CONTINUOUS_LINE_METRICS
    assert "center_x_norm" not in CONTINUOUS_LINE_METRICS


def test_orientation_validity_in_summary() -> None:
    rows = (
        _row(aspect_ratio=10.0, orientation_deg=8.0, orientation_direction_valid=True),
        _row(
            line_id="0:1",
            aspect_ratio=1.2,
            orientation_deg=80.0,
            orientation_direction_valid=False,
        ),
        _row(line_id="0:2", orientation_deg=1.0),
        _row(line_id="0:3", orientation_deg=6.0),
    )
    result = LineLevelAnalysisResult(
        config=_config(),
        page_metrics=(),
        line_metrics=rows,
        invalid_rows=(),
        processing_errors=(),
    )
    summary = build_dataset_summary(result)
    assert "orientation_thresholds" not in summary
    assert summary["orientation_validity"] == {
        "all_lines": 4,
        "orientation_valid": 3,
        "orientation_excluded": 1,
    }
    assert "overlap_iou_tiers" not in summary
