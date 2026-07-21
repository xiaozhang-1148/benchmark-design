"""Tests for line bbox-outside ink ratio and cross-dataset line geometry."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from benchmark_design.line_level.bbox_ink import (
    compute_bbox_outside_ink,
    load_calibration_result,
    load_normalized_ink_mask,
)
from benchmark_design.line_level.dataset_aspect import (
    DatasetLineGeometryRow,
    collect_dataset_image_sizes,
    dataset_size_summary_frame,
    geometry_from_box,
    geometry_from_image_size,
    rows_from_line_metrics,
)
from benchmark_design.report.line_level.plotting import export_external_dataset_aspect_plots


def test_bbox_outside_ink_detects_gap_pixels(tmp_path: Path) -> None:
    array = np.full((40, 60), 240, dtype=np.uint8)
    array[5:35, 5:55] = 20
    image_path = tmp_path / "page.png"
    Image.fromarray(array, mode="L").save(image_path)

    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "dark_reference": 10.0,
                "light_reference": 255.0,
                "gray_threshold": 128.0,
                "darkness_threshold": 128.0 / 255.0,
                "dark_percentile": 1.0,
                "light_percentile": 99.5,
                "threshold_method": "test",
                "image_count": 1,
            }
        ),
        encoding="utf-8",
    )
    calibration = load_calibration_result(calibration_path)
    ink_mask = load_normalized_ink_mask(image_path, calibration)
    shape = Polygon([(30, 8), (52, 20), (30, 32), (8, 20)])
    stats = compute_bbox_outside_ink(ink_mask, shape)
    assert stats.bbox_pixel_count > 0
    assert stats.bbox_outside_pixel_count > 0
    assert stats.bbox_outside_ink_count >= 0
    assert 0.0 <= stats.bbox_outside_ink_ratio <= 1.0


def test_axis_aligned_rectangle_mask_has_no_outside_region(tmp_path: Path) -> None:
    """When mask fills its pixel AABB, outside count is 0 → 区域内墨=0."""
    array = np.full((40, 60), 240, dtype=np.uint8)
    array[10:30, 10:50] = 20  # ink inside the rectangle
    image_path = tmp_path / "page.png"
    Image.fromarray(array, mode="L").save(image_path)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "dark_reference": 10.0,
                "light_reference": 255.0,
                "gray_threshold": 128.0,
                "darkness_threshold": 128.0 / 255.0,
                "dark_percentile": 1.0,
                "light_percentile": 99.5,
                "threshold_method": "test",
                "image_count": 1,
            }
        ),
        encoding="utf-8",
    )
    calibration = load_calibration_result(calibration_path)
    ink_mask = load_normalized_ink_mask(image_path, calibration)
    # Axis-aligned rectangle label (mask ≡ AABB after rasterization).
    shape = Polygon([(10, 10), (50, 10), (50, 30), (10, 30)])
    stats = compute_bbox_outside_ink(ink_mask, shape)
    assert stats.bbox_outside_pixel_count == 0
    assert stats.bbox_outside_ink_count == 0
    assert stats.bbox_outside_ink_ratio == 0.0


def test_tilted_quad_counts_outside_mask_interference(tmp_path: Path) -> None:
    """Mask/bbox gaps with foreground pixels are counted as interference."""
    array = np.full((80, 120), 240, dtype=np.uint8)
    array[20:50, 20:90] = 20
    image_path = tmp_path / "page.png"
    Image.fromarray(array, mode="L").save(image_path)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "dark_reference": 10.0,
                "light_reference": 255.0,
                "gray_threshold": 128.0,
                "darkness_threshold": 128.0 / 255.0,
                "dark_percentile": 1.0,
                "light_percentile": 99.5,
                "threshold_method": "test",
                "image_count": 1,
            }
        ),
        encoding="utf-8",
    )
    calibration = load_calibration_result(calibration_path)
    ink_mask = load_normalized_ink_mask(image_path, calibration)
    shape = Polygon([(20, 20), (90, 21), (90, 50), (20, 49)])
    stats = compute_bbox_outside_ink(ink_mask, shape)
    assert stats.bbox_outside_pixel_count > 0
    assert stats.has_interference
    assert stats.interference_ratio == pytest.approx(
        stats.interference_pixels / stats.bbox_area,
    )


def test_geometry_from_image_size_keeps_axes() -> None:
    width, height, aspect = geometry_from_image_size(200, 50)
    assert width == 200.0
    assert height == 50.0
    assert abs(aspect - 4.0) < 1e-9
    # Portrait image: height > width, aspect < 1 — must not swap.
    width2, height2, aspect2 = geometry_from_image_size(50, 200)
    assert width2 == 50.0
    assert height2 == 200.0
    assert abs(aspect2 - 0.25) < 1e-9
    # Alias still returns (width, height, aspect).
    assert geometry_from_box(200, 50) == (200.0, 50.0, 4.0)


def test_external_dataset_summary_and_plots(tmp_path: Path) -> None:
    rows = [
        DatasetLineGeometryRow("ours", "o1", width_px=200.0, height_px=40.0, aspect_ratio=5.0),
        DatasetLineGeometryRow("ours", "o2", width_px=250.0, height_px=50.0, aspect_ratio=5.0),
        DatasetLineGeometryRow("A", "a1", width_px=120.0, height_px=30.0, aspect_ratio=4.0),
        DatasetLineGeometryRow("A", "a2", width_px=140.0, height_px=35.0, aspect_ratio=4.0),
        DatasetLineGeometryRow("B", "b1", width_px=300.0, height_px=80.0, aspect_ratio=3.75),
        DatasetLineGeometryRow("B", "b2", width_px=330.0, height_px=90.0, aspect_ratio=3.67),
    ]
    summary = dataset_size_summary_frame(rows)
    assert set(summary["dataset"]) == {"ours", "A", "B"}
    assert "line_count" in summary.columns
    assert "height_px_median" in summary.columns
    assert "width_px_median" in summary.columns
    plots_dir = tmp_path / "plots"
    outputs = export_external_dataset_aspect_plots(rows, plots_dir)
    assert len(outputs) == 3


def test_rows_from_line_metrics_and_merge_ours() -> None:
    lines = [
        SimpleNamespace(
            image_id="p1",
            line_id="0:0",
            is_valid=True,
            bbox_height_px=40.0,
            bbox_width_px=200.0,
            aspect_ratio=5.0,
        ),
        SimpleNamespace(
            image_id="p1",
            line_id="0:1",
            is_valid=False,
            bbox_height_px=10.0,
            bbox_width_px=20.0,
            aspect_ratio=2.0,
        ),
        SimpleNamespace(
            image_id="p2",
            line_id="0:0",
            is_valid=True,
            bbox_height_px=50.0,
            bbox_width_px=100.0,
            aspect_ratio=2.0,
        ),
    ]
    ours = rows_from_line_metrics(lines, dataset="ours")
    assert len(ours) == 2
    assert ours[0].height_px == 40.0
    assert ours[0].width_px == 200.0
    assert abs(ours[0].aspect_ratio - 5.0) < 1e-9

    empty = Path("/tmp/definitely_missing_external_datasets_for_test")
    merged = collect_dataset_image_sizes(empty, workers=1, show_progress=False, ours_rows=ours)
    assert [row.dataset for row in merged] == ["ours", "ours"]
