"""Tests for line-level AABB size + mask-contour OBB signed orientation."""

from __future__ import annotations

from shapely.geometry import Polygon

from benchmark_design.line_level.geometry import (
    _long_side_alpha_deg,
    _long_side_signed_deg,
    _nearest_axis_tilt_deg,
    compute_line_geometry,
    geometry_from_line,
    validate_aabb_geometry,
)
from benchmark_design.line_level.models import LineAnnotation


def _line(polygon: tuple[tuple[float, float], ...]) -> LineAnnotation:
    return LineAnnotation(
        image_id="test",
        image_name="test.jpg",
        line_id="0:0",
        block_order=0,
        line_order=0,
        block_type="Txtblock",
        polygon=polygon,
        ocr="",
        source_file="test.json",
    )


def _page_size() -> tuple[int, int]:
    return 300, 200


def test_horizontal_rectangle_orientation_near_zero() -> None:
    shape = Polygon([(10, 20), (110, 20), (110, 30), (10, 30)])
    page_width, page_height = _page_size()
    geom = compute_line_geometry(shape, page_width=page_width, page_height=page_height)
    assert geom["bbox_width_px"] == 100.0
    assert geom["bbox_height_px"] == 10.0
    assert abs(geom["aspect_ratio"] - 10.0) < 0.01
    assert -90.0 <= geom["orientation_deg"] < 90.0
    assert abs(geom["orientation_deg"]) < 2.0
    assert geom["obb_long_side_px"] >= geom["obb_short_side_px"]
    assert validate_aabb_geometry(geom) == []


def test_vertical_rectangle_orientation_near_ninety() -> None:
    shape = Polygon([(10, 10), (30, 10), (30, 110), (10, 110)])
    page_width, page_height = _page_size()
    geom = compute_line_geometry(shape, page_width=page_width, page_height=page_height)
    assert geom["bbox_width_px"] == 20.0
    assert geom["bbox_height_px"] == 100.0
    assert abs(geom["aspect_ratio"] - 0.2) < 1e-9
    assert abs(geom["orientation_deg"]) > 80.0
    assert validate_aabb_geometry(geom) == []


def test_orientation_sign_distinguishes_tilt_direction() -> None:
    """Clockwise / counterclockwise tilts get opposite-signed α."""
    plus = Polygon([(0, 10), (100, 28), (97, 48), (-3, 30)])
    minus = Polygon([(0, 30), (100, 12), (97, -8), (-3, 10)])
    page_width, page_height = _page_size()
    g_plus = compute_line_geometry(plus, page_width=page_width, page_height=page_height)
    g_minus = compute_line_geometry(minus, page_width=page_width, page_height=page_height)
    assert g_plus["orientation_deg"] > 0.0
    assert g_minus["orientation_deg"] < 0.0
    assert abs(g_plus["orientation_deg"] + g_minus["orientation_deg"]) < 3.0


def test_near_square_axis_aligned_orientation_near_zero() -> None:
    shape = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    page_width, page_height = _page_size()
    geom = compute_line_geometry(shape, page_width=page_width, page_height=page_height)
    angle = abs(float(geom["orientation_deg"]))
    assert angle < 2.0 or abs(angle - 90.0) < 2.0
    assert geom["obb_long_side_px"] / geom["obb_short_side_px"] < 2.0


def test_tilted_rectangle_orientation_in_signed_interval() -> None:
    shape = Polygon([(0, 0), (40, 10), (30, 40), (-10, 30)])
    page_width, page_height = _page_size()
    geom = compute_line_geometry(shape, page_width=page_width, page_height=page_height)
    assert -90.0 <= geom["orientation_deg"] < 90.0
    assert geom["bbox_height_px"] > 0
    assert geom["bbox_width_px"] > 0
    assert geom["obb_long_side_px"] > 0
    assert geom["obb_short_side_px"] > 0


def test_ten_and_eighty_degree_tilts_keep_distinct_signed_angles() -> None:
    page_width, page_height = _page_size()
    ten = Polygon([(0, 0), (100, 17.6), (95, 26.3), (-5, 8.7)])
    eighty = Polygon([(0, 0), (17.6, 100), (26.3, 95), (8.7, -5)])
    g_ten = compute_line_geometry(ten, page_width=page_width, page_height=page_height)
    g_eighty = compute_line_geometry(eighty, page_width=page_width, page_height=page_height)
    assert 8.0 <= g_ten["orientation_deg"] <= 12.0
    assert 75.0 <= abs(g_eighty["orientation_deg"]) <= 85.0
    assert abs(g_ten["orientation_deg"] - abs(g_eighty["orientation_deg"])) > 60.0


def test_nearest_axis_tilt_formula() -> None:
    assert _nearest_axis_tilt_deg(0.0) == 0.0
    assert _nearest_axis_tilt_deg(90.0) == 0.0
    assert _nearest_axis_tilt_deg(-90.0) == 0.0
    assert _nearest_axis_tilt_deg(10.0) == 10.0
    assert _nearest_axis_tilt_deg(-10.0) == 10.0
    assert _nearest_axis_tilt_deg(80.0) == 10.0
    assert _nearest_axis_tilt_deg(45.0) == 45.0


def test_long_side_signed_and_unsigned() -> None:
    assert _long_side_signed_deg(100.0, 10.0, 12.0) == 12.0
    assert _long_side_alpha_deg(100.0, 10.0, 12.0) == 12.0
    assert abs(_long_side_signed_deg(10.0, 100.0, 0.0)) == 90.0
    assert _long_side_alpha_deg(10.0, 100.0, 0.0) == 90.0


def test_geometry_from_line_returns_size_metrics() -> None:
    line = _line(((50, 50), (150, 50), (150, 70), (50, 70)))
    shape = Polygon(line.polygon)
    geom = geometry_from_line(line, shape, page_width=200, page_height=160)
    assert geom["bbox_width_px"] == 100.0
    assert geom["bbox_height_px"] == 20.0
    assert abs(geom["aspect_ratio"] - 5.0) < 0.01
    assert abs(geom["orientation_deg"]) < 2.0
    assert "center_x_norm" not in geom
    assert "long_side_px" not in geom
    assert "line_height_px" not in geom
