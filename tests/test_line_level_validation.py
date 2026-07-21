"""Tests for line polygon validation."""

from __future__ import annotations

from benchmark_design.line_level.models import LineAnnotation
from benchmark_design.line_level.validation import normalize_line_polygon, validate_line_polygon
from shapely.geometry import Polygon


def _line(polygon: tuple[tuple[float, float], ...]) -> LineAnnotation:
    return LineAnnotation(
        image_id="page",
        image_name="page.png",
        line_id="0:0",
        block_order=0,
        line_order=0,
        block_type="Txtblock",
        polygon=polygon,
        ocr="sample",
        source_file="page.png.json",
    )


def test_self_intersecting_polygon_is_accepted() -> None:
    polygon = ((285, 373), (365, 374), (367, 421), (285, 421), (285, 372), (285, 373))
    ok, reason, shape = validate_line_polygon(
        _line(polygon),
        image_width=680,
        image_height=657,
    )

    assert ok is True
    assert reason == ""
    assert shape is not None
    assert shape.area > 0


def test_normalize_line_polygon_prefers_repaired_part() -> None:
    polygon = Polygon(((285, 373), (365, 374), (367, 421), (285, 421), (285, 372), (285, 373)))
    normalized = normalize_line_polygon(polygon)

    assert normalized.area > 0
    assert normalized.is_valid
