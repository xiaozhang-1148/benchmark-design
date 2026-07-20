"""Validate line polygon annotations."""

from __future__ import annotations

import math

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from dataclasses import dataclass

from benchmark_design.line_level.models import InvalidAnnotationRow, LineAnnotation


@dataclass(frozen=True, slots=True)
class ValidatedLine:
    line: LineAnnotation
    shape: Polygon


def _is_finite_point(point: tuple[float, float]) -> bool:
    return math.isfinite(point[0]) and math.isfinite(point[1])


def _iter_polygon_parts(geometry: BaseGeometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            yield from _iter_polygon_parts(part)
        return
    if isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygon_parts(part)


def _largest_polygon_part(geometry: BaseGeometry) -> Polygon | None:
    parts = list(_iter_polygon_parts(geometry))
    if not parts:
        return None
    return max(parts, key=lambda item: item.area)


def normalize_line_polygon(shape: Polygon) -> Polygon:
    """Return a usable polygon for metrics, accepting self-intersecting inputs."""
    if shape.is_valid:
        return shape

    repaired = make_valid(shape)
    normalized = _largest_polygon_part(repaired)
    if normalized is not None and normalized.area > 0:
        return normalized
    return shape


def validate_line_polygon(
    line: LineAnnotation,
    *,
    image_width: int,
    image_height: int,
) -> tuple[bool, str, Polygon | None]:
    polygon = line.polygon
    if len(polygon) < 3:
        return False, "insufficient_points", None
    if not all(_is_finite_point(point) for point in polygon):
        return False, "non_finite_coordinates", None

    shape = normalize_line_polygon(Polygon(polygon))

    if shape.area <= 0:
        return False, "zero_area", None

    if image_width > 0 and image_height > 0:
        minx, miny, maxx, maxy = shape.bounds
        if minx < 0 or miny < 0 or maxx > image_width or maxy > image_height:
            return False, "out_of_bounds", shape

    return True, "", shape


def validate_page_lines(
    lines: tuple[LineAnnotation, ...],
    *,
    image_width: int,
    image_height: int,
) -> tuple[list[ValidatedLine], list[InvalidAnnotationRow]]:
    valid_lines: list[ValidatedLine] = []
    invalid_rows: list[InvalidAnnotationRow] = []
    seen_ids: set[str] = set()

    for line in lines:
        if line.line_id in seen_ids:
            invalid_rows.append(
                InvalidAnnotationRow(
                    image_id=line.image_id,
                    line_id=line.line_id,
                    block_type=line.block_type,
                    reason="duplicate_line_id",
                    polygon_point_count=len(line.polygon),
                )
            )
            continue
        seen_ids.add(line.line_id)

        ok, reason, shape = validate_line_polygon(
            line,
            image_width=image_width,
            image_height=image_height,
        )
        if not ok or shape is None:
            invalid_rows.append(
                InvalidAnnotationRow(
                    image_id=line.image_id,
                    line_id=line.line_id,
                    block_type=line.block_type,
                    reason=reason,
                    polygon_point_count=len(line.polygon),
                )
            )
            continue
        valid_lines.append(ValidatedLine(line=line, shape=shape))

    return valid_lines, invalid_rows
