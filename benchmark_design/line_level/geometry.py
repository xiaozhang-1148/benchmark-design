"""Line geometry: axis-aligned bbox for size; mask-contour OBB for direction."""

from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Polygon

from benchmark_design.line_level.bbox_ink import rasterize_line_polygon
from benchmark_design.line_level.models import LineAnnotation


def _normalize_angle_deg(angle: float) -> float:
    """Map angle into [-90, 90)."""
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def polygon_to_shapely(polygon: tuple[tuple[float, float], ...]) -> Polygon:
    return Polygon(polygon)


def page_orientation(width: int, height: int) -> str:
    if width < height:
        return "portrait"
    if width > height:
        return "landscape"
    return "square"


def axis_aligned_bbox(shape: Polygon) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) in original image pixel coordinates."""
    minx, miny, maxx, maxy = shape.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _unsigned_deflection_deg(angle: float) -> float:
    """Absolute deflection of an undirected edge from image +x, in [0, 90]."""
    return abs(_normalize_angle_deg(angle))


def _long_side_alpha_deg(rect_w: float, rect_h: float, angle: float) -> float:
    """Unsigned |α|: long-side angle vs image +x, in [0, 90]."""
    return abs(_long_side_signed_deg(rect_w, rect_h, angle))


def _long_side_signed_deg(rect_w: float, rect_h: float, angle: float) -> float:
    """Signed α: long-side angle vs image +x, in [-90, 90)."""
    if max(rect_w, rect_h) <= 0:
        return 0.0
    if rect_w >= rect_h:
        return _normalize_angle_deg(angle)
    return _normalize_angle_deg(angle + 90.0)


def _nearest_axis_tilt_deg(alpha: float) -> float:
    """θ = min(|α|, 90° − |α|): nearest-axis tilt magnitude, in [0, 45]."""
    alpha_abs = abs(_normalize_angle_deg(alpha))
    return float(min(alpha_abs, 90.0 - alpha_abs))


def _largest_contour_points(mask: np.ndarray) -> np.ndarray | None:
    uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0 or largest.shape[0] < 3:
        return None
    return largest.reshape(-1, 2).astype(np.float32)


def _contour_points_for_orientation(
    shape: Polygon,
    *,
    page_width: int,
    page_height: int,
) -> np.ndarray:
    """GT line mask contour; fall back to polygon vertices when rasterization is empty."""
    if page_width > 0 and page_height > 0:
        mask = rasterize_line_polygon(shape, page_height, page_width)
        contour = _largest_contour_points(mask)
        if contour is not None:
            return contour
    return np.asarray(shape.exterior.coords[:-1], dtype=np.float32)


def _orientation_deg_from_contour_points(coords: np.ndarray) -> float:
    if coords.shape[0] < 3:
        return 0.0
    rect = cv2.minAreaRect(coords)
    (rect_w, rect_h) = rect[1]
    angle = float(rect[2])
    return _long_side_signed_deg(rect_w, rect_h, angle)


def compute_line_geometry(
    shape: Polygon,
    *,
    page_width: int,
    page_height: int,
) -> dict[str, float]:
    """AABB size metrics + mask-contour OBB orientation metrics.

    Chapter 4.1.1 uses bbox_width/height/aspect_ratio (AABB Δx / Δy).
    Chapter 4.1.2 uses orientation_deg = signed α (OBB long side vs image +x),
    in [-90°, 90°); positive tilts downward in image coordinates.
    """
    x_min, y_min, x_max, y_max = axis_aligned_bbox(shape)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width <= 0 or bbox_height <= 0:
        bbox_width = max(bbox_width, 0.0)
        bbox_height = max(bbox_height, 0.0)

    contour_coords = _contour_points_for_orientation(
        shape,
        page_width=page_width,
        page_height=page_height,
    )
    rect = cv2.minAreaRect(contour_coords)
    (rect_w, rect_h) = rect[1]
    obb_long = float(max(rect_w, rect_h))
    obb_short = float(min(rect_w, rect_h))
    if obb_long <= 0:
        obb_long = 1.0
        obb_short = 1.0

    orientation = _orientation_deg_from_contour_points(contour_coords)

    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0.0
    return {
        "bbox_width_px": float(bbox_width),
        "bbox_height_px": float(bbox_height),
        "aspect_ratio": float(aspect_ratio),
        "obb_long_side_px": obb_long,
        "obb_short_side_px": obb_short,
        "orientation_deg": float(orientation),
    }


def geometry_from_line(
    line: LineAnnotation,
    shape: Polygon,
    *,
    page_width: int,
    page_height: int,
) -> dict[str, float]:
    _ = line
    return compute_line_geometry(
        shape,
        page_width=page_width,
        page_height=page_height,
    )


def validate_aabb_geometry(geom: dict[str, float], *, atol: float = 1e-6) -> list[str]:
    """Validate AABB fields; does not require width >= height."""
    errors: list[str] = []
    width = float(geom.get("bbox_width_px", 0.0))
    height = float(geom.get("bbox_height_px", 0.0))
    aspect = float(geom.get("aspect_ratio", 0.0))
    if not math.isfinite(width) or width <= 0:
        errors.append("bbox_width_px must be finite and > 0")
    if not math.isfinite(height) or height <= 0:
        errors.append("bbox_height_px must be finite and > 0")
    if not math.isfinite(aspect) or aspect <= 0:
        errors.append("aspect_ratio must be finite and > 0")
    if height > 0 and abs(aspect - width / height) > atol:
        errors.append("aspect_ratio must equal bbox_width_px / bbox_height_px")
    orientation = float(geom.get("orientation_deg", 0.0))
    if not math.isfinite(orientation) or orientation < -90.0 or orientation >= 90.0:
        errors.append("orientation_deg must be finite and in [-90, 90)")
    return errors


def validate_line_metrics_aabb(rows: list, *, atol: float = 1e-5) -> list[str]:
    """Post-run checklist for chapter 4.1.1 AABB fields on LineMetricsRow rows."""
    errors: list[str] = []
    taller_than_wide = 0
    for row in rows:
        if not getattr(row, "is_valid", False):
            continue
        width = float(getattr(row, "bbox_width_px", 0.0))
        height = float(getattr(row, "bbox_height_px", 0.0))
        aspect = float(getattr(row, "aspect_ratio", 0.0))
        orientation = float(getattr(row, "orientation_deg", 0.0))
        row_errors = validate_aabb_geometry(
            {
                "bbox_width_px": width,
                "bbox_height_px": height,
                "aspect_ratio": aspect,
                "orientation_deg": orientation,
            },
            atol=atol,
        )
        image_id = getattr(row, "image_id", "?")
        line_id = getattr(row, "line_id", "?")
        for message in row_errors:
            errors.append(f"{image_id}:{line_id} {message}")
            if len(errors) >= 50:
                errors.append("... truncated")
                return errors
        if width < height:
            taller_than_wide += 1
    if taller_than_wide == 0 and any(getattr(r, "is_valid", False) for r in rows):
        # Not an error — only note absence when useful; checklist allows width < height.
        pass
    return errors
