"""Black-pixel interference in the line bbox outside the polygon mask."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import math
import numpy as np
from shapely.geometry import Polygon

from benchmark_design.foreground.calibration import calibration_to_threshold_config
from benchmark_design.foreground.mask import foreground_mask_from_gray_threshold
from benchmark_design.foreground.models import ForegroundThresholdConfig
from benchmark_design.foreground.threshold import load_foreground_threshold_config
from benchmark_design.io.image import load_grayscale_image
from benchmark_design.page_level.models import CalibrationResult


@dataclass(frozen=True, slots=True)
class BBoxOutsideInkStats:
    interference_ratio: float
    outside_area: int
    interference_pixels: int
    bbox_area: int
    mask_area: int
    has_interference: bool

    @property
    def bbox_outside_ink_ratio(self) -> float:
        return self.interference_ratio

    @property
    def bbox_outside_pixel_count(self) -> int:
        return self.outside_area

    @property
    def bbox_outside_ink_count(self) -> int:
        return self.interference_pixels

    @property
    def bbox_pixel_count(self) -> int:
        return self.bbox_area


def load_calibration_result(path: Path) -> CalibrationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dark_reference = float(payload["dark_reference"])
    light_reference = float(payload["light_reference"])
    gray_threshold = float(
        payload.get(
            "gray_threshold",
            payload.get("global_threshold", payload.get("t_I", 128.0)),
        )
    )
    if "darkness_threshold" in payload:
        tau_d = float(payload["darkness_threshold"])
    elif "tau_D" in payload:
        tau_d = float(payload["tau_D"])
    elif "tau_d" in payload:
        tau_d = float(payload["tau_d"])
    elif payload.get("foreground_valley_threshold") is not None:
        tau_d = float(payload["foreground_valley_threshold"])
    else:
        from benchmark_design.foreground.threshold import gray_threshold_to_tau_d

        tau_d = gray_threshold_to_tau_d(
            gray_threshold,
            dark_reference=dark_reference,
            light_reference=light_reference,
        )
    gray_hist = payload.get("gray_histogram", payload.get("darkness_histogram"))
    return CalibrationResult(
        dark_reference=dark_reference,
        light_reference=light_reference,
        gray_threshold=gray_threshold,
        tau_d=float(tau_d),
        dark_percentile=float(payload.get("dark_percentile", 1.0)),
        light_percentile=float(payload.get("light_percentile", 99.5)),
        threshold_method=str(payload.get("threshold_method", "global_pooled_otsu")),
        image_count=int(payload.get("num_pages", payload.get("image_count", 0))),
        gray_histogram=tuple(int(v) for v in gray_hist) if gray_hist else (),
    )


def load_foreground_threshold(path: Path) -> ForegroundThresholdConfig:
    return load_foreground_threshold_config(path)


def calibration_to_foreground_config(calibration: CalibrationResult) -> ForegroundThresholdConfig:
    return calibration_to_threshold_config(calibration)


def load_normalized_ink_mask(
    image_path: Path,
    calibration: CalibrationResult,
) -> np.ndarray:
    """Load page-level foreground mask F using shared gray threshold (I <= t_I)."""
    gray = load_grayscale_image(image_path)
    return foreground_mask_from_gray_threshold(gray, calibration.gray_threshold)


def rasterize_line_polygon(shape: Polygon, height: int, width: int) -> np.ndarray:
    """Rasterize a GT line polygon to a boolean mask."""
    return _raster_polygon(shape, height, width)


def _raster_polygon(shape: Polygon, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    coords = np.asarray(shape.exterior.coords[:-1], dtype=np.float32)
    if coords.shape[0] < 3:
        return mask.astype(bool)
    cv2.fillPoly(mask, [np.round(coords).astype(np.int32)], 1)
    return mask.astype(bool)


def _bbox_mask_from_line_mask(line_mask: np.ndarray) -> np.ndarray:
    """AABB of the rasterized mask itself (inclusive pixel bounds)."""
    if not np.any(line_mask):
        return np.zeros_like(line_mask, dtype=bool)
    ys, xs = np.where(line_mask)
    bbox = np.zeros_like(line_mask, dtype=bool)
    bbox[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1] = True
    return bbox


def compute_bbox_outside_ink(
    foreground_mask: np.ndarray,
    shape: Polygon,
) -> BBoxOutsideInkStats:
    """Interference pixels in (AABB \\ mask) using unified foreground mask F."""
    height, width = foreground_mask.shape[:2]
    line_mask = _raster_polygon(shape, height, width)
    bbox_mask = _bbox_mask_from_line_mask(line_mask)
    outside = bbox_mask & ~line_mask
    bbox_area = int(bbox_mask.sum())
    mask_area = int(line_mask.sum())
    outside_area = int(outside.sum())
    interference_pixels = int((foreground_mask & outside).sum()) if outside_area else 0
    has_interference = interference_pixels > 0
    ratio = float(interference_pixels / bbox_area) if bbox_area else 0.0
    return BBoxOutsideInkStats(
        interference_ratio=ratio,
        outside_area=outside_area,
        interference_pixels=interference_pixels,
        bbox_area=bbox_area,
        mask_area=mask_area,
        has_interference=has_interference,
    )
