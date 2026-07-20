"""Process a single page image and all its lines (geometry + bbox ink metrics)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from benchmark_design.line_level.bbox_ink import (
    compute_bbox_outside_ink,
    load_normalized_ink_mask,
)
from benchmark_design.line_level.geometry import geometry_from_line, page_orientation
from benchmark_design.line_level.layout import compute_target_pairs, summarize_target_pairs
from benchmark_design.line_level.models import (
    InvalidAnnotationRow,
    LineAnnotation,
    LineLevelConfig,
    LineMetricsRow,
    PageMetricsRow,
    PageProcessResult,
    PageTask,
    ProcessingErrorRow,
)
from benchmark_design.line_level.validation import ValidatedLine, validate_page_lines
from benchmark_design.page_level.models import CalibrationResult


def _resolve_page_size(page: PageTask) -> tuple[int, int, ProcessingErrorRow | None]:
    width = int(page.width)
    height = int(page.height)
    if width > 0 and height > 0:
        return width, height, None
    if page.image_path is None or not Path(page.image_path).is_file():
        return 0, 0, ProcessingErrorRow(
            image_id=page.image_id,
            error_type="missing_image",
            error_message=f"Missing image for {page.image_name}",
        )
    try:
        with Image.open(page.image_path) as image:
            width, height = image.size
    except OSError as exc:
        return 0, 0, ProcessingErrorRow(
            image_id=page.image_id,
            error_type="image_read_error",
            error_message=str(exc),
        )
    return int(width), int(height), None


def _empty_page_height_stats() -> tuple[float, float, float]:
    return 0.0, 0.0, 0.0


def _page_bbox_height_stats(line_rows: list[LineMetricsRow]) -> tuple[float, float, float]:
    heights = [row.bbox_height_px for row in line_rows if row.is_valid and row.bbox_height_px > 0]
    if not heights:
        return _empty_page_height_stats()
    values = np.array(heights, dtype=np.float64)
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.05)),
        float(np.quantile(values, 0.95)),
    )


def _page_calibration(config: LineLevelConfig) -> CalibrationResult | None:
    if not config.bbox_outside_ink_enabled:
        return None
    return config.calibration


def _empty_line_row(
    line: LineAnnotation,
    *,
    reason: str,
    is_ignore: bool,
    page_orientation_value: str = "",
) -> LineMetricsRow:
    return LineMetricsRow(
        image_id=line.image_id,
        line_id=line.line_id,
        bbox_width_px=0.0,
        bbox_height_px=0.0,
        aspect_ratio=0.0,
        orientation_deg=0.0,
        orientation_direction_valid=False,
        is_ignore=is_ignore,
        is_valid=False,
        invalid_reason=reason,
        page_orientation=page_orientation_value,
        block_type=line.block_type,
    )


def process_page(page: PageTask, config: LineLevelConfig) -> PageProcessResult:
    started = time.perf_counter()
    page_width, page_height, size_error = _resolve_page_size(page)
    orientation = page_orientation(page_width, page_height)

    if size_error is not None or page_width <= 0 or page_height <= 0:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        error = size_error or ProcessingErrorRow(
            image_id=page.image_id,
            error_type="invalid_page_size",
            error_message=f"Invalid page size for {page.image_name}: {page_width}x{page_height}",
        )
        return PageProcessResult(
            page_metrics=PageMetricsRow(
                image_id=page.image_id,
                width=page_width,
                height=page_height,
                total_pixels=max(page_width, 0) * max(page_height, 0),
                dpi=page.dpi,
                line_count=len(page.lines),
                valid_line_count=0,
                ignore_line_count=sum(1 for line in page.lines if line.is_ignore),
                ioa_positive_pair_count=0,
                horizontal_adjacent_pair_count=0,
                page_orientation=orientation,
                median_bbox_height_px=0.0,
                p05_bbox_height_px=0.0,
                p95_bbox_height_px=0.0,
                processing_time_ms=elapsed_ms,
                status="error",
                error_message=error.error_message,
            ),
            line_metrics=(),
            invalid_rows=tuple(
                InvalidAnnotationRow(
                    image_id=line.image_id,
                    line_id=line.line_id,
                    block_type=line.block_type,
                    reason=error.error_type,
                    polygon_point_count=len(line.polygon),
                )
                for line in page.lines
            ),
            error=error,
        )

    valid_entries, invalid_rows = validate_page_lines(
        page.lines,
        image_width=page_width,
        image_height=page_height,
    )
    valid_ids = {entry.line.line_id for entry in valid_entries}

    line_metric_rows: list[LineMetricsRow] = []
    for line in page.lines:
        if line.line_id in valid_ids:
            continue
        if line.is_ignore:
            line_metric_rows.append(
                _empty_line_row(line, reason="ignored_or_invalid", is_ignore=True, page_orientation_value=orientation)
            )
        else:
            line_metric_rows.append(
                _empty_line_row(line, reason="invalid_polygon", is_ignore=False, page_orientation_value=orientation)
            )

    prepared: list[tuple[ValidatedLine, dict[str, float]]] = []
    pair_line_ids: list[str] = []
    pair_shapes: list[Polygon] = []

    for entry in valid_entries:
        line = entry.line
        shape = entry.shape
        geom = geometry_from_line(
            line,
            shape,
            page_width=page_width,
            page_height=page_height,
        )
        prepared.append((entry, geom))
        # Pair stats: valid, non-ignore, positive-area polygons only.
        if (not line.is_ignore) and float(shape.area) > 0.0:
            pair_line_ids.append(line.line_id)
            pair_shapes.append(shape)

    pair_rows = (
        compute_target_pairs(
            image_id=page.image_id,
            line_ids=pair_line_ids,
            shapes=pair_shapes,
            height_similarity_threshold=config.height_similarity_threshold,
            vertical_overlap_ratio_threshold=config.vertical_overlap_ratio_threshold,
            horizontal_gap_px_threshold=config.horizontal_gap_px_threshold,
        )
        if len(pair_shapes) > 1
        else []
    )
    pair_summary = summarize_target_pairs(pair_rows)

    ink_mask: np.ndarray | None = None
    calibration = _page_calibration(config)
    if calibration is not None and page.image_path is not None and Path(page.image_path).is_file():
        try:
            ink_mask = load_normalized_ink_mask(page.image_path, calibration)
        except OSError:
            ink_mask = None

    for entry, geom in prepared:
        line = entry.line
        # Orientation is always assigned (θ from mask-contour OBB long side).
        orientation_direction_valid = True
        ink_ratio = None
        outside_pixels = 0
        outside_ink = 0
        bbox_pixels = 0
        mask_area = 0
        has_interference = False
        if ink_mask is not None:
            ink_stats = compute_bbox_outside_ink(ink_mask, entry.shape)
            ink_ratio = ink_stats.interference_ratio
            outside_pixels = ink_stats.outside_area
            outside_ink = ink_stats.interference_pixels
            bbox_pixels = ink_stats.bbox_area
            mask_area = ink_stats.mask_area
            has_interference = ink_stats.has_interference
        line_metric_rows.append(
            LineMetricsRow(
                image_id=line.image_id,
                line_id=line.line_id,
                bbox_width_px=geom["bbox_width_px"],
                bbox_height_px=geom["bbox_height_px"],
                aspect_ratio=geom["aspect_ratio"],
                orientation_deg=geom["orientation_deg"],
                orientation_direction_valid=orientation_direction_valid,
                is_ignore=line.is_ignore,
                is_valid=not line.is_ignore,
                invalid_reason="",
                page_orientation=orientation,
                block_type=line.block_type,
                bbox_outside_ink_ratio=ink_ratio,
                bbox_outside_pixel_count=outside_pixels,
                bbox_outside_ink_count=outside_ink,
                bbox_pixel_count=bbox_pixels,
                mask_area=mask_area,
                has_interference=has_interference,
                obb_long_side_px=geom["obb_long_side_px"],
                obb_short_side_px=geom["obb_short_side_px"],
            )
        )

    line_metric_rows.sort(key=lambda row: row.line_id)
    valid_count = sum(1 for row in line_metric_rows if row.is_valid)
    ignore_count = sum(1 for row in line_metric_rows if row.is_ignore)
    median_height, p05_height, p95_height = _page_bbox_height_stats(line_metric_rows)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    total_pixels = page_width * page_height

    return PageProcessResult(
        page_metrics=PageMetricsRow(
            image_id=page.image_id,
            width=page_width,
            height=page_height,
            total_pixels=total_pixels,
            dpi=page.dpi,
            line_count=len(page.lines),
            valid_line_count=valid_count,
            ignore_line_count=ignore_count,
            ioa_positive_pair_count=pair_summary["ioa_positive_pair_count"],
            horizontal_adjacent_pair_count=pair_summary["horizontal_adjacent_pair_count"],
            page_orientation=orientation,
            median_bbox_height_px=median_height,
            p05_bbox_height_px=p05_height,
            p95_bbox_height_px=p95_height,
            processing_time_ms=elapsed_ms,
            status="ok",
        ),
        line_metrics=tuple(line_metric_rows),
        invalid_rows=tuple(invalid_rows),
        pair_rows=tuple(pair_rows),
    )
