"""Polygon and bbox geometry helpers."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure.models import PageBlockAnnotation, TxtBlockGeometry


def polygon_area(polygon: tuple[tuple[float, float], ...]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_core_x_interval(
    polygon: tuple[tuple[float, float], ...],
    *,
    lower_percentile: float = 0.15,
    upper_percentile: float = 0.85,
) -> tuple[float, float]:
    """Trim extreme x vertices so jagged mask wings do not widen column separation tests."""
    if not polygon:
        return 0.0, 0.0
    if len(polygon) < 4:
        x1, _, x2, _ = polygon_bbox(polygon)
        return x1, x2
    xs = sorted(point[0] for point in polygon)
    lower_index = min(max(int(len(xs) * lower_percentile), 0), len(xs) - 1)
    upper_index = min(max(int(len(xs) * upper_percentile), lower_index), len(xs) - 1)
    return xs[lower_index], xs[upper_index]


def core_x_gap_norm(
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    page_width: int,
) -> float:
    """Positive gap between core x intervals; negative when they overlap."""
    if page_width <= 0:
        return 0.0
    return (right[0] - left[1]) / page_width


def polygon_bbox(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    if not polygon:
        return 0.0, 0.0, 0.0, 0.0
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def normalize_polygon(
    polygon: tuple[tuple[float, float], ...],
    *,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float], ...]:
    if image_width <= 0 or image_height <= 0:
        return polygon
    return tuple((x / image_width, y / image_height) for x, y in polygon)


def txt_block_geometry(
    block: PageBlockAnnotation,
    *,
    image_width: int,
    image_height: int,
) -> TxtBlockGeometry | None:
    polygon = block.polygon
    if len(polygon) < 3:
        return None
    area = polygon_area(polygon)
    x1, y1, x2, y2 = polygon_bbox(polygon)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    block_width = max(x2 - x1, 0.0)
    block_height = max(y2 - y1, 0.0)
    page_width = float(image_width)
    page_height = float(image_height)
    core_x1, core_x2 = polygon_core_x_interval(polygon)
    core_center_x = (core_x1 + core_x2) / 2.0
    return TxtBlockGeometry(
        page_id=block.page_id,
        block_id=block.block_id,
        block_order=block.block_order,
        mask_area=area,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_x2=x2,
        bbox_y2=y2,
        width=block_width,
        height=block_height,
        center_x=center_x,
        center_y=center_y,
        x_interval=(x1, x2),
        y_interval=(y1, y2),
        core_x_interval=(core_x1, core_x2),
        core_center_x=core_center_x,
        norm_center_x=center_x / page_width,
        norm_center_y=center_y / page_height,
        norm_bbox_x1=x1 / page_width,
        norm_bbox_y1=y1 / page_height,
        norm_bbox_x2=x2 / page_width,
        norm_bbox_y2=y2 / page_height,
        norm_mask_area=area / (page_width * page_height),
    )


def y_range_overlap(y1_min: float, y1_max: float, y2_min: float, y2_max: float) -> float:
    return max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))


def bbox_y_overlap_norm(
    blocks_a: list[TxtBlockGeometry],
    blocks_b: list[TxtBlockGeometry],
    *,
    page_height: int,
) -> float:
    if not blocks_a or not blocks_b or page_height <= 0:
        return 0.0
    y_a_min = min(block.bbox_y1 for block in blocks_a)
    y_a_max = max(block.bbox_y2 for block in blocks_a)
    y_b_min = min(block.bbox_y1 for block in blocks_b)
    y_b_max = max(block.bbox_y2 for block in blocks_b)
    return y_range_overlap(y_a_min, y_a_max, y_b_min, y_b_max) / page_height
