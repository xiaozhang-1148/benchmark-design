"""center_x clustering for column detection (re-exports column_bands)."""

from __future__ import annotations

from benchmark_design.vision.flow_structure.column_bands import (
    _column_x_ranges,
    best_column_cluster,
    best_two_column_cluster,
)
from benchmark_design.vision.flow_structure.models import TxtBlockGeometry
from benchmark_design.vision.flow_structure.thresholds import BRIDGE_COLUMN_OVERLAP_RATIO


def _block_overlaps_column(block: TxtBlockGeometry, x_range: tuple[float, float]) -> float:
    span = max(block.bbox_x2 - block.bbox_x1, 1.0)
    overlap = min(block.bbox_x2, x_range[1]) - max(block.bbox_x1, x_range[0])
    return max(0.0, overlap) / span


def _block_center_column(block: TxtBlockGeometry, column_ranges: dict[int, tuple[float, float]]) -> int | None:
    for column_id, x_range in column_ranges.items():
        if x_range[0] <= block.center_x <= x_range[1]:
            return column_id
    return None


def true_cross_column_bridge_ids(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_width: int,
    inter_band_gaps: tuple[tuple[float, float], ...] | None = None,
) -> set[str]:
    if len(set(column_ids)) <= 1:
        return set()

    column_ranges = _column_x_ranges(txt_blocks, column_ids)
    sorted_columns = sorted(column_ranges)
    bridge_ids: set[str] = set()

    gaps = inter_band_gaps or ()
    if not gaps and len(sorted_columns) >= 2:
        for left_id, right_id in zip(sorted_columns, sorted_columns[1:], strict=False):
            left_right = column_ranges[left_id][1]
            right_left = column_ranges[right_id][0]
            if right_left > left_right:
                gaps = (*gaps, (left_right / page_width, right_left / page_width))

    for block, assigned in zip(txt_blocks, column_ids, strict=True):
        span = max(block.bbox_x2 - block.bbox_x1, 1.0)
        center_col = _block_center_column(block, column_ranges)
        overlapping_columns = [
            column_id
            for column_id in sorted_columns
            if _block_overlaps_column(block, column_ranges[column_id]) >= BRIDGE_COLUMN_OVERLAP_RATIO
        ]

        if len(overlapping_columns) < 2:
            continue

        spans_gap = False
        for gap_left_norm, gap_right_norm in gaps:
            gap_left = gap_left_norm * page_width
            gap_right = gap_right_norm * page_width
            if block.bbox_x1 <= gap_left and block.bbox_x2 >= gap_right:
                spans_gap = True
                break
            if gap_left <= block.center_x <= gap_right:
                spans_gap = True
                break

        if not spans_gap and center_col is not None and center_col == assigned:
            left_overlap = _block_overlaps_column(block, column_ranges[sorted_columns[0]])
            right_overlap = _block_overlaps_column(block, column_ranges[sorted_columns[-1]])
            if len(sorted_columns) == 2 and left_overlap >= BRIDGE_COLUMN_OVERLAP_RATIO and right_overlap >= BRIDGE_COLUMN_OVERLAP_RATIO:
                gutter_left = column_ranges[sorted_columns[0]][1]
                gutter_right = column_ranges[sorted_columns[1]][0]
                gutter_mid = (gutter_left + gutter_right) / 2.0
                if gutter_left <= block.center_x <= gutter_right or abs(block.center_x - gutter_mid) < span * 0.15:
                    bridge_ids.add(block.block_id)
            continue

        overlaps = [
            _block_overlaps_column(block, column_ranges[column_id]) for column_id in overlapping_columns[:2]
        ]
        if spans_gap or all(ratio >= BRIDGE_COLUMN_OVERLAP_RATIO for ratio in overlaps):
            bridge_ids.add(block.block_id)

    return bridge_ids


def cross_column_block_ids(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_width: int,
) -> set[str]:
    return true_cross_column_bridge_ids(txt_blocks, column_ids, page_width=page_width)


def has_cross_column_block(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_width: int,
) -> bool:
    return bool(true_cross_column_bridge_ids(txt_blocks, column_ids, page_width=page_width))
