"""Single-flow existence detection (evaluated before column layout)."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure.geometry import core_x_gap_norm, y_range_overlap
from benchmark_design.block_level.flow_structure.metrics import sort_blocks_by_y, vertical_sequential_score
from benchmark_design.block_level.flow_structure.models import ColumnClusterResult, TxtBlockGeometry
from benchmark_design.block_level.flow_structure.thresholds import (
    ADJACENT_Y_OVERLAP_MAX_NORM,
    COLUMN_BAND_GAP_NORM,
    PARALLEL_BAND_X_SEPARATION_NORM,
    PARALLEL_BAND_Y_OVERLAP_NORM,
    VERTICAL_SEQUENTIAL_SCORE_MIN,
    VERTICAL_STACK_MIN_SEQUENTIAL_SCORE,
)


def forms_vertical_reading_stack(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_height: int,
) -> bool:
    """True when blocks follow a top-to-bottom path despite x offsets between parts."""
    if len(txt_blocks) <= 1:
        return len(txt_blocks) == 1
    score, max_overlap = vertical_sequential_score(txt_blocks, page_height=page_height)
    if score < VERTICAL_STACK_MIN_SEQUENTIAL_SCORE:
        return False
    if max_overlap >= ADJACENT_Y_OVERLAP_MAX_NORM:
        return False

    ordered = sort_blocks_by_y(txt_blocks)
    for upper, lower in zip(ordered, ordered[1:], strict=False):
        if lower.bbox_y1 + page_height * 0.02 < upper.bbox_y2:
            overlap = y_range_overlap(upper.bbox_y1, upper.bbox_y2, lower.bbox_y1, lower.bbox_y2)
            overlap_norm = overlap / page_height if page_height > 0 else 0.0
            if overlap_norm >= ADJACENT_Y_OVERLAP_MAX_NORM:
                return False
    return True


def _has_parallel_horizontal_bands(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_width: int,
    page_height: int,
) -> bool:
    if len(txt_blocks) < 2:
        return False
    for left in txt_blocks:
        for right in txt_blocks:
            if left.block_id >= right.block_id:
                continue
            y_overlap = y_range_overlap(left.bbox_y1, left.bbox_y2, right.bbox_y1, right.bbox_y2)
            y_overlap_norm = y_overlap / page_height if page_height > 0 else 0.0
            if y_overlap_norm < PARALLEL_BAND_Y_OVERLAP_NORM:
                continue
            x_sep_norm = max(
                core_x_gap_norm(left.core_x_interval, right.core_x_interval, page_width=page_width),
                0.0,
            )
            if x_sep_norm >= PARALLEL_BAND_X_SEPARATION_NORM:
                return True
    return False


def _has_clear_column_separation(
    cluster: ColumnClusterResult | None,
    *,
    txt_blocks: list[TxtBlockGeometry],
    page_height: int,
) -> bool:
    if forms_vertical_reading_stack(txt_blocks, page_height=page_height):
        return False
    if cluster is None:
        return False
    if cluster.column_layout_exists:
        return True
    return (
        cluster.max_column_gap_norm >= COLUMN_BAND_GAP_NORM
        and cluster.num_columns >= 2
    )


def _has_x_backtrack(txt_blocks: list[TxtBlockGeometry], *, page_width: int) -> bool:
    ordered = sort_blocks_by_y(txt_blocks)
    if len(ordered) < 2:
        return False
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_core = left.core_center_x / page_width if page_width > 0 else left.norm_center_x
        right_core = right.core_center_x / page_width if page_width > 0 else right.norm_center_x
        if right_core + 0.08 < left_core:
            return True
    return False


def single_flow_exists(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_width: int,
    page_height: int,
    column_cluster: ColumnClusterResult | None = None,
) -> bool:
    if len(txt_blocks) <= 1:
        return len(txt_blocks) == 1

    vertical_stack = forms_vertical_reading_stack(txt_blocks, page_height=page_height)
    score, max_overlap = vertical_sequential_score(txt_blocks, page_height=page_height)

    if vertical_stack:
        # Vertically separated sections may shift in x; backtrack is not a veto here.
        return max_overlap < ADJACENT_Y_OVERLAP_MAX_NORM * 2

    if score < VERTICAL_SEQUENTIAL_SCORE_MIN:
        return False
    if max_overlap >= ADJACENT_Y_OVERLAP_MAX_NORM * 2:
        return False
    if _has_parallel_horizontal_bands(txt_blocks, page_width=page_width, page_height=page_height):
        return False
    if _has_clear_column_separation(cluster=column_cluster, txt_blocks=txt_blocks, page_height=page_height):
        return False
    if _has_x_backtrack(txt_blocks, page_width=page_width):
        return False
    return True
