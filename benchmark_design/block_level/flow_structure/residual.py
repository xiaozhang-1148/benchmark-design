"""Residual txtBlock detection for flow-structure recovery."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure.geometry import y_range_overlap
from benchmark_design.block_level.flow_structure.metrics import (
    _pair_is_inserted,
    sort_blocks_by_y,
    vertical_sequential_score,
)
from benchmark_design.block_level.flow_structure.models import FlowStructureMetrics, SkeletonAssessment, TxtBlockGeometry
from benchmark_design.block_level.flow_structure.skeleton import assess_skeleton
from benchmark_design.block_level.flow_structure.thresholds import (
    ADJACENT_Y_OVERLAP_MAX_NORM,
    INSERTED_BLOCK_X_OFFSET_NORM,
    VERTICAL_SEQUENTIAL_SCORE_MIN,
)


def _heavy_overlap_pair(
    left: TxtBlockGeometry,
    right: TxtBlockGeometry,
    *,
    page_height: int,
) -> bool:
    overlap = y_range_overlap(left.bbox_y1, left.bbox_y2, right.bbox_y1, right.bbox_y2)
    overlap_norm = overlap / page_height if page_height > 0 else 0.0
    return overlap_norm >= ADJACENT_Y_OVERLAP_MAX_NORM


def _block_fits_any_column(
    block: TxtBlockGeometry,
    metrics: FlowStructureMetrics,
) -> bool:
    cluster = metrics.column_cluster
    if cluster is None or not cluster.band_x_ranges:
        return False
    for x_range in cluster.band_x_ranges:
        overlap = min(block.core_x_interval[1], x_range[1]) - max(block.core_x_interval[0], x_range[0])
        span = max(block.core_x_interval[1] - block.core_x_interval[0], 1.0)
        if overlap / span >= 0.5 and x_range[0] <= block.core_center_x <= x_range[1]:
            return True
    return False


def detect_residual_block_ids(
    txt_blocks: list[TxtBlockGeometry],
    metrics: FlowStructureMetrics,
    *,
    page_height: int,
    column_layout_exists: bool,
    single_flow_exists: bool,
) -> set[str]:
    """Identify txtBlocks that prevent a stable single or column skeleton."""
    if len(txt_blocks) <= 2:
        return set()

    residual_ids: set[str] = set()

    if column_layout_exists and metrics.column_cluster is not None:
        cluster = metrics.column_cluster
        grouped: dict[int, list[TxtBlockGeometry]] = {}
        for block, column_id in zip(txt_blocks, list(cluster.column_ids), strict=True):
            grouped.setdefault(column_id, []).append(block)
        for blocks in grouped.values():
            ordered = sort_blocks_by_y(blocks)
            for left, right in zip(ordered, ordered[1:], strict=False):
                if _heavy_overlap_pair(left, right, page_height=page_height):
                    residual_ids.add(right.block_id)
        for block in txt_blocks:
            if not _block_fits_any_column(block, metrics):
                residual_ids.add(block.block_id)
        return residual_ids

    if not single_flow_exists:
        ordered = sort_blocks_by_y(txt_blocks)
        for left, right in zip(ordered, ordered[1:], strict=False):
            if _heavy_overlap_pair(left, right, page_height=page_height):
                residual_ids.add(right.block_id)

        score, _ = vertical_sequential_score(txt_blocks, page_height=page_height)
        if score < VERTICAL_SEQUENTIAL_SCORE_MIN:
            for block in txt_blocks:
                remaining = [item for item in txt_blocks if item.block_id != block.block_id]
                if len(remaining) < 2:
                    continue
                reduced_score, _ = vertical_sequential_score(remaining, page_height=page_height)
                if reduced_score >= VERTICAL_SEQUENTIAL_SCORE_MIN:
                    residual_ids.add(block.block_id)

    return residual_ids


def all_txtblocks_fit_column_flow(
    txt_blocks: list[TxtBlockGeometry],
    metrics: FlowStructureMetrics,
    *,
    page_height: int,
) -> bool:
    if metrics.num_detected_columns < 2 or metrics.column_cluster is None:
        return False
    if not metrics.column_cluster.column_layout_exists:
        return False
    residuals = detect_residual_block_ids(
        txt_blocks,
        metrics,
        page_height=page_height,
        column_layout_exists=True,
        single_flow_exists=False,
    )
    return not residuals


def recover_skeleton_after_residual_removal(
    txt_blocks: list[TxtBlockGeometry],
    residual_ids: set[str],
    *,
    page_width: int,
    page_height: int,
) -> SkeletonAssessment | None:
    from benchmark_design.block_level.flow_structure.metrics import compute_flow_metrics
    from benchmark_design.block_level.flow_structure.models import PageAnnotation

    remaining = [block for block in txt_blocks if block.block_id not in residual_ids]
    if len(remaining) < 2:
        return None

    stub_page = PageAnnotation(
        page_id="residual-recovery",
        image_name="",
        source_file="",
        image_width=page_width,
        image_height=page_height,
        blocks=(),
    )
    metrics = compute_flow_metrics(stub_page, remaining)
    skeleton = assess_skeleton(remaining, metrics, page_height=page_height, page_width=page_width)
    if skeleton.column_layout_exists or skeleton.single_flow_exists:
        return skeleton
    return None


def _block_in_y_gap(
    block: TxtBlockGeometry,
    *,
    gap_top: float,
    gap_bottom: float,
) -> bool:
    return gap_top <= block.center_y <= gap_bottom or (block.bbox_y1 <= gap_bottom and block.bbox_y2 >= gap_top)


def _block_in_x_side_wing(
    block: TxtBlockGeometry,
    main_blocks: list[TxtBlockGeometry],
    *,
    page_width: int,
) -> bool:
    if not main_blocks:
        return False
    main_x_min = min(item.bbox_x1 for item in main_blocks)
    main_x_max = max(item.bbox_x2 for item in main_blocks)
    if block.bbox_x2 < main_x_min or block.bbox_x1 > main_x_max:
        return True
    x_offset = min(
        abs(block.norm_center_x - item.norm_center_x) for item in main_blocks
    )
    return x_offset > INSERTED_BLOCK_X_OFFSET_NORM


def residual_blocks_satisfy_inserted(
    txt_blocks: list[TxtBlockGeometry],
    residual_ids: set[str],
    *,
    metrics: FlowStructureMetrics,
    page_height: int,
    page_width: int,
) -> bool:
    if not residual_ids:
        return False

    main_blocks = [block for block in txt_blocks if block.block_id not in residual_ids]
    if len(main_blocks) < 2:
        return False

    ordered = sort_blocks_by_y(txt_blocks)
    touched = set(residual_ids)

    for block_id in residual_ids:
        block = next(item for item in txt_blocks if item.block_id == block_id)
        if _block_fits_any_column(block, metrics):
            continue
        in_gap = False
        for left, right in zip(ordered, ordered[1:], strict=False):
            if left.block_id in touched or right.block_id in touched:
                continue
            if left.block_id in residual_ids or right.block_id in residual_ids:
                gap_top = left.bbox_y2 if left.block_id not in residual_ids else block.bbox_y2
                gap_bottom = right.bbox_y1 if right.block_id not in residual_ids else block.bbox_y1
                if _block_in_y_gap(block, gap_top=gap_top, gap_bottom=gap_bottom):
                    in_gap = True
                    break
        side_wing = _block_in_x_side_wing(block, main_blocks, page_width=page_width)
        if not in_gap and not side_wing:
            return False

    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.block_id not in touched and right.block_id not in touched:
            continue
        if _pair_is_inserted(left, right, page_height=page_height):
            return True

    return len(residual_ids) >= 1 and any(
        not _block_fits_any_column(next(item for item in txt_blocks if item.block_id == block_id), metrics)
        for block_id in residual_ids
    )
