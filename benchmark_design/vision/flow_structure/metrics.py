"""Derived geometry metrics for flow structure classification."""

from __future__ import annotations

import statistics

from benchmark_design.vision.flow_structure.column_bands import best_column_cluster
from benchmark_design.vision.flow_structure.clustering import true_cross_column_bridge_ids
from benchmark_design.vision.flow_structure.geometry import (
    polygon_bbox,
    txt_block_geometry,
    y_range_overlap,
)
from benchmark_design.vision.flow_structure.models import (
    BlockGeometryRecord,
    FlowStructureMetrics,
    PageAnnotation,
    SkeletonType,
    TxtBlockGeometry,
)
from benchmark_design.vision.flow_structure.thresholds import (
    ADJACENT_Y_OVERLAP_MAX_NORM,
    INSERTED_BLOCK_X_OFFSET_NORM,
    INSERTED_BLOCK_Y_BACKTRACK_NORM,
    is_txt_block,
)


def extract_txt_blocks(page: PageAnnotation) -> list[TxtBlockGeometry]:
    blocks: list[TxtBlockGeometry] = []
    for block in page.blocks:
        if not is_txt_block(block.block_type):
            continue
        geometry = txt_block_geometry(
            block,
            image_width=page.image_width,
            image_height=page.image_height,
        )
        if geometry is not None:
            blocks.append(geometry)
    return blocks


def sort_blocks_by_y(txt_blocks: list[TxtBlockGeometry]) -> list[TxtBlockGeometry]:
    return sorted(txt_blocks, key=lambda block: (block.bbox_y1, block.bbox_x1))


def vertical_sequential_score(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_height: int,
) -> tuple[float, float]:
    ordered = sort_blocks_by_y(txt_blocks)
    if len(ordered) <= 1:
        return 1.0, 0.0
    non_overlap_count = 0
    max_overlap_norm = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        overlap = y_range_overlap(left.bbox_y1, left.bbox_y2, right.bbox_y1, right.bbox_y2)
        overlap_norm = overlap / page_height if page_height > 0 else 0.0
        max_overlap_norm = max(max_overlap_norm, overlap_norm)
        if overlap_norm < ADJACENT_Y_OVERLAP_MAX_NORM:
            non_overlap_count += 1
    score = non_overlap_count / (len(ordered) - 1)
    return score, max_overlap_norm


def x_center_span_norm(txt_blocks: list[TxtBlockGeometry]) -> float:
    if not txt_blocks:
        return 0.0
    centers = [block.norm_center_x for block in txt_blocks]
    return max(centers) - min(centers)


def x_center_std_norm(txt_blocks: list[TxtBlockGeometry]) -> float:
    if len(txt_blocks) <= 1:
        return 0.0
    return statistics.pstdev(block.norm_center_x for block in txt_blocks)


def _pair_is_inserted(
    previous: TxtBlockGeometry,
    current: TxtBlockGeometry,
    *,
    page_height: int,
) -> bool:
    y_backtrack = previous.bbox_y2 - current.bbox_y1
    if y_backtrack <= INSERTED_BLOCK_Y_BACKTRACK_NORM * page_height:
        return False
    x_offset = abs(current.norm_center_x - previous.norm_center_x)
    return x_offset > INSERTED_BLOCK_X_OFFSET_NORM


def has_inserted_answer_block(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_height: int,
    skeleton_type: SkeletonType | None,
    column_ids: list[int] | None,
) -> bool:
    if len(txt_blocks) <= 1:
        return False

    if skeleton_type == "columnar" and column_ids is not None:
        grouped: dict[int, list[TxtBlockGeometry]] = {}
        for block, column_id in zip(txt_blocks, column_ids, strict=True):
            grouped.setdefault(column_id, []).append(block)
        for blocks in grouped.values():
            ordered = sort_blocks_by_y(blocks)
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if _pair_is_inserted(previous, current, page_height=page_height):
                    return True
        return False

    ordered = sort_blocks_by_y(txt_blocks)
    id_to_column = {block.block_id: column_ids[index] for index, block in enumerate(txt_blocks)}
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if column_ids is not None:
            if id_to_column[previous.block_id] != id_to_column[current.block_id]:
                continue
        if _pair_is_inserted(previous, current, page_height=page_height):
            return True
    return False


def compute_flow_metrics(
    page: PageAnnotation,
    txt_blocks: list[TxtBlockGeometry],
    *,
    stable_column_layout: bool = False,
    skeleton_type: SkeletonType | None = None,
) -> FlowStructureMetrics:
    page_width = page.image_width
    page_height = page.image_height
    v_score, max_adj_overlap = vertical_sequential_score(txt_blocks, page_height=page_height)
    x_span = x_center_span_norm(txt_blocks)
    x_std = x_center_std_norm(txt_blocks)
    cluster = best_column_cluster(txt_blocks, page_width=page_width, page_height=page_height)
    column_ids = list(cluster.column_ids)
    bridge_ids = (
        true_cross_column_bridge_ids(
            txt_blocks,
            column_ids,
            page_width=page_width,
            inter_band_gaps=cluster.inter_band_gaps if cluster else None,
        )
        if stable_column_layout
        else set()
    )
    inserted = has_inserted_answer_block(
        txt_blocks,
        page_height=page_height,
        skeleton_type=skeleton_type or "unstable",
        column_ids=column_ids,
    )
    return FlowStructureMetrics(
        vertical_sequential_score=v_score,
        max_adjacent_y_overlap_norm=max_adj_overlap,
        x_center_span_norm=x_span,
        x_center_std_norm=x_std,
        num_detected_columns=cluster.num_columns,
        column_center_distance_norm=cluster.column_center_distance_norm,
        max_column_gap_norm=cluster.max_column_gap_norm,
        column_y_overlap_norm=cluster.column_y_overlap_norm,
        column_area_balance=cluster.column_area_balance,
        x_cluster_separation_norm=cluster.x_cluster_separation_norm,
        largest_column_area_ratio=cluster.largest_column_area_ratio,
        second_largest_column_area_ratio=cluster.second_largest_column_area_ratio,
        has_cross_column_block=bool(bridge_ids),
        true_cross_column_bridge=bool(bridge_ids),
        has_interrupting_chart_or_figure=False,
        has_inserted_block=inserted,
        inserted_answer_block=inserted,
        column_cluster=cluster,
    )


def build_block_records(
    page: PageAnnotation,
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_width: int,
    bridge_ids: set[str] | None = None,
) -> tuple[BlockGeometryRecord, ...]:
    ordered = sort_blocks_by_y(txt_blocks)
    sort_index_by_id = {block.block_id: index for index, block in enumerate(ordered)}
    column_by_id = {block.block_id: column_ids[index] for index, block in enumerate(txt_blocks)}
    if bridge_ids is None:
        bridge_ids = true_cross_column_bridge_ids(
            txt_blocks,
            column_ids,
            page_width=page_width,
        )
    polygon_by_id = {block.block_id: block.polygon for block in page.blocks}
    return tuple(
        BlockGeometryRecord(
            page_id=page.page_id,
            block_id=block.block_id,
            block_type="Txtblock",
            mask_area=block.mask_area,
            bbox_x1=block.bbox_x1,
            bbox_y1=block.bbox_y1,
            bbox_x2=block.bbox_x2,
            bbox_y2=block.bbox_y2,
            center_x=block.center_x,
            center_y=block.center_y,
            norm_center_x=block.norm_center_x,
            norm_center_y=block.norm_center_y,
            sort_index=sort_index_by_id[block.block_id],
            assigned_column_id=column_by_id.get(block.block_id),
            is_cross_column_block=block.block_id in bridge_ids,
            polygon=polygon_by_id.get(block.block_id, ()),
        )
        for block in txt_blocks
    )
