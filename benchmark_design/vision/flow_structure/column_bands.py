"""Multi-signal column band detection and layout assessment."""

from __future__ import annotations

from benchmark_design.vision.flow_structure.geometry import bbox_y_overlap_norm, core_x_gap_norm
from benchmark_design.vision.flow_structure.models import ColumnClusterResult, TxtBlockGeometry
from benchmark_design.vision.flow_structure.thresholds import (
    BRIDGE_COLUMN_OVERLAP_RATIO,
    CLUSTER_STABILITY_MIN_AGREEMENT,
    COLUMN_BAND_CENTER_SEP_NORM,
    COLUMN_BAND_GAP_NORM,
    COLUMN_BAND_MIN_INNER_VERTICAL,
    COLUMN_CENTER_DISTANCE_NORM,
    COLUMN_GAP_NORM,
    COLUMN_Y_OVERLAP_NORM,
    CONFIDENCE_WEIGHT_CLUSTER_STABILITY,
    CONFIDENCE_WEIGHT_INNER_VERTICAL,
    CONFIDENCE_WEIGHT_MULTI_BLOCK_COLUMNS,
    CONFIDENCE_WEIGHT_SECOND_COLUMN_AREA,
    CONFIDENCE_WEIGHT_Y_OVERLAP,
    MIN_SECOND_COLUMN_AREA_RATIO,
    VERTICAL_SEQUENTIAL_SCORE_MIN,
)


def _column_area_stats(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
) -> tuple[float, float, float]:
    total_area = sum(block.mask_area for block in txt_blocks)
    if total_area <= 0:
        return 0.0, 0.0, 0.0
    by_column: dict[int, float] = {}
    for block, column_id in zip(txt_blocks, column_ids, strict=True):
        by_column[column_id] = by_column.get(column_id, 0.0) + block.mask_area
    ratios = sorted((area / total_area for area in by_column.values()), reverse=True)
    largest = ratios[0] if ratios else 0.0
    second = ratios[1] if len(ratios) > 1 else 0.0
    balance = (min(ratios) / max(ratios)) if ratios and max(ratios) > 0 else 0.0
    return largest, second, balance


def _column_x_ranges(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
) -> dict[int, tuple[float, float]]:
    ranges: dict[int, tuple[float, float]] = {}
    for block, column_id in zip(txt_blocks, column_ids, strict=True):
        core_left, core_right = block.core_x_interval
        left, right = ranges.get(column_id, (core_left, core_right))
        ranges[column_id] = (min(left, core_left), max(right, core_right))
    return ranges


def _inter_band_gaps(
    band_ranges: dict[int, tuple[float, float]],
    *,
    page_width: int,
) -> tuple[tuple[float, float], ...]:
    sorted_columns = sorted(band_ranges)
    gaps: list[tuple[float, float]] = []
    for left_id, right_id in zip(sorted_columns, sorted_columns[1:], strict=False):
        left_right = band_ranges[left_id][1]
        right_left = band_ranges[right_id][0]
        if right_left > left_right:
            gaps.append((left_right / page_width, right_left / page_width))
    return tuple(gaps)


def _x_interval_overlap_ratio(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return 0.0
    span = max(min(left[1] - left[0], right[1] - right[0]), 1.0)
    return overlap / span


def _block_spans_bands(
    block: TxtBlockGeometry,
    band_ranges: dict[int, tuple[float, float]],
    inter_gaps: tuple[tuple[float, float], ...],
    *,
    page_width: int,
) -> bool:
    if len(band_ranges) < 2:
        return False
    sorted_columns = sorted(band_ranges)
    overlapping = [
        column_id
        for column_id in sorted_columns
        if _x_interval_overlap_ratio(block.core_x_interval, band_ranges[column_id]) >= BRIDGE_COLUMN_OVERLAP_RATIO
    ]
    if len(overlapping) < 2:
        return False
    for gap_left_norm, gap_right_norm in inter_gaps:
        gap_left = gap_left_norm * page_width
        gap_right = gap_right_norm * page_width
        gap_mid = (gap_left + gap_right) / 2.0
        if block.core_x_interval[0] <= gap_left and block.core_x_interval[1] >= gap_right:
            return True
        if gap_left <= block.core_center_x <= gap_right:
            return True
        if abs(block.core_center_x - gap_mid) < max(block.core_x_interval[1] - block.core_x_interval[0], 1.0) * 0.15:
            return True
    return len(overlapping) >= 2 and all(
        _x_interval_overlap_ratio(block.core_x_interval, band_ranges[column_id]) >= BRIDGE_COLUMN_OVERLAP_RATIO
        for column_id in overlapping[:2]
    )


def _bands_have_local_reading_order(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_height: int,
) -> bool:
    from benchmark_design.vision.flow_structure.metrics import vertical_sequential_score

    grouped: dict[int, list[TxtBlockGeometry]] = {}
    for block, column_id in zip(txt_blocks, column_ids, strict=True):
        grouped.setdefault(column_id, []).append(block)
    for blocks in grouped.values():
        if len(blocks) <= 1:
            continue
        score, _ = vertical_sequential_score(blocks, page_height=page_height)
        if score < COLUMN_BAND_MIN_INNER_VERTICAL:
            return False
    return True


def _column_inner_vertical_scores(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_height: int,
) -> list[float]:
    from benchmark_design.vision.flow_structure.metrics import vertical_sequential_score

    grouped: dict[int, list[TxtBlockGeometry]] = {}
    for block, column_id in zip(txt_blocks, column_ids, strict=True):
        grouped.setdefault(column_id, []).append(block)
    return [
        vertical_sequential_score(blocks, page_height=page_height)[0]
        for blocks in grouped.values()
        if len(blocks) >= 2
    ]


def _layout_exists(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_width: int,
    page_height: int,
    max_gap_norm: float,
    center_distance_norm: float,
) -> bool:
    from benchmark_design.vision.flow_structure.single_flow import forms_vertical_reading_stack

    if forms_vertical_reading_stack(txt_blocks, page_height=page_height):
        return False

    num_columns = len(set(column_ids))
    if num_columns < 2:
        return False
    band_ranges = _column_x_ranges(txt_blocks, column_ids)
    inter_gaps = _inter_band_gaps(band_ranges, page_width=page_width)
    has_separation = (
        max_gap_norm >= COLUMN_BAND_GAP_NORM
        or center_distance_norm >= COLUMN_BAND_CENTER_SEP_NORM
        or bool(inter_gaps)
    )
    if not has_separation:
        return False
    if not _bands_have_local_reading_order(txt_blocks, column_ids, page_height=page_height):
        return False
    for block in txt_blocks:
        if _block_spans_bands(block, band_ranges, inter_gaps, page_width=page_width):
            return False
    return True


def _layout_confidence(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    metrics: ColumnClusterResult,
    *,
    page_height: int,
    is_stable: bool,
) -> float:
    score = 0.0
    if metrics.second_largest_column_area_ratio > MIN_SECOND_COLUMN_AREA_RATIO:
        score += CONFIDENCE_WEIGHT_SECOND_COLUMN_AREA
    if metrics.column_y_overlap_norm > COLUMN_Y_OVERLAP_NORM:
        score += CONFIDENCE_WEIGHT_Y_OVERLAP
    grouped: dict[int, list[TxtBlockGeometry]] = {}
    for block, column_id in zip(txt_blocks, column_ids, strict=True):
        grouped.setdefault(column_id, []).append(block)
    if all(len(blocks) >= 2 for blocks in grouped.values()) and len(grouped) >= 2:
        score += CONFIDENCE_WEIGHT_MULTI_BLOCK_COLUMNS
    inner_scores = _column_inner_vertical_scores(txt_blocks, column_ids, page_height=page_height)
    if inner_scores and min(inner_scores) >= VERTICAL_SEQUENTIAL_SCORE_MIN:
        score += CONFIDENCE_WEIGHT_INNER_VERTICAL
    elif inner_scores and min(inner_scores) >= COLUMN_BAND_MIN_INNER_VERTICAL:
        score += CONFIDENCE_WEIGHT_INNER_VERTICAL * 0.5
    if is_stable:
        score += CONFIDENCE_WEIGHT_CLUSTER_STABILITY
    if metrics.max_column_gap_norm >= COLUMN_GAP_NORM or metrics.column_center_distance_norm >= COLUMN_CENTER_DISTANCE_NORM:
        score = min(1.0, score + 0.05)
    return min(1.0, score)


def _normalize_column_ids(column_ids: list[int]) -> list[int]:
    unique = sorted(set(column_ids))
    mapping = {old: index for index, old in enumerate(unique)}
    return [mapping[column_id] for column_id in column_ids]


def _cluster_metrics(
    txt_blocks: list[TxtBlockGeometry],
    column_ids: list[int],
    *,
    page_width: int,
    page_height: int,
    is_stable: bool = True,
) -> ColumnClusterResult:
    column_ids = _normalize_column_ids(column_ids)
    num_columns = max(column_ids) + 1 if column_ids else 0
    grouped: dict[int, list[TxtBlockGeometry]] = {column_id: [] for column_id in range(num_columns)}
    for block, column_id in zip(txt_blocks, column_ids, strict=True):
        grouped[column_id].append(block)

    sorted_columns = sorted(grouped)
    max_gap_norm = 0.0
    max_y_overlap_norm = 0.0
    center_distance_norm = 0.0
    x_separation_norm = 0.0

    if len(sorted_columns) >= 2:
        left_col, right_col = sorted_columns[0], sorted_columns[-1]
        left_blocks = grouped[left_col]
        right_blocks = grouped[right_col]
        left_center = sum(block.center_x for block in left_blocks) / len(left_blocks)
        right_center = sum(block.center_x for block in right_blocks) / len(right_blocks)
        center_distance_norm = abs(left_center - right_center) / page_width

        left_right = max(block.core_x_interval[1] for block in left_blocks)
        right_left = min(block.core_x_interval[0] for block in right_blocks)
        max_gap_norm = max(0.0, (right_left - left_right) / page_width)
        x_separation_norm = max_gap_norm

        max_y_overlap_norm = bbox_y_overlap_norm(left_blocks, right_blocks, page_height=page_height)

        if len(sorted_columns) > 2:
            for left_id, right_id in zip(sorted_columns, sorted_columns[1:], strict=False):
                lb = grouped[left_id]
                rb = grouped[right_id]
                lr = max(block.core_x_interval[1] for block in lb)
                rl = min(block.core_x_interval[0] for block in rb)
                max_gap_norm = max(max_gap_norm, max(0.0, (rl - lr) / page_width))
                max_y_overlap_norm = max(
                    max_y_overlap_norm,
                    bbox_y_overlap_norm(lb, rb, page_height=page_height),
                )

    largest, second, balance = _column_area_stats(txt_blocks, column_ids)
    band_ranges = _column_x_ranges(txt_blocks, column_ids)
    inter_gaps = _inter_band_gaps(band_ranges, page_width=page_width)
    band_x_ranges = tuple(band_ranges[column_id] for column_id in sorted_columns)

    layout_exists = _layout_exists(
        txt_blocks,
        column_ids,
        page_width=page_width,
        page_height=page_height,
        max_gap_norm=max_gap_norm,
        center_distance_norm=center_distance_norm,
    )

    base = ColumnClusterResult(
        column_ids=tuple(column_ids),
        num_columns=num_columns,
        column_center_distance_norm=center_distance_norm,
        column_y_overlap_norm=max_y_overlap_norm,
        x_cluster_separation_norm=x_separation_norm,
        max_column_gap_norm=max_gap_norm,
        largest_column_area_ratio=largest,
        second_largest_column_area_ratio=second,
        column_area_balance=balance,
        is_stable=is_stable,
        band_x_ranges=band_x_ranges,
        inter_band_gaps=inter_gaps,
        column_layout_exists=layout_exists,
        column_layout_confidence=0.0,
    )
    confidence = _layout_confidence(
        txt_blocks,
        column_ids,
        base,
        page_height=page_height,
        is_stable=is_stable,
    )
    return ColumnClusterResult(
        column_ids=base.column_ids,
        num_columns=base.num_columns,
        column_center_distance_norm=base.column_center_distance_norm,
        column_y_overlap_norm=base.column_y_overlap_norm,
        x_cluster_separation_norm=base.x_cluster_separation_norm,
        max_column_gap_norm=base.max_column_gap_norm,
        largest_column_area_ratio=base.largest_column_area_ratio,
        second_largest_column_area_ratio=base.second_largest_column_area_ratio,
        column_area_balance=base.column_area_balance,
        is_stable=base.is_stable,
        band_x_ranges=base.band_x_ranges,
        inter_band_gaps=base.inter_band_gaps,
        column_layout_exists=layout_exists,
        column_layout_confidence=confidence,
    )


def _kmeans_clusters(centers: list[float], k: int, max_iterations: int = 20) -> list[int]:
    if len(centers) <= 1:
        return [0] * len(centers)
    k = min(k, len(centers))
    sorted_centers = sorted(centers)
    step = max(len(sorted_centers) - 1, 1) / max(k - 1, 1)
    centroids = [sorted_centers[int(round(index * step))] for index in range(k)]
    assignments = [0] * len(centers)
    for _ in range(max_iterations):
        changed = False
        for index, value in enumerate(centers):
            nearest = min(range(k), key=lambda cluster: abs(value - centroids[cluster]))
            if nearest != assignments[index]:
                changed = True
            assignments[index] = nearest
        new_centroids: list[float] = []
        for cluster in range(k):
            group = [centers[i] for i, cid in enumerate(assignments) if cid == cluster]
            new_centroids.append(sum(group) / len(group) if group else centroids[cluster])
        if new_centroids == centroids and not changed:
            break
        centroids = new_centroids
    return assignments


def _two_column_assignment(txt_blocks: list[TxtBlockGeometry], split_index: int, order: list[int]) -> list[int]:
    column_ids = [0] * len(txt_blocks)
    left_set = set(order[:split_index])
    for index in range(len(txt_blocks)):
        column_ids[index] = 0 if index in left_set else 1
    return column_ids


def _multi_column_assignment(
    txt_blocks: list[TxtBlockGeometry],
    split_points: list[int],
    order: list[int],
) -> list[int]:
    column_ids = [0] * len(txt_blocks)
    boundaries = [0, *split_points, len(order)]
    for column_id in range(len(boundaries) - 1):
        group = set(order[boundaries[column_id] : boundaries[column_id + 1]])
        for index in range(len(txt_blocks)):
            if index in group:
                column_ids[index] = column_id
    return column_ids


def _overlap_graph_assignment(txt_blocks: list[TxtBlockGeometry]) -> list[int]:
    n = len(txt_blocks)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(n):
        for right in range(left + 1, n):
            if _x_interval_overlap_ratio(
                txt_blocks[left].core_x_interval,
                txt_blocks[right].core_x_interval,
            ) >= 0.35:
                union(left, right)

    roots = [find(index) for index in range(n)]
    root_order = sorted(set(roots), key=lambda root: min(txt_blocks[i].center_x for i, r in enumerate(roots) if r == root))
    root_to_column = {root: index for index, root in enumerate(root_order)}
    return [root_to_column[root] for root in roots]


def _gap_split_assignment(txt_blocks: list[TxtBlockGeometry], *, page_width: int) -> list[int] | None:
    order = sorted(range(len(txt_blocks)), key=lambda index: txt_blocks[index].center_x)
    column_ids = [0] * len(txt_blocks)
    current_column = 0
    for split_at in range(1, len(order)):
        gap_norm = core_x_gap_norm(
            txt_blocks[order[split_at - 1]].core_x_interval,
            txt_blocks[order[split_at]].core_x_interval,
            page_width=page_width,
        )
        if gap_norm >= COLUMN_BAND_GAP_NORM:
            current_column += 1
        column_ids[order[split_at]] = current_column
    if len(set(column_ids)) >= 2:
        return column_ids
    return None


def _column_candidates(txt_blocks: list[TxtBlockGeometry], *, page_width: int) -> list[list[int]]:
    if len(txt_blocks) <= 1:
        return [[0] * len(txt_blocks)]

    centers = [block.center_x for block in txt_blocks]
    candidates: list[list[int]] = []
    max_k = min(len(txt_blocks), 4)

    for k in range(2, max_k + 1):
        candidates.append(_kmeans_clusters(centers, k))

    order = sorted(range(len(txt_blocks)), key=lambda index: txt_blocks[index].center_x)
    for split_at in range(1, len(order)):
        candidates.append(_two_column_assignment(txt_blocks, split_at, order))

    if len(txt_blocks) >= 3:
        for split_a in range(1, len(order)):
            for split_b in range(split_a + 1, len(order)):
                candidates.append(_multi_column_assignment(txt_blocks, [split_a, split_b], order))

    overlap_assignment = _overlap_graph_assignment(txt_blocks)
    if len(set(overlap_assignment)) >= 2:
        candidates.append(overlap_assignment)

    gap_assignment = _gap_split_assignment(txt_blocks, page_width=page_width)
    if gap_assignment is not None:
        candidates.append(gap_assignment)

    return candidates


def _assignment_agreement(left: list[int], right: list[int]) -> float:
    if not left:
        return 0.0
    direct = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    swapped = sum(1 for a, b in zip(left, right, strict=True) if a != b)
    return max(direct, swapped) / len(left)


def _candidate_score(cluster: ColumnClusterResult) -> tuple[float, ...]:
    return (
        1.0 if cluster.column_layout_exists else 0.0,
        cluster.column_layout_confidence,
        cluster.max_column_gap_norm,
        cluster.column_center_distance_norm,
        cluster.column_y_overlap_norm,
        cluster.second_largest_column_area_ratio,
        float(cluster.num_columns),
    )


def best_column_cluster(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_width: int,
    page_height: int,
) -> ColumnClusterResult:
    if len(txt_blocks) <= 1:
        return _cluster_metrics(txt_blocks, [0] * len(txt_blocks), page_width=page_width, page_height=page_height)

    candidates = _column_candidates(txt_blocks, page_width=page_width)
    evaluated = [
        _cluster_metrics(txt_blocks, column_ids, page_width=page_width, page_height=page_height)
        for column_ids in candidates
    ]

    existing = [cluster for cluster in evaluated if cluster.column_layout_exists]
    pool = existing if existing else evaluated
    best = max(pool, key=_candidate_score)

    agreements = [
        _assignment_agreement(list(candidate.column_ids), list(best.column_ids)) for candidate in evaluated
    ]
    is_stable = max(agreements) >= CLUSTER_STABILITY_MIN_AGREEMENT if agreements else True

    return _cluster_metrics(
        txt_blocks,
        list(best.column_ids),
        page_width=page_width,
        page_height=page_height,
        is_stable=is_stable,
    )


def best_two_column_cluster(
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_width: int,
    page_height: int,
) -> ColumnClusterResult:
    return best_column_cluster(txt_blocks, page_width=page_width, page_height=page_height)
