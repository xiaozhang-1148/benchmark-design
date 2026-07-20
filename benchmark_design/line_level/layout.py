"""Same-page nearby target-pair IoA and horizontal-adjacency analysis."""

from __future__ import annotations

from shapely.geometry import Polygon

from benchmark_design.line_level.models import TargetPairRow

DEFAULT_HEIGHT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_VERTICAL_OVERLAP_RATIO_THRESHOLD = 0.7
DEFAULT_HORIZONTAL_GAP_PX_THRESHOLD = 50.0


def axis_aligned_bbox(shape: Polygon) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) in original image pixel coordinates."""
    minx, miny, maxx, maxy = shape.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _ordered_line_ids(line_id_a: str, line_id_b: str) -> tuple[str, str]:
    if line_id_a <= line_id_b:
        return line_id_a, line_id_b
    return line_id_b, line_id_a


def _aabb_vertical_overlap_px(
    y_a_min: float,
    y_a_max: float,
    y_b_min: float,
    y_b_max: float,
) -> float:
    return max(0.0, min(y_a_max, y_b_max) - max(y_a_min, y_b_min))


def _aabb_horizontal_relation(
    x_a_min: float,
    x_a_max: float,
    x_b_min: float,
    x_b_max: float,
) -> tuple[float, bool, bool]:
    """Return (gap_px, horizontally_separated, x_ranges_overlap)."""
    if x_a_max <= x_b_min:
        return x_b_min - x_a_max, True, False
    if x_b_max <= x_a_min:
        return x_a_min - x_b_max, True, False
    return 0.0, False, True


def _pair_is_nearby_candidate(
    *,
    y_a_min: float,
    y_a_max: float,
    y_b_min: float,
    y_b_max: float,
    x_a_min: float,
    x_a_max: float,
    x_b_min: float,
    x_b_max: float,
    horizontal_gap_px_threshold: float,
) -> bool:
    """Keep only pairs that can yield IoA>0 or horizontal adjacency.

    - No shared vertical interval → skip (neither IoA nor adjacency).
    - Vertically overlapping but far in x (gap > threshold and no x-overlap)
      → skip (polygon intersection empty; adjacency impossible).
    """
    if _aabb_vertical_overlap_px(y_a_min, y_a_max, y_b_min, y_b_max) <= 0.0:
        return False
    gap, _separated, x_overlap = _aabb_horizontal_relation(x_a_min, x_a_max, x_b_min, x_b_max)
    if x_overlap:
        return True
    return gap <= horizontal_gap_px_threshold


def evaluate_target_pair(
    *,
    image_id: str,
    line_id_a: str,
    shape_a: Polygon,
    line_id_b: str,
    shape_b: Polygon,
    height_similarity_threshold: float = DEFAULT_HEIGHT_SIMILARITY_THRESHOLD,
    vertical_overlap_ratio_threshold: float = DEFAULT_VERTICAL_OVERLAP_RATIO_THRESHOLD,
    horizontal_gap_px_threshold: float = DEFAULT_HORIZONTAL_GAP_PX_THRESHOLD,
) -> TargetPairRow:
    """Evaluate one unordered same-page target pair."""
    id_a, id_b = _ordered_line_ids(line_id_a, line_id_b)
    if (id_a, id_b) != (line_id_a, line_id_b):
        shape_a, shape_b = shape_b, shape_a

    area_a = float(shape_a.area)
    area_b = float(shape_b.area)
    intersection = shape_a.intersection(shape_b)
    intersection_area = float(intersection.area)
    min_area = min(area_a, area_b)
    ioa = intersection_area / min_area if min_area > 0 else 0.0
    ioa_positive = intersection_area > 0.0

    x_a_min, y_a_min, x_a_max, y_a_max = axis_aligned_bbox(shape_a)
    x_b_min, y_b_min, x_b_max, y_b_max = axis_aligned_bbox(shape_b)
    h_a = y_a_max - y_a_min
    h_b = y_b_max - y_b_min
    height_similarity = min(h_a, h_b) / max(h_a, h_b) if max(h_a, h_b) > 0 else 0.0
    vertical_overlap_px = _aabb_vertical_overlap_px(y_a_min, y_a_max, y_b_min, y_b_max)
    vertical_overlap_ratio = vertical_overlap_px / min(h_a, h_b) if min(h_a, h_b) > 0 else 0.0

    horizontal_gap_px, horizontally_separated, _x_overlap = _aabb_horizontal_relation(
        x_a_min, x_a_max, x_b_min, x_b_max
    )

    horizontal_adjacent = (
        (not ioa_positive)
        and horizontally_separated
        and height_similarity >= height_similarity_threshold
        and vertical_overlap_ratio >= vertical_overlap_ratio_threshold
        and horizontal_gap_px <= horizontal_gap_px_threshold
    )

    return TargetPairRow(
        image_id=image_id,
        line_id_a=id_a,
        line_id_b=id_b,
        intersection_area=intersection_area,
        ioa=ioa,
        horizontal_gap_px=horizontal_gap_px,
        height_similarity=height_similarity,
        vertical_overlap_px=vertical_overlap_px,
        vertical_overlap_ratio=vertical_overlap_ratio,
        ioa_positive=ioa_positive,
        horizontal_adjacent=horizontal_adjacent,
    )


def compute_target_pairs(
    *,
    image_id: str,
    line_ids: list[str],
    shapes: list[Polygon],
    height_similarity_threshold: float = DEFAULT_HEIGHT_SIMILARITY_THRESHOLD,
    vertical_overlap_ratio_threshold: float = DEFAULT_VERTICAL_OVERLAP_RATIO_THRESHOLD,
    horizontal_gap_px_threshold: float = DEFAULT_HORIZONTAL_GAP_PX_THRESHOLD,
) -> list[TargetPairRow]:
    """Enumerate nearby unordered same-page pairs (not full C(n,2)).

    Only pairs with a shared vertical AABB interval are considered. Among those,
    pairs that are far apart horizontally (x-gap above the adjacency threshold
    and no x-overlap) are skipped — they cannot have polygon IoA>0 or be
    horizontally adjacent.
    """
    if len(shapes) != len(line_ids):
        raise ValueError("line_ids and shapes must have the same length")

    boxes = [axis_aligned_bbox(shape) for shape in shapes]
    order = sorted(range(len(shapes)), key=lambda index: (boxes[index][1], boxes[index][0], line_ids[index]))

    pairs: list[TargetPairRow] = []
    for position, i in enumerate(order):
        _x_i0, y_i0, _x_i1, y_i1 = boxes[i]
        for j in order[position + 1 :]:
            _x_j0, y_j0, _x_j1, y_j1 = boxes[j]
            # Sorted by y_min: once j starts at or below i's bottom, later j only higher.
            if y_j0 >= y_i1:
                break
            x_i0, y_i0b, x_i1, y_i1b = boxes[i]
            x_j0, y_j0b, x_j1, y_j1b = boxes[j]
            if not _pair_is_nearby_candidate(
                y_a_min=y_i0b,
                y_a_max=y_i1b,
                y_b_min=y_j0b,
                y_b_max=y_j1b,
                x_a_min=x_i0,
                x_a_max=x_i1,
                x_b_min=x_j0,
                x_b_max=x_j1,
                horizontal_gap_px_threshold=horizontal_gap_px_threshold,
            ):
                continue
            pairs.append(
                evaluate_target_pair(
                    image_id=image_id,
                    line_id_a=line_ids[i],
                    shape_a=shapes[i],
                    line_id_b=line_ids[j],
                    shape_b=shapes[j],
                    height_similarity_threshold=height_similarity_threshold,
                    vertical_overlap_ratio_threshold=vertical_overlap_ratio_threshold,
                    horizontal_gap_px_threshold=horizontal_gap_px_threshold,
                )
            )
    return pairs


def summarize_target_pairs(pairs: list[TargetPairRow]) -> dict[str, int]:
    line_ids: set[str] = set()
    page_ids: set[str] = set()
    for row in pairs:
        line_ids.add(row.line_id_a)
        line_ids.add(row.line_id_b)
        page_ids.add(row.image_id)
    return {
        "pair_count": len(pairs),
        "ioa_positive_pair_count": sum(1 for row in pairs if row.ioa_positive),
        "horizontal_adjacent_pair_count": sum(1 for row in pairs if row.horizontal_adjacent),
        "unique_line_count": len(line_ids),
        "unique_page_count": len(page_ids),
    }


def summarize_horizontal_adjacent_pairs(pairs: list[TargetPairRow]) -> dict[str, int]:
    """Summarize horizontal-adjacent pairs only (IoA=0 + linking rules)."""
    line_ids: set[str] = set()
    page_ids: set[str] = set()
    pair_count = 0
    for row in pairs:
        if not row.horizontal_adjacent:
            continue
        pair_count += 1
        line_ids.add(row.line_id_a)
        line_ids.add(row.line_id_b)
        page_ids.add(row.image_id)
    return {
        "pair_count": pair_count,
        "unique_line_count": len(line_ids),
        "unique_page_count": len(page_ids),
    }


def horizontal_adjacent_scope_rows(
    pairs: list[TargetPairRow],
    *,
    valid_line_count: int,
    page_count: int,
) -> list[dict[str, object]]:
    """Chapter Table 2 rows: adjacent-line counts, pages, and ratios."""
    summary = summarize_horizontal_adjacent_pairs(pairs)
    unique_line_count = int(summary["unique_line_count"])
    unique_page_count = int(summary["unique_page_count"])
    pair_count = int(summary["pair_count"])
    return [
        {
            "item": "水平相邻 line（去重）",
            "count": unique_line_count,
            "ratio": (unique_line_count / valid_line_count) if valid_line_count else 0.0,
        },
        {
            "item": "涉及页面",
            "count": unique_page_count,
            "ratio": (unique_page_count / page_count) if page_count else 0.0,
        },
        {
            "item": "水平相邻无序 pair",
            "count": pair_count,
            "ratio": (pair_count / valid_line_count) if valid_line_count else 0.0,
        },
    ]


def target_pair_scope_rows(
    pairs: list[TargetPairRow],
    *,
    valid_line_count: int,
    page_count: int,
) -> list[dict[str, object]]:
    summary = summarize_target_pairs(pairs)
    pair_count = int(summary["pair_count"])
    unique_line_count = int(summary["unique_line_count"])
    unique_page_count = int(summary["unique_page_count"])
    return [
        {
            "item": "满足条件的无序 pair",
            "count": pair_count,
            "ratio": (pair_count / valid_line_count) if valid_line_count else 0.0,
        },
        {
            "item": "涉及的唯一 line",
            "count": unique_line_count,
            "ratio": (unique_line_count / valid_line_count) if valid_line_count else 0.0,
        },
        {
            "item": "涉及页面",
            "count": unique_page_count,
            "ratio": (unique_page_count / page_count) if page_count else 0.0,
        },
    ]
