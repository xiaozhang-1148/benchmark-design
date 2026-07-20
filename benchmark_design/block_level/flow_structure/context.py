"""Context block impact assessment."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure.geometry import polygon_area, polygon_bbox, y_range_overlap
from benchmark_design.block_level.flow_structure.metrics import sort_blocks_by_y
from benchmark_design.block_level.flow_structure.models import (
    ContextAssessment,
    ContextStatus,
    PageBlockAnnotation,
    SkeletonAssessment,
    TxtBlockGeometry,
)
from benchmark_design.block_level.flow_structure.thresholds import CONTEXT_MIN_AREA_RATIO, INTERRUPT_OVERLAP_RATIO


def _txt_y_span(txt_blocks: list[TxtBlockGeometry]) -> tuple[float, float, float]:
    if not txt_blocks:
        return 0.0, 0.0, 1.0
    y_min = min(block.bbox_y1 for block in txt_blocks)
    y_max = max(block.bbox_y2 for block in txt_blocks)
    return y_min, y_max, max(y_max - y_min, 1.0)


def _context_is_meaningful(
    block: PageBlockAnnotation,
    *,
    txt_blocks: list[TxtBlockGeometry],
    page_area: float,
    page_width: int,
) -> bool:
    if len(block.polygon) < 3:
        return False
    area = polygon_area(block.polygon)
    if page_area > 0 and area / page_area < CONTEXT_MIN_AREA_RATIO:
        return False
    _, y1, _, y2 = polygon_bbox(block.polygon)
    x1, _, x2, _ = polygon_bbox(block.polygon)
    txt_y_min = min(block.bbox_y1 for block in txt_blocks)
    txt_y_max = max(block.bbox_y2 for block in txt_blocks)
    txt_x_min = min(block.bbox_x1 for block in txt_blocks)
    txt_x_max = max(block.bbox_x2 for block in txt_blocks)
    txt_span = max(txt_y_max - txt_y_min, 1.0)
    y_overlap = y_range_overlap(y1, y2, txt_y_min, txt_y_max) / txt_span
    if y_overlap >= INTERRUPT_OVERLAP_RATIO * 0.5:
        return True
    x_overlap = max(0.0, min(x2, txt_x_max) - max(x1, txt_x_min))
    x_span = max(txt_x_max - txt_x_min, 1.0)
    if x_overlap / x_span >= 0.05:
        return True
    return area / page_area >= CONTEXT_MIN_AREA_RATIO * 2


def _context_interrupts_adjacent_txt_pair(
    block: PageBlockAnnotation,
    *,
    left: TxtBlockGeometry,
    right: TxtBlockGeometry,
) -> tuple[bool, str]:
    if len(block.polygon) < 3:
        return False, ""
    _, y1, _, y2 = polygon_bbox(block.polygon)
    x1, _, x2, _ = polygon_bbox(block.polygon)

    gap_top = left.bbox_y2
    gap_bottom = right.bbox_y1
    if gap_bottom <= gap_top:
        return False, ""

    block_center_y = (y1 + y2) / 2.0
    gap_height = gap_bottom - gap_top
    if gap_top <= block_center_y <= gap_bottom:
        txt_x_min = min(left.bbox_x1, right.bbox_x1)
        txt_x_max = max(left.bbox_x2, right.bbox_x2)
        if x1 <= txt_x_max and x2 >= txt_x_min:
            return True, "context_in_txt_gap"

    if gap_height > 0 and gap_top <= y2 and y1 <= gap_bottom:
        gap_overlap = y_range_overlap(y1, y2, gap_top, gap_bottom)
        if gap_overlap / max(gap_height, 1.0) >= 0.30:
            txt_x_min = min(left.bbox_x1, right.bbox_x1)
            txt_x_max = max(left.bbox_x2, right.bbox_x2)
            if x1 <= txt_x_max and x2 >= txt_x_min:
                return True, "context_spans_txt_gap"

    return False, ""


def _context_interrupts_txt_path(
    block: PageBlockAnnotation,
    *,
    txt_blocks: list[TxtBlockGeometry],
) -> tuple[bool, str]:
    if len(block.polygon) < 3 or len(txt_blocks) < 2:
        return False, ""

    ordered = sort_blocks_by_y(txt_blocks)
    for left, right in zip(ordered, ordered[1:], strict=False):
        interrupts, reason = _context_interrupts_adjacent_txt_pair(block, left=left, right=right)
        if interrupts:
            return True, reason
    return False, ""


def assess_page_context(
    context_blocks: tuple[PageBlockAnnotation, ...],
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_width: int,
    page_height: int,
    skeleton: SkeletonAssessment,
) -> ContextAssessment:
    if not context_blocks or not txt_blocks:
        return ContextAssessment(context_status="no_context", context_impact_reason="")

    page_area = max(page_width * page_height, 1)
    meaningful_blocks: list[PageBlockAnnotation] = []
    for block in context_blocks:
        if _context_is_meaningful(
            block,
            txt_blocks=txt_blocks,
            page_area=page_area,
            page_width=page_width,
        ):
            meaningful_blocks.append(block)

    if not meaningful_blocks:
        return ContextAssessment(context_status="no_context", context_impact_reason="")

    interrupt_reasons: list[str] = []
    for block in meaningful_blocks:
        interrupts, reason = _context_interrupts_txt_path(block, txt_blocks=txt_blocks)
        if interrupts:
            interrupt_reasons.append(reason)

    if interrupt_reasons:
        return ContextAssessment(
            context_status="context_interrupted",
            context_impact_reason=interrupt_reasons[0],
            diagnostic_tags=("context_interrupted",),
        )

    preserved_tag = "context_preserved"
    if skeleton.column_layout_exists:
        preserved_tag = "context_preserved_columnar"
    return ContextAssessment(
        context_status="context_preserved",
        context_impact_reason="side_or_outer_context",
        diagnostic_tags=(preserved_tag,),
    )
