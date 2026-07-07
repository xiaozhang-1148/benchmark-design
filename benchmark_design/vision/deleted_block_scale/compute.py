"""Per-page Deleted-Block Scale computation."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.flow_structure.thresholds import is_deleted_text_block, is_txt_block
from benchmark_design.vision.deleted_block_scale.masks import AnswerDeletedMaskBundle
from benchmark_design.vision.deleted_block_scale.models import (
    BlockDeletedScaleGeometryRecord,
    PageDeletedBlockScaleResult,
)
from benchmark_design.vision.masks import UnifiedPageMaskBundle, build_unified_page_masks
from benchmark_design.vision.processing import _resolve_image_path


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _block_records_from_bundle(block_records: tuple) -> tuple[BlockDeletedScaleGeometryRecord, ...]:
    return tuple(
        BlockDeletedScaleGeometryRecord(
            page_id=record["page_id"],
            block_id=record["block_id"],
            block_order=record["block_order"],
            block_type=record["block_type"],
            polygon_area=record["polygon_area"],
            mask_area=record["mask_area"],
            bbox_x1=record["bbox_x1"],
            bbox_y1=record["bbox_y1"],
            bbox_x2=record["bbox_x2"],
            bbox_y2=record["bbox_y2"],
            is_valid_answer_block=record["is_valid_answer_block"],
            is_deleted_text_block=record["is_deleted_text_block"],
            mask_out_of_bounds=record["mask_out_of_bounds"],
            geometry_valid=record["geometry_valid"],
        )
        for record in block_records
    )


def compute_page_deleted_block_scale(
    page: PageAnnotation,
    *,
    input_dir: Path | None = None,
    unified_masks: UnifiedPageMaskBundle | None = None,
) -> PageDeletedBlockScaleResult:
    image_dir = input_dir or Path(page.source_file).parent
    review_path = str(image_dir / page.image_name)
    image_path = _resolve_image_path(page.image_name, image_dir)
    if image_path is None:
        image_path = image_dir / page.image_name

    num_txt = sum(1 for block in page.blocks if is_txt_block(block.block_type))
    num_deleted = sum(1 for block in page.blocks if is_deleted_text_block(block.block_type))

    reasons: list[str] = []
    if not image_path.is_file():
        _append_reason(reasons, "missing_image")

    masks: AnswerDeletedMaskBundle = (
        unified_masks.deleted_block_masks()
        if unified_masks is not None
        else build_unified_page_masks(
            page.blocks,
            image_width=page.image_width,
            image_height=page.image_height,
        ).deleted_block_masks()
    )
    block_records = _block_records_from_bundle(masks.block_records)

    for record in block_records:
        if not record.geometry_valid:
            _append_reason(reasons, "invalid_polygon")
        if record.mask_out_of_bounds:
            _append_reason(reasons, "polygon_out_of_bounds")

    valid_area = int(masks.valid_union.sum())
    deleted_area = int(masks.deleted_union.sum())
    answer_related_area = int(masks.answer_related_union.sum())

    r_del: float | None = None
    if answer_related_area == 0:
        _append_reason(reasons, "empty_answer_related_area")
    else:
        r_del = deleted_area / answer_related_area
        if deleted_area > answer_related_area:
            _append_reason(reasons, "deleted_area_exceeds_answer_related_area")
        if r_del < 0:
            _append_reason(reasons, "negative_deleted_area_ratio")
        elif r_del > 1:
            _append_reason(reasons, "deleted_area_ratio_above_one")

    if num_deleted > 0 and deleted_area == 0:
        _append_reason(reasons, "deleted_text_block_present_but_zero_area")

    return PageDeletedBlockScaleResult(
        page_id=page.page_id,
        image_name=page.image_name,
        image_width=page.image_width,
        image_height=page.image_height,
        num_txtBlock=num_txt,
        num_deleted_text_block=num_deleted,
        valid_area=valid_area,
        deleted_area=deleted_area,
        answer_related_area=answer_related_area,
        r_del=r_del,
        has_deleted_text_block=num_deleted > 0,
        needs_manual_review=bool(reasons),
        review_reason=";".join(reasons),
        review_image_path=review_path,
        block_records=block_records,
    )
