"""Data models for Deleted-Block Scale."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BlockDeletedScaleGeometryRecord:
    page_id: str
    block_id: str
    block_order: int
    block_type: str
    polygon_area: float
    mask_area: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    is_valid_answer_block: bool
    is_deleted_text_block: bool
    mask_out_of_bounds: bool
    geometry_valid: bool


@dataclass(frozen=True, slots=True)
class PageDeletedBlockScaleResult:
    page_id: str
    image_name: str
    image_width: int
    image_height: int
    num_txtBlock: int
    num_deleted_text_block: int
    valid_area: int
    deleted_area: int
    answer_related_area: int
    r_del: float | None
    has_deleted_text_block: bool
    needs_manual_review: bool
    review_reason: str
    review_image_path: str
    block_records: tuple[BlockDeletedScaleGeometryRecord, ...] = field(default_factory=tuple)
