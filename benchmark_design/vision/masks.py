"""Unified per-page mask rasterization for vision metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark_design.vision.flow_structure.geometry import polygon_area, polygon_bbox
from benchmark_design.vision.flow_structure.models import PageBlockAnnotation
from benchmark_design.vision.flow_structure.block_roles import is_figure_or_chart
from benchmark_design.vision.flow_structure.thresholds import is_deleted_text_block, is_txt_block
from benchmark_design.vision.foreground_load.thresholds import is_effective_block


def polygon_out_of_bounds(
    polygon: tuple[tuple[float, float], ...],
    *,
    image_width: int,
    image_height: int,
) -> bool:
    for x, y in polygon:
        if x < 0 or y < 0 or x >= image_width or y >= image_height:
            return True
    return False


def _polygon_points(polygon: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)


def _fill_polygon(mask: np.ndarray, polygon: tuple[tuple[float, float], ...], value: int) -> None:
    if len(polygon) < 3:
        return
    try:
        import cv2
    except ImportError:
        _fill_polygon_pil(mask, polygon, value=value)
        return
    cv2.fillPoly(mask, [_polygon_points(polygon)], value)


def _fill_polygon_pil(mask: np.ndarray, polygon: tuple[tuple[float, float], ...], *, value: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for foreground load metrics. Install with: pip install Pillow"
        ) from exc
    layer = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    draw = ImageDraw.Draw(layer)
    draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    filled = np.array(layer, dtype=np.uint8) > 0
    mask[filled] = value


@dataclass(frozen=True, slots=True)
class PageMaskBundle:
    effective_union: np.ndarray
    txt_block_masks: dict[str, np.ndarray]
    num_effective_blocks: int
    out_of_bounds: bool


@dataclass(frozen=True, slots=True)
class AnswerDeletedMaskBundle:
    valid_union: np.ndarray
    deleted_union: np.ndarray
    answer_related_union: np.ndarray
    block_records: tuple[dict[str, object], ...]
    out_of_bounds: bool

    @property
    def answer_visible_union(self) -> np.ndarray:
        return self.answer_related_union


@dataclass(frozen=True, slots=True)
class UnifiedPageMaskBundle:
    effective_union: np.ndarray
    txt_block_masks: dict[str, np.ndarray]
    num_effective_blocks: int
    out_of_bounds: bool
    valid_union: np.ndarray
    deleted_union: np.ndarray
    answer_related_union: np.ndarray
    dbs_block_records: tuple[dict[str, object], ...]

    @property
    def answer_visible_union(self) -> np.ndarray:
        return self.answer_related_union

    def foreground_masks(self) -> PageMaskBundle:
        return PageMaskBundle(
            effective_union=self.effective_union,
            txt_block_masks=self.txt_block_masks,
            num_effective_blocks=self.num_effective_blocks,
            out_of_bounds=self.out_of_bounds,
        )

    def deleted_block_masks(self) -> AnswerDeletedMaskBundle:
        return AnswerDeletedMaskBundle(
            valid_union=self.valid_union,
            deleted_union=self.deleted_union,
            answer_related_union=self.answer_related_union,
            block_records=self.dbs_block_records,
            out_of_bounds=self.out_of_bounds,
        )


def _empty_unified(image_width: int, image_height: int) -> UnifiedPageMaskBundle:
    empty = np.zeros((max(image_height, 0), max(image_width, 0)), dtype=bool)
    return UnifiedPageMaskBundle(
        effective_union=empty,
        txt_block_masks={},
        num_effective_blocks=0,
        out_of_bounds=False,
        valid_union=empty.copy(),
        deleted_union=empty.copy(),
        answer_related_union=empty.copy(),
        dbs_block_records=(),
    )


def build_unified_page_masks(
    blocks: tuple[PageBlockAnnotation, ...],
    *,
    image_width: int,
    image_height: int,
) -> UnifiedPageMaskBundle:
    if image_width <= 0 or image_height <= 0:
        return _empty_unified(image_width, image_height)

    fill_buffer = np.zeros((image_height, image_width), dtype=np.uint8)
    block_mask = np.zeros((image_height, image_width), dtype=bool)
    txt_block_masks: dict[str, np.ndarray] = {}
    valid_union = np.zeros((image_height, image_width), dtype=bool)
    deleted_union = np.zeros((image_height, image_width), dtype=bool)
    effective_union = np.zeros((image_height, image_width), dtype=bool)
    dbs_block_records: list[dict[str, object]] = []
    num_effective_blocks = 0
    out_of_bounds = False

    for block in blocks:
        is_txt = is_txt_block(block.block_type)
        is_deleted = is_deleted_text_block(block.block_type)
        is_valid_region = is_txt or is_figure_or_chart(block.block_type)
        is_effective = is_effective_block(block.block_type)
        participates_raster = is_valid_region or is_deleted or is_effective
        geometry_valid = len(block.polygon) >= 3
        block_out_of_bounds = False

        if geometry_valid and polygon_out_of_bounds(
            block.polygon,
            image_width=image_width,
            image_height=image_height,
        ):
            block_out_of_bounds = True
            out_of_bounds = True

        mask_area = 0
        if participates_raster and geometry_valid:
            fill_buffer.fill(0)
            _fill_polygon(fill_buffer, block.polygon, 1)
            np.greater(fill_buffer, 0, out=block_mask)
            mask_area = int(block_mask.sum())
            effective_union |= block_mask
            if is_txt:
                txt_block_masks[block.block_id] = block_mask.copy()
            if is_valid_region:
                valid_union |= block_mask
            if is_deleted:
                deleted_union |= block_mask
            if is_effective:
                num_effective_blocks += 1

        if is_txt or is_deleted:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = polygon_bbox(block.polygon)
            dbs_block_records.append(
                {
                    "page_id": block.page_id,
                    "block_id": block.block_id,
                    "block_order": block.block_order,
                    "block_type": block.block_type,
                    "polygon_area": polygon_area(block.polygon),
                    "mask_area": mask_area,
                    "bbox_x1": bbox_x1,
                    "bbox_y1": bbox_y1,
                    "bbox_x2": bbox_x2,
                    "bbox_y2": bbox_y2,
                    "is_valid_answer_block": is_txt,
                    "is_deleted_text_block": is_deleted,
                    "mask_out_of_bounds": block_out_of_bounds,
                    "geometry_valid": geometry_valid,
                }
            )

    answer_related_union = valid_union | deleted_union
    return UnifiedPageMaskBundle(
        effective_union=effective_union,
        txt_block_masks=txt_block_masks,
        num_effective_blocks=num_effective_blocks,
        out_of_bounds=out_of_bounds,
        valid_union=valid_union,
        deleted_union=deleted_union,
        answer_related_union=answer_related_union,
        dbs_block_records=tuple(dbs_block_records),
    )


def build_page_masks(
    blocks: tuple[PageBlockAnnotation, ...],
    *,
    image_width: int,
    image_height: int,
    is_effective_block=None,
    is_txt_block_fn=None,
) -> PageMaskBundle:
    del is_effective_block, is_txt_block_fn
    return build_unified_page_masks(
        blocks,
        image_width=image_width,
        image_height=image_height,
    ).foreground_masks()


def build_answer_deleted_masks(
    blocks: tuple[PageBlockAnnotation, ...],
    *,
    image_width: int,
    image_height: int,
) -> AnswerDeletedMaskBundle:
    return build_unified_page_masks(
        blocks,
        image_width=image_width,
        image_height=image_height,
    ).deleted_block_masks()


def rasterize_polygon(
    polygon: tuple[tuple[float, float], ...],
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    _fill_polygon(mask, polygon, 1)
    return mask > 0


def union_effective_masks(
    blocks: tuple[PageBlockAnnotation, ...],
    *,
    image_width: int,
    image_height: int,
    is_effective_block,
) -> tuple[np.ndarray, int, bool]:
    del is_effective_block
    bundle = build_unified_page_masks(
        blocks,
        image_width=image_width,
        image_height=image_height,
    )
    return bundle.effective_union, bundle.num_effective_blocks, bundle.out_of_bounds
