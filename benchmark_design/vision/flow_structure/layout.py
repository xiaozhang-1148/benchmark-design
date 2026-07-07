"""Geometry cleaning and block role assignment."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_design.vision.flow_structure.block_roles import block_role, is_flow_context_block
from benchmark_design.vision.flow_structure.geometry import polygon_area
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageBlockAnnotation


@dataclass(frozen=True, slots=True)
class LayoutBlockRecord:
    block: PageBlockAnnotation
    role: str
    geometry_valid: bool
    mask_area: float
    qa_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PageLayoutContext:
    layout_blocks: tuple[LayoutBlockRecord, ...]
    qa_tags: tuple[str, ...]
    primary_answer_blocks: tuple[PageBlockAnnotation, ...]
    deleted_answer_blocks: tuple[PageBlockAnnotation, ...]
    visual_structural_blocks: tuple[PageBlockAnnotation, ...]
    flow_context_blocks: tuple[PageBlockAnnotation, ...]
    other_blocks: tuple[PageBlockAnnotation, ...]

    @property
    def answer_blocks(self) -> tuple[PageBlockAnnotation, ...]:
        return self.primary_answer_blocks

    @property
    def context_blocks(self) -> tuple[PageBlockAnnotation, ...]:
        return self.flow_context_blocks

    @property
    def num_context_blocks(self) -> int:
        return len(self.flow_context_blocks)


def _block_qa_tags(block: PageBlockAnnotation) -> tuple[str, ...]:
    tags: list[str] = []
    if len(block.polygon) < 3:
        tags.append("invalid_geometry")
    area = polygon_area(block.polygon)
    if area <= 0:
        tags.append("zero_area")
    return tuple(tags)


def build_page_layout_context(page: PageAnnotation) -> PageLayoutContext:
    layout_blocks: list[LayoutBlockRecord] = []
    page_qa: list[str] = []
    primary: list[PageBlockAnnotation] = []
    deleted: list[PageBlockAnnotation] = []
    visual: list[PageBlockAnnotation] = []
    flow_context: list[PageBlockAnnotation] = []
    other: list[PageBlockAnnotation] = []

    for block in page.blocks:
        qa_tags = _block_qa_tags(block)
        geometry_valid = not qa_tags
        mask_area = polygon_area(block.polygon) if len(block.polygon) >= 3 else 0.0
        role = block_role(block.block_type)
        record = LayoutBlockRecord(
            block=block,
            role=role,
            geometry_valid=geometry_valid,
            mask_area=mask_area,
            qa_tags=qa_tags,
        )
        layout_blocks.append(record)
        for tag in qa_tags:
            if tag not in page_qa:
                page_qa.append(tag)

        if not geometry_valid:
            continue

        if role == "primary_answer":
            primary.append(block)
        elif role == "deleted_answer":
            deleted.append(block)
            flow_context.append(block)
        elif role == "visual_structural":
            visual.append(block)
            if is_flow_context_block(block.block_type):
                flow_context.append(block)
        else:
            other.append(block)

    return PageLayoutContext(
        layout_blocks=tuple(layout_blocks),
        qa_tags=tuple(page_qa),
        primary_answer_blocks=tuple(primary),
        deleted_answer_blocks=tuple(deleted),
        visual_structural_blocks=tuple(visual),
        flow_context_blocks=tuple(flow_context),
        other_blocks=tuple(other),
    )
