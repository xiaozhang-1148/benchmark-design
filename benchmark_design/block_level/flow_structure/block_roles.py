"""Block role taxonomy for flow structure layout analysis."""

from __future__ import annotations

PRIMARY_ANSWER_TYPES = frozenset({"txtblock", "txt_block"})
DELETED_ANSWER_TYPES = frozenset({"deleted_text_block"})
VISUAL_STRUCTURAL_TYPES = frozenset(
    {
        "figure",
        "chart",
        "table",
        "drawing",
        "printed_graphic",
    }
)


def normalize_block_type(block_type: str) -> str:
    return block_type.strip().lower().replace("-", "_")


def is_txt_block(block_type: str) -> bool:
    return normalize_block_type(block_type) in PRIMARY_ANSWER_TYPES


def is_deleted_text_block(block_type: str) -> bool:
    return normalize_block_type(block_type) in DELETED_ANSWER_TYPES


def is_figure_or_chart(block_type: str) -> bool:
    normalized = normalize_block_type(block_type)
    return normalized in {"figure", "chart"}


def is_flow_context_block(block_type: str) -> bool:
    return is_deleted_text_block(block_type) or is_figure_or_chart(block_type)


def is_visual_structural_block(block_type: str) -> bool:
    return normalize_block_type(block_type) in VISUAL_STRUCTURAL_TYPES


def is_interrupting_block(block_type: str) -> bool:
    normalized = normalize_block_type(block_type)
    return normalized in {"figure", "chart"}


def is_auxiliary_non_txt(block_type: str) -> bool:
    normalized = normalize_block_type(block_type)
    return normalized in VISUAL_STRUCTURAL_TYPES | DELETED_ANSWER_TYPES


def block_role(block_type: str) -> str:
    if is_txt_block(block_type):
        return "primary_answer"
    if is_deleted_text_block(block_type):
        return "deleted_answer"
    if is_visual_structural_block(block_type):
        return "visual_structural"
    return "other"
