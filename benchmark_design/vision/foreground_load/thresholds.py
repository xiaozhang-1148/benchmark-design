"""Thresholds for foreground load metrics."""

from __future__ import annotations

from benchmark_design.vision.flow_structure.thresholds import normalize_block_type

D_REVIEW_LOW = 0.02
D_REVIEW_HIGH = 0.98
MIN_MASK_PIXELS = 32

LEVEL_LOW_MAX = 0.06
LEVEL_MEDIUM_MAX = 0.12
TAG_VERY_HIGH_MIN = 0.18

EFFECTIVE_BLOCK_TYPES = frozenset(
    {
        "txtblock",
        "txt_block",
        "figure",
        "deleted_text_block",
        "chart",
    }
)


def is_effective_block(block_type: str) -> bool:
    normalized = normalize_block_type(block_type)
    return normalized in EFFECTIVE_BLOCK_TYPES


def count_blocks_by_type(blocks) -> tuple[int, int, int, int]:
    num_txt = 0
    num_figure = 0
    num_chart = 0
    num_deleted = 0
    for block in blocks:
        normalized = normalize_block_type(block.block_type)
        if normalized in {"txtblock", "txt_block"}:
            num_txt += 1
        elif normalized == "figure":
            num_figure += 1
        elif normalized == "chart":
            num_chart += 1
        elif normalized == "deleted_text_block":
            num_deleted += 1
    return num_txt, num_figure, num_chart, num_deleted
