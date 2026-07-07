"""Diagnostic-only foreground density classification (not used in main reports)."""

from __future__ import annotations

from benchmark_design.vision.foreground_load.thresholds import (
    LEVEL_LOW_MAX,
    LEVEL_MEDIUM_MAX,
    TAG_VERY_HIGH_MIN,
)

FOREGROUND_LOAD_LEVELS: tuple[str, ...] = ("low", "medium", "high")
RELATIVE_LOAD_TERTILES: tuple[str, ...] = ("lower", "middle", "upper")
EXTREME_DENSITY_TAG = "extreme_foreground_pixel_density_candidate"

_REVIEW_TO_TAG = {
    "density_saturated_low": "saturated_low",
    "density_saturated_high": "saturated_high",
}


def diagnostic_level(density: float | None) -> str:
    """Internal QA bucket; not exported as a benchmark density level."""
    return foreground_load_level(density)


def foreground_load_level(density: float | None) -> str:
    if density is None:
        return ""
    if density < LEVEL_LOW_MAX:
        return "low"
    if density < LEVEL_MEDIUM_MAX:
        return "medium"
    return "high"


def relative_load_tertile(density: float | None, *, p33: float, p66: float) -> str:
    if density is None:
        return ""
    if density <= p33:
        return "lower"
    if density <= p66:
        return "middle"
    return "upper"


def diagnostic_tags(density: float | None, review_reason: str) -> str:
    tags: list[str] = []
    for reason in review_reason.split(";"):
        if not reason:
            continue
        tag = _REVIEW_TO_TAG.get(reason, reason)
        if tag not in tags:
            tags.append(tag)
    if density is not None and density >= TAG_VERY_HIGH_MIN and EXTREME_DENSITY_TAG not in tags:
        tags.append(EXTREME_DENSITY_TAG)
    return ";".join(tags)


def foreground_load_tags(density: float | None, review_reason: str) -> str:
    """Backward-compatible alias for diagnostic_tags."""
    return diagnostic_tags(density, review_reason)
