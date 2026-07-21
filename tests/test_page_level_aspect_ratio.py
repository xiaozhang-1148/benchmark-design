"""Tests for aspect-ratio grouping."""

from __future__ import annotations

from benchmark_design.config.page_level import DEFAULT_ASPECT_RATIO_BINS
from benchmark_design.page_level.features import assign_aspect_ratio_group


def test_assign_aspect_ratio_group_default_bins() -> None:
    assert assign_aspect_ratio_group(0.52, DEFAULT_ASPECT_RATIO_BINS) == "portrait"
    assert assign_aspect_ratio_group(0.72, DEFAULT_ASPECT_RATIO_BINS) == "portrait"
    assert assign_aspect_ratio_group(1.05, DEFAULT_ASPECT_RATIO_BINS) == "near_square"
    assert assign_aspect_ratio_group(1.45, DEFAULT_ASPECT_RATIO_BINS) == "landscape"


def test_assign_aspect_ratio_group_boundary_values() -> None:
    assert assign_aspect_ratio_group(0.90, DEFAULT_ASPECT_RATIO_BINS) == "near_square"
    assert assign_aspect_ratio_group(0.899, DEFAULT_ASPECT_RATIO_BINS) == "portrait"
    assert assign_aspect_ratio_group(1.20, DEFAULT_ASPECT_RATIO_BINS) == "landscape"
