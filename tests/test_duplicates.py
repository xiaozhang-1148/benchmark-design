"""Tests for duplicate detection (full normalized_latex equality)."""

from __future__ import annotations

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.duplicates import (
    DuplicateIndex,
    build_duplicate_groups,
    duplicate_stats_from_expressions,
    normalize_expression_latex,
)


def test_normalize_expression_latex_strips_whitespace() -> None:
    assert normalize_expression_latex("  x+y  ") == "x+y"
    assert normalize_expression_latex("x") == "x"


def test_partial_latex_strings_are_not_duplicates() -> None:
    records = [
        ExpressionRecord("a", 0, 0, "", r"x", dataset="ours"),
        ExpressionRecord("b", 0, 1, "", r"x+y", dataset="ours"),
        ExpressionRecord("c", 0, 2, "", r"x", dataset="ours"),
    ]
    stats = duplicate_stats_from_expressions(records)
    assert stats.expression_count == 3
    assert stats.unique_normalized_latex_count == 2
    assert stats.redundant_expression_count == 1
    assert stats.duplicate_rate == 1 / 3
    assert stats.duplicate_group_count == 1
    assert stats.max_duplicate_group_size == 2


def test_build_duplicate_groups_exact_match_only() -> None:
    records = [
        ExpressionRecord("a", 0, 0, "", "x", dataset="ours"),
        ExpressionRecord("b", 0, 1, "", "x", dataset="ours"),
        ExpressionRecord("c", 0, 2, "", "y", dataset="ours"),
    ]
    group_ids, group_sizes = build_duplicate_groups(records)
    assert group_sizes[0] == 2
    assert group_sizes[1] == 2
    assert group_sizes[2] == 1
    assert group_ids[0] == group_ids[1]
    assert group_ids[0] != group_ids[2]


def test_duplicate_index_whitespace_normalization() -> None:
    records = [
        ExpressionRecord("a", 0, 0, "", " x ", dataset="ours"),
        ExpressionRecord("b", 0, 1, "", "x", dataset="ours"),
    ]
    index = DuplicateIndex.from_expressions(records)
    stats = index.stats()
    assert stats.unique_normalized_latex_count == 1
    assert stats.max_duplicate_group_size == 2
    assert index.is_duplicate(0)
    assert index.is_duplicate(1)
