"""Tests for matrix environment detection."""

from __future__ import annotations

from benchmark_design.ocr.matrix_environments import (
    expression_has_matrix_environment,
    find_valid_matrix_environment_span,
    matrix_environment_stats,
)


def _cases_block(*body: str) -> list[str]:
    tokens: list[str] = [r"\begin", "{", "cases", "}"]
    tokens.extend(body)
    tokens.extend([r"\end", "{", "cases", "}"])
    return tokens


def test_valid_cases_environment_counts_once() -> None:
    tokens = _cases_block("a", r"\\", "b")
    assert matrix_environment_stats(tokens).count == 1
    assert expression_has_matrix_environment(tokens)


def test_cases_without_row_break_is_not_counted() -> None:
    tokens = _cases_block("a")
    assert matrix_environment_stats(tokens).count == 0
    assert not expression_has_matrix_environment(tokens)


def test_loose_tokens_do_not_count() -> None:
    tokens = ["cases", r"\\", r"\begin", r"\end"]
    assert matrix_environment_stats(tokens).count == 0


def test_nested_matrix_environments() -> None:
    tokens = [
        r"\begin",
        "{",
        "cases",
        "}",
        r"\begin",
        "{",
        "pmatrix",
        "}",
        "a",
        r"\\",
        "b",
        r"\end",
        "{",
        "pmatrix",
        "}",
        r"\\",
        "c",
        r"\end",
        "{",
        "cases",
        "}",
    ]
    stats = matrix_environment_stats(tokens)
    assert stats.count == 2
    assert stats.max_depth == 2


def test_find_valid_matrix_environment_span() -> None:
    tokens = _cases_block("a", r"\\", "b")
    span = find_valid_matrix_environment_span(tokens, 0)
    assert span == (4, 7, 11)
