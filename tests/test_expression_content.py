"""Tests for expression content classification."""

from __future__ import annotations

from benchmark_design.ocr.expression_content import (
    ExpressionContentKind,
    classify_expression_content,
    compute_ocr_expression_content_from_token_sequences,
)
from benchmark_design.ocr.tokenizer import tokenize_greedy


def test_classify_expression_content_fixture_cases() -> None:
    assert classify_expression_content(tokenize_greedy(r"\frac { a } { b }")) is ExpressionContentKind.LATEX_COMMAND
    assert classify_expression_content(tokenize_greedy("解")) is ExpressionContentKind.CJK
    assert classify_expression_content(tokenize_greedy("解 : ( 1 ) 依 题")) is ExpressionContentKind.MIXED


def test_compute_ocr_expression_content_from_token_sequences() -> None:
    sequences = [
        tokenize_greedy(r"\frac { a } { b }"),
        tokenize_greedy("解"),
        tokenize_greedy("解 : x"),
    ]
    metrics = compute_ocr_expression_content_from_token_sequences(sequences)
    assert metrics.expression_count == 3
    rows = dict((kind, count) for kind, count, _ in metrics.as_rows())
    assert rows["pure latex_command"] == 1
    assert rows["pure CJK"] == 1
    assert rows["mixed"] == 1
