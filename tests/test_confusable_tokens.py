"""Tests for confusable token group statistics."""

from __future__ import annotations

from benchmark_design.ocr.confusable_tokens import (
    PRIMARY_CONFUSABLE_GROUPS,
    compute_ocr_confusable_token_metrics_from_token_sequences,
)
from benchmark_design.ocr.tokenizer import tokenize_greedy


def test_compute_confusable_token_metrics_basic() -> None:
    sequences = [
        tokenize_greedy("x + 2 z"),
        tokenize_greedy(r"\rho = p"),
        tokenize_greedy("8"),
    ]
    metrics = compute_ocr_confusable_token_metrics_from_token_sequences(sequences)
    assert metrics.expression_count == 3
    assert metrics.total_token_count == sum(len(seq) for seq in sequences)

    digit_letter = next(item for item in metrics.primary_groups if item.group.name == "digit-letter")
    assert digit_letter.token_count >= 2
    assert digit_letter.expression_count == 1
    assert digit_letter.co_occurrence_expression_count == 1

    latin_greek = next(item for item in metrics.primary_groups if item.group.name == "latin-greek")
    assert latin_greek.expression_count == 1
    assert "p" in latin_greek.dominant_tokens or r"\rho" in latin_greek.dominant_tokens

    infinity_like = next(item for item in metrics.primary_groups if item.group.name == "infinity-like")
    assert infinity_like.expression_count == 1
    assert infinity_like.token_count == 1


def test_primary_group_count() -> None:
    assert len(PRIMARY_CONFUSABLE_GROUPS) == 8
