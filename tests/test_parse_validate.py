"""Tests for LaTeX parse validation."""

from __future__ import annotations

from benchmark_design.ocr.parse_validate import validate_parse_status
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


def _status(latex: str) -> str:
    tokens = tokenize_greedy(latex, build_latex_vocab())
    return validate_parse_status(tokens)


def test_validate_parse_status_ok() -> None:
    assert _status(r"\frac{1}{2}") == "ok"


def test_validate_parse_status_unbalanced_braces() -> None:
    assert _status(r"\frac{1}{2") == "unbalanced_braces"


def test_validate_parse_status_incomplete_substructure() -> None:
    assert _status(r"x^") == "incomplete_substructure"
    assert _status(r"\frac{1}") == "incomplete_substructure"
