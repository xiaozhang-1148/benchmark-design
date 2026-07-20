"""Tests for LaTeX parse validation."""

from __future__ import annotations

from benchmark_design.ocr.latex_ast_parser import dictionary_ok
from benchmark_design.ocr.parse_validate import validate_parse_status
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


def _status(latex: str) -> str:
    tokens = tokenize_greedy(latex, build_latex_vocab())
    return validate_parse_status(tokens)


def test_validate_parse_status_ok() -> None:
    assert _status(r"\frac{1}{2}") == "ok"
    assert _status(r"\frac12") == "ok"
    assert _status(r"l^2") == "ok"
    assert _status(r"l_p") == "ok"
    assert _status(r"\sqrt x") == "ok"
    assert _status(r"\sqrt{x+1}") == "ok"
    assert _status(r"l_{p+1}") == "ok"
    assert _status(r"^ { 0 } ( x : 2 )") == "ok"
    assert _status(r"_ { 1 } = 0 x _ { 2 } = - 5") == "ok"
    assert _status(r"^ { ^ { \prime } }") == "ok"
    assert _status(r"x { ^ { 2 } }") == "ok"
    assert _status(r"{ _ { 1 } }") == "ok"


def test_validate_parse_status_unbalanced_braces() -> None:
    assert _status(r"\frac{1}{2") == "unbalanced_braces"
    assert _status(r"{1+2") == "unbalanced_braces"


def test_validate_parse_status_incomplete_substructure() -> None:
    assert _status(r"x^") == "incomplete_substructure"
    assert _status(r"\frac{1}") == "incomplete_substructure"
    assert _status(r"x_{") == "unbalanced_braces"


def test_validate_parse_status_unknown_token() -> None:
    assert _status(r"100%") == "unknown_token"
    ok, missing = dictionary_ok(tokenize_greedy(r"100%", build_latex_vocab()))
    assert not ok
    assert missing == "%"
