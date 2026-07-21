"""Tests for structural LaTeX AST validation."""

from __future__ import annotations

from benchmark_design.ocr.latex_ast_parser import (
    dictionary_ok,
    validate_structural_ast,
)
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


def _tokens(latex: str) -> list[str]:
    return tokenize_greedy(latex, build_latex_vocab())


def test_dictionary_ok_uses_frozen_latex_dict() -> None:
    ok, missing = dictionary_ok(_tokens(r"\alpha + 1"))
    assert ok
    assert missing is None


def test_validate_structural_ast_single_atom_arguments() -> None:
    assert validate_structural_ast(_tokens(r"\frac12")) is None
    assert validate_structural_ast(_tokens(r"l^2")) is None
    assert validate_structural_ast(_tokens(r"\sqrt x")) is None


def test_validate_structural_ast_leading_and_grouped_scripts_without_base() -> None:
    assert validate_structural_ast(_tokens(r"^ { 0 } ( x : 2 )")) is None
    assert validate_structural_ast(_tokens(r"_ { 1 } = 0")) is None
    assert validate_structural_ast(_tokens(r"^ { ^ { \prime } }")) is None
    assert validate_structural_ast(_tokens(r"x { ^ { 2 } }")) is None
    assert validate_structural_ast(_tokens(r"{ _ { 1 } }")) is None


def test_validate_structural_ast_sqrt_optional_root_in_frac_denominator() -> None:
    latex = r"\frac { 1 } { 2 \sqrt [ 3 ] { x } }"
    assert validate_structural_ast(_tokens(latex)) is None


def test_validate_structural_ast_mamaotb54auqdwgja_delete_sqrt() -> None:
    latex = (
        r"( 2 ) \delete 设 T _ { r + 1 } = C _ { h } ^ { r } ( \sqrt [ 3 ] { x } ) ^ { n - r } "
        r"( - \frac { 1 } { 2 \sqrt [ 3 ] { x } } ) ^ { r }"
    )
    assert validate_structural_ast(_tokens(latex)) is None


def test_validate_structural_ast_only_reports_missing_args_and_braces() -> None:
    assert validate_structural_ast(_tokens(r"x^")) == "incomplete_substructure"
    assert validate_structural_ast(_tokens(r"\frac{1}")) == "incomplete_substructure"
    assert validate_structural_ast(_tokens(r"\frac{1}{2")) == "unbalanced_braces"
