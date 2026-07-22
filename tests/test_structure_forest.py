"""Tests for unified structure forest AST metrics."""

from __future__ import annotations

import pytest

from benchmark_design.ocr.structure_forest import compute_ast_forest_metrics
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

VOCAB = build_latex_vocab()


def _metrics(latex: str):
    return compute_ast_forest_metrics(tokenize_greedy(latex, VOCAB))


def test_frac_creates_parent_node_and_depth() -> None:
    metrics = _metrics(r"\frac{1}{2}")
    assert metrics.structure_flags["frac"]
    assert metrics.ast_node_count == 1
    assert metrics.ast_depth == 1


def test_nested_frac_depth() -> None:
    metrics = _metrics(r"\frac{\frac{a}{b}}{c}")
    assert metrics.ast_depth == 2
    assert metrics.ast_node_count == 2


def test_accent_creates_parent_node() -> None:
    metrics = _metrics(r"\vec{x}")
    assert metrics.structure_flags["accent"]
    assert metrics.ast_node_count == 1
    assert metrics.ast_depth == 1


def test_bigop_groups_sum_and_lim() -> None:
    assert _metrics(r"\sum_{i=1}^{n} i").structure_flags["bigop"]
    assert _metrics(r"\lim_{x \to 0} f(x)").structure_flags["bigop"]


def test_mean_ast_node_depth() -> None:
    plain = _metrics("x")
    assert plain.mean_ast_node_depth == 0.0
    frac = _metrics(r"\frac{\frac{a}{b}}{c}")
    assert frac.mean_ast_node_depth == pytest.approx(1.5)


def test_stackrel_and_textcircled_nodes() -> None:
    stackrel = _metrics(r"\stackrel{!}{=}")
    textcircled = _metrics(r"\textcircled{1}")
    assert stackrel.structure_flags["stackrel"]
    assert textcircled.structure_flags["textcircled"]
    assert stackrel.ast_node_count == 1
    assert textcircled.ast_node_count == 1
