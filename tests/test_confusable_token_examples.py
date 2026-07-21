"""Tests for confusable token example exports."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.ocr.confusable_tokens import (
    ocr_non_whitespace_char_count,
    select_confusable_token_examples,
)
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.report.confusable_token_examples import write_confusable_token_examples_csv


def _feature(
    expression_id: str,
    latex: str,
    tokens: tuple[str, ...],
) -> ExpressionFeatures:
    return ExpressionFeatures(
        expression_id=expression_id,
        dataset="ours",
        source_file="sample.json",
        line_id=expression_id,
        normalized_latex=latex,
        token_sequence=tokens,
        token_length=len(tokens),
        length_bin="short",
        is_duplicate=False,
        duplicate_group_id=0,
        duplicate_count=1,
        token_type_counts={},
        has_rare_1=False,
        has_rare_5=False,
        has_rare_10=False,
        structure_types=(),
        structure_type_count=0,
        structure_max_depths={},
        ast_depth=0,
        mean_token_nested_level=0.0,
        parse_status="ok",
    )


def test_ocr_non_whitespace_char_count() -> None:
    assert ocr_non_whitespace_char_count(r"x + 4") == 3
    assert ocr_non_whitespace_char_count(r"abcd") == 4


def test_select_confusable_token_examples_filters_short_ocr() -> None:
    features = [
        _feature("e1", "4", ("4",)),
        _feature("e2", r"x + 4 + \varphi", ("x", "+", "4", "+", r"\varphi")),
        _feature("e3", r"\varphi = 1", (r"\varphi", "=", "1")),
    ]
    rows = select_confusable_token_examples(features, tokens=("4", r"\varphi"), per_token=10)
    ids = {feature.expression_id for _, feature in rows}
    assert "e1" not in ids
    assert "e2" in ids
    assert "e3" in ids


def test_write_confusable_token_examples_csv(tmp_path: Path) -> None:
    features = [
        _feature("e2", r"x + 4 + \varphi", ("x", "+", "4", "+", r"\varphi")),
        _feature("e3", r"\varphi = 1", (r"\varphi", "=", "1")),
    ]
    output_path = tmp_path / "examples.csv"
    count = write_confusable_token_examples_csv(features, output_path, per_token=1)
    assert count == 2
    text = output_path.read_text(encoding="utf-8")
    assert "greek-variant" in text
    assert r"\varphi" in text
