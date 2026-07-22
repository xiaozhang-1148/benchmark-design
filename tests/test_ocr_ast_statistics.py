"""Tests for PosFormer position forest and AST depth statistics."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.ast_statistics import compute_ocr_ast_statistics
from benchmark_design.ocr.position_forest import encode_position_forest_tokens, max_nested_level
from benchmark_design.ocr.tokenizer import tokenize_greedy, build_latex_vocab
from benchmark_design.report.ast_statistics_table import write_ast_statistics_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
VOCAB = build_latex_vocab()


def _encode(text: str):
    return encode_position_forest_tokens(tokenize_greedy(text, VOCAB))


def test_max_nested_level_simple_fraction() -> None:
    encoding = _encode(r"\frac{1}{n}")
    assert encoding.max_nested_level == 1
    assert encoding.identifiers[0] == "ML"
    assert encoding.identifiers[-1] == "MR"


def test_max_nested_level_nested_superscript_fraction() -> None:
    encoding = _encode(r"4^{x-\frac{1}{4}}")
    assert encoding.max_nested_level == 2
    assert "MLL" in encoding.identifiers
    assert "MLR" in encoding.identifiers


def test_max_nested_level_paper_fraction_example() -> None:
    text = r"Ay^{3}_{1}+\frac{y^{\beta_{1}}_{2}B}{C}"
    encoding = _encode(text)
    assert encoding.max_nested_level == 3
    assert "MLLR" in encoding.identifiers


def test_max_nested_level_helper() -> None:
    tokens = tokenize_greedy(r"x^{2}", VOCAB)
    assert max_nested_level(tokens) == 1


def test_max_nested_level_matrix_environment() -> None:
    encoding = _encode(r"\begin{cases} a \\ b \end{cases}")
    assert encoding.max_nested_level == 1


def test_max_nested_level_matrix_without_row_break() -> None:
    encoding = _encode(r"\begin{cases} a \end{cases}")
    assert encoding.max_nested_level == 0


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_compute_ocr_ast_statistics_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_ast_statistics(sample_benchmark_dir)

    assert metrics.expression_count == 3
    assert metrics.max_max_nested_level == 1
    assert metrics.nested_level_0_ratio == pytest.approx(1 / 3)
    assert metrics.nested_level_1_ratio == pytest.approx(2 / 3)
    assert metrics.complex_expression_ratio == 0.0


def test_write_ast_statistics_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_ast_statistics(sample_benchmark_dir)
    output_root = tmp_path / "benchmark_output"
    paths = write_ast_statistics_report(
        metrics,
        output_root / "tables",
        input_dir=sample_benchmark_dir,
        output_root=output_root,
    )

    csv_text = paths["summary_csv"].read_text(encoding="utf-8")
    assert paths["summary_csv"].name == "ast_depth_summary.csv"
    assert "metric,definition,value" in csv_text
    assert "Complex expression ratio (>2)" in csv_text
    assert paths["metadata"].name == "ast_depth_metadata.json"
    assert paths["metadata"].exists()
    assert "markdown" not in paths
