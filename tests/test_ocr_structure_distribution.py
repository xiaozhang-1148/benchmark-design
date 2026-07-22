"""Tests for OCR structure-type distribution."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.structure_distribution import (
    STRUCTURE_TYPES,
    compute_ocr_structure_distribution,
    max_structure_depth,
)
from benchmark_design.report.structure_distribution_table import write_structure_distribution_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_max_structure_depth_nested_fraction() -> None:
    tokens = [r"\frac", "{", r"\frac", "{", "a", "}", "{", "b", "}", "}", "{", "c", "}"]
    spec = next(s for s in STRUCTURE_TYPES if s.structure_type == "分式")
    assert max_structure_depth(tokens, spec) == 2


def test_max_structure_depth_matrix_environment() -> None:
    tokens = [r"\begin", "{", "cases", "}", "a", r"\\", "b", r"\end", "{", "cases", "}"]
    spec = next(s for s in STRUCTURE_TYPES if s.structure_type == "Environment")
    assert max_structure_depth(tokens, spec) == 1


def test_max_structure_depth_matrix_without_row_break() -> None:
    tokens = [r"\begin", "{", "cases", "}", "a", r"\end", "{", "cases", "}"]
    spec = next(s for s in STRUCTURE_TYPES if s.structure_type == "Environment")
    assert max_structure_depth(tokens, spec) == 0


def test_matrix_occurrence_counts_complete_environment_only() -> None:
    tokens = [r"\begin", "{", "cases", "}", "a", r"\\", "b", r"\end", "{", "cases", "}"]
    from benchmark_design.ocr.structure_distribution import compute_ocr_structure_distribution_from_token_sequences

    metrics = compute_ocr_structure_distribution_from_token_sequences([tokens])
    matrix_row = next(row for row in metrics.rows if row.structure_type == "Environment")
    assert matrix_row.occurrence_count == 1
    assert matrix_row.expression_count == 1


def test_max_structure_depth_nested_matrix() -> None:
    tokens = [
        r"\begin", "{", "cases", "}",
        r"\begin", "{", "pmatrix", "}", "a", r"\\", "b", r"\end", "{", "pmatrix", "}",
        r"\\",
        "c",
        r"\end", "{", "cases", "}",
    ]
    spec = next(s for s in STRUCTURE_TYPES if s.structure_type == "Environment")
    assert max_structure_depth(tokens, spec) == 2


def test_compute_ocr_structure_distribution_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_structure_distribution(sample_benchmark_dir)
    rows = {row.structure_type: row for row in metrics.rows}

    assert metrics.expression_count == 3
    assert metrics.structural_token_count >= 4
    assert rows["下标"].expression_ratio == pytest.approx(2 / 3)
    assert rows["下标"].occurrence_ratio == pytest.approx(4 / 12)
    assert rows["下标"].max_depth == 1
    assert rows["下标"].occurrence_count == 4
    assert rows["Environment"].expression_ratio == 0.0
    assert rows["大运算符及极限"].occurrence_count == 0


def test_write_structure_distribution_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_structure_distribution(sample_benchmark_dir)
    paths = write_structure_distribution_report(
        metrics,
        tmp_path / "ocr",
        input_dir=sample_benchmark_dir,
    )

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "structure_type,trigger_tokens,expr_ratio" in csv_text
    assert "下标,sub,0.666667" in csv_text
    assert paths["markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_structure_distribution_full_benchmark() -> None:
    metrics = compute_ocr_structure_distribution(FULL_BENCHMARK)
    rows = {row.structure_type: row for row in metrics.rows}

    assert metrics.expression_count == 152_113
    assert rows["分式"].expression_ratio == pytest.approx(0.33160873824065007)
    assert rows["分式"].max_depth == 5
    assert rows["上标"].expression_ratio == pytest.approx(0.27050942391511573)
    assert rows["根式"].expression_ratio == pytest.approx(0.12046307679159572)
    assert rows["根式"].occurrence_count == 28_037
    assert rows["Environment"].expression_ratio == pytest.approx(0.03819528902855114)
    assert rows["Environment"].max_depth == 2
    assert rows["Environment"].occurrence_count == 6_879
    assert rows["大运算符及极限"].occurrence_count >= 74
