"""Tests for OCR token taxonomy composition."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.token_taxonomy import (
    TokenCategory,
    classify_token,
    compute_ocr_token_taxonomy,
)
from benchmark_design.report.token_taxonomy_table import write_token_taxonomy_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_classify_token_examples() -> None:
    assert classify_token("a") is TokenCategory.ENGLISH
    assert classify_token("2") is TokenCategory.DIGIT
    assert classify_token("解") is TokenCategory.CJK
    assert classify_token("+") is TokenCategory.OPERATOR
    assert classify_token("(") is TokenCategory.GROUPING
    assert classify_token("{") is TokenCategory.STRUCTURAL
    assert classify_token("}") is TokenCategory.STRUCTURAL
    assert classify_token(r"\{") is TokenCategory.GROUPING
    assert classify_token(r"\}") is TokenCategory.GROUPING
    assert classify_token(r"\frac") is TokenCategory.STRUCTURAL
    assert classify_token(r"\alpha") is TokenCategory.GREEK
    assert classify_token(",") is TokenCategory.PUNCTUATION
    assert classify_token(":") is TokenCategory.PUNCTUATION
    assert classify_token(r"\\") is TokenCategory.LAYOUT_ALIGNMENT
    assert classify_token("&") is TokenCategory.LAYOUT_ALIGNMENT
    assert classify_token("\\") is TokenCategory.LAYOUT_ALIGNMENT
    assert classify_token("|") is TokenCategory.GROUPING
    assert classify_token("、") is TokenCategory.PUNCTUATION
    assert classify_token("。") is TokenCategory.PUNCTUATION
    assert classify_token('"') is TokenCategory.PUNCTUATION
    assert classify_token(r"\_") is TokenCategory.GROUPING
    assert classify_token(r"\#") is TokenCategory.GROUPING
    assert classify_token(r"\%") is TokenCategory.GROUPING
    assert classify_token("%") is TokenCategory.OTHER
    assert classify_token("√") is TokenCategory.SPECIAL_SYMBOL


def test_compute_ocr_token_taxonomy_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_token_taxonomy(sample_benchmark_dir)
    rows = {token_type: (count, share) for token_type, count, share in metrics.as_rows()}
    assert metrics.total_token_count == 40
    assert rows["english tokens"] == (10, pytest.approx(0.25))
    assert rows["digit tokens"] == (5, pytest.approx(0.125))
    assert rows["greek tokens"] == (0, pytest.approx(0.0))
    assert rows["special symbol tokens"] == (1, pytest.approx(0.025))
    assert rows["operator tokens"] == (6, pytest.approx(0.15))
    assert rows["grouping tokens"] == (2, pytest.approx(0.05))
    assert rows["structural tokens"] == (12, pytest.approx(0.3))
    assert rows["CJK tokens"] == (3, pytest.approx(0.075))
    assert rows["punctuation tokens"] == (1, pytest.approx(0.025))
    assert rows["layout / alignment tokens"] == (0, pytest.approx(0.0))
    assert rows["other / unknown tokens"] == (0, pytest.approx(0.0))
    assert metrics.other_unknown_ratio == pytest.approx(0.0)


def test_write_token_taxonomy_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_token_taxonomy(sample_benchmark_dir)
    paths = write_token_taxonomy_report(metrics, tmp_path / "ocr", input_dir=sample_benchmark_dir)

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "english tokens,10," in csv_text
    assert "other / unknown token ratio" in csv_text
    assert paths["markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_token_taxonomy_full_benchmark() -> None:
    metrics = compute_ocr_token_taxonomy(FULL_BENCHMARK)
    rows = {token_type: (count, share) for token_type, count, share in metrics.as_rows()}
    assert metrics.total_token_count == 3_552_617
    assert rows["english tokens"][0] == 554_816
    assert rows["digit tokens"][0] == 622_006
    assert rows["greek tokens"][0] > 0
    assert rows["special symbol tokens"][0] < 194_436
    assert rows["operator tokens"][0] == 413_652
    assert rows["grouping tokens"][0] < 1_085_920
    assert rows["structural tokens"][0] > 345_162
    assert rows["CJK tokens"][0] == 257_285
    assert rows["punctuation tokens"][0] == 70_634
    assert rows["layout / alignment tokens"][0] == 8_706
    assert rows["other / unknown tokens"][0] == 0
    assert metrics.other_unknown_ratio == pytest.approx(0.0)
    assert sum(count for count, _ in rows.values()) == 3_552_617
