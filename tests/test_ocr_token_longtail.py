"""Tests for OCR token long-tail statistics."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.token_longtail import (
    compute_ocr_token_longtail,
    gini_coefficient,
    top_k_coverage,
)
from benchmark_design.report.token_longtail_table import write_token_longtail_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_gini_coefficient_uniform() -> None:
    assert gini_coefficient([1, 1, 1, 1]) == pytest.approx(0.0)


def test_gini_coefficient_concentrated() -> None:
    assert gini_coefficient([0, 0, 0, 10]) == pytest.approx(0.75)


def test_top_k_coverage() -> None:
    from collections import Counter

    counter = Counter({"a": 5, "b": 3, "c": 2})
    assert top_k_coverage(counter, 1) == pytest.approx(0.5)
    assert top_k_coverage(counter, 2) == pytest.approx(0.8)


def test_compute_ocr_token_longtail_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_token_longtail(sample_benchmark_dir)
    summary = dict(metrics.summary_rows())

    assert metrics.vocabulary_size == 19
    assert metrics.total_token_count == 40
    assert metrics.expression_count == 3
    assert metrics.gini == pytest.approx(0.28157894736842104)
    assert summary["top-10 coverage"] == pytest.approx(0.725)
    assert summary["top-50 coverage"] == pytest.approx(1.0)
    assert summary["rare_1 vocab ratio"] == pytest.approx(0.3684210526315789)
    assert summary["rare_5 vocab ratio"] == pytest.approx(1.0)
    assert summary["rare_10 vocab ratio"] == pytest.approx(1.0)
    assert summary["rare_1 expression ratio"] == pytest.approx(2 / 3)
    assert summary["rare_5 expression ratio"] == pytest.approx(1.0)
    assert len(metrics.frequency_distribution) == 19
    assert metrics.frequency_distribution[0].rank == 1
    assert metrics.frequency_distribution[-1].cumulative_share == pytest.approx(1.0)


def test_write_token_longtail_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_token_longtail(sample_benchmark_dir)
    paths = write_token_longtail_report(metrics, tmp_path / "ocr", input_dir=sample_benchmark_dir)

    assert "gini," in paths["summary_csv"].read_text(encoding="utf-8")
    assert paths["frequency_csv"].read_text(encoding="utf-8").count("\n") == 20
    assert paths["summary_markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_token_longtail_full_benchmark() -> None:
    metrics = compute_ocr_token_longtail(FULL_BENCHMARK)
    summary = dict(metrics.summary_rows())

    assert metrics.vocabulary_size == 1_005
    assert metrics.total_token_count == 3_552_617
    assert metrics.expression_count == 152_113
    assert metrics.gini == pytest.approx(0.9469881187733546)
    assert summary["top-10 coverage"] == pytest.approx(0.5331233285209185)
    assert summary["top-50 coverage"] == pytest.approx(0.8656508708932035)
    assert summary["top-100 coverage"] == pytest.approx(0.939349217773827)
    assert summary["top-500 coverage"] == pytest.approx(0.9980147029640403)
    assert summary["rare_1 vocab ratio"] == pytest.approx(0.10149253731343283)
    assert summary["rare_5 vocab ratio"] == pytest.approx(0.22686567164179106)
    assert summary["rare_10 vocab ratio"] == pytest.approx(0.29054726368159206)
    assert summary["rare_1 expression ratio"] == pytest.approx(0.0006245357070072907)
    assert summary["rare_5 expression ratio"] == pytest.approx(0.002715086810463274)
    assert summary["rare_10 expression ratio"] == pytest.approx(0.005154063097828588)
    assert len(metrics.frequency_distribution) == 1_005
