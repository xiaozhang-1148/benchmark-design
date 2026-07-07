"""Tests for OCR expression length distribution."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.length_distribution import (
    compute_ocr_length_distribution,
    percentile,
)
from benchmark_design.report.length_table import write_length_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_percentile_interpolation() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 50) == 10.0


def test_compute_ocr_length_distribution_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_length_distribution(sample_benchmark_dir)
    assert metrics.expression_count == 3
    assert metrics.mean_length == pytest.approx(40 / 3)
    assert metrics.std == pytest.approx(4.4969125210773475)
    assert metrics.cv == pytest.approx(0.33726843908080106)
    assert metrics.p50 == pytest.approx(16.0)
    assert metrics.p90 == pytest.approx(16.8)
    assert metrics.max_length == 17


def test_write_length_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_length_distribution(sample_benchmark_dir)
    paths = write_length_report(metrics, tmp_path / "ocr", input_dir=sample_benchmark_dir)

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "mean length," in csv_text
    assert "max,17" in csv_text
    assert paths["markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_length_distribution_full_benchmark() -> None:
    metrics = compute_ocr_length_distribution(FULL_BENCHMARK)
    assert metrics.expression_count == 152_113
    assert metrics.mean_length == pytest.approx(23.35511757706442)
    assert metrics.std == pytest.approx(18.26402405143241)
    assert metrics.cv == pytest.approx(0.7820137916740078)
    assert metrics.p50 == pytest.approx(19.0)
    assert metrics.p90 == pytest.approx(45.0)
    assert metrics.max_length == 270
