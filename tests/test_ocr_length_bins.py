"""Tests for fixed OCR length-bin statistics."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.length_bins import (
    assign_length_bin,
    compute_ocr_length_bins,
)
from benchmark_design.report.length_bins_table import write_length_bins_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_assign_length_bin_boundaries() -> None:
    assert assign_length_bin(1) == "1-10 tokens"
    assert assign_length_bin(10) == "1-10 tokens"
    assert assign_length_bin(11) == "11-20 tokens"
    assert assign_length_bin(20) == "11-20 tokens"
    assert assign_length_bin(21) == "21-40 tokens"
    assert assign_length_bin(40) == "21-40 tokens"
    assert assign_length_bin(41) == "41-80 tokens"
    assert assign_length_bin(80) == "41-80 tokens"
    assert assign_length_bin(81) == "> 80 tokens"


def test_compute_ocr_length_bins_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_length_bins(sample_benchmark_dir)
    assert metrics.expression_count == 3
    rows = {label: (count, share) for label, count, share in metrics.as_rows()}
    assert rows["1-10 tokens"] == (1, pytest.approx(1 / 3))
    assert rows["11-20 tokens"] == (2, pytest.approx(2 / 3))
    assert rows["21-40 tokens"] == (0, 0.0)
    assert rows["41-80 tokens"] == (0, 0.0)
    assert rows["> 80 tokens"] == (0, 0.0)


def test_write_length_bins_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_length_bins(sample_benchmark_dir)
    paths = write_length_bins_report(metrics, tmp_path / "ocr", input_dir=sample_benchmark_dir)

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "1-10 tokens,1," in csv_text
    assert "11-20 tokens,2," in csv_text
    assert paths["markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_length_bins_full_benchmark() -> None:
    metrics = compute_ocr_length_bins(FULL_BENCHMARK)
    rows = {label: (count, share) for label, count, share in metrics.as_rows()}
    assert metrics.expression_count == 152_113
    assert rows["1-10 tokens"][0] == 34_920
    assert rows["11-20 tokens"][0] == 48_008
    assert rows["21-40 tokens"][0] == 49_099
    assert rows["41-80 tokens"][0] == 17_706
    assert rows["> 80 tokens"][0] == 2_380
    assert rows["1-10 tokens"][1] == pytest.approx(0.22956617777573252)
    assert sum(count for count, _ in rows.values()) == 152_113
