"""Tests for OCR structure combination complexity."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.structure_complexity import (
    compute_ocr_structure_complexity,
    compute_ocr_structure_complexity_from_counts,
)
from benchmark_design.ocr.structure_distribution import count_structure_types_in_tokens
from benchmark_design.report.structure_complexity_table import write_structure_complexity_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_count_structure_types_in_tokens_multi_type() -> None:
    tokens = [r"\frac", "{", "a", "}", "{", "b", "}", "^", "{", "2", "}"]
    assert count_structure_types_in_tokens(tokens) == 2


def test_compute_ocr_structure_complexity_from_counts() -> None:
    metrics = compute_ocr_structure_complexity_from_counts([0, 1, 1, 2, 4])
    assert metrics.expression_count == 5
    assert metrics.structural_expression_ratio == pytest.approx(0.8)
    assert metrics.mean_structure_type_count == pytest.approx(1.6)
    assert metrics.p50_structure_type_count == pytest.approx(1.0)
    assert metrics.p90_structure_type_count == pytest.approx(3.2)
    assert metrics.max_structure_type_count == 4
    assert metrics.multi_structure_ratio_ge_2 == pytest.approx(0.4)
    assert metrics.multi_structure_ratio_ge_3 == pytest.approx(0.2)
    assert metrics.multi_structure_ratio_ge_4 == pytest.approx(0.2)


def test_compute_ocr_structure_complexity_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_structure_complexity(sample_benchmark_dir)

    assert metrics.expression_count == 3
    assert metrics.structural_expression_ratio == pytest.approx(2 / 3)
    assert metrics.mean_structure_type_count == pytest.approx(2 / 3)
    assert metrics.p50_structure_type_count == pytest.approx(1.0)
    assert metrics.p90_structure_type_count == pytest.approx(1.0)
    assert metrics.max_structure_type_count == 1
    assert metrics.multi_structure_ratio_ge_2 == 0.0
    assert metrics.multi_structure_ratio_ge_3 == 0.0
    assert metrics.multi_structure_ratio_ge_4 == 0.0


def test_write_structure_complexity_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_structure_complexity(sample_benchmark_dir)
    paths = write_structure_complexity_report(
        metrics,
        tmp_path / "ocr",
        input_dir=sample_benchmark_dir,
    )

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "metric,definition,value" in csv_text
    assert "Structural expression ratio,含至少一种结构类型的表达式比例,0.666667" in csv_text
    assert paths["markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_structure_complexity_full_benchmark() -> None:
    metrics = compute_ocr_structure_complexity(FULL_BENCHMARK)

    assert metrics.expression_count == 152_113
    assert metrics.structural_expression_ratio == pytest.approx(0.6510028728642522)
    assert metrics.mean_structure_type_count == pytest.approx(1.0127799727833913)
    assert metrics.p50_structure_type_count == pytest.approx(1.0)
    assert metrics.p90_structure_type_count == pytest.approx(2.0)
    assert metrics.max_structure_type_count == 6
    assert metrics.multi_structure_ratio_ge_2 == pytest.approx(0.2713114592441146)
    assert metrics.multi_structure_ratio_ge_3 == pytest.approx(0.07855344382136964)
    assert metrics.multi_structure_ratio_ge_4 == pytest.approx(0.011050994983992164)
