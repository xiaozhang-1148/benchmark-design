"""Tests for consolidated OCR benchmark report."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark_design.ocr.consolidated import compute_ocr_consolidated_metrics
from benchmark_design.report.consolidated_table import (
    build_consolidated_csv_rows,
    write_consolidated_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_build_consolidated_csv_rows_has_all_tables(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_consolidated_metrics(sample_benchmark_dir)
    rows = build_consolidated_csv_rows(metrics)
    table_ids = {row.table_id for row in rows}
    assert table_ids == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_write_consolidated_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    from benchmark_design.ocr.processing import build_enriched_corpus

    metrics = compute_ocr_consolidated_metrics(sample_benchmark_dir)
    enriched = build_enriched_corpus("ours", sample_benchmark_dir)
    output_root = tmp_path / "benchmark_output"
    paths = write_consolidated_report(
        metrics,
        output_root / "tables",
        output_root=output_root,
        features=list(enriched.features),
    )

    md_text = paths["markdown"].read_text(encoding="utf-8")
    assert "# OCR Benchmark Statistics Summary" in md_text
    assert "## Table 1. OCR Data Scale / OCR 数据规模" in md_text
    assert "## Table 7. Structure Combination Complexity / 结构组合复杂度" in md_text
    assert "## Table 8. Expression Content Type / 表达式内容类型" in md_text
    assert "## Table 9. Confusable Token Group Statistics / 易混 Token 组统计" in md_text
    assert "## Table 10. Expression-level Structural Difficulty / 表达式结构难度" in md_text
    assert "Expression-level Structural Difficulty" in md_text
    assert not (output_root / "tables" / "all_metrics_long.csv").exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_consolidated_metrics_full_benchmark() -> None:
    metrics = compute_ocr_consolidated_metrics(FULL_BENCHMARK)

    assert metrics.scale.expression_count == 152_113
    assert metrics.scale.total_token_count == 3_552_617
    assert metrics.length.mean_length == pytest.approx(23.355118, rel=1e-4)
    assert metrics.complexity.structural_expression_ratio == pytest.approx(0.6510028728642522)
    assert len(metrics.structure.rows) == 8
    assert len(build_consolidated_csv_rows(metrics)) >= 40
