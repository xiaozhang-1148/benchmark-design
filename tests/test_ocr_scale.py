"""Tests for OCR scale statistics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmark_design.io.benchmark_loader import iter_expressions
from benchmark_design.ocr.scale import compute_ocr_scale
from benchmark_design.ocr.tokenizer import tokenize_greedy
from benchmark_design.report.scale_table import write_scale_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"
FULL_BENCHMARK = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")


@pytest.fixture
def sample_benchmark_dir(tmp_path: Path) -> Path:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    return tmp_path


def test_tokenize_greedy_latex_fraction() -> None:
    assert tokenize_greedy(r"\frac { a } { b }") == ["\\frac", "{", "a", "}", "{", "b", "}"]


def test_tokenize_greedy_cjk_mixed() -> None:
    assert tokenize_greedy("解 : ( 1 ) 依 题") == ["解", ":", "(", "1", ")", "依", "题"]


def test_tokenize_greedy_delete_marker() -> None:
    assert tokenize_greedy(r"\delete x") == ["\\delete", "x"]


def test_iter_expressions_skips_empty_ocr(sample_benchmark_dir: Path) -> None:
    page = json.loads((sample_benchmark_dir / "sample.jpg.json").read_text(encoding="utf-8"))
    page["blocks"][0]["lines"].append({"order": 3, "ocr": "   ", "polygon": []})
    (sample_benchmark_dir / "sample.jpg.json").write_text(json.dumps(page), encoding="utf-8")

    records = list(iter_expressions(sample_benchmark_dir))
    assert len(records) == 3


def test_compute_ocr_scale_fixture(sample_benchmark_dir: Path) -> None:
    metrics = compute_ocr_scale(sample_benchmark_dir)
    assert metrics.expression_count == 3
    assert metrics.total_token_count == 40
    assert metrics.vocabulary_size == 19
    assert metrics.unique_normalized_latex_count == 3
    assert metrics.duplicate_rate == 0.0
    assert metrics.json_file_count == 1


def test_write_scale_report(sample_benchmark_dir: Path, tmp_path: Path) -> None:
    metrics = compute_ocr_scale(sample_benchmark_dir)
    paths = write_scale_report(metrics, tmp_path / "ocr", input_dir=sample_benchmark_dir)

    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "expression count,3" in csv_text
    assert paths["markdown"].exists()
    assert paths["metadata"].exists()


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_compute_ocr_scale_full_benchmark() -> None:
    metrics = compute_ocr_scale(FULL_BENCHMARK)
    assert metrics.expression_count == 152_113
    assert metrics.total_token_count == 3_552_617
    assert metrics.vocabulary_size == 1_005
    assert metrics.unique_normalized_latex_count == 120_322
    assert metrics.duplicate_rate == pytest.approx(0.2089959438049345)
    assert metrics.json_file_count == 9_911
