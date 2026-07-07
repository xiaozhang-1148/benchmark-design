"""Tests for expression feature extraction."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_design.ocr.duplicates import build_duplicate_groups
from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus
from benchmark_design.ocr.expression_features import parse_success_rate

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_build_duplicate_groups() -> None:
    from benchmark_design.io.benchmark_loader import ExpressionRecord

    records = [
        ExpressionRecord("a", 0, 0, "", "x", dataset="ours"),
        ExpressionRecord("b", 0, 1, "", "x", dataset="ours"),
        ExpressionRecord("c", 0, 2, "", "y", dataset="ours"),
    ]
    group_ids, group_sizes = build_duplicate_groups(records)
    assert group_sizes[0] == 2
    assert group_sizes[1] == 2
    assert group_sizes[2] == 1
    assert group_ids[0] == group_ids[1]


def test_build_enriched_corpus_fixture(tmp_path: Path) -> None:
    target = tmp_path / "sample.jpg.json"
    shutil.copy(FIXTURE_PATH, target)
    enriched = build_enriched_corpus("ours", tmp_path, ProcessingOptions(show_progress=False))
    assert len(enriched.features) == 3
    assert enriched.features[0].token_length > 0
    assert 0.0 <= parse_success_rate(list(enriched.features)) <= 1.0
