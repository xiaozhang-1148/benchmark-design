"""Cross-benchmark integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_design.config import CROSS_BENCHMARK_SETS
from benchmark_design.ocr.cross_benchmark import compute_cross_benchmark_rows
from benchmark_design.ocr.processing import ProcessingOptions

CROHME2014 = CROSS_BENCHMARK_SETS["CROHME2014"]


@pytest.mark.integration
@pytest.mark.skipif(not (CROHME2014 / "caption.txt").is_file(), reason="CROHME2014 unavailable")
def test_cross_benchmark_crohme2014() -> None:
    rows = compute_cross_benchmark_rows(
        ["CROHME2014"],
        processing=ProcessingOptions(show_progress=False),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.dataset == "CROHME2014"
    assert row.expression_count == 986
    assert 0.0 <= row.parse_success_rate <= 1.0
    assert row.vocabulary_size > 0
