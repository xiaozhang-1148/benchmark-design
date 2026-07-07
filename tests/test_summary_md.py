"""Tests for benchmark summary.md generation."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_design.ocr.ast_statistics import compute_ocr_ast_statistics_from_token_sequences
from benchmark_design.ocr.consolidated import compute_ocr_consolidated_metrics_from_corpus
from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus
from benchmark_design.report.summary_md import build_benchmark_summary_markdown

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_build_benchmark_summary_markdown(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_PATH, input_dir / "sample.jpg.json")
    enriched = build_enriched_corpus("ours", input_dir, ProcessingOptions(show_progress=False))
    consolidated = compute_ocr_consolidated_metrics_from_corpus(enriched)
    ast_metrics = compute_ocr_ast_statistics_from_token_sequences(enriched.token_sequences)

    text = build_benchmark_summary_markdown(enriched, consolidated, ast_metrics)

    assert text.startswith("# OCR Benchmark Summary\n")
    assert "Dataset: ours  \n" in text
    assert "Tokenizer: LATEX_DICT greedy longest-match tokenizer  \n" in text
    assert "## 1. Dataset Scale" in text
    assert "## 2. Expression Length" in text
    assert "## 3. Duplicate Summary" in text
    assert "Duplicate definition: two expression records are duplicates iff their full" in text
    assert "## 4. Token Taxonomy" in text
    assert "## 5. Token Long-Tail" in text
    assert "## 6. Structure Complexity" in text
    assert "| Fraction | `\\frac` |" in text
    assert "## 7. AST Depth" in text
    assert "| Complex expression ratio, depth > 2 |" in text
    assert "## 8. Expression Content Type" in text
    assert "| Pure latex_command |" in text
    assert "| Pure CJK |" in text
    assert "| Mixed |" in text
    assert "## Output index" not in text
