"""Tests for tokenizer documentation export."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus
from benchmark_design.ocr.tokenizer_docs import write_tokenizer_docs

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_write_tokenizer_docs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_PATH, input_dir / "sample.jpg.json")
    enriched = build_enriched_corpus("ours", input_dir, ProcessingOptions(show_progress=False))
    docs_dir = tmp_path / "docs"
    resources_dir = tmp_path / "resources"
    paths = write_tokenizer_docs(
        list(enriched.features),
        docs_dir,
        resources_dir,
    )

    assert paths["tokenizer_rules"].read_text(encoding="utf-8").startswith("# LaTeX Tokenizer Rules")
    assert paths["latex_dictionary"].read_text(encoding="utf-8").startswith("# LaTeX Dictionary")
    assert "latex_vocab.csv" in paths["latex_dictionary"].read_text(encoding="utf-8")
    assert paths["taxonomy_rules"].read_text(encoding="utf-8").startswith("# Token Taxonomy Rules")
    assert "punctuation tokens" in paths["taxonomy_rules"].read_text(encoding="utf-8")
    assert paths["metric_definitions"].is_file()
    assert paths["data_schema"].read_text(encoding="utf-8").startswith("# Data Schema")
    assert paths["known_limitations"].read_text(encoding="utf-8").startswith("# Known Limitations")
    assert paths["latex_vocab"].is_file()
    assert paths["token_taxonomy_map"].is_file()
