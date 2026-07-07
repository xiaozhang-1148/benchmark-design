"""Tests for example CSV exports."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus
from benchmark_design.report.export_examples import write_all_examples

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_write_all_examples_schemas(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_PATH, input_dir / "sample.jpg.json")
    enriched = build_enriched_corpus("ours", input_dir, ProcessingOptions(show_progress=False))
    features = list(enriched.features)
    paths = write_all_examples(features, tmp_path / "examples")

    assert not (tmp_path / "examples" / "duplicate_examples.csv").exists()
    assert paths["duplicate_group_examples"].is_file()

    with paths["duplicate_group_examples"].open(encoding="utf-8") as handle:
        duplicate_reader = csv.DictReader(handle)
        assert duplicate_reader.fieldnames == [
            "rank",
            "normalized_latex",
            "duplicate_count",
            "expression_ids_sample",
            "source_file_count",
            "token_length",
            "structure_type_count",
            "ast_depth",
        ]
        duplicate_rows = list(duplicate_reader)
    assert len(duplicate_rows) == 0

    with paths["rare_token_examples"].open(encoding="utf-8") as handle:
        rare_rows = list(csv.DictReader(handle))
    assert len(rare_rows) <= 50
    assert rare_rows[0].keys() == {
        "rank",
        "expression_id",
        "rare_tokens",
        "rare_token_counts",
        "rare_threshold",
        "normalized_latex",
        "token_length",
    }

    with paths["unknown_token_examples"].open(encoding="utf-8") as handle:
        unknown_header = handle.readline().strip().split(",")
    assert unknown_header == [
        "rank",
        "expression_id",
        "unknown_tokens",
        "unknown_token_count",
        "normalized_latex",
        "token_length",
    ]

    with paths["high_structure_examples"].open(encoding="utf-8") as handle:
        high_rows = list(csv.DictReader(handle))
    assert high_rows[0].keys() == {
        "rank",
        "expression_id",
        "structure_type_count",
        "structure_types",
        "structure_max_depths",
        "ast_depth",
        "token_length",
        "normalized_latex",
    }
