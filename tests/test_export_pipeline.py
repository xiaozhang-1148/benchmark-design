"""Smoke tests for full benchmark export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmark_design.ocr.processing import ProcessingOptions
from benchmark_design.report.export_pipeline import run_benchmark_export

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_page.json"

EXPECTED_CORE_TABLES = (
    "dataset_scale.csv",
    "parse_status_summary.csv",
    "length_summary.csv",
    "length_bins.csv",
    "duplicate_summary.csv",
    "duplicate_group_size_distribution.csv",
    "token_taxonomy_composition.csv",
    "token_frequency_top100.csv",
    "token_long_tail_summary.csv",
    "punctuation_layout_token_summary.csv",
    "unclassified_token_summary.csv",
    "structure_type_distribution.csv",
    "structure_combination_summary.csv",
    "ast_depth_summary.csv",
    "ast_depth_distribution.csv",
    "expression_content_summary.csv",
    "confusable_token_group_summary.csv",
    "README.md",
)

EXPECTED_APPENDIX_TABLES = (
    "length_distribution.csv",
    "rare_token_summary.csv",
    "structure_pattern_distribution.csv",
    "structure_cooccurrence_matrix.csv",
    "confusable_token_counts.csv",
)


def test_run_benchmark_export_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_PATH, input_dir / "sample.jpg.json")
    output_root = tmp_path / "benchmark_output"

    manifest = run_benchmark_export(
        input_dir,
        output_root,
        processing=ProcessingOptions(show_progress=False),
        skip_cross_benchmark=True,
    )

    assert (output_root / "summary.md").is_file()
    summary_text = (output_root / "summary.md").read_text(encoding="utf-8")
    assert summary_text.startswith("# OCR Benchmark Summary")
    assert "## 1. Dataset Scale" in summary_text
    assert "## 8. Expression Content Type" in summary_text
    assert "## 9. Confusable Token Groups" in summary_text
    assert "## Output index" not in summary_text
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["expression_count"] == 3
    assert "parse_success_rate" in metadata
    assert not metadata["input_dir"].startswith("/")
    assert all(not value.startswith("/") for value in metadata["manifest"].values())

    for name in EXPECTED_CORE_TABLES:
        assert (output_root / "tables" / name).is_file(), name
    for name in EXPECTED_APPENDIX_TABLES:
        assert (output_root / "tables" / "appendix" / name).is_file(), name

    assert not (output_root / "tables" / "all_metrics_long.csv").exists()
    assert not (output_root / "tables" / "token_frequency_topk.csv").exists()
    assert not (output_root / "tables" / "unclassified_token_ratio_by_token.csv").exists()
    assert not (output_root / "tables" / "token_taxonomy_map.csv").exists()

    scale_header = (output_root / "tables" / "dataset_scale.csv").read_text(encoding="utf-8").splitlines()[0]
    assert scale_header == (
        "dataset,json_file_count,expression_count,total_token_count,unique_normalized_latex_count,vocabulary_size"
    )

    assert (output_root / "ocr_benchmark_summary.md").is_file()
    assert (output_root / "summary.md").is_file()
    assert not (output_root / "report").exists()
    assert not (output_root / "ast_depth_summary.md").exists()
    assert (output_root / "docs" / "metadata" / "ocr_benchmark_metadata.json").is_file()
    assert (output_root / "resources" / "latex_vocab.csv").is_file()
    assert (output_root / "resources" / "token_taxonomy_map.csv").is_file()
    assert not (output_root / "tables" / "ocr_benchmark_summary.md").exists()
    assert not (output_root / "tables" / "taxonomy_unknown_tokens.csv").exists()

    assert (output_root / "details" / "expression_level_statistics.csv").is_file()
    assert not (output_root / "details" / "structure_patterns.csv").exists()
    assert not (output_root / "details" / "duplicate_count_distribution.csv").exists()
    assert not (output_root / "details" / "parse_failure_samples.csv").exists()
    assert not (output_root / "details" / "rare_token_expressions.csv").exists()
    assert (output_root / "docs" / "tokenizer_rules.md").is_file()
    assert (output_root / "figures" / "length_histogram.png").is_file()
    assert str(manifest["summary"]) == "summary.md"
    assert str(manifest["metadata"]) == "metadata.json"

    parse_header = (
        output_root / "tables" / "parse_status_summary.csv"
    ).read_text(encoding="utf-8").splitlines()
    assert parse_header[0] == "parse_status,count,share"
    assert parse_header[1].startswith("ok,")

    duplicate_header = (
        output_root / "details" / "duplicate_groups.csv"
    ).read_text(encoding="utf-8").splitlines()[0]
    assert duplicate_header == (
        "normalized_latex,duplicate_count,expression_ids_sample,source_file_count,first_seen_expression_id"
    )

    duplicate_summary = (output_root / "tables" / "duplicate_summary.csv").read_text(encoding="utf-8")
    assert "max_duplicate_group_size" in duplicate_summary
    assert "max_duplicate_count" not in duplicate_summary
