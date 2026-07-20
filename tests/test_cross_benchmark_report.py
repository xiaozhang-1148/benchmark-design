"""Cross-benchmark comparison report tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_design.config import CROSS_BENCHMARK_SETS
from benchmark_design.ocr.cross_benchmark import CrossBenchmarkProfile, LengthBinStat, compute_cross_benchmark_profiles
from benchmark_design.ocr.processing import ProcessingOptions
from benchmark_design.report.cross_benchmark_report_md import (
    _structure_within_structured_share,
    build_cross_benchmark_comparison_markdown,
)

FULL_BENCHMARK = CROSS_BENCHMARK_SETS["ours"]
HME100K = CROSS_BENCHMARK_SETS["HME100K"]


def _make_profile(**overrides: object) -> CrossBenchmarkProfile:
    defaults = {
        "display_name": "Sample",
        "expression_count": 100,
        "unique_expression_count": 80,
        "duplicate_rate": 0.2,
        "vocabulary_size": 50,
        "parse_success_rate": 1.0,
        "mean_length": 10.0,
        "p50_length": 9.0,
        "p90_length": 20.0,
        "p95_length": 25.0,
        "p99_length": 30.0,
        "max_length": 40,
        "length_bins": (LengthBinStat(label="1-10", count=60, share=0.6),),
        "gini": 0.5,
        "top_10_coverage": 0.4,
        "top_50_coverage": 0.7,
        "top_100_coverage": 0.8,
        "rare_1_vocab_ratio": 0.1,
        "rare_5_vocab_ratio": 0.2,
        "rare_10_expression_ratio": 0.3,
        "structural_expression_ratio": 0.65,
        "mean_structure_type_count": 1.5,
        "multi_structure_ratio_ge_2": 0.2,
        "multi_structure_ratio_ge_3": 0.1,
        "matrix_structure_ratio": 0.05,
        "mean_ast_depth": 1.0,
        "p50_ast_depth": 1.0,
        "p90_ast_depth": 2.0,
        "max_ast_depth": 3,
        "ast_ge_3_ratio": 0.05,
        "count_gt_40_tokens": 1,
        "count_gt_80_tokens": 0,
        "count_any_structure": 65,
        "count_multi_struct_ge_3": 5,
        "structure_type_count_bins": (40, 15, 7, 3),
        "count_matrix_structure": 4,
        "count_ast_ge_3": 5,
        "ast_depth_counts": (35, 40, 20, 4, 1, 0),
        "count_gt_40_and_ast_ge_2": 1,
        "count_gt_80_and_ast_ge_3": 0,
        "total_token_count": 1000,
        "taxonomy_token_counts": (100, 100, 100, 100, 100, 100, 100, 0),
        "latin_variable_token_ratio": 0.1,
        "digit_token_ratio": 0.1,
        "special_symbol_token_ratio": 0.1,
        "operator_token_ratio": 0.1,
        "grouping_token_ratio": 0.1,
        "structural_token_ratio": 0.1,
        "cjk_token_ratio": 0.1,
        "other_unknown_token_ratio": 0.0,
        "notes": "",
        "structural_difficulty_counts": (50, 20, 20, 10),
    }
    defaults.update(overrides)
    return CrossBenchmarkProfile(**defaults)  # type: ignore[arg-type]


def test_structure_within_structured_share_helper() -> None:
    assert _structure_within_structured_share(1319, 2444) == pytest.approx(1319 / 2444)
    assert _structure_within_structured_share(0, 0) == 0.0


def test_structure_combination_uses_structured_denominator() -> None:
    profile = _make_profile(
        display_name="Ours",
        expression_count=100,
        count_any_structure=65,
        structure_type_count_bins=(40, 15, 7, 3),
        count_matrix_structure=4,
    )
    text = build_cross_benchmark_comparison_markdown([profile])

    assert "| Ours | 65 (65.00%) | 40 (61.54%) | 15 (23.08%) | 7 (10.77%) | 3 (4.62%) | 4 (6.15%) |" in text


def test_structural_difficulty_section_in_markdown() -> None:
    profiles = [
        _make_profile(display_name="Sample"),
        _make_profile(display_name="Ours"),
    ]
    text = build_cross_benchmark_comparison_markdown(profiles)
    assert "## 7. Expression-level Structural Difficulty (L1–L4)" in text
    assert "| Sample | 50 (50.00%) | 20 (20.00%) | 20 (20.00%) | 10 (10.00%) |" in text


@pytest.mark.integration
@pytest.mark.skipif(not FULL_BENCHMARK.is_dir(), reason="full benchmark dataset unavailable")
def test_cross_benchmark_profiles_ours_values() -> None:
    profiles = compute_cross_benchmark_profiles(processing=ProcessingOptions(show_progress=False))
    ours = next(profile for profile in profiles if profile.display_name == "Ours")
    assert ours.expression_count == 152_012
    assert ours.unique_expression_count == 120_322
    assert ours.duplicate_rate == pytest.approx(0.208996, rel=1e-4)
    assert ours.vocabulary_size == 1_005
    assert ours.mean_length == pytest.approx(23.36, rel=1e-2)
    assert ours.count_gt_80_tokens == 2_380
    assert ours.count_multi_struct_ge_3 == 11_949


@pytest.mark.integration
@pytest.mark.skipif(not HME100K.is_dir(), reason="HME100K dataset unavailable")
def test_cross_benchmark_comparison_markdown_structure(tmp_path: Path) -> None:
    profiles = compute_cross_benchmark_profiles(processing=ProcessingOptions(show_progress=False))
    text = build_cross_benchmark_comparison_markdown(profiles)

    assert text.startswith("# Cross-Benchmark Comparison\n")
    assert "## 1. Dataset Scale and Effective Diversity" in text
    assert "## 10. Summary of Advantages" in text
    assert "## 5. Structure Combination" in text
    assert "Matrix/Layout" not in text
    assert "1 Structure Type" in text
    assert "Any Structure ≥1" in text
    assert "≥4 Structure Types" in text
    assert "Env." in text
    assert "Do **not** sum **Env.**" in text
    assert "Multi-Struct ≥2" not in text
    assert "## 6. AST Depth Distribution" in text
    assert "## 7. Expression-level Structural Difficulty (L1–L4)" in text
    assert "depth 0" in text
    assert "count (share%)" in text
    assert "## 8. Token Taxonomy Composition" in text
    assert "count (share%)" in text
    assert "Latin variable" in text
    assert "Mixed Text-Math Expr. Ratio" not in text
    assert "| Ours | 152,113 |" in text or "| Ours |" in text
    assert "TBD" not in text
    assert "| CROHME |" in text
    assert "| MNE |" in text

    hme = next(profile for profile in profiles if profile.display_name == "HME100K")
    assert hme.expression_count == 99_109
    assert f"| HME100K | {hme.expression_count:,} |" in text
