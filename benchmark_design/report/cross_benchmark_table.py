"""Write cross-benchmark comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_design.config import CROSS_BENCHMARK_PROVENANCE, CROSS_BENCHMARK_REPORT_GROUPS
from benchmark_design.ocr.cross_benchmark import (
    CrossBenchmarkProfile,
    CrossBenchmarkRow,
    STRUCTURAL_DIFFICULTY_TIERS,
    compute_cross_benchmark_results,
    length_bin_rows_for_profile,
)
from benchmark_design.ocr.processing import EnrichedCorpus, ProcessingOptions
from benchmark_design.report.cross_benchmark_report_md import write_cross_benchmark_comparison_markdown


def _sanitize_provenance_source(source: str) -> str:
    if source.startswith("/"):
        return Path(source).name
    return source


def write_cross_benchmark_summary_csv(rows: list[CrossBenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "expression_count",
                "vocabulary_size",
                "total_token_count",
                "duplicate_rate",
                "mean_length",
                "p50_length",
                "p90_length",
                "max_length",
                "gini",
                "top_10_coverage",
                "unclassified_token_ratio",
                "rare_token_occurrence_ratio",
                "mean_structure_type_count",
                "mean_ast_depth",
                "parse_success_rate",
                "vocab_coverage",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    row.expression_count,
                    row.vocabulary_size,
                    row.total_token_count,
                    f"{row.duplicate_rate:.6f}",
                    f"{row.mean_length:.6f}",
                    f"{row.p50_length:.6f}",
                    f"{row.p90_length:.6f}",
                    row.max_length,
                    f"{row.gini:.6f}",
                    f"{row.top_10_coverage:.6f}",
                    f"{row.unknown_token_ratio:.6f}",
                    f"{row.rare_token_occurrence_ratio:.6f}",
                    f"{row.mean_structure_type_count:.6f}",
                    f"{row.mean_ast_depth:.6f}",
                    f"{row.parse_success_rate:.6f}",
                    f"{row.vocab_coverage:.6f}",
                ]
            )


def write_cross_benchmark_profiles_csv(profiles: list[CrossBenchmarkProfile], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "expression_count",
                "unique_expression_count",
                "duplicate_rate",
                "vocabulary_size",
                "parse_success_rate",
                "mean_length",
                "p50_length",
                "p90_length",
                "max_length",
                "gini",
                "top_10_coverage",
                "top_50_coverage",
                "top_100_coverage",
                "rare_1_vocab_ratio",
                "rare_5_vocab_ratio",
                "rare_10_expression_ratio",
                "structural_expression_ratio",
                "mean_structure_type_count",
                "multi_structure_ratio_ge_2",
                "multi_structure_ratio_ge_3",
                "matrix_structure_ratio",
                "mean_ast_depth",
                "p50_ast_depth",
                "p90_ast_depth",
                "max_ast_depth",
                "ast_ge_3_ratio",
                "count_gt_40_tokens",
                "count_gt_80_tokens",
                "count_ast_ge_3",
                "ast_depth_count_0",
                "ast_depth_count_1",
                "ast_depth_count_2",
                "ast_depth_count_3",
                "ast_depth_count_4",
                "ast_depth_count_ge_5",
                "count_gt_40_and_ast_ge_2",
                "count_gt_80_and_ast_ge_3",
                "count_multi_struct_ge_3",
                "total_token_count",
                "english_token_count",
                "digit_token_count",
                "greek_token_count",
                "special_symbol_token_count",
                "operator_token_count",
                "grouping_token_count",
                "structural_token_count",
                "cjk_token_count",
                "other_unknown_token_count",
                "english_token_ratio",
                "digit_token_ratio",
                "greek_token_ratio",
                "special_symbol_token_ratio",
                "operator_token_ratio",
                "grouping_token_ratio",
                "structural_token_ratio",
                "cjk_token_ratio",
                "other_unknown_token_ratio",
                "structural_difficulty_l1_count",
                "structural_difficulty_l2_count",
                "structural_difficulty_l3_count",
                "structural_difficulty_l4_count",
                "structural_difficulty_l1_ratio",
                "structural_difficulty_l2_ratio",
                "structural_difficulty_l3_ratio",
                "structural_difficulty_l4_ratio",
            ]
        )
        for profile in profiles:
            tier_ratios = [
                count / profile.expression_count if profile.expression_count else 0.0
                for count in profile.structural_difficulty_counts
            ]
            writer.writerow(
                [
                    profile.display_name,
                    profile.expression_count,
                    profile.unique_expression_count,
                    f"{profile.duplicate_rate:.6f}",
                    profile.vocabulary_size,
                    f"{profile.parse_success_rate:.6f}",
                    f"{profile.mean_length:.6f}",
                    f"{profile.p50_length:.6f}",
                    f"{profile.p90_length:.6f}",
                    profile.max_length,
                    f"{profile.gini:.6f}",
                    f"{profile.top_10_coverage:.6f}",
                    f"{profile.top_50_coverage:.6f}",
                    f"{profile.top_100_coverage:.6f}",
                    f"{profile.rare_1_vocab_ratio:.6f}",
                    f"{profile.rare_5_vocab_ratio:.6f}",
                    f"{profile.rare_10_expression_ratio:.6f}",
                    f"{profile.structural_expression_ratio:.6f}",
                    f"{profile.mean_structure_type_count:.6f}",
                    f"{profile.multi_structure_ratio_ge_2:.6f}",
                    f"{profile.multi_structure_ratio_ge_3:.6f}",
                    f"{profile.matrix_structure_ratio:.6f}",
                    f"{profile.mean_ast_depth:.6f}",
                    f"{profile.p50_ast_depth:.6f}",
                    f"{profile.p90_ast_depth:.6f}",
                    profile.max_ast_depth,
                    f"{profile.ast_ge_3_ratio:.6f}",
                    profile.count_gt_40_tokens,
                    profile.count_gt_80_tokens,
                    profile.count_ast_ge_3,
                    profile.ast_depth_counts[0],
                    profile.ast_depth_counts[1],
                    profile.ast_depth_counts[2],
                    profile.ast_depth_counts[3],
                    profile.ast_depth_counts[4],
                    profile.ast_depth_counts[5],
                    profile.count_gt_40_and_ast_ge_2,
                    profile.count_gt_80_and_ast_ge_3,
                    profile.count_multi_struct_ge_3,
                    profile.total_token_count,
                    profile.taxonomy_token_counts[0],
                    profile.taxonomy_token_counts[1],
                    profile.taxonomy_token_counts[2],
                    profile.taxonomy_token_counts[3],
                    profile.taxonomy_token_counts[4],
                    profile.taxonomy_token_counts[5],
                    profile.taxonomy_token_counts[6],
                    profile.taxonomy_token_counts[7],
                    profile.taxonomy_token_counts[8],
                    f"{profile.english_token_ratio:.6f}",
                    f"{profile.digit_token_ratio:.6f}",
                    f"{profile.greek_token_ratio:.6f}",
                    f"{profile.special_symbol_token_ratio:.6f}",
                    f"{profile.operator_token_ratio:.6f}",
                    f"{profile.grouping_token_ratio:.6f}",
                    f"{profile.structural_token_ratio:.6f}",
                    f"{profile.cjk_token_ratio:.6f}",
                    f"{profile.other_unknown_token_ratio:.6f}",
                    profile.structural_difficulty_counts[0],
                    profile.structural_difficulty_counts[1],
                    profile.structural_difficulty_counts[2],
                    profile.structural_difficulty_counts[3],
                    f"{tier_ratios[0]:.6f}",
                    f"{tier_ratios[1]:.6f}",
                    f"{tier_ratios[2]:.6f}",
                    f"{tier_ratios[3]:.6f}",
                ]
            )


def write_cross_benchmark_length_bins_csv(
    profiles: list[CrossBenchmarkProfile],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "length_bin", "count", "share"])
        for profile in profiles:
            for dataset, label, count, share in length_bin_rows_for_profile(profile):
                writer.writerow([dataset, label, count, f"{share:.6f}"])


def write_cross_benchmark_structural_difficulty_csv(
    profiles: list[CrossBenchmarkProfile],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "structural_difficulty", "count", "share"])
        for profile in profiles:
            for tier, count in zip(
                STRUCTURAL_DIFFICULTY_TIERS,
                profile.structural_difficulty_counts,
                strict=True,
            ):
                share = count / profile.expression_count if profile.expression_count else 0.0
                writer.writerow([profile.display_name, tier, count, f"{share:.6f}"])


def write_cross_benchmark_tokenizer_coverage_csv(rows: list[CrossBenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "unclassified_token_ratio", "vocab_coverage", "parse_success_rate"])
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    f"{row.unknown_token_ratio:.6f}",
                    f"{row.vocab_coverage:.6f}",
                    f"{row.parse_success_rate:.6f}",
                ]
            )


def write_cross_benchmark_provenance_csv(
    provenance_rows: tuple[dict[str, str], ...],
    output_path: Path,
    *,
    dataset_names: list[str],
) -> None:
    selected = {row["dataset"] for row in provenance_rows if row["dataset"] in dataset_names}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["dataset", "version_or_year", "split", "source", "license_or_access", "preprocessing_note"]
        )
        for row in provenance_rows:
            if row["dataset"] not in selected:
                continue
            writer.writerow(
                [
                    row["dataset"],
                    row["version_or_year"],
                    row["split"],
                    _sanitize_provenance_source(row["source"]),
                    row["license_or_access"],
                    row["preprocessing_note"],
                ]
            )


def write_cross_benchmark_report(
    output_dir: Path,
    *,
    dataset_names: list[str] | None = None,
    processing: ProcessingOptions | None = None,
    corpus_cache: dict[str, EnrichedCorpus] | None = None,
) -> dict[str, Path]:
    processing = processing or ProcessingOptions()
    raw_names = dataset_names or sorted(
        {name for sources in CROSS_BENCHMARK_REPORT_GROUPS.values() for name in sources}
    )
    profiles, raw_rows = compute_cross_benchmark_results(
        processing=processing,
        dataset_names=raw_names,
        corpus_cache=corpus_cache,
    )
    cross_dir = output_dir / "cross_benchmark"
    cross_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": cross_dir / "cross_benchmark_summary.csv",
        "profiles_csv": cross_dir / "cross_benchmark_profiles.csv",
        "summary_md": output_dir / "cross_benchmark_summary.md",
        "length_bins_csv": cross_dir / "cross_benchmark_length_bins.csv",
        "structural_difficulty_csv": cross_dir / "cross_benchmark_structural_difficulty.csv",
        "tokenizer_coverage_csv": cross_dir / "cross_benchmark_tokenizer_coverage.csv",
        "provenance_csv": cross_dir / "cross_benchmark_provenance.csv",
    }
    write_cross_benchmark_summary_csv(raw_rows, paths["summary_csv"])
    write_cross_benchmark_profiles_csv(profiles, paths["profiles_csv"])
    write_cross_benchmark_comparison_markdown(profiles, paths["summary_md"])
    write_cross_benchmark_length_bins_csv(profiles, paths["length_bins_csv"])
    write_cross_benchmark_structural_difficulty_csv(profiles, paths["structural_difficulty_csv"])
    write_cross_benchmark_tokenizer_coverage_csv(raw_rows, paths["tokenizer_coverage_csv"])
    write_cross_benchmark_provenance_csv(CROSS_BENCHMARK_PROVENANCE, paths["provenance_csv"], dataset_names=raw_names)
    return paths
