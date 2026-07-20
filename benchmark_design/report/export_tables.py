"""Write refactored benchmark summary tables under ``tables/``."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_design.ocr.confusable_tokens import OcrConfusableTokenMetrics
from benchmark_design.ocr.consolidated import OcrConsolidatedMetrics
from benchmark_design.ocr.expression_content import compute_ocr_expression_content_from_token_sequences
from benchmark_design.ocr.expression_features import ExpressionFeatures, RARE_THRESHOLDS, resolve_token_counter
from benchmark_design.ocr.duplicates import DuplicateIndex, duplicate_stats_from_features
from benchmark_design.ocr.processing import EnrichedCorpus
from benchmark_design.ocr.token_taxonomy import (
    LAYOUT_ALIGNMENT_TOKENS,
    PUNCTUATION_TOKENS,
    TOKEN_CATEGORY_ORDER,
    TokenCategory,
    classify_token,
)
from benchmark_design.report.export_details import (
    write_length_distribution_csv,
    write_structure_cooccurrence_matrix_csv,
    write_structure_patterns_csv,
)
from benchmark_design.report.lbd_coordinate_table import (
    write_expression_lbd_coordinate_counts_csv,
    write_expression_structural_difficulty_counts_csv,
)
from benchmark_design.report.structure_complexity_table import write_structure_complexity_csv
from benchmark_design.report.structure_distribution_table import write_structure_distribution_csv

TOKEN_FREQUENCY_TOP_K = 100

BIN_EXPORT_LABELS: tuple[str, ...] = ("1-10", "11-20", "21-40", "41-80", ">80")

TAXONOMY_EXPORT_LABELS: dict[TokenCategory, str] = {
    TokenCategory.LATIN_VARIABLE: "latin variable",
    TokenCategory.DIGIT: "digit",
    TokenCategory.SPECIAL_SYMBOL: "special symbol",
    TokenCategory.OPERATOR: "operator",
    TokenCategory.GROUPING: "grouping",
    TokenCategory.STRUCTURAL: "structural",
    TokenCategory.CJK: "CJK",
    TokenCategory.PUNCTUATION: "punctuation",
    TokenCategory.LAYOUT_ALIGNMENT: "layout / alignment",
    TokenCategory.OTHER: "unclassified",
}

PUNCTUATION_LAYOUT_TOKENS: frozenset[str] = PUNCTUATION_TOKENS | LAYOUT_ALIGNMENT_TOKENS

CORE_TABLES: tuple[str, ...] = (
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
    "expression_lbd_coordinate_counts.csv",
    "expression_structural_difficulty_counts.csv",
)

APPENDIX_TABLES: tuple[str, ...] = (
    "length_distribution.csv",
    "rare_token_summary.csv",
    "structure_pattern_distribution.csv",
    "structure_cooccurrence_matrix.csv",
    "confusable_token_counts.csv",
)


def _write_metric_value_csv(rows: list[tuple[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in rows:
            if isinstance(value, float):
                writer.writerow([metric, f"{value:.6f}"])
            else:
                writer.writerow([metric, value])


def write_duplicate_summary_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    stats = duplicate_stats_from_features(features)
    rows = [
        ("expression_count", stats.expression_count),
        ("unique_normalized_latex_count", stats.unique_normalized_latex_count),
        ("redundant_expression_count", stats.redundant_expression_count),
        ("redundancy_rate", stats.duplicate_rate),
        ("duplicated_group_expression_count", stats.duplicated_group_expression_count),
        ("duplicated_group_expression_ratio", stats.duplicated_group_expression_ratio),
        ("duplicate_group_count", stats.duplicate_group_count),
        ("max_duplicate_group_size", stats.max_duplicate_group_size),
    ]
    _write_metric_value_csv(rows, output_path)


def write_duplicate_group_size_distribution_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    index = DuplicateIndex.from_normalized_latex_values(features)
    stats = index.stats()
    expression_count = stats.expression_count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["duplicate_count", "num_groups", "num_expressions", "expression_share"])
        group_size_counter = Counter(
            len(indices) for indices in index.groups.values() if len(indices) >= 2
        )
        for duplicate_count in sorted(group_size_counter):
            num_groups = group_size_counter[duplicate_count]
            num_expressions = duplicate_count * num_groups
            writer.writerow(
                [
                    duplicate_count,
                    num_groups,
                    num_expressions,
                    f"{num_expressions / expression_count:.6f}" if expression_count else "0",
                ]
            )


def write_length_summary_csv(metrics: OcrConsolidatedMetrics, output_path: Path) -> None:
    length = metrics.length
    rows = [
        ("mean", length.mean_length),
        ("std", length.std),
        ("cv", length.cv),
        ("p50", length.p50),
        ("p90", length.p90),
        ("max", length.max_length),
    ]
    _write_metric_value_csv(rows, output_path)


def write_token_long_tail_summary_csv(metrics: OcrConsolidatedMetrics, output_path: Path) -> None:
    longtail = metrics.longtail
    rows: list[tuple[str, float | int]] = [("gini", longtail.gini)]
    rows.extend((f"top-{k} coverage", coverage) for k, coverage in longtail.top_k_coverage)
    rows.extend((f"rare_{threshold} vocab ratio", ratio) for threshold, ratio in longtail.rare_vocab_ratio)
    rows.extend(
        (f"rare_{threshold} expression ratio", ratio) for threshold, ratio in longtail.rare_expression_ratio
    )
    _write_metric_value_csv(rows, output_path)


def write_dataset_scale_csv(enriched: EnrichedCorpus, metrics: OcrConsolidatedMetrics, output_path: Path) -> None:
    scale = metrics.scale
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "json_file_count",
                "expression_count",
                "total_token_count",
                "unique_normalized_latex_count",
                "vocabulary_size",
            ]
        )
        writer.writerow(
            [
                enriched.dataset,
                enriched.json_file_count,
                scale.expression_count,
                scale.total_token_count,
                scale.unique_normalized_latex_count,
                scale.vocabulary_size,
            ]
        )


def write_parse_status_summary_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    total = len(features)
    ok_count = sum(1 for feature in features if feature.parse_status == "ok")
    failed_count = total - ok_count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parse_status", "count", "share"])
        writer.writerow(["ok", ok_count, f"{ok_count / total:.6f}" if total else "0"])
        writer.writerow(["failed", failed_count, f"{failed_count / total:.6f}" if total else "0"])


def write_length_bins_export_csv(metrics: OcrConsolidatedMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["length_bin", "count", "share"])
        for export_label, (_, count, share) in zip(
            BIN_EXPORT_LABELS,
            metrics.bins.as_rows(),
            strict=True,
        ):
            writer.writerow([export_label, count, f"{share:.6f}"])


def write_token_taxonomy_composition_csv(metrics: OcrConsolidatedMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token_type", "count", "share"])
        for category in TOKEN_CATEGORY_ORDER:
            item = next(row for row in metrics.taxonomy.categories if row.category is category)
            writer.writerow(
                [
                    TAXONOMY_EXPORT_LABELS[category],
                    item.count,
                    f"{item.share:.6f}",
                ]
            )


def write_token_frequency_top100_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    top_k: int = TOKEN_FREQUENCY_TOP_K,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    total = sum(counter.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "token", "count", "share", "category"])
        for rank, (token, count) in enumerate(counter.most_common(top_k), start=1):
            category = TAXONOMY_EXPORT_LABELS[classify_token(token)]
            writer.writerow([rank, token, count, f"{count / total:.6f}" if total else "0", category])


def write_rare_token_summary_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    expression_count = len(features)
    expr_hits: dict[str, int] = defaultdict(int)
    for feature in features:
        for token in set(feature.token_sequence):
            expr_hits[token] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "token", "count", "category", "expression_hit_count", "expression_hit_ratio"])
        for threshold in RARE_THRESHOLDS:
            for token, count in sorted(counter.items(), key=lambda item: (item[1], item[0])):
                if count > threshold:
                    continue
                hits = expr_hits[token]
                writer.writerow(
                    [
                        threshold,
                        token,
                        count,
                        TAXONOMY_EXPORT_LABELS[classify_token(token)],
                        hits,
                        f"{hits / expression_count:.6f}" if expression_count else "0",
                    ]
                )


def write_punctuation_layout_token_summary_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    total = sum(counter.values())
    expression_count = len(features)
    expr_hits: dict[str, int] = defaultdict(int)
    for feature in features:
        for token in set(feature.token_sequence):
            if token in PUNCTUATION_LAYOUT_TOKENS:
                expr_hits[token] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "count", "share", "category", "expression_hit_count", "expression_hit_ratio"])
        for token in sorted(PUNCTUATION_LAYOUT_TOKENS):
            count = counter.get(token, 0)
            category = TAXONOMY_EXPORT_LABELS[classify_token(token)]
            hits = expr_hits.get(token, 0)
            writer.writerow(
                [
                    token,
                    count,
                    f"{count / total:.6f}" if total else "0",
                    category,
                    hits,
                    f"{hits / expression_count:.6f}" if expression_count else "0",
                ]
            )


def write_unclassified_token_summary_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    total = sum(counter.values())
    expression_count = len(features)
    expr_hits: dict[str, int] = defaultdict(int)
    for feature in features:
        for token in set(feature.token_sequence):
            if classify_token(token) is TokenCategory.OTHER:
                expr_hits[token] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "count", "share", "expression_hit_count"])
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if classify_token(token) is not TokenCategory.OTHER:
                continue
            hits = expr_hits[token]
            writer.writerow([token, count, f"{count / total:.6f}" if total else "0", hits])


def write_expression_content_summary_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
) -> None:
    metrics = compute_ocr_expression_content_from_token_sequences(
        feature.token_sequence for feature in features
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["content_type", "expression_count", "share"])
        for label, count, share in metrics.as_rows():
            writer.writerow([label, count, f"{share:.6f}"])


def write_confusable_token_group_summary_csv(
    metrics: OcrConfusableTokenMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "tier",
                "representative_tokens",
                "token_count",
                "token_ratio",
                "expression_count",
                "expression_ratio",
                "co_occurrence_expression_count",
                "dominant_tokens",
                "rare_side_tokens",
            ]
        )
        for group_metrics in metrics.primary_groups:
            writer.writerow(
                [
                    group_metrics.group.name,
                    group_metrics.group.tier.value,
                    group_metrics.group.representative_tokens,
                    group_metrics.token_count,
                    f"{group_metrics.token_ratio:.6f}",
                    group_metrics.expression_count,
                    f"{group_metrics.expression_ratio:.6f}",
                    group_metrics.co_occurrence_expression_count,
                    "|".join(group_metrics.dominant_tokens),
                    "|".join(group_metrics.rare_side_tokens),
                ]
            )


def write_confusable_token_counts_appendix_csv(
    metrics: OcrConfusableTokenMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "group",
                "tier",
                "token",
                "count",
                "share_of_group",
                "share_of_corpus",
            ]
        )
        for group_metrics in metrics.primary_groups:
            for token_count in group_metrics.token_counts:
                writer.writerow(
                    [
                        token_count.group,
                        group_metrics.group.tier.value,
                        token_count.token,
                        token_count.count,
                        f"{token_count.share_of_group:.6f}",
                        f"{token_count.share_of_corpus:.6f}",
                    ]
                )


def write_ast_depth_distribution_table_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    counter = Counter(feature.ast_depth for feature in features)
    total = len(features)
    max_depth = max(counter) if counter else 0
    upper = max(max_depth, 5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ast_depth", "expression_count", "share"])
        for depth in range(upper + 1):
            count = counter.get(depth, 0)
            writer.writerow([depth, count, f"{count / total:.6f}" if total else "0"])


def write_tables_readme(output_path: Path) -> None:
    lines = [
        "# Benchmark Tables",
        "",
        "Machine-readable summary tables for the OCR benchmark export.",
        "",
        "## Core (`tables/`)",
        "",
        "Primary tables referenced by `summary.md`, `ocr_benchmark_summary.md`, and figures.",
        "",
        "- `dataset_scale.csv` — corpus scale with `dataset` column.",
        "- `parse_status_summary.csv` — parse success / failure counts.",
        "- `length_summary.csv` / `length_bins.csv` — length statistics and fixed bins for the main histogram.",
        "- `duplicate_summary.csv` — duplicate = full `normalized_latex` exact match between expression records.",
        "- `duplicate_group_size_distribution.csv` — histogram of duplicate **group** sizes; rows with "
        "`duplicate_count >= 2` only (unique expressions are omitted).",
        "- `token_taxonomy_composition.csv` — token category composition.",
        "- `token_frequency_top100.csv` — top-100 token counts.",
        "- `token_long_tail_summary.csv` — Gini and top-k coverage.",
        "- `punctuation_layout_token_summary.csv` — punctuation / layout token audit.",
        "- `unclassified_token_summary.csv` — unclassified taxonomy residue (header-only when empty).",
        "- `structure_type_distribution.csv` / `structure_combination_summary.csv` — structure metrics.",
        "- `ast_depth_summary.csv` / `ast_depth_distribution.csv` — PosFormer AST depth summary and histogram source.",
        "- `expression_content_summary.csv` — pure latex_command / pure CJK / mixed expression counts.",
        "- `confusable_token_group_summary.csv` — Table 9 confusable token group metrics.",
        "- `expression_lbd_coordinate_counts.csv` — 27-cell L/B/D coordinate counts with structural difficulty tier.",
        "- `expression_structural_difficulty_counts.csv` — Expression-level Structural Difficulty tier counts.",
        "",
        "## Appendix (`tables/appendix/`)",
        "",
        "Detailed distributions and long-tail listings; not required for the main narrative.",
        "",
        "- `length_distribution.csv` — per-length expression counts (long-tail curve source).",
        "- `rare_token_summary.csv` — low-frequency token listing by threshold.",
        "- `structure_pattern_distribution.csv` — structure combination pattern detail.",
        "- `structure_cooccurrence_matrix.csv` — pairwise structure co-occurrence counts (appendix).",
        "- `figures/structure_cooccurrence_heatmap.png` — joint distribution of structure type count "
        "and maximum AST nesting depth in Ours (log-scaled counts; percentages over all instances).",
        "- `figures/lbd_coordinate_examples/<L1|L2|L3|L4>/` — up to 20 OCR crop examples per tier.",
        "- `figures/stc_high_complexity/` — top 20 NTC/CBC high-complexity expression crops.",
        "- `confusable_token_counts.csv` — Table 9 token-by-token counts for all confusable groups.",
        "- `examples/confusable_token_4_varphi_examples.csv` — 20 sample expressions for `4` and `\\varphi` (OCR length > 3).",
        "",
        "## Cross-benchmark (`cross_benchmark/`)",
        "",
        "Unified-tokenizer comparison across Ours, CROHME, HME100K, MathWriting, and MNE.",
        "Narrative: `cross_benchmark_summary.md` at the output root.",
        "",
        "- `cross_benchmark_profiles.csv` — full multi-metric profile table.",
        "- `cross_benchmark_summary.csv` — compact comparison.",
        "- `cross_benchmark_length_bins.csv` — length bins by dataset.",
        "- `cross_benchmark_structural_difficulty.csv` — L1–L4 structural difficulty tiers by dataset.",
        "- `cross_benchmark_tokenizer_coverage.csv` — `unclassified_token_ratio` and vocab coverage.",
        "- `cross_benchmark_provenance.csv` — external dataset sources (no absolute filesystem paths).",
        "",
        "## Resources (`resources/`)",
        "",
        "- `latex_vocab.csv` — `LATEX_DICT` vocabulary.",
        "- `token_taxonomy_map.csv` — token → category mapping for dictionary + corpus tokens.",
        "",
        "Human-readable summaries: output root (`summary.md`, `ocr_benchmark_summary.md`).",
        "Expression-level detail: `details/`; audit samples: `examples/`.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_tables(
    enriched: EnrichedCorpus,
    metrics: OcrConsolidatedMetrics,
    tables_dir: Path,
    *,
    appendix_dir: Path,
    token_counter: Counter[str] | None = None,
) -> dict[str, Path]:
    """Write the refactored ``tables/`` CSV set and return manifest paths."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    features = list(enriched.features)
    resolved_counter = resolve_token_counter(features, token_counter)
    paths = {
        "dataset_scale": tables_dir / "dataset_scale.csv",
        "parse_status_summary": tables_dir / "parse_status_summary.csv",
        "length_summary": tables_dir / "length_summary.csv",
        "length_bins": tables_dir / "length_bins.csv",
        "length_distribution": appendix_dir / "length_distribution.csv",
        "duplicate_summary": tables_dir / "duplicate_summary.csv",
        "duplicate_group_size_distribution": tables_dir / "duplicate_group_size_distribution.csv",
        "token_taxonomy_composition": tables_dir / "token_taxonomy_composition.csv",
        "token_frequency_top100": tables_dir / "token_frequency_top100.csv",
        "token_long_tail_summary": tables_dir / "token_long_tail_summary.csv",
        "rare_token_summary": appendix_dir / "rare_token_summary.csv",
        "punctuation_layout_token_summary": tables_dir / "punctuation_layout_token_summary.csv",
        "unclassified_token_summary": tables_dir / "unclassified_token_summary.csv",
        "structure_type_distribution": tables_dir / "structure_type_distribution.csv",
        "structure_combination_summary": tables_dir / "structure_combination_summary.csv",
        "structure_pattern_distribution": appendix_dir / "structure_pattern_distribution.csv",
        "structure_cooccurrence_matrix": appendix_dir / "structure_cooccurrence_matrix.csv",
        "ast_depth_distribution": tables_dir / "ast_depth_distribution.csv",
        "expression_content_summary": tables_dir / "expression_content_summary.csv",
        "confusable_token_group_summary": tables_dir / "confusable_token_group_summary.csv",
        "expression_lbd_coordinate_counts": tables_dir / "expression_lbd_coordinate_counts.csv",
        "expression_structural_difficulty_counts": tables_dir / "expression_structural_difficulty_counts.csv",
        "confusable_token_counts": appendix_dir / "confusable_token_counts.csv",
        "readme": tables_dir / "README.md",
    }

    write_dataset_scale_csv(enriched, metrics, paths["dataset_scale"])
    write_parse_status_summary_csv(features, paths["parse_status_summary"])
    write_length_summary_csv(metrics, paths["length_summary"])
    write_length_bins_export_csv(metrics, paths["length_bins"])
    write_length_distribution_csv(features, paths["length_distribution"])
    write_duplicate_summary_csv(features, paths["duplicate_summary"])
    write_duplicate_group_size_distribution_csv(features, paths["duplicate_group_size_distribution"])
    write_token_taxonomy_composition_csv(metrics, paths["token_taxonomy_composition"])
    write_token_frequency_top100_csv(features, paths["token_frequency_top100"], token_counter=resolved_counter)
    write_token_long_tail_summary_csv(metrics, paths["token_long_tail_summary"])
    write_rare_token_summary_csv(features, paths["rare_token_summary"], token_counter=resolved_counter)
    write_punctuation_layout_token_summary_csv(
        features,
        paths["punctuation_layout_token_summary"],
        token_counter=resolved_counter,
    )
    write_unclassified_token_summary_csv(
        features,
        paths["unclassified_token_summary"],
        token_counter=resolved_counter,
    )
    write_structure_distribution_csv(metrics.structure, paths["structure_type_distribution"])
    write_structure_complexity_csv(metrics.complexity, paths["structure_combination_summary"])
    write_structure_patterns_csv(features, paths["structure_pattern_distribution"])
    write_structure_cooccurrence_matrix_csv(features, paths["structure_cooccurrence_matrix"])
    write_ast_depth_distribution_table_csv(features, paths["ast_depth_distribution"])
    write_expression_content_summary_csv(features, paths["expression_content_summary"])
    write_confusable_token_group_summary_csv(metrics.confusable, paths["confusable_token_group_summary"])
    write_expression_lbd_coordinate_counts_csv(features, paths["expression_lbd_coordinate_counts"])
    write_expression_structural_difficulty_counts_csv(
        features,
        paths["expression_structural_difficulty_counts"],
    )
    write_confusable_token_counts_appendix_csv(metrics.confusable, paths["confusable_token_counts"])
    write_tables_readme(paths["readme"])

    return paths
