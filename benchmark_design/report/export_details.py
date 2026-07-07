"""Write detailed benchmark CSV exports derived from expression features."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_design.ocr.duplicates import duplicate_feature_groups
from benchmark_design.ocr.expression_features import ExpressionFeatures, RARE_THRESHOLDS, resolve_token_counter
from benchmark_design.ocr.structure_distribution import STRUCTURE_TYPES
from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token
from benchmark_design.report.output_layout import relativize_source_file


def write_expression_level_statistics_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    input_dir: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "expression_id",
                "dataset",
                "source_file",
                "line_id",
                "normalized_latex",
                "token_sequence",
                "token_length",
                "length_bin",
                "is_duplicate",
                "duplicate_group_id",
                "duplicate_count",
                "token_type_counts",
                "has_rare_1",
                "has_rare_5",
                "has_rare_10",
                "structure_types",
                "structure_type_count",
                "structure_max_depths",
                "ast_depth",
                "parse_status",
            ]
        )
        for feature in features:
            writer.writerow(
                [
                    feature.expression_id,
                    feature.dataset,
                    relativize_source_file(feature.source_file, input_dir=input_dir),
                    feature.line_id,
                    feature.normalized_latex,
                    " ".join(feature.token_sequence),
                    feature.token_length,
                    feature.length_bin,
                    int(feature.is_duplicate),
                    feature.duplicate_group_id,
                    feature.duplicate_count,
                    feature.token_type_counts_json(),
                    int(feature.has_rare_1),
                    int(feature.has_rare_5),
                    int(feature.has_rare_10),
                    feature.structure_types_str(),
                    feature.structure_type_count,
                    feature.structure_max_depths_json(),
                    feature.ast_depth,
                    feature.parse_status,
                ]
            )


def write_duplicate_groups_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    groups = duplicate_feature_groups(features)
    first_seen: dict[str, ExpressionFeatures] = {}
    for feature in features:
        if feature.normalized_latex not in first_seen:
            first_seen[feature.normalized_latex] = feature

    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "normalized_latex",
                "duplicate_count",
                "expression_ids_sample",
                "source_file_count",
                "first_seen_expression_id",
            ]
        )
        for normalized_latex, items in ranked:
            duplicate_count = len(items)
            sample_ids = "|".join(item.expression_id for item in items[:5])
            source_file_count = len({item.source_file for item in items if item.source_file})
            writer.writerow(
                [
                    normalized_latex,
                    duplicate_count,
                    sample_ids,
                    source_file_count,
                    first_seen[normalized_latex].expression_id,
                ]
            )


def write_top_duplicate_expressions_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    groups = duplicate_feature_groups(features)
    ranked = sorted(groups.values(), key=lambda items: (-len(items), items[0].normalized_latex))[:100]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "duplicate_count", "normalized_latex", "expression_ids"])
        for rank, items in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    len(items),
                    items[0].normalized_latex,
                    "|".join(item.expression_id for item in items),
                ]
            )


def write_length_distribution_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    counter = Counter(feature.token_length for feature in features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["length_value", "expression_count"])
        for length_value in sorted(counter):
            writer.writerow([length_value, counter[length_value]])


def write_longest_expressions_csv(features: list[ExpressionFeatures], output_path: Path, *, top_n: int = 100) -> None:
    ranked = sorted(features, key=lambda feature: (-feature.token_length, feature.expression_id))[:top_n]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "expression_id", "token_length", "length_bin", "normalized_latex"])
        for rank, feature in enumerate(ranked, start=1):
            writer.writerow([rank, feature.expression_id, feature.token_length, feature.length_bin, feature.normalized_latex])


def write_long_bin_examples_csv(features: list[ExpressionFeatures], output_path: Path, *, per_bin: int = 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["length_bin", "expression_id", "token_length", "normalized_latex"])
        for label in ("41-80 tokens", "> 80 tokens"):
            candidates = [feature for feature in features if feature.length_bin == label]
            candidates.sort(key=lambda feature: -feature.token_length)
            for feature in candidates[:per_bin]:
                writer.writerow([label, feature.expression_id, feature.token_length, feature.normalized_latex])


def write_token_frequency_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    total = sum(counter.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "token", "count", "share", "category", "is_unknown"])
        for rank, (token, count) in enumerate(counter.most_common(), start=1):
            category = classify_token(token)
            writer.writerow(
                [
                    rank,
                    token,
                    count,
                    f"{count / total:.6f}" if total else "0",
                    category.value,
                    int(category is TokenCategory.OTHER),
                ]
            )


def write_rare_tokens_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rare_threshold", "token", "count"])
        for threshold in RARE_THRESHOLDS:
            for token, count in sorted(counter.items(), key=lambda item: (item[1], item[0])):
                if count <= threshold:
                    writer.writerow([threshold, token, count])


def write_structure_expression_stats_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["expression_id", "structure_types", "structure_type_count", "structure_max_depths", "ast_depth", "parse_status"]
        )
        for feature in features:
            writer.writerow(
                [
                    feature.expression_id,
                    feature.structure_types_str(),
                    feature.structure_type_count,
                    feature.structure_max_depths_json(),
                    feature.ast_depth,
                    feature.parse_status,
                ]
            )


def write_structure_cooccurrence_matrix_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    type_names = [spec.structure_type for spec in STRUCTURE_TYPES]
    counts = Counter[tuple[str, str]]()
    for feature in features:
        present = list(feature.structure_types)
        for left in present:
            for right in present:
                counts[(left, right)] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure_type_a", "structure_type_b", "cooccurrence_count", "cooccurrence_ratio"])
        expression_count = len(features)
        for left in type_names:
            for right in type_names:
                count = counts.get((left, right), 0)
                writer.writerow(
                    [
                        left,
                        right,
                        count,
                        f"{count / expression_count:.6f}" if expression_count else "0",
                    ]
                )


def write_structure_patterns_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    pattern_counter = Counter(feature.structure_types for feature in features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure_pattern", "structure_type_count", "expression_count", "share"])
        total = len(features)
        for pattern, count in pattern_counter.most_common():
            writer.writerow(["|".join(pattern), len(pattern), count, f"{count / total:.6f}" if total else "0"])


def write_ast_depth_distribution_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    counter = Counter(feature.ast_depth for feature in features)
    total = len(features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ast_depth", "expression_count", "share"])
        for depth in sorted(counter):
            writer.writerow([depth, counter[depth], f"{counter[depth] / total:.6f}" if total else "0"])


def write_all_details(
    features: list[ExpressionFeatures],
    details_dir: Path,
    tables_dir: Path,
    *,
    input_dir: Path | None = None,
    token_counter: Counter[str] | None = None,
) -> dict[str, Path]:
    details_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    resolved_counter = resolve_token_counter(features, token_counter)
    paths = {
        "expression_level_statistics": details_dir / "expression_level_statistics.csv",
        "duplicate_groups": details_dir / "duplicate_groups.csv",
        "top_duplicate_expressions": details_dir / "top_duplicate_expressions.csv",
        "longest_expressions": details_dir / "longest_expressions.csv",
        "long_bin_examples": details_dir / "long_bin_examples.csv",
        "token_frequency": details_dir / "token_frequency.csv",
        "rare_tokens": details_dir / "rare_tokens.csv",
        "structure_expression_stats": details_dir / "structure_expression_stats.csv",
    }
    write_expression_level_statistics_csv(
        features,
        paths["expression_level_statistics"],
        input_dir=input_dir,
    )
    write_duplicate_groups_csv(features, paths["duplicate_groups"])
    write_top_duplicate_expressions_csv(features, paths["top_duplicate_expressions"])
    write_longest_expressions_csv(features, paths["longest_expressions"])
    write_long_bin_examples_csv(features, paths["long_bin_examples"])
    write_token_frequency_csv(features, paths["token_frequency"], token_counter=resolved_counter)
    write_rare_tokens_csv(features, paths["rare_tokens"], token_counter=resolved_counter)
    write_structure_expression_stats_csv(features, paths["structure_expression_stats"])
    return paths
