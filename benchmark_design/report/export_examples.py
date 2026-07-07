"""Write example CSV samples for manual audit."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from benchmark_design.ocr.duplicates import duplicate_feature_groups
from benchmark_design.ocr.expression_features import ExpressionFeatures, RARE_THRESHOLDS, resolve_token_counter
from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token

RARE_EXAMPLE_THRESHOLD = 10


def write_longest_expressions_csv(features: list[ExpressionFeatures], output_path: Path, *, top_n: int = 50) -> None:
    ranked = sorted(features, key=lambda feature: (-feature.token_length, feature.expression_id))[:top_n]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "expression_id", "normalized_latex", "token_length", "structure_type_count", "ast_depth"])
        for rank, feature in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    feature.expression_id,
                    feature.normalized_latex,
                    feature.token_length,
                    feature.structure_type_count,
                    feature.ast_depth,
                ]
            )


def write_deepest_ast_examples_csv(features: list[ExpressionFeatures], output_path: Path, *, top_n: int = 50) -> None:
    ranked = sorted(
        features,
        key=lambda feature: (-feature.ast_depth, -feature.token_length, feature.expression_id),
    )[:top_n]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "expression_id", "normalized_latex", "token_length", "structure_type_count", "ast_depth"])
        for rank, feature in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    feature.expression_id,
                    feature.normalized_latex,
                    feature.token_length,
                    feature.structure_type_count,
                    feature.ast_depth,
                ]
            )


def write_duplicate_group_examples_csv(features: list[ExpressionFeatures], output_path: Path, *, top_n: int = 50) -> None:
    groups = duplicate_feature_groups(features)
    first_seen = {
        feature.normalized_latex: feature
        for feature in features
        if feature.normalized_latex in groups
    }

    ranked = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "normalized_latex",
                "duplicate_count",
                "expression_ids_sample",
                "source_file_count",
                "token_length",
                "structure_type_count",
                "ast_depth",
            ]
        )
        for rank, (normalized_latex, items) in enumerate(ranked, start=1):
            representative = first_seen[normalized_latex]
            writer.writerow(
                [
                    rank,
                    normalized_latex,
                    len(items),
                    "|".join(item.expression_id for item in items[:5]),
                    len({item.source_file for item in items if item.source_file}),
                    representative.token_length,
                    representative.structure_type_count,
                    representative.ast_depth,
                ]
            )


def write_rare_token_examples_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    threshold: int = RARE_EXAMPLE_THRESHOLD,
    top_n: int = 50,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    rare_tokens = {token for token, count in counter.items() if count <= threshold}

    def rare_info(feature: ExpressionFeatures) -> tuple[list[str], list[int]]:
        tokens = sorted({token for token in feature.token_sequence if token in rare_tokens})
        return tokens, [counter[token] for token in tokens]

    candidates = [feature for feature in features if any(token in rare_tokens for token in feature.token_sequence)]
    ranked = sorted(
        candidates,
        key=lambda feature: (
            -len(rare_info(feature)[0]),
            -feature.token_length,
            feature.expression_id,
        ),
    )[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "expression_id",
                "rare_tokens",
                "rare_token_counts",
                "rare_threshold",
                "normalized_latex",
                "token_length",
            ]
        )
        for rank, feature in enumerate(ranked, start=1):
            tokens, counts = rare_info(feature)
            writer.writerow(
                [
                    rank,
                    feature.expression_id,
                    "|".join(tokens),
                    "|".join(str(count) for count in counts),
                    threshold,
                    feature.normalized_latex,
                    feature.token_length,
                ]
            )


def write_unknown_token_examples_csv(features: list[ExpressionFeatures], output_path: Path, *, top_n: int = 50) -> None:
    def unknown_tokens(feature: ExpressionFeatures) -> list[str]:
        return sorted({token for token in feature.token_sequence if classify_token(token) is TokenCategory.OTHER})

    candidates = [feature for feature in features if unknown_tokens(feature)]
    ranked = sorted(
        candidates,
        key=lambda feature: (
            -len(unknown_tokens(feature)),
            -feature.token_length,
            feature.expression_id,
        ),
    )[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "expression_id",
                "unknown_tokens",
                "unknown_token_count",
                "normalized_latex",
                "token_length",
            ]
        )
        for rank, feature in enumerate(ranked, start=1):
            tokens = unknown_tokens(feature)
            writer.writerow(
                [
                    rank,
                    feature.expression_id,
                    "|".join(tokens),
                    len(tokens),
                    feature.normalized_latex,
                    feature.token_length,
                ]
            )


def write_high_structure_examples_csv(features: list[ExpressionFeatures], output_path: Path, *, top_n: int = 50) -> None:
    ranked = sorted(
        features,
        key=lambda feature: (-feature.structure_type_count, -feature.token_length, feature.expression_id),
    )[:top_n]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "expression_id",
                "structure_type_count",
                "structure_types",
                "structure_max_depths",
                "ast_depth",
                "token_length",
                "normalized_latex",
            ]
        )
        for rank, feature in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    feature.expression_id,
                    feature.structure_type_count,
                    feature.structure_types_str(),
                    feature.structure_max_depths_json(),
                    feature.ast_depth,
                    feature.token_length,
                    feature.normalized_latex,
                ]
            )


def write_all_examples(
    features: list[ExpressionFeatures],
    examples_dir: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> dict[str, Path]:
    examples_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "longest_expressions": examples_dir / "longest_expressions.csv",
        "high_structure_examples": examples_dir / "high_structure_examples.csv",
        "deepest_ast_examples": examples_dir / "deepest_ast_examples.csv",
        "unknown_token_examples": examples_dir / "unknown_token_examples.csv",
        "duplicate_group_examples": examples_dir / "duplicate_group_examples.csv",
        "rare_token_examples": examples_dir / "rare_token_examples.csv",
    }
    write_longest_expressions_csv(features, paths["longest_expressions"])
    write_high_structure_examples_csv(features, paths["high_structure_examples"])
    write_deepest_ast_examples_csv(features, paths["deepest_ast_examples"])
    write_unknown_token_examples_csv(features, paths["unknown_token_examples"])
    write_duplicate_group_examples_csv(features, paths["duplicate_group_examples"])
    write_rare_token_examples_csv(features, paths["rare_token_examples"], token_counter=token_counter)
    return paths
