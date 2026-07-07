"""Per-expression feature extraction for export and cross-benchmark analysis."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.duplicates import DuplicateIndex, normalize_expression_latex
from benchmark_design.ocr.length_bin_specs import assign_length_bin
from benchmark_design.ocr.parse_validate import validate_parse_status
from benchmark_design.ocr.position_forest import encode_position_forest_tokens
from benchmark_design.ocr.structure_distribution import (
    STRUCTURE_TYPES,
    max_structure_depth,
    structure_types_present_in_tokens,
)
from benchmark_design.ocr.token_taxonomy import TOKEN_CATEGORY_ORDER, TokenCategory, classify_token

RARE_THRESHOLDS: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True, slots=True)
class ExpressionFeatures:
    expression_id: str
    dataset: str
    source_file: str
    line_id: str
    normalized_latex: str
    token_sequence: tuple[str, ...]
    token_length: int
    length_bin: str
    is_duplicate: bool
    duplicate_group_id: int
    duplicate_count: int
    token_type_counts: dict[str, int]
    has_rare_1: bool
    has_rare_5: bool
    has_rare_10: bool
    structure_types: tuple[str, ...]
    structure_type_count: int
    structure_max_depths: dict[str, int]
    ast_depth: int
    mean_token_nested_level: float
    parse_status: str

    def token_type_counts_json(self) -> str:
        return json.dumps(self.token_type_counts, ensure_ascii=False)

    def structure_types_str(self) -> str:
        return "|".join(self.structure_types)

    def structure_max_depths_json(self) -> str:
        return json.dumps(self.structure_max_depths, ensure_ascii=False)


def _token_type_counts(tokens: tuple[str, ...]) -> dict[str, int]:
    counts = {category.value: 0 for category in TOKEN_CATEGORY_ORDER}
    for token in tokens:
        counts[classify_token(token).value] += 1
    return counts


def _structure_max_depths(tokens: list[str]) -> dict[str, int]:
    return {spec.structure_type: max_structure_depth(tokens, spec) for spec in STRUCTURE_TYPES}


def _build_rare_sets(token_counter: Counter[str]) -> dict[int, set[str]]:
    return {
        threshold: {token for token, count in token_counter.items() if count <= threshold}
        for threshold in RARE_THRESHOLDS
    }


def extract_single_features(
    record: ExpressionRecord,
    tokens: tuple[str, ...],
    *,
    duplicate_group_id: int,
    duplicate_count: int,
    rare_sets: dict[int, set[str]],
) -> ExpressionFeatures:
    token_list = list(tokens)
    structure_types = tuple(sorted(structure_types_present_in_tokens(token_list)))
    encoding = encode_position_forest_tokens(token_list)
    if encoding.nested_levels:
        mean_token_nested_level = sum(encoding.nested_levels) / len(encoding.nested_levels)
    else:
        mean_token_nested_level = 0.0

    return ExpressionFeatures(
        expression_id=record.expression_id or f"{record.dataset}:{record.image_name}",
        dataset=record.dataset,
        source_file=record.source_file,
        line_id=record.line_id,
        normalized_latex=normalize_expression_latex(record.ocr),
        token_sequence=tokens,
        token_length=len(tokens),
        length_bin=assign_length_bin(len(tokens)),
        is_duplicate=duplicate_count > 1,
        duplicate_group_id=duplicate_group_id,
        duplicate_count=duplicate_count,
        token_type_counts=_token_type_counts(tokens),
        has_rare_1=any(token in rare_sets[1] for token in tokens),
        has_rare_5=any(token in rare_sets[5] for token in tokens),
        has_rare_10=any(token in rare_sets[10] for token in tokens),
        structure_types=structure_types,
        structure_type_count=len(structure_types),
        structure_max_depths=_structure_max_depths(token_list),
        ast_depth=encoding.max_nested_level,
        mean_token_nested_level=mean_token_nested_level,
        parse_status=validate_parse_status(token_list),
    )


def build_expression_features(
    expressions: list[ExpressionRecord],
    token_sequences: list[tuple[str, ...]],
) -> list[ExpressionFeatures]:
    token_counter: Counter[str] = Counter()
    for tokens in token_sequences:
        token_counter.update(tokens)

    duplicate_index = DuplicateIndex.from_expressions(expressions)
    rare_sets = _build_rare_sets(token_counter)
    features: list[ExpressionFeatures] = []
    for index, (record, tokens) in enumerate(zip(expressions, token_sequences, strict=True)):
        features.append(
            extract_single_features(
                record,
                tokens,
                duplicate_group_id=duplicate_index.group_id_by_index[index],
                duplicate_count=duplicate_index.group_size_by_index[index],
                rare_sets=rare_sets,
            )
        )
    return features


def corpus_token_counter(features: list[ExpressionFeatures]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for feature in features:
        counter.update(feature.token_sequence)
    return counter


def resolve_token_counter(
    features: list[ExpressionFeatures],
    token_counter: Counter[str] | None = None,
) -> Counter[str]:
    if token_counter is not None:
        return token_counter
    return corpus_token_counter(features)


def parse_success_rate(features: list[ExpressionFeatures]) -> float:
    if not features:
        return 0.0
    ok_count = sum(feature.parse_status == "ok" for feature in features)
    return ok_count / len(features)
