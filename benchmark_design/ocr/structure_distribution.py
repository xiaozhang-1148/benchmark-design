"""LaTeX structure-type distribution statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.matrix_environments import MATRIX_STRUCTURE_TRIGGER_TOKENS
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.structure_forest import (
    AST_STRUCTURE_SPECS,
    compute_ast_forest_metrics,
    max_structure_depth_for_type,
)
from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

if TYPE_CHECKING:
    from benchmark_design.ocr.expression_features import ExpressionFeatures


@dataclass(frozen=True, slots=True)
class StructureTypeSpec:
    structure_tier: str
    structure_type: str
    trigger_tokens: str
    triggers: frozenset[str]
    structure_id: str


STRUCTURE_TYPES: tuple[StructureTypeSpec, ...] = tuple(
    StructureTypeSpec(
        "核心结构" if spec.structure_tier == "core" else "扩展结构",
        spec.display_name,
        spec.trigger_tokens,
        spec.triggers,
        spec.structure_type,
    )
    for spec in AST_STRUCTURE_SPECS
)

MATRIX_STRUCTURE_TYPE = "Environment"
MATRIX_STRUCTURE_REPORT_COLUMN = MATRIX_STRUCTURE_TYPE


def max_structure_depth(tokens: list[str], spec: StructureTypeSpec) -> int:
    return max_structure_depth_for_type(tokens, spec.structure_id)


def _structure_type_present(tokens: list[str], spec: StructureTypeSpec) -> bool:
    metrics = compute_ast_forest_metrics(tokens)
    return metrics.structure_flags.get(spec.structure_id, False)


def _structure_occurrence_count(tokens: list[str], spec: StructureTypeSpec) -> int:
    if spec.structure_id == "env":
        from benchmark_design.ocr.matrix_environments import matrix_environment_stats

        return matrix_environment_stats(tokens).count
    return sum(1 for token in tokens if token in spec.triggers)


def count_structure_types_in_tokens(tokens: list[str]) -> int:
    """Return how many distinct AST structure types appear in *tokens*."""
    return compute_ast_forest_metrics(tokens).structure_type_count


def structure_types_present_in_tokens(tokens: list[str]) -> frozenset[str]:
    metrics = compute_ast_forest_metrics(tokens)
    return frozenset(
        spec.structure_type
        for spec in STRUCTURE_TYPES
        if metrics.structure_flags.get(spec.structure_id, False)
    )


@dataclass(frozen=True, slots=True)
class StructureTypeRow:
    structure_type: str
    trigger_tokens: str
    expression_ratio: float
    occurrence_ratio: float
    max_depth: int
    expression_count: int
    occurrence_count: int


@dataclass(frozen=True, slots=True)
class OcrStructureDistributionMetrics:
    expression_count: int
    structural_token_count: int
    rows: tuple[StructureTypeRow, ...]

    def as_rows(self) -> list[tuple[str, str, float, float, int, int, int]]:
        return [
            (
                row.structure_type,
                row.trigger_tokens,
                row.expression_ratio,
                row.occurrence_ratio,
                row.max_depth,
                row.expression_count,
                row.occurrence_count,
            )
            for row in self.rows
        ]


def _accumulate_structure_stats(
    token_list: list[str],
    *,
    expression_hits: Counter[str],
    occurrence_counts: Counter[str],
    max_depths: Counter[str],
    structural_token_count: int,
) -> int:
    for token in token_list:
        if classify_token(token) is TokenCategory.STRUCTURAL:
            structural_token_count += 1

    for spec in STRUCTURE_TYPES:
        count = _structure_occurrence_count(token_list, spec)
        if count:
            occurrence_counts[spec.structure_type] += count
            expression_hits[spec.structure_type] += 1
        max_depths[spec.structure_type] = max(
            max_depths[spec.structure_type],
            max_structure_depth(token_list, spec),
        )
    return structural_token_count


def compute_ocr_structure_distribution_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
) -> OcrStructureDistributionMetrics:
    expression_hits = Counter[str]()
    occurrence_counts = Counter[str]()
    max_depths = Counter[str]()
    structural_token_count = 0
    expression_count = 0

    for tokens in token_sequences:
        expression_count += 1
        structural_token_count = _accumulate_structure_stats(
            list(tokens),
            expression_hits=expression_hits,
            occurrence_counts=occurrence_counts,
            max_depths=max_depths,
            structural_token_count=structural_token_count,
        )

    rows: list[StructureTypeRow] = []
    for spec in STRUCTURE_TYPES:
        expr_count = expression_hits[spec.structure_type]
        occ_count = occurrence_counts[spec.structure_type]
        rows.append(
            StructureTypeRow(
                structure_type=spec.structure_type,
                trigger_tokens=spec.trigger_tokens,
                expression_ratio=expr_count / expression_count if expression_count else 0.0,
                occurrence_ratio=occ_count / structural_token_count if structural_token_count else 0.0,
                max_depth=max_depths[spec.structure_type],
                expression_count=expr_count,
                occurrence_count=occ_count,
            )
        )

    return OcrStructureDistributionMetrics(
        expression_count=expression_count,
        structural_token_count=structural_token_count,
        rows=tuple(rows),
    )


def compute_ocr_structure_distribution_from_features(
    features: Sequence["ExpressionFeatures"],
) -> OcrStructureDistributionMetrics:
    expression_hits = Counter[str]()
    occurrence_counts = Counter[str]()
    max_depths = Counter[str]()
    structural_token_count = 0
    expression_count = len(features)

    for feature in features:
        structural_token_count = _accumulate_structure_stats(
            list(feature.token_sequence),
            expression_hits=expression_hits,
            occurrence_counts=occurrence_counts,
            max_depths=max_depths,
            structural_token_count=structural_token_count,
        )

    rows: list[StructureTypeRow] = []
    for spec in STRUCTURE_TYPES:
        expr_count = expression_hits[spec.structure_type]
        occ_count = occurrence_counts[spec.structure_type]
        rows.append(
            StructureTypeRow(
                structure_type=spec.structure_type,
                trigger_tokens=spec.trigger_tokens,
                expression_ratio=expr_count / expression_count if expression_count else 0.0,
                occurrence_ratio=occ_count / structural_token_count if structural_token_count else 0.0,
                max_depth=max_depths[spec.structure_type],
                expression_count=expr_count,
                occurrence_count=occ_count,
            )
        )

    return OcrStructureDistributionMetrics(
        expression_count=expression_count,
        structural_token_count=structural_token_count,
        rows=tuple(rows),
    )


def compute_ocr_structure_distribution_from_expressions(
    expressions: Iterable[ExpressionRecord],
) -> OcrStructureDistributionMetrics:
    expression_list = list(expressions)
    vocab = build_latex_vocab()

    expression_hits = Counter[str]()
    occurrence_counts = Counter[str]()
    max_depths = Counter[str]()
    structural_token_count = 0

    for record in expression_list:
        tokens = tokenize_greedy(record.ocr, vocab)
        structural_token_count = _accumulate_structure_stats(
            tokens,
            expression_hits=expression_hits,
            occurrence_counts=occurrence_counts,
            max_depths=max_depths,
            structural_token_count=structural_token_count,
        )

    expression_count = len(expression_list)
    rows: list[StructureTypeRow] = []
    for spec in STRUCTURE_TYPES:
        expr_count = expression_hits[spec.structure_type]
        occ_count = occurrence_counts[spec.structure_type]
        rows.append(
            StructureTypeRow(
                structure_type=spec.structure_type,
                trigger_tokens=spec.trigger_tokens,
                expression_ratio=expr_count / expression_count if expression_count else 0.0,
                occurrence_ratio=occ_count / structural_token_count if structural_token_count else 0.0,
                max_depth=max_depths[spec.structure_type],
                expression_count=expr_count,
                occurrence_count=occ_count,
            )
        )

    return OcrStructureDistributionMetrics(
        expression_count=expression_count,
        structural_token_count=structural_token_count,
        rows=tuple(rows),
    )


def compute_ocr_structure_distribution(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrStructureDistributionMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_structure_distribution_from_token_sequences(corpus.token_sequences)
