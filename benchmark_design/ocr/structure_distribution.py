"""LaTeX structure-type distribution statistics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.matrix_environments import (
    MATRIX_ENVIRONMENT_NAMES,
    MATRIX_STRUCTURE_TRIGGER_TOKENS,
    expression_has_matrix_environment,
    matrix_environment_stats,
)
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

if TYPE_CHECKING:
    from benchmark_design.ocr.expression_features import ExpressionFeatures


@dataclass(frozen=True, slots=True)
class StructureTypeSpec:
    structure_type: str
    trigger_tokens: str
    triggers: frozenset[str]


STRUCTURE_TYPES: tuple[StructureTypeSpec, ...] = (
    StructureTypeSpec("分式", r"\frac", frozenset({r"\frac", r"\dfrac", r"\tfrac", r"\cfrac"})),
    StructureTypeSpec("上标", "^", frozenset({"^"})),
    StructureTypeSpec("下标", "_", frozenset({"_"})),
    StructureTypeSpec("根式", r"\sqrt", frozenset({r"\sqrt"})),
    StructureTypeSpec("求和", r"\sum", frozenset({r"\sum"})),
    StructureTypeSpec(
        "积分",
        r"\int",
        frozenset({r"\int", r"\iint", r"\iiint", r"\iiiint", r"\oint"}),
    ),
    StructureTypeSpec(
        "Env.",
        MATRIX_STRUCTURE_TRIGGER_TOKENS,
        MATRIX_ENVIRONMENT_NAMES,
    ),
    StructureTypeSpec("极限", r"\lim", frozenset({r"\lim", r"\limsup", r"\liminf"})),
)

MATRIX_STRUCTURE_TYPE = "Env."
MATRIX_STRUCTURE_REPORT_COLUMN = "Env."


def _trigger_brace_depth(tokens: list[str], triggers: frozenset[str]) -> int:
    """Generic depth: each trigger opens a layer; ``}`` closes one layer."""
    depth = 0
    max_depth = 0
    for token in tokens:
        if token in triggers:
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == "}":
            depth = max(0, depth - 1)
    return max_depth


def max_structure_depth(tokens: list[str], spec: StructureTypeSpec) -> int:
    if spec.structure_type == MATRIX_STRUCTURE_TYPE:
        return matrix_environment_stats(tokens).max_depth
    return _trigger_brace_depth(tokens, spec.triggers)


def _structure_type_present(tokens: list[str], spec: StructureTypeSpec) -> bool:
    if spec.structure_type == MATRIX_STRUCTURE_TYPE:
        return expression_has_matrix_environment(tokens)
    return any(token in spec.triggers for token in tokens)


def _structure_occurrence_count(tokens: list[str], spec: StructureTypeSpec) -> int:
    if spec.structure_type == MATRIX_STRUCTURE_TYPE:
        return matrix_environment_stats(tokens).count
    return sum(1 for token in tokens if token in spec.triggers)


def count_structure_types_in_tokens(tokens: list[str]) -> int:
    """Return how many distinct table-6 structure types appear in *tokens*."""
    return sum(1 for spec in STRUCTURE_TYPES if _structure_type_present(tokens, spec))


def structure_types_present_in_tokens(tokens: list[str]) -> frozenset[str]:
    return frozenset(spec.structure_type for spec in STRUCTURE_TYPES if _structure_type_present(tokens, spec))


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
