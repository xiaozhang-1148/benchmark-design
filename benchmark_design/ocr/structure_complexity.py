"""Structure-type combination complexity per expression."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.length_distribution import percentile
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.structure_distribution import count_structure_types_in_tokens
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

if TYPE_CHECKING:
    from benchmark_design.ocr.expression_features import ExpressionFeatures


@dataclass(frozen=True, slots=True)
class OcrStructureComplexityMetrics:
    expression_count: int
    structural_expression_ratio: float
    mean_structure_type_count: float
    p50_structure_type_count: float
    p90_structure_type_count: float
    max_structure_type_count: int
    multi_structure_ratio_ge_2: float
    multi_structure_ratio_ge_3: float
    multi_structure_ratio_ge_4: float

    def as_rows(self) -> list[tuple[str, str, float | int]]:
        return [
            (
                "Structural expression ratio",
                "含至少一种结构类型的表达式比例",
                self.structural_expression_ratio,
            ),
            (
                "Mean structure type count",
                "每个表达式平均包含多少种结构类型",
                self.mean_structure_type_count,
            ),
            (
                "P50 structure type count",
                "典型表达式结构类型数",
                self.p50_structure_type_count,
            ),
            (
                "P90 structure type count",
                "高结构组合表达式的结构类型数",
                self.p90_structure_type_count,
            ),
            (
                "Max structure type count",
                "单个表达式最多包含多少种结构类型",
                self.max_structure_type_count,
            ),
            (
                "Multi-structure ratio >=2",
                "含至少 2 种结构类型的表达式比例",
                self.multi_structure_ratio_ge_2,
            ),
            (
                "Multi-structure ratio >=3",
                "含至少 3 种结构类型的表达式比例",
                self.multi_structure_ratio_ge_3,
            ),
            (
                "Multi-structure ratio >=4",
                "含至少 4 种结构类型的表达式比例",
                self.multi_structure_ratio_ge_4,
            ),
        ]


def compute_structure_type_counts_from_features(
    features: Sequence["ExpressionFeatures"],
) -> list[int]:
    return [feature.structure_type_count for feature in features]


def compute_structure_type_counts_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
) -> list[int]:
    return [count_structure_types_in_tokens(list(tokens)) for tokens in token_sequences]


def compute_structure_type_counts(expressions: Iterable[ExpressionRecord]) -> list[int]:
    vocab = build_latex_vocab()
    counts: list[int] = []
    for record in expressions:
        tokens = tokenize_greedy(record.ocr, vocab)
        counts.append(count_structure_types_in_tokens(tokens))
    return counts


def compute_ocr_structure_complexity_from_counts(counts: Sequence[int]) -> OcrStructureComplexityMetrics:
    expression_count = len(counts)
    if expression_count == 0:
        return OcrStructureComplexityMetrics(
            expression_count=0,
            structural_expression_ratio=0.0,
            mean_structure_type_count=0.0,
            p50_structure_type_count=0.0,
            p90_structure_type_count=0.0,
            max_structure_type_count=0,
            multi_structure_ratio_ge_2=0.0,
            multi_structure_ratio_ge_3=0.0,
            multi_structure_ratio_ge_4=0.0,
        )

    def ratio_at_least(threshold: int) -> float:
        return sum(count >= threshold for count in counts) / expression_count

    return OcrStructureComplexityMetrics(
        expression_count=expression_count,
        structural_expression_ratio=ratio_at_least(1),
        mean_structure_type_count=sum(counts) / expression_count,
        p50_structure_type_count=percentile(counts, 50),
        p90_structure_type_count=percentile(counts, 90),
        max_structure_type_count=max(counts),
        multi_structure_ratio_ge_2=ratio_at_least(2),
        multi_structure_ratio_ge_3=ratio_at_least(3),
        multi_structure_ratio_ge_4=ratio_at_least(4),
    )


def compute_ocr_structure_complexity_from_expressions(
    expressions: Iterable[ExpressionRecord],
) -> OcrStructureComplexityMetrics:
    return compute_ocr_structure_complexity_from_counts(compute_structure_type_counts(expressions))


def compute_ocr_structure_complexity(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrStructureComplexityMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_structure_complexity_from_counts(
        compute_structure_type_counts_from_token_sequences(corpus.token_sequences)
    )
