"""Expression token-length distribution statistics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


def percentile(values: Sequence[float | int], p: float) -> float:
    """Linear-interpolated percentile on sorted *values*."""
    if not values:
        return math.nan
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]

    rank = (p / 100.0) * (count - 1)
    lower = int(rank)
    upper = min(lower + 1, count - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@dataclass(frozen=True, slots=True)
class OcrLengthDistributionMetrics:
    expression_count: int
    mean_length: float
    std: float
    cv: float
    p50: float
    p90: float
    p95: float
    p99: float
    max_length: int

    def as_rows(self) -> list[tuple[str, float | int]]:
        return [
            ("mean length", self.mean_length),
            ("std", self.std),
            ("cv", self.cv),
            ("p50", self.p50),
            ("p90", self.p90),
            ("p95", self.p95),
            ("p99", self.p99),
            ("max", self.max_length),
        ]


def compute_expression_token_lengths_from_tokenized(
    token_sequences: Iterable[Sequence[str]],
) -> list[int]:
    return [len(tokens) for tokens in token_sequences]


def compute_expression_token_lengths(
    expressions: Iterable[ExpressionRecord],
) -> list[int]:
    vocab = build_latex_vocab()
    return [len(tokenize_greedy(record.ocr, vocab)) for record in expressions]


def compute_ocr_length_distribution_from_lengths(
    lengths: Sequence[int],
) -> OcrLengthDistributionMetrics:
    count = len(lengths)
    if count == 0:
        return OcrLengthDistributionMetrics(
            expression_count=0,
            mean_length=math.nan,
            std=math.nan,
            cv=math.nan,
            p50=math.nan,
            p90=math.nan,
            p95=math.nan,
            p99=math.nan,
            max_length=0,
        )

    mean_length = sum(lengths) / count
    variance = sum((length - mean_length) ** 2 for length in lengths) / count
    std = math.sqrt(variance)
    cv = std / mean_length if mean_length else math.nan

    return OcrLengthDistributionMetrics(
        expression_count=count,
        mean_length=mean_length,
        std=std,
        cv=cv,
        p50=percentile(lengths, 50),
        p90=percentile(lengths, 90),
        p95=percentile(lengths, 95),
        p99=percentile(lengths, 99),
        max_length=max(lengths),
    )


def compute_ocr_length_distribution_from_expressions(
    expressions: Iterable[ExpressionRecord],
) -> OcrLengthDistributionMetrics:
    return compute_ocr_length_distribution_from_lengths(
        compute_expression_token_lengths(expressions)
    )


def compute_ocr_length_distribution(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrLengthDistributionMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_length_distribution_from_lengths(
        compute_expression_token_lengths_from_tokenized(corpus.token_sequences)
    )
