"""Aggregate OCR data-scale metrics over a benchmark corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord, iter_benchmark_json_paths
from benchmark_design.ocr.duplicates import duplicate_stats_from_expressions
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


@dataclass(frozen=True, slots=True)
class OcrScaleMetrics:
    expression_count: int
    total_token_count: int
    vocabulary_size: int
    unique_normalized_latex_count: int
    duplicate_rate: float
    json_file_count: int

    def as_rows(self) -> list[tuple[str, float | int]]:
        return [
            ("expression count", self.expression_count),
            ("total token count", self.total_token_count),
            ("vocabulary size", self.vocabulary_size),
            ("unique normalized LaTeX count", self.unique_normalized_latex_count),
            ("duplicate rate", self.duplicate_rate),
        ]


def compute_ocr_scale_from_tokenized(
    expressions: Iterable[ExpressionRecord],
    token_sequences: Iterable[Sequence[str]],
    *,
    json_file_count: int = 0,
) -> OcrScaleMetrics:
    token_counter: Counter[str] = Counter()
    expression_count = 0
    total_token_count = 0

    for record, tokens in zip(expressions, token_sequences, strict=True):
        expression_count += 1
        total_token_count += len(tokens)
        token_counter.update(tokens)

    duplicate_stats = duplicate_stats_from_expressions(expressions)

    return OcrScaleMetrics(
        expression_count=expression_count,
        total_token_count=total_token_count,
        vocabulary_size=len(token_counter),
        unique_normalized_latex_count=duplicate_stats.unique_normalized_latex_count,
        duplicate_rate=duplicate_stats.duplicate_rate,
        json_file_count=json_file_count,
    )


def compute_ocr_scale_from_expressions(
    expressions: Iterable[ExpressionRecord],
    *,
    json_file_count: int = 0,
) -> OcrScaleMetrics:
    """Compute scale metrics from pre-loaded expression records."""
    vocab = build_latex_vocab()
    token_counter: Counter[str] = Counter()
    expression_count = 0
    total_token_count = 0

    for record in expressions:
        expression_count += 1
        tokens = tokenize_greedy(record.ocr, vocab)
        total_token_count += len(tokens)
        token_counter.update(tokens)

    duplicate_stats = duplicate_stats_from_expressions(expressions)

    return OcrScaleMetrics(
        expression_count=expression_count,
        total_token_count=total_token_count,
        vocabulary_size=len(token_counter),
        unique_normalized_latex_count=duplicate_stats.unique_normalized_latex_count,
        duplicate_rate=duplicate_stats.duplicate_rate,
        json_file_count=json_file_count,
    )


def compute_ocr_scale(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrScaleMetrics:
    """Scan *input_dir* and compute OCR scale metrics."""
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_scale_from_tokenized(
        corpus.expressions,
        corpus.token_sequences,
        json_file_count=corpus.json_file_count,
    )
