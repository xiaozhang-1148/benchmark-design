"""Fixed token-length bin counts for cross-benchmark comparison."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.length_bin_specs import DEFAULT_LENGTH_BINS, LengthBinSpec, assign_length_bin
from benchmark_design.ocr.length_distribution import (
    compute_expression_token_lengths,
    compute_expression_token_lengths_from_tokenized,
)
from benchmark_design.ocr.processing_options import ProcessingOptions


@dataclass(frozen=True, slots=True)
class LengthBinCount:
    label: str
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class OcrLengthBinMetrics:
    expression_count: int
    bins: tuple[LengthBinCount, ...]

    def as_rows(self) -> list[tuple[str, int, float]]:
        return [(item.label, item.count, item.share) for item in self.bins]


def compute_ocr_length_bins_from_lengths(
    lengths: Sequence[int],
    *,
    bins: Sequence[LengthBinSpec] = DEFAULT_LENGTH_BINS,
) -> OcrLengthBinMetrics:
    counts = {spec.label: 0 for spec in bins}
    for length in lengths:
        counts[assign_length_bin(length, bins)] += 1

    total = len(lengths)
    bin_rows = tuple(
        LengthBinCount(
            label=spec.label,
            count=counts[spec.label],
            share=counts[spec.label] / total if total else 0.0,
        )
        for spec in bins
    )
    return OcrLengthBinMetrics(expression_count=total, bins=bin_rows)


def compute_ocr_length_bins_from_expressions(
    expressions: Iterable[ExpressionRecord],
    *,
    bins: Sequence[LengthBinSpec] = DEFAULT_LENGTH_BINS,
) -> OcrLengthBinMetrics:
    return compute_ocr_length_bins_from_lengths(
        compute_expression_token_lengths(expressions),
        bins=bins,
    )


def compute_ocr_length_bins(
    input_dir: Path,
    *,
    bins: Sequence[LengthBinSpec] = DEFAULT_LENGTH_BINS,
    processing: ProcessingOptions | None = None,
) -> OcrLengthBinMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_length_bins_from_lengths(
        compute_expression_token_lengths_from_tokenized(corpus.token_sequences),
        bins=bins,
    )
