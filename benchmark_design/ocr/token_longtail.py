"""Token frequency long-tail statistics for OCR corpora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

DEFAULT_TOP_K: tuple[int, ...] = (10, 50, 100, 500)
RARE_THRESHOLDS: tuple[int, ...] = (1, 5, 10)


def gini_coefficient(counts: Sequence[int]) -> float:
    """Gini index over token-type frequencies (0 = uniform, 1 = maximally concentrated)."""
    if not counts:
        return 0.0
    sorted_counts = sorted(counts)
    total = sum(sorted_counts)
    if total == 0:
        return 0.0
    count = len(sorted_counts)
    weighted = sum((2 * index - count - 1) * value for index, value in enumerate(sorted_counts, start=1))
    return weighted / (count * total)


def top_k_coverage(token_counter: Counter[str], k: int) -> float:
    total = sum(token_counter.values())
    if total == 0:
        return 0.0
    top_counts = sorted(token_counter.values(), reverse=True)[:k]
    return sum(top_counts) / total


def rare_vocab_ratio(token_counter: Counter[str], threshold: int) -> float:
    vocabulary_size = len(token_counter)
    if vocabulary_size == 0:
        return 0.0
    rare_types = sum(1 for count in token_counter.values() if count <= threshold)
    return rare_types / vocabulary_size


def rare_expression_ratio_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
    token_counter: Counter[str],
    threshold: int,
) -> float:
    rare_tokens = {token for token, count in token_counter.items() if count <= threshold}
    expression_count = 0
    rare_expression_count = 0

    for tokens in token_sequences:
        expression_count += 1
        if any(token in rare_tokens for token in tokens):
            rare_expression_count += 1

    if expression_count == 0:
        return 0.0
    return rare_expression_count / expression_count


def rare_expression_ratio(
    expressions: Iterable[ExpressionRecord],
    token_counter: Counter[str],
    threshold: int,
) -> float:
    rare_tokens = {token for token, count in token_counter.items() if count <= threshold}
    expression_count = 0
    rare_expression_count = 0
    vocab = build_latex_vocab()

    for record in expressions:
        expression_count += 1
        tokens = tokenize_greedy(record.ocr, vocab)
        if any(token in rare_tokens for token in tokens):
            rare_expression_count += 1

    if expression_count == 0:
        return 0.0
    return rare_expression_count / expression_count


@dataclass(frozen=True, slots=True)
class TokenFrequencyRow:
    rank: int
    token: str
    count: int
    frequency_share: float
    cumulative_share: float


@dataclass(frozen=True, slots=True)
class OcrTokenLongtailMetrics:
    vocabulary_size: int
    total_token_count: int
    expression_count: int
    gini: float
    top_k_coverage: tuple[tuple[int, float], ...]
    rare_vocab_ratio: tuple[tuple[int, float], ...]
    rare_expression_ratio: tuple[tuple[int, float], ...]
    frequency_distribution: tuple[TokenFrequencyRow, ...]

    def summary_rows(self) -> list[tuple[str, float | int]]:
        rows: list[tuple[str, float | int]] = [
            ("vocabulary size", self.vocabulary_size),
            ("total token count", self.total_token_count),
            ("expression count", self.expression_count),
            ("gini", self.gini),
        ]
        for k, coverage in self.top_k_coverage:
            rows.append((f"top-{k} coverage", coverage))
        for threshold, ratio in self.rare_vocab_ratio:
            rows.append((f"rare_{threshold} vocab ratio", ratio))
        for threshold, ratio in self.rare_expression_ratio:
            rows.append((f"rare_{threshold} expression ratio", ratio))
        return rows


def build_token_counter_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
) -> Counter[str]:
    token_counter: Counter[str] = Counter()
    for tokens in token_sequences:
        token_counter.update(tokens)
    return token_counter


def build_token_counter(expressions: Iterable[ExpressionRecord]) -> Counter[str]:
    vocab = build_latex_vocab()
    token_counter: Counter[str] = Counter()
    for record in expressions:
        token_counter.update(tokenize_greedy(record.ocr, vocab))
    return token_counter


def build_frequency_distribution(token_counter: Counter[str]) -> tuple[TokenFrequencyRow, ...]:
    total = sum(token_counter.values())
    if total == 0:
        return ()

    ranked = token_counter.most_common()
    cumulative = 0
    rows: list[TokenFrequencyRow] = []
    for rank, (token, count) in enumerate(ranked, start=1):
        cumulative += count
        rows.append(
            TokenFrequencyRow(
                rank=rank,
                token=token,
                count=count,
                frequency_share=count / total,
                cumulative_share=cumulative / total,
            )
        )
    return tuple(rows)


def compute_ocr_token_longtail_from_token_sequences(
    token_sequences: Sequence[Sequence[str]],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    rare_thresholds: Sequence[int] = RARE_THRESHOLDS,
) -> OcrTokenLongtailMetrics:
    token_counter = build_token_counter_from_token_sequences(token_sequences)
    frequencies = list(token_counter.values())
    frequency_distribution = build_frequency_distribution(token_counter)

    return OcrTokenLongtailMetrics(
        vocabulary_size=len(token_counter),
        total_token_count=sum(token_counter.values()),
        expression_count=len(token_sequences),
        gini=gini_coefficient(frequencies),
        top_k_coverage=tuple((k, top_k_coverage(token_counter, k)) for k in top_k),
        rare_vocab_ratio=tuple((t, rare_vocab_ratio(token_counter, t)) for t in rare_thresholds),
        rare_expression_ratio=tuple(
            (t, rare_expression_ratio_from_token_sequences(token_sequences, token_counter, t))
            for t in rare_thresholds
        ),
        frequency_distribution=frequency_distribution,
    )


def compute_ocr_token_longtail_from_expressions(
    expressions: Iterable[ExpressionRecord],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    rare_thresholds: Sequence[int] = RARE_THRESHOLDS,
) -> OcrTokenLongtailMetrics:
    expression_list = list(expressions)
    token_counter = build_token_counter(expression_list)
    frequencies = list(token_counter.values())
    frequency_distribution = build_frequency_distribution(token_counter)

    return OcrTokenLongtailMetrics(
        vocabulary_size=len(token_counter),
        total_token_count=sum(token_counter.values()),
        expression_count=len(expression_list),
        gini=gini_coefficient(frequencies),
        top_k_coverage=tuple((k, top_k_coverage(token_counter, k)) for k in top_k),
        rare_vocab_ratio=tuple((t, rare_vocab_ratio(token_counter, t)) for t in rare_thresholds),
        rare_expression_ratio=tuple(
            (t, rare_expression_ratio(expression_list, token_counter, t)) for t in rare_thresholds
        ),
        frequency_distribution=frequency_distribution,
    )


def compute_ocr_token_longtail(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrTokenLongtailMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_token_longtail_from_token_sequences(corpus.token_sequences)
