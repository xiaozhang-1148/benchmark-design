"""Chapter-6 Rare-8 (corpus frequency <= 8) helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow

# Rare-8 = token types whose corpus instance frequency is at most this value.
RARE8_MAX_CORPUS_FREQUENCY = 8


@dataclass(frozen=True, slots=True)
class Rare8Summary:
    rare_vocab_count: int
    token_instance_count: int
    expression_count: int
    page_count: int
    page_ratio: float


def compute_rare8_token_set(
    token_counter: Counter[str],
    *,
    max_frequency: int = RARE8_MAX_CORPUS_FREQUENCY,
) -> set[str]:
    return {token for token, count in token_counter.items() if count <= max_frequency}


def compute_rare8_page_stats(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    rare8_tokens: set[str],
) -> dict[str, tuple[int, int]]:
    """Return page_id -> (has_rare8, rare8_token_instance_count)."""
    counts: dict[str, int] = defaultdict(int)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        instance_count = sum(1 for token in row.tokens if token in rare8_tokens)
        if instance_count:
            counts[row.image_id] += instance_count
    return {page_id: (1, count) for page_id, count in counts.items()}


def summarize_rare8(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    rare8_tokens: set[str],
    *,
    total_pages: int,
) -> Rare8Summary:
    page_stats = compute_rare8_page_stats(expression_rows, rare8_tokens)
    token_instances = sum(count for _, count in page_stats.values())
    expression_count = sum(
        1
        for row in expression_rows
        if row.valid_for_latex and any(token in rare8_tokens for token in row.tokens)
    )
    page_count = len(page_stats)
    return Rare8Summary(
        rare_vocab_count=len(rare8_tokens),
        token_instance_count=token_instances,
        expression_count=expression_count,
        page_count=page_count,
        page_ratio=page_count / total_pages if total_pages else 0.0,
    )
