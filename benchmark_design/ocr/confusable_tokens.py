"""Confusable / visually similar token group statistics (Table 9)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from benchmark_design.ocr.tokenizer import tokenize_greedy

if TYPE_CHECKING:
    from benchmark_design.ocr.expression_features import ExpressionFeatures


class ConfusableGroupTier(StrEnum):
    PRIMARY = "primary"
    APPENDIX = "appendix"


@dataclass(frozen=True, slots=True)
class ConfusableSubgroupSpec:
    tokens: tuple[str, ...]

    @property
    def label(self) -> str:
        return " / ".join(self.tokens)


@dataclass(frozen=True, slots=True)
class ConfusableGroupSpec:
    name: str
    representative_tokens: str
    subgroups: tuple[ConfusableSubgroupSpec, ...]
    tier: ConfusableGroupTier

    @property
    def tokens(self) -> tuple[str, ...]:
        seen: list[str] = []
        for subgroup in self.subgroups:
            for token in subgroup.tokens:
                if token not in seen:
                    seen.append(token)
        return tuple(seen)


def _group(
    name: str,
    representative_tokens: str,
    *subgroups: tuple[str, ...],
    tier: ConfusableGroupTier = ConfusableGroupTier.PRIMARY,
) -> ConfusableGroupSpec:
    return ConfusableGroupSpec(
        name=name,
        representative_tokens=representative_tokens,
        subgroups=tuple(ConfusableSubgroupSpec(tokens=subgroup) for subgroup in subgroups),
        tier=tier,
    )


PRIMARY_CONFUSABLE_GROUPS: tuple[ConfusableGroupSpec, ...] = (
    _group(
        "digit-letter",
        "1/l, 2/z, 5/s, 6/b, 9/g/q",
        ("1", "l"),
        ("2", "z"),
        ("5", "s"),
        ("6", "b"),
        ("9", "g", "q"),
    ),
    _group(
        "circle-like",
        "0/o/\\theta",
        ("0", "o"),
        (r"\theta",),
    ),
    _group(
        "latin-greek",
        "p/\\rho, u/\\mu",
        ("p", r"\rho"),
        ("u", r"\mu"),
    ),
    _group(
        "greek-variant",
        "4/\\varphi",
        ("4", r"\varphi"),
    ),
    _group(
        "operator-variable",
        "x/\\times",
        ("x", r"\times"),
    ),
    _group(
        "relation-stroke",
        "</\\leq, >/\\geq",
        ("<", r"\leq", r"\le"),
        (">", r"\geq", r"\ge"),
    ),
)

APPENDIX_ONLY_CONFUSABLE_GROUPS: tuple[ConfusableGroupSpec, ...] = ()

ALL_CONFUSABLE_GROUPS: tuple[ConfusableGroupSpec, ...] = PRIMARY_CONFUSABLE_GROUPS + APPENDIX_ONLY_CONFUSABLE_GROUPS


def iter_unique_confusable_tokens() -> tuple[tuple[str, str], ...]:
    """Return ``(group_name, token)`` pairs for each distinct confusable token."""
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for group in ALL_CONFUSABLE_GROUPS:
        for token in group.tokens:
            if token in seen:
                continue
            seen.add(token)
            items.append((group.name, token))
    return tuple(items)


@dataclass(frozen=True, slots=True)
class ConfusableTokenCount:
    group: str
    token: str
    count: int
    share_of_group: float
    share_of_corpus: float


@dataclass(frozen=True, slots=True)
class ConfusableGroupMetrics:
    group: ConfusableGroupSpec
    token_count: int
    token_ratio: float
    expression_count: int
    expression_ratio: float
    co_occurrence_expression_count: int
    dominant_tokens: tuple[str, ...]
    rare_side_tokens: tuple[str, ...]
    token_counts: tuple[ConfusableTokenCount, ...]

    def main_table_row(self) -> tuple[str, str, int, float, int, float]:
        return (
            self.group.name,
            self.group.representative_tokens,
            self.token_count,
            self.token_ratio,
            self.expression_count,
            self.expression_ratio,
        )


@dataclass(frozen=True, slots=True)
class OcrConfusableTokenMetrics:
    expression_count: int
    total_token_count: int
    primary_groups: tuple[ConfusableGroupMetrics, ...]
    appendix_groups: tuple[ConfusableGroupMetrics, ...]

    def all_groups(self) -> tuple[ConfusableGroupMetrics, ...]:
        return self.primary_groups + self.appendix_groups


def _group_token_set(group: ConfusableGroupSpec) -> frozenset[str]:
    return frozenset(group.tokens)


def _co_occurrence_count(
    token_sequences: Sequence[Sequence[str]],
    group_tokens: frozenset[str],
) -> int:
    total = 0
    for tokens in token_sequences:
        present = {token for token in tokens if token in group_tokens}
        if len(present) >= 2:
            total += 1
    return total


def _expression_hit_count(token_sequences: Sequence[Sequence[str]], group_tokens: frozenset[str]) -> int:
    return sum(1 for tokens in token_sequences if any(token in group_tokens for token in tokens))


def _dominant_and_rare_tokens(
    group: ConfusableGroupSpec,
    token_counter: Counter[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dominant: list[str] = []
    rare_side: list[str] = []

    for subgroup in group.subgroups:
        counts = [(token, token_counter.get(token, 0)) for token in subgroup.tokens]
        positive = [(token, count) for token, count in counts if count > 0]
        if not positive:
            continue
        if len(subgroup.tokens) == 1:
            dominant.append(positive[0][0])
            continue
        positive.sort(key=lambda item: (-item[1], item[0]))
        max_count = positive[0][1]
        subgroup_dominant = [token for token, count in positive if count == max_count]
        subgroup_rare = [token for token, count in positive if count < max_count]
        dominant.extend(token for token in subgroup_dominant if token not in dominant)
        rare_side.extend(token for token in subgroup_rare if token not in rare_side)

    return tuple(dominant), tuple(rare_side)


def _compute_group_metrics(
    group: ConfusableGroupSpec,
    *,
    token_counter: Counter[str],
    token_sequences: Sequence[Sequence[str]],
    expression_count: int,
    total_token_count: int,
) -> ConfusableGroupMetrics:
    group_tokens = _group_token_set(group)
    token_count = sum(token_counter.get(token, 0) for token in group.tokens)
    expr_count = _expression_hit_count(token_sequences, group_tokens)
    dominant_tokens, rare_side_tokens = _dominant_and_rare_tokens(group, token_counter)
    token_counts = tuple(
        ConfusableTokenCount(
            group=group.name,
            token=token,
            count=token_counter.get(token, 0),
            share_of_group=token_counter.get(token, 0) / token_count if token_count else 0.0,
            share_of_corpus=token_counter.get(token, 0) / total_token_count if total_token_count else 0.0,
        )
        for token in group.tokens
    )
    return ConfusableGroupMetrics(
        group=group,
        token_count=token_count,
        token_ratio=token_count / total_token_count if total_token_count else 0.0,
        expression_count=expr_count,
        expression_ratio=expr_count / expression_count if expression_count else 0.0,
        co_occurrence_expression_count=_co_occurrence_count(token_sequences, group_tokens),
        dominant_tokens=dominant_tokens,
        rare_side_tokens=rare_side_tokens,
        token_counts=token_counts,
    )


def compute_ocr_confusable_token_metrics_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
    *,
    token_counter: Counter[str] | None = None,
) -> OcrConfusableTokenMetrics:
    sequences = [tuple(tokens) for tokens in token_sequences]
    counter = token_counter if token_counter is not None else Counter()
    if token_counter is None:
        for tokens in sequences:
            counter.update(tokens)

    expression_count = len(sequences)
    total_token_count = sum(counter.values())
    primary_groups = tuple(
        _compute_group_metrics(
            group,
            token_counter=counter,
            token_sequences=sequences,
            expression_count=expression_count,
            total_token_count=total_token_count,
        )
        for group in PRIMARY_CONFUSABLE_GROUPS
    )
    appendix_groups = tuple(
        _compute_group_metrics(
            group,
            token_counter=counter,
            token_sequences=sequences,
            expression_count=expression_count,
            total_token_count=total_token_count,
        )
        for group in APPENDIX_ONLY_CONFUSABLE_GROUPS
    )
    return OcrConfusableTokenMetrics(
        expression_count=expression_count,
        total_token_count=total_token_count,
        primary_groups=primary_groups,
        appendix_groups=appendix_groups,
    )


def group_tokens_present(tokens: Sequence[str], group: ConfusableGroupSpec) -> tuple[str, ...]:
    group_set = _group_token_set(group)
    return tuple(token for token in tokens if token in group_set)


def subgroup_co_occurrence_count(
    tokens: Sequence[str],
    subgroup: ConfusableSubgroupSpec,
) -> int:
    return sum(1 for token in subgroup.tokens if token in tokens)


CONFUSABLE_EXAMPLE_MIN_OCR_CHARS = 3
CONFUSABLE_EXAMPLE_COUNT_PER_TOKEN = 10


def ocr_non_whitespace_char_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def select_confusable_token_examples(
    features: Sequence[ExpressionFeatures],
    *,
    tokens: Sequence[str],
    min_ocr_chars: int = CONFUSABLE_EXAMPLE_MIN_OCR_CHARS,
    per_token: int = CONFUSABLE_EXAMPLE_COUNT_PER_TOKEN,
) -> tuple[tuple[str, ExpressionFeatures], ...]:
    """Return up to *per_token* examples per token with OCR length strictly > *min_ocr_chars*."""
    selected: list[tuple[str, ExpressionFeatures]] = []
    seen_ids: set[str] = set()

    for token in tokens:
        candidates = [
            feature
            for feature in features
            if token in feature.token_sequence
            and ocr_non_whitespace_char_count(feature.normalized_latex) > min_ocr_chars
            and feature.expression_id not in seen_ids
        ]
        candidates.sort(
            key=lambda feature: (
                -ocr_non_whitespace_char_count(feature.normalized_latex),
                -feature.token_length,
                feature.expression_id,
            )
        )
        for feature in candidates[:per_token]:
            selected.append((token, feature))
            seen_ids.add(feature.expression_id)

    return tuple(selected)
