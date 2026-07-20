"""Vocabulary indexes, feasibility checks, and train-cover locking before stratification."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd

from benchmark_design.page_level_latex_split.config import SplitConfig
from benchmark_design.page_level_latex_split.labels import PageLabels
from benchmark_design.page_level_latex_split.refine_state import page_tokens_index
from benchmark_design.page_level_latex_split.stratify import (
    SPLITS,
    LabelQuota,
    classify_formal_labels,
    compute_label_support,
    largest_remainder_counts,
)
from benchmark_design.page_level_latex_split.tie_break import sort_key_for_page


@dataclass(frozen=True, slots=True)
class VocabFeasibilityResult:
    feasible: bool
    train_capacity: int
    min_cover_pages: int
    impossible_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainVocabCoverResult:
    locked_train_pages: frozenset[str]
    locked_assignment: dict[str, str]
    cover_token_count: int
    cover_page_count: int


@dataclass
class VocabIndex:
    token_to_pages: dict[str, tuple[str, ...]]
    page_to_tokens: dict[str, frozenset[str]]
    all_tokens: frozenset[str]

    @classmethod
    def from_token_counts(cls, token_counts: pd.DataFrame) -> VocabIndex:
        page_to_tokens_raw = page_tokens_index(token_counts)
        token_to_pages: dict[str, list[str]] = defaultdict(list)
        for page_id, tokens in page_to_tokens_raw.items():
            for token in tokens:
                token_to_pages[token].append(page_id)
        token_to_pages_sorted = {
            token: tuple(sorted(pages)) for token, pages in sorted(token_to_pages.items())
        }
        page_to_tokens = {page_id: frozenset(tokens) for page_id, tokens in page_to_tokens_raw.items()}
        all_tokens = frozenset(token_to_pages_sorted)
        return cls(
            token_to_pages=token_to_pages_sorted,
            page_to_tokens=page_to_tokens,
            all_tokens=all_tokens,
        )


def check_vocab_feasibility(
    vocab_index: VocabIndex,
    *,
    page_count: int,
    train_ratio: float,
) -> VocabFeasibilityResult:
    """Every corpus token must appear on at least one page assignable to train."""
    train_capacity = largest_remainder_counts(
        page_count,
        {"train": train_ratio, "val": (1 - train_ratio) / 2, "test": (1 - train_ratio) / 2},
        SPLITS,
    )["train"]
    impossible: list[str] = []
    for token in sorted(vocab_index.all_tokens):
        if not vocab_index.token_to_pages.get(token):
            impossible.append(token)
    # Greedy lower bound: rarest tokens need distinct pages unless one page covers many.
    min_cover = _estimate_min_cover_pages(vocab_index)
    feasible = not impossible and min_cover <= train_capacity
    return VocabFeasibilityResult(
        feasible=feasible,
        train_capacity=train_capacity,
        min_cover_pages=min_cover,
        impossible_tokens=tuple(impossible),
    )


def _estimate_min_cover_pages(vocab_index: VocabIndex) -> int:
    """Greedy set-cover size upper bound used as feasibility sanity check."""
    uncovered = set(vocab_index.all_tokens)
    locked: set[str] = set()
    while uncovered:
        best_page: str | None = None
        best_gain = 0
        pages_seen: set[str] = set()
        for token in sorted(uncovered, key=lambda t: len(vocab_index.token_to_pages.get(t, ()))):
            for page_id in vocab_index.token_to_pages.get(token, ()):
                if page_id in locked or page_id in pages_seen:
                    continue
                pages_seen.add(page_id)
                gain = len(vocab_index.page_to_tokens.get(page_id, frozenset()) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_page = page_id
        if best_page is None:
            break
        locked.add(best_page)
        uncovered -= vocab_index.page_to_tokens.get(best_page, frozenset())
    return len(locked)


def build_train_vocab_cover(
    vocab_index: VocabIndex,
    page_labels: list[PageLabels],
    config: SplitConfig,
    *,
    seed: int,
) -> TrainVocabCoverResult:
    """Lock a small train page set so every corpus token has train support before stratify."""
    n = len(page_labels)
    capacities = largest_remainder_counts(n, config.ratios, SPLITS)
    train_capacity = capacities["train"]

    tokens_by_rarity = sorted(
        vocab_index.all_tokens,
        key=lambda token: (len(vocab_index.token_to_pages.get(token, ())), token),
    )
    locked_pages: set[str] = set()
    covered_tokens: set[str] = set()

    for token in tokens_by_rarity:
        if token in covered_tokens:
            continue
        candidate_pages = [
            page_id
            for page_id in vocab_index.token_to_pages.get(token, ())
            if page_id not in locked_pages
        ]
        if not candidate_pages:
            # Re-use an already locked page that contains this token.
            for page_id in vocab_index.token_to_pages.get(token, ()):
                if page_id in locked_pages:
                    covered_tokens.add(token)
                    break
            continue

        best_page = max(
            candidate_pages,
            key=lambda page_id: (
                len(vocab_index.page_to_tokens.get(page_id, frozenset()) - covered_tokens),
                sort_key_for_page(config, seed, page_id),
            ),
        )
        locked_pages.add(best_page)
        covered_tokens |= vocab_index.page_to_tokens.get(best_page, frozenset())
        if len(locked_pages) >= train_capacity:
            break

    if covered_tokens != vocab_index.all_tokens:
        # Fallback: lock any page needed for still-uncovered tokens.
        for token in tokens_by_rarity:
            if token in covered_tokens:
                continue
            for page_id in vocab_index.token_to_pages.get(token, ()):
                if page_id not in locked_pages:
                    if len(locked_pages) >= train_capacity:
                        break
                    locked_pages.add(page_id)
                    covered_tokens |= vocab_index.page_to_tokens.get(page_id, frozenset())
            if token in covered_tokens:
                continue

    locked_assignment = {page_id: "train" for page_id in locked_pages}
    return TrainVocabCoverResult(
        locked_train_pages=frozenset(locked_pages),
        locked_assignment=locked_assignment,
        cover_token_count=len(covered_tokens),
        cover_page_count=len(locked_pages),
    )


@dataclass
class AdjustedQuotas:
    quotas: dict[str, LabelQuota]
    remaining_capacity: dict[str, int]
    initial_filled: dict[str, Counter[str]] = field(default_factory=dict)


def deduct_locked_train_quotas(
    page_labels: list[PageLabels],
    locked_assignment: dict[str, str],
    config: SplitConfig,
) -> AdjustedQuotas:
    """Reduce train label targets and capacity by pages locked to train for vocab cover."""
    n = len(page_labels)
    capacities = largest_remainder_counts(n, config.ratios, SPLITS)
    support = compute_label_support(page_labels)
    quotas = classify_formal_labels(support, capacities, config)

    by_id = {page.page_id: page for page in page_labels}
    filled: dict[str, Counter[str]] = {label: Counter() for label in quotas}
    for page_id, split in locked_assignment.items():
        for label in by_id[page_id].labels:
            filled[label][split] += 1

    remaining = dict(capacities)
    for split in SPLITS:
        remaining[split] -= sum(1 for pid, sp in locked_assignment.items() if sp == split)

    adjusted: dict[str, LabelQuota] = {}
    for label, quota in quotas.items():
        locked_train = filled[label]["train"]
        new_targets = dict(quota.targets)
        new_targets["train"] = max(0, quota.targets["train"] - locked_train)
        adjusted[label] = LabelQuota(
            label=quota.label,
            support=quota.support,
            formal=quota.formal,
            reason=quota.reason,
            targets=new_targets,
            expected_raw=dict(quota.expected_raw),
        )

    return AdjustedQuotas(
        quotas=adjusted,
        remaining_capacity=remaining,
        initial_filled=filled,
    )
