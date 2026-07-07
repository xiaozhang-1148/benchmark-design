"""Duplicate detection for benchmark expression records.

Duplicate definition
--------------------
Two expression records are duplicates iff their **full** ``normalized_latex``
strings are exactly equal (character-for-character). Only whole-expression
equality counts; partial, token-level, or fuzzy matching is never used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from benchmark_design.io.benchmark_loader import ExpressionRecord


def normalize_expression_latex(raw: str) -> str:
    """Return the canonical ``normalized_latex`` duplicate key for one expression."""
    return raw.strip()


@dataclass(frozen=True, slots=True)
class DuplicateStats:
    expression_count: int
    unique_normalized_latex_count: int
    redundant_expression_count: int
    duplicate_rate: float
    duplicated_group_expression_count: int
    duplicated_group_expression_ratio: float
    duplicate_group_count: int
    max_duplicate_group_size: int


@dataclass(frozen=True, slots=True)
class DuplicateIndex:
    """Expression records grouped by exact ``normalized_latex`` equality."""

    normalized_latex_by_index: tuple[str, ...]
    group_id_by_index: tuple[int, ...]
    group_size_by_index: tuple[int, ...]
    groups: dict[str, tuple[int, ...]]

    @classmethod
    def from_normalized_latex(cls, normalized_latex: Sequence[str]) -> DuplicateIndex:
        latex_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, latex in enumerate(normalized_latex):
            latex_to_indices[latex].append(index)

        count = len(normalized_latex)
        group_id_by_index = [0] * count
        group_size_by_index = [0] * count
        groups: dict[str, tuple[int, ...]] = {}

        for group_id, (latex, indices) in enumerate(latex_to_indices.items()):
            size = len(indices)
            groups[latex] = tuple(indices)
            for index in indices:
                group_id_by_index[index] = group_id
                group_size_by_index[index] = size

        return cls(
            normalized_latex_by_index=tuple(normalized_latex),
            group_id_by_index=tuple(group_id_by_index),
            group_size_by_index=tuple(group_size_by_index),
            groups=groups,
        )

    @classmethod
    def from_expressions(cls, expressions: Sequence[ExpressionRecord]) -> DuplicateIndex:
        return cls.from_normalized_latex(
            [normalize_expression_latex(record.ocr) for record in expressions]
        )

    @classmethod
    def from_normalized_latex_values(
        cls,
        features: Sequence[_HasNormalizedLatex],
    ) -> DuplicateIndex:
        return cls.from_normalized_latex([feature.normalized_latex for feature in features])

    def stats(self) -> DuplicateStats:
        expression_count = len(self.normalized_latex_by_index)
        unique_count = len(self.groups)
        redundant_expression_count = expression_count - unique_count
        duplicate_groups = [indices for indices in self.groups.values() if len(indices) >= 2]
        duplicated_group_expression_count = sum(len(indices) for indices in duplicate_groups)
        return DuplicateStats(
            expression_count=expression_count,
            unique_normalized_latex_count=unique_count,
            redundant_expression_count=redundant_expression_count,
            duplicate_rate=(
                redundant_expression_count / expression_count if expression_count else 0.0
            ),
            duplicated_group_expression_count=duplicated_group_expression_count,
            duplicated_group_expression_ratio=(
                duplicated_group_expression_count / expression_count if expression_count else 0.0
            ),
            duplicate_group_count=len(duplicate_groups),
            max_duplicate_group_size=max((len(indices) for indices in self.groups.values()), default=0),
        )

    def is_duplicate(self, index: int) -> bool:
        return self.group_size_by_index[index] > 1


class _HasNormalizedLatex(Protocol):
    normalized_latex: str


TFeature = TypeVar("TFeature", bound=_HasNormalizedLatex)


def build_duplicate_groups(
    expressions: Sequence[ExpressionRecord],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return per-index duplicate group id and group size."""
    index = DuplicateIndex.from_expressions(expressions)
    group_id_by_index = dict(enumerate(index.group_id_by_index))
    group_size_by_index = dict(enumerate(index.group_size_by_index))
    return group_id_by_index, group_size_by_index


def duplicate_stats_from_features(features: Sequence[_HasNormalizedLatex]) -> DuplicateStats:
    return DuplicateIndex.from_normalized_latex_values(features).stats()


def duplicate_stats_from_expressions(expressions: Sequence[ExpressionRecord]) -> DuplicateStats:
    return DuplicateIndex.from_expressions(expressions).stats()


def group_features_by_normalized_latex(
    features: Sequence[TFeature],
) -> dict[str, list[TFeature]]:
    groups: dict[str, list[TFeature]] = defaultdict(list)
    for feature in features:
        groups[feature.normalized_latex].append(feature)
    return dict(groups)


def duplicate_feature_groups(
    features: Sequence[TFeature],
) -> dict[str, list[TFeature]]:
    """Groups with at least two expressions sharing the same ``normalized_latex``."""
    return {
        latex: items
        for latex, items in group_features_by_normalized_latex(features).items()
        if len(items) >= 2
    }
