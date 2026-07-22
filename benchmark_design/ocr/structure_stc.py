"""Nested Traversal Cost (NTC) and Co-occurrence Breadth Count (CBC) for LaTeX expressions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark_design.ocr.structure_forest import (
    StructureNode,
    build_structure_forest,
    stc_display_type,
)

if TYPE_CHECKING:
    from benchmark_design.ocr.expression_features import ExpressionFeatures

STC_TYPE_LABELS: tuple[str, ...] = (
    "Env",
    "Frac",
    "Sqrt",
    "Sup",
    "Sub",
    "Lim",
    "Sum",
    "Int",
    "Accent",
    "Stackrel",
    "Textcircled",
)


@dataclass(frozen=True, slots=True)
class StructurePath:
    segments: tuple[str, ...]
    raw_repeat_lens: tuple[int, ...]
    raw_length: int
    cost: int
    frequency: int


@dataclass(frozen=True, slots=True)
class SingleNodeCount:
    stc_type: str
    frequency: int


@dataclass(frozen=True, slots=True)
class NtcCbcResult:
    ntc: int
    cbc: int
    max_recursive_len: int
    nested_paths: tuple[StructurePath, ...]
    single_nodes: tuple[SingleNodeCount, ...]
    structure_roots: int


def is_stc_export_cohort(feature: ExpressionFeatures) -> bool:
    """Return True for heatmap bottom-right cohort: depth >= 4 and types >= 3."""
    return feature.ast_depth >= 4 and feature.structure_type_count >= 3


def format_chain_segments(segments: tuple[str, ...]) -> str:
    return " -> ".join(segments)


def format_structure_path(path: StructurePath) -> str:
    label = format_chain_segments(path.segments)
    if path.frequency > 1:
        return f"{label} ×{path.frequency}"
    return label


def format_single_node(node: SingleNodeCount) -> str:
    if node.frequency > 1:
        return f"{node.stc_type} ×{node.frequency}"
    return node.stc_type


def _to_stc_node(node: StructureNode) -> StructureNode:
    return StructureNode(
        stc_display_type(node.stc_type),
        tuple(_to_stc_node(child) for child in node.children),
    )


def _stc_forest(tokens: list[str]) -> tuple[StructureNode, ...]:
    return tuple(_to_stc_node(root) for root in build_structure_forest(tokens))


def _compress_types(types: list[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not types:
        return (), ()

    segments: list[str] = []
    raw_repeat_lens: list[int] = []
    index = 0
    while index < len(types):
        run_start = index
        while index < len(types) and types[index] == types[run_start]:
            index += 1
        run_len = index - run_start
        if run_len >= 2:
            segments.extend([types[run_start], f"{types[run_start]}*"])
            raw_repeat_lens.append(run_len)
        else:
            segments.append(types[run_start])
    return tuple(segments), tuple(raw_repeat_lens)


def _compressed_path_cost(segments: tuple[str, ...], raw_repeat_lens: tuple[int, ...]) -> int:
    cost = 0
    repeat_index = 0
    for segment in segments:
        if segment.endswith("*"):
            repeat_len = raw_repeat_lens[repeat_index]
            repeat_index += 1
            cost += 1 + math.ceil(math.log2(repeat_len))
            continue
        cost += 1
    return cost


def _compressed_cost(types: list[str]) -> int:
    segments, raw_repeat_lens = _compress_types(types)
    return _compressed_path_cost(segments, raw_repeat_lens)


def _longest_chain_types(node: StructureNode) -> list[str]:
    chain = [node.stc_type]
    if not node.children:
        return chain

    best_suffix: list[str] = []
    best_raw_len = -1
    best_cost = -1
    for child in node.children:
        suffix = _longest_chain_types(child)
        suffix_cost = _compressed_cost(suffix)
        if len(suffix) > best_raw_len or (len(suffix) == best_raw_len and suffix_cost > best_cost):
            best_suffix = suffix
            best_raw_len = len(suffix)
            best_cost = suffix_cost
    return chain + best_suffix


def _collect_chain_entries(
    node: StructureNode,
    ancestors: tuple[StructureNode, ...],
) -> list[tuple[tuple[StructureNode, ...], list[str]]]:
    entries: list[tuple[tuple[StructureNode, ...], list[str]]] = [
        (ancestors, _longest_chain_types(node))
    ]
    extended_ancestors = ancestors + (node,)
    for child in node.children:
        entries.extend(_collect_chain_entries(child, extended_ancestors))
    return entries


def _collect_all_chain_entries(roots: tuple[StructureNode, ...]) -> list[tuple[tuple[StructureNode, ...], list[str]]]:
    entries: list[tuple[tuple[StructureNode, ...], list[str]]] = []
    for root in roots:
        entries.extend(_collect_chain_entries(root, ()))
    return entries


def _is_ancestor_redundant(ancestors: tuple[StructureNode, ...], chain: list[str]) -> bool:
    for ancestor in ancestors:
        ancestor_chain = _longest_chain_types(ancestor)
        if len(ancestor_chain) > len(chain) and _is_proper_suffix(chain, ancestor_chain):
            return True
    return False


def _dedup_subchains(
    entries: list[tuple[tuple[StructureNode, ...], list[str]]],
) -> list[list[str]]:
    kept: list[list[str]] = []
    for ancestors, chain in entries:
        if _is_ancestor_redundant(ancestors, chain):
            continue
        kept.append(chain)
    return kept


def _is_proper_suffix(short: list[str], long: list[str]) -> bool:
    if len(short) >= len(long):
        return False
    suffix_len = len(short)
    return any(long[start : start + suffix_len] == short for start in range(1, len(long) - suffix_len + 1))


def compute_ntc_cbc(tokens: list[str]) -> NtcCbcResult:
    """Compute nested traversal cost and co-occurrence breadth count."""
    roots = _stc_forest(tokens)
    chain_entries = _collect_all_chain_entries(roots)
    kept_chains = _dedup_subchains(chain_entries)
    raw_counter: Counter[tuple[str, ...]] = Counter(tuple(chain) for chain in kept_chains)

    nested_counter: Counter[tuple[tuple[str, ...], tuple[int, ...], int, int]] = Counter()
    single_counter: Counter[str] = Counter()
    max_recursive_len = 0

    for chain_tuple, frequency in raw_counter.items():
        chain = list(chain_tuple)
        if len(chain) >= 2:
            segments, raw_repeat_lens = _compress_types(chain)
            max_recursive_len = max(max_recursive_len, max(raw_repeat_lens, default=0))
            cost = _compressed_path_cost(segments, raw_repeat_lens)
            nested_counter[(segments, raw_repeat_lens, cost, len(chain))] += frequency
            continue
        single_counter[chain[0]] += frequency

    nested_paths: list[StructurePath] = []
    total_ntc = 0
    for (segments, raw_repeat_lens, cost, raw_length), frequency in nested_counter.items():
        nested_paths.append(
            StructurePath(
                segments=segments,
                raw_repeat_lens=raw_repeat_lens,
                raw_length=raw_length,
                cost=cost,
                frequency=frequency,
            )
        )
        total_ntc += cost * frequency

    nested_paths.sort(key=lambda path: (-path.cost, path.segments))

    single_nodes = tuple(
        SingleNodeCount(stc_type=stc_type, frequency=frequency)
        for stc_type, frequency in sorted(single_counter.items(), key=lambda item: (-item[1], item[0]))
    )
    total_cbc = sum(node.frequency for node in single_nodes)

    return NtcCbcResult(
        ntc=total_ntc,
        cbc=total_cbc,
        max_recursive_len=max_recursive_len,
        nested_paths=tuple(nested_paths),
        single_nodes=single_nodes,
        structure_roots=len(roots),
    )


def ntc_cbc_sort_key(
    *,
    ast_depth: int,
    structure_type_count: int,
    ntc: int,
    cbc: int,
    max_recursive_len: int,
    expression_id: str,
) -> tuple[int, int, int, int, int, str]:
    """Sort cohort rows: depth/types block, then NTC > CBC > max_recursive_len descending."""
    return (
        -ast_depth,
        -structure_type_count,
        -ntc,
        -cbc,
        -max_recursive_len,
        expression_id,
    )
