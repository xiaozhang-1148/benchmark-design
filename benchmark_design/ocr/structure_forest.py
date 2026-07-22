"""Unified LaTeX structure forest for AST depth, node count, and structure-type stats.

Every structure type in ``AST_STRUCTURE_SPECS`` is parsed into a parent ``StructureNode``
with argument subtrees as children (not treated as flat tokens). ``ast_depth`` and
``ast_node_count`` are derived from the same forest for all benchmark pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from benchmark_design.ocr.matrix_environments import (
    MATRIX_ENVIRONMENT_NAMES,
    expression_has_matrix_environment,
    find_valid_matrix_environment_span,
    matrix_environment_stats,
)
from benchmark_design.ocr.position_forest import (
    FRACTION_TRIGGERS,
    SUPERSCRIPT_TRIGGERS,
    _find_fraction_arg_ends,
    _find_single_arg_substructure_end,
)

StructureTier = Literal["core", "extended"]

FRAC_TRIGGERS: frozenset[str] = FRACTION_TRIGGERS
SUP_TRIGGERS: frozenset[str] = SUPERSCRIPT_TRIGGERS
SUB_TRIGGERS: frozenset[str] = frozenset({"_"})
SQRT_TRIGGERS: frozenset[str] = frozenset({r"\sqrt"})
ENV_TRIGGERS: frozenset[str] = MATRIX_ENVIRONMENT_NAMES

BIGOP_TRIGGERS: frozenset[str] = frozenset(
    {
        r"\sum",
        r"\int",
        r"\iint",
        r"\iiint",
        r"\iiiint",
        r"\oint",
        r"\lim",
        r"\limsup",
        r"\liminf",
    }
)

ACCENT_TRIGGERS: frozenset[str] = frozenset(
    {
        r"\vec",
        r"\bar",
        r"\hat",
        r"\overline",
        r"\widehat",
        r"\dot",
    }
)

STACKREL_TRIGGERS: frozenset[str] = frozenset({r"\stackrel"})
TEXTCIRCLED_TRIGGERS: frozenset[str] = frozenset({r"\textcircled"})

_LIM_TRIGGERS: frozenset[str] = frozenset({r"\lim", r"\limsup", r"\liminf"})
_INT_TRIGGERS: frozenset[str] = frozenset({r"\int", r"\iint", r"\iiint", r"\iiiint", r"\oint"})
_BIGOP_NODE_TYPES: frozenset[str] = frozenset({"sum", "lim", "int"})

STRUCTURE_TYPE_ORDER: tuple[str, ...] = (
    "frac",
    "sup",
    "sub",
    "sqrt",
    "env",
    "bigop",
    "accent",
    "stackrel",
    "textcircled",
)


@dataclass(frozen=True, slots=True)
class AstStructureSpec:
    structure_tier: StructureTier
    structure_type: str
    display_name: str
    trigger_tokens: str
    triggers: frozenset[str]


AST_STRUCTURE_SPECS: tuple[AstStructureSpec, ...] = (
    AstStructureSpec("core", "frac", "分式", "frac", FRAC_TRIGGERS),
    AstStructureSpec("core", "sup", "上标", "sup", SUP_TRIGGERS),
    AstStructureSpec("core", "sub", "下标", "sub", SUB_TRIGGERS),
    AstStructureSpec("core", "sqrt", "根式", "sqrt", SQRT_TRIGGERS),
    AstStructureSpec("core", "env", "Environment", "env", ENV_TRIGGERS),
    AstStructureSpec("core", "bigop", "大运算符及极限", "sum、lim 等", BIGOP_TRIGGERS),
    AstStructureSpec(
        "extended",
        "accent",
        "重音与上下修饰",
        "vec、bar、hat、overline、widehat、dot",
        ACCENT_TRIGGERS,
    ),
    AstStructureSpec("extended", "stackrel", "堆叠标注", "stackrel", STACKREL_TRIGGERS),
    AstStructureSpec("extended", "textcircled", "包围结构", "textcircled", TEXTCIRCLED_TRIGGERS),
)


@dataclass(frozen=True, slots=True)
class StructureNode:
    stc_type: str
    children: tuple[StructureNode, ...]


@dataclass(frozen=True, slots=True)
class AstForestMetrics:
    roots: tuple[StructureNode, ...]
    ast_node_count: int
    ast_depth: int
    structure_flags: dict[str, bool]

    @property
    def structure_type_count(self) -> int:
        return sum(1 for present in self.structure_flags.values() if present)

    def present_types(self) -> tuple[str, ...]:
        return tuple(name for name in STRUCTURE_TYPE_ORDER if self.structure_flags.get(name, False))

    @property
    def mean_ast_node_depth(self) -> float:
        """Average depth of all structure nodes in the forest (0 when empty)."""
        depths = _all_node_depths(self.roots)
        return sum(depths) / len(depths) if depths else 0.0


def _parse_big_operator(
    tokens: list[str],
    index: int,
    end: int,
    op_type: str,
) -> tuple[StructureNode, int]:
    children: list[StructureNode] = []
    next_idx = index + 1
    while next_idx < end and tokens[next_idx] in ("_", "^"):
        if tokens[next_idx] == "_":
            sub_end = _find_single_arg_substructure_end(tokens, next_idx)
            arg_roots = _parse_sequence(tokens, next_idx + 1, sub_end + 1)
            children.append(StructureNode("sub", tuple(arg_roots)))
            next_idx = sub_end + 1
            continue
        sub_end = _find_single_arg_substructure_end(tokens, next_idx)
        arg_roots = _parse_sequence(tokens, next_idx + 1, sub_end + 1)
        children.append(StructureNode("sup", tuple(arg_roots)))
        next_idx = sub_end + 1
    return StructureNode(op_type, tuple(children)), next_idx


def _parse_sequence(tokens: list[str], start: int, end: int) -> list[StructureNode]:
    roots: list[StructureNode] = []
    index = start
    while index < end:
        matrix_span = find_valid_matrix_environment_span(tokens, index)
        if matrix_span is not None:
            body_start, end_begin, after_block = matrix_span
            body_roots = _parse_sequence(tokens, body_start, end_begin)
            roots.append(StructureNode("env", tuple(body_roots)))
            index = after_block
            continue

        token = tokens[index]
        if token in FRAC_TRIGGERS:
            first_end, second_end = _find_fraction_arg_ends(tokens, index)
            num_roots = _parse_sequence(tokens, index + 1, first_end + 1)
            den_roots = _parse_sequence(tokens, first_end + 1, second_end + 1)
            roots.append(StructureNode("frac", tuple(num_roots + den_roots)))
            index = second_end + 1
            continue

        if token in SUP_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            arg_roots = _parse_sequence(tokens, index + 1, sub_end + 1)
            roots.append(StructureNode("sup", tuple(arg_roots)))
            index = sub_end + 1
            continue

        if token in SUB_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            arg_roots = _parse_sequence(tokens, index + 1, sub_end + 1)
            roots.append(StructureNode("sub", tuple(arg_roots)))
            index = sub_end + 1
            continue

        if token in SQRT_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            arg_roots = _parse_sequence(tokens, index + 1, sub_end + 1)
            roots.append(StructureNode("sqrt", tuple(arg_roots)))
            index = sub_end + 1
            continue

        if token == r"\sum":
            node, index = _parse_big_operator(tokens, index, end, "sum")
            roots.append(node)
            continue

        if token in _LIM_TRIGGERS:
            node, index = _parse_big_operator(tokens, index, end, "lim")
            roots.append(node)
            continue

        if token in _INT_TRIGGERS:
            node, index = _parse_big_operator(tokens, index, end, "int")
            roots.append(node)
            continue

        if token in ACCENT_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            arg_roots = _parse_sequence(tokens, index + 1, sub_end + 1)
            roots.append(StructureNode("accent", tuple(arg_roots)))
            index = sub_end + 1
            continue

        if token in STACKREL_TRIGGERS:
            first_end, second_end = _find_fraction_arg_ends(tokens, index)
            top_roots = _parse_sequence(tokens, index + 1, first_end + 1)
            base_roots = _parse_sequence(tokens, first_end + 1, second_end + 1)
            roots.append(StructureNode("stackrel", tuple(top_roots + base_roots)))
            index = second_end + 1
            continue

        if token in TEXTCIRCLED_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            arg_roots = _parse_sequence(tokens, index + 1, sub_end + 1)
            roots.append(StructureNode("textcircled", tuple(arg_roots)))
            index = sub_end + 1
            continue

        index += 1
    return roots


def build_structure_forest(tokens: list[str]) -> tuple[StructureNode, ...]:
    """Return top-level structure roots parsed from *tokens*."""
    return tuple(_parse_sequence(tokens, 0, len(tokens)))


def _node_depth(node: StructureNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_node_depth(child) for child in node.children)


def _count_nodes(node: StructureNode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def _forest_depth(roots: tuple[StructureNode, ...]) -> int:
    if not roots:
        return 0
    return max(_node_depth(root) for root in roots)


def _forest_node_count(roots: tuple[StructureNode, ...]) -> int:
    return sum(_count_nodes(root) for root in roots)


def _all_node_depths(roots: tuple[StructureNode, ...]) -> list[int]:
    depths: list[int] = []

    def visit(node: StructureNode) -> None:
        depths.append(_node_depth(node))
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return depths


def _structure_flags_from_tokens(tokens: list[str]) -> dict[str, bool]:
    token_set = set(tokens)
    return {
        "frac": bool(token_set & FRAC_TRIGGERS),
        "sup": bool(token_set & SUP_TRIGGERS),
        "sub": bool(token_set & SUB_TRIGGERS),
        "sqrt": bool(token_set & SQRT_TRIGGERS),
        "env": expression_has_matrix_environment(tokens),
        "bigop": bool(token_set & BIGOP_TRIGGERS),
        "accent": bool(token_set & ACCENT_TRIGGERS),
        "stackrel": bool(token_set & STACKREL_TRIGGERS),
        "textcircled": bool(token_set & TEXTCIRCLED_TRIGGERS),
    }


def _structure_flags_from_forest(roots: tuple[StructureNode, ...]) -> dict[str, bool]:
    flags = {name: False for name in STRUCTURE_TYPE_ORDER}

    def visit(node: StructureNode) -> None:
        if node.stc_type in flags:
            flags[node.stc_type] = True
        elif node.stc_type in _BIGOP_NODE_TYPES:
            flags["bigop"] = True
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)
    return flags


def compute_ast_forest_metrics(tokens: list[str]) -> AstForestMetrics:
    """Build the structure forest and derive AST depth / node count / type flags."""
    roots = build_structure_forest(tokens)
    token_flags = _structure_flags_from_tokens(tokens)
    forest_flags = _structure_flags_from_forest(roots)
    structure_flags = {name: token_flags[name] or forest_flags[name] for name in STRUCTURE_TYPE_ORDER}
    return AstForestMetrics(
        roots=roots,
        ast_node_count=_forest_node_count(roots),
        ast_depth=_forest_depth(roots),
        structure_flags=structure_flags,
    )


def structure_types_present_in_tokens(tokens: list[str]) -> frozenset[str]:
    metrics = compute_ast_forest_metrics(tokens)
    return frozenset(metrics.present_types())


def max_structure_depth_for_type(tokens: list[str], structure_type: str) -> int:
    """Return max nesting depth for *structure_type* within the forest."""
    if structure_type == "env":
        return matrix_environment_stats(tokens).max_depth

    spec_triggers = next(
        (spec.triggers for spec in AST_STRUCTURE_SPECS if spec.structure_type == structure_type),
        frozenset(),
    )
    if structure_type == "bigop":
        spec_triggers = BIGOP_TRIGGERS

    roots = build_structure_forest(tokens)

    def max_depth_for_nodes(nodes: tuple[StructureNode, ...]) -> int:
        best = 0
        for node in nodes:
            if node.stc_type == structure_type or (
                structure_type == "bigop" and node.stc_type in _BIGOP_NODE_TYPES
            ):
                best = max(best, _node_depth(node))
            best = max(best, max_depth_for_nodes(node.children))
        return best

    depth = 0
    max_depth = 0
    for token in tokens:
        if token in spec_triggers:
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == "}":
            depth = max(0, depth - 1)
    return max(max_depth, max_depth_for_nodes(roots))


def stc_display_type(stc_type: str) -> str:
    """Map forest node types to Title Case labels used by STC/NTC exports."""
    mapping = {
        "frac": "Frac",
        "sup": "Sup",
        "sub": "Sub",
        "sqrt": "Sqrt",
        "env": "Env",
        "sum": "Sum",
        "lim": "Lim",
        "int": "Int",
        "accent": "Accent",
        "stackrel": "Stackrel",
        "textcircled": "Textcircled",
    }
    return mapping.get(stc_type, stc_type.title())


__all__ = [
    "ACCENT_TRIGGERS",
    "AST_STRUCTURE_SPECS",
    "BIGOP_TRIGGERS",
    "AstForestMetrics",
    "AstStructureSpec",
    "ENV_TRIGGERS",
    "FRAC_TRIGGERS",
    "STACKREL_TRIGGERS",
    "STRUCTURE_TYPE_ORDER",
    "SQRT_TRIGGERS",
    "SUB_TRIGGERS",
    "SUP_TRIGGERS",
    "TEXTCIRCLED_TRIGGERS",
    "StructureNode",
    "build_structure_forest",
    "compute_ast_forest_metrics",
    "max_structure_depth_for_type",
    "stc_display_type",
    "structure_types_present_in_tokens",
]
