"""Map flow_structure (3-class) to hierarchical flow_group subtypes."""

from __future__ import annotations

from typing import Literal

FlowGroupLabel = Literal[
    "Single-block flow",
    "Sequential multi-block flow",
    "Context-preserved single flow",
    "Two-column flow",
    "Multi-column flow",
    "Context-preserved columnar flow",
    "Cross-column hybrid",
    "Interrupted-context hybrid",
    "Inserted-block hybrid",
    "Irregular-layout hybrid",
    "no_valid_answer_block",
]

FlowGroupId = Literal[
    "single_block",
    "sequential_multi_block",
    "context_preserved_single",
    "two_column",
    "multi_column",
    "context_preserved_columnar",
    "cross_column_hybrid",
    "interrupted_context_hybrid",
    "inserted_block_hybrid",
    "irregular_layout_hybrid",
    "no_valid_answer_block",
]

FlowStructureSlug = Literal["single_flow", "columnar_flow", "hybrid_layout", "na"]

FLOW_STRUCTURE_SLUG_BY_LABEL: dict[str, FlowStructureSlug] = {
    "Single-flow": "single_flow",
    "Columnar-flow": "columnar_flow",
    "Hybrid-flow": "hybrid_layout",
    "NA": "na",
}

FLOW_GROUP_HIERARCHY: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Single-flow",
        (
            ("Single-block flow", "single_block"),
            ("Sequential multi-block flow", "sequential_multi_block"),
            ("Context-preserved single flow", "context_preserved_single"),
        ),
    ),
    (
        "Columnar-flow",
        (
            ("Two-column flow", "two_column"),
            ("Multi-column flow", "multi_column"),
            ("Context-preserved columnar flow", "context_preserved_columnar"),
        ),
    ),
    (
        "Hybrid-flow",
        (
            ("Cross-column hybrid", "cross_column_hybrid"),
            ("Interrupted-context hybrid", "interrupted_context_hybrid"),
            ("Inserted-block hybrid", "inserted_block_hybrid"),
            ("Irregular-layout hybrid", "irregular_layout_hybrid"),
        ),
    ),
)

FLOW_GROUP_LABELS: tuple[FlowGroupLabel, ...] = tuple(
    label for _, groups in FLOW_GROUP_HIERARCHY for label, _ in groups  # type: ignore[misc]
)

FLOW_GROUP_ID_BY_LABEL: dict[str, FlowGroupId] = {
    label: group_id  # type: ignore[misc]
    for _, groups in FLOW_GROUP_HIERARCHY
    for label, group_id in groups
}
FLOW_GROUP_ID_BY_LABEL["no_valid_answer_block"] = "no_valid_answer_block"

REGULAR_FLOW_GROUPS: frozenset[str] = frozenset(
    {
        "Single-block flow",
        "Sequential multi-block flow",
        "Context-preserved single flow",
        "Two-column flow",
        "Multi-column flow",
        "Context-preserved columnar flow",
    }
)


def flow_group_figure_parts(*, flow_structure: str, flow_group_id: str) -> tuple[str, str]:
    slug = FLOW_STRUCTURE_SLUG_BY_LABEL.get(flow_structure, "na")
    if flow_structure == "NA":
        return "na", "no_valid_answer_block"
    return slug, flow_group_id


def derive_legacy_hybrid_reason(hybrid_rule_id: str) -> str:
    mapping = {
        "cross_column": "cross_column_block",
        "interrupted_context": "interrupted_context",
        "inserted_block": "inserted_block",
        "irregular_layout": "unstable_txtblock_flow",
    }
    return mapping.get(hybrid_rule_id, "")


def derive_legacy_tags(
    *,
    flow_structure: str,
    skeleton_type: str | None,
    context_status: str,
    hybrid_rule_id: str,
    num_txtBlock: int,
    num_columns: int,
) -> tuple[str, ...]:
    tags: list[str] = []
    if flow_structure == "NA":
        return tuple(tags)
    if skeleton_type == "single" or num_txtBlock == 1:
        tags.append("single_block")
    if skeleton_type == "vertical_single_flow":
        tags.append("sequential_multi_block")
    if skeleton_type == "columnar" or flow_structure == "Columnar-flow":
        tags.append("columnar")
    if context_status == "context_preserved":
        tags.append("context_preserved")
    if hybrid_rule_id == "cross_column":
        tags.append("cross_column_block")
    if hybrid_rule_id == "interrupted_context":
        tags.append("interrupted_context")
    if hybrid_rule_id == "inserted_block":
        tags.append("inserted_block")
    if hybrid_rule_id == "irregular_layout":
        tags.append("unstable_txtblock_flow")
    if num_columns >= 3 and flow_structure == "Columnar-flow":
        tags.append("multi_column")
    return tuple(dict.fromkeys(tags))


def derive_flow_group_fields(
    *,
    flow_structure: str,
    num_txtBlock: int,
    hybrid_reason: str,
    decision_reason: str,
    x_center_span_norm: float,
    has_cross_column_block: bool,
    has_interrupting_visual_block: bool,
    has_interrupting_deleted_block: bool,
    num_detected_columns: int,
    has_preserved_context: bool,
    has_inserted_block: bool,
) -> tuple[str, str, str, str, bool]:
    """Backward-compatible wrapper for tests and legacy callers."""
    del x_center_span_norm, has_interrupting_visual_block, has_interrupting_deleted_block
    del has_preserved_context, has_inserted_block, num_txtBlock
    if flow_structure == "NA":
        return "no_valid_answer_block", "no_valid_answer_block", decision_reason, "", False
    if flow_structure == "Hybrid-flow":
        if hybrid_reason == "cross_column_block" or has_cross_column_block:
            group = "Cross-column hybrid"
        elif hybrid_reason == "interrupted_context":
            group = "Interrupted-context hybrid"
        elif hybrid_reason == "inserted_block":
            group = "Inserted-block hybrid"
        else:
            group = "Irregular-layout hybrid"
        group_id = FLOW_GROUP_ID_BY_LABEL[group]
        return group, group_id, hybrid_reason or decision_reason, "", False
    if flow_structure == "Columnar-flow":
        if num_detected_columns >= 3:
            group = "Multi-column flow"
        else:
            group = "Two-column flow"
        group_id = FLOW_GROUP_ID_BY_LABEL[group]
        return group, group_id, decision_reason, "columnar", True
    group = "Single-block flow" if num_txtBlock == 1 else "Sequential multi-block flow"
    group_id = FLOW_GROUP_ID_BY_LABEL[group]
    return group, group_id, decision_reason, "single_block", True
