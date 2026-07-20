"""Flow-structure decision tree (flow_structure + flow_group classification)."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure.context import assess_page_context
from benchmark_design.block_level.flow_structure.flow_group import derive_legacy_hybrid_reason, derive_legacy_tags
from benchmark_design.block_level.flow_structure.models import (
    ContextAssessment,
    FlowClassification,
    FlowStructureMetrics,
    SkeletonAssessment,
    TxtBlockGeometry,
)
from benchmark_design.block_level.flow_structure.residual import (
    all_txtblocks_fit_column_flow,
    detect_residual_block_ids,
    recover_skeleton_after_residual_removal,
    residual_blocks_satisfy_inserted,
)


def _na_classification() -> FlowClassification:
    return FlowClassification(
        flow_structure="NA",
        flow_group="no_valid_answer_block",
        flow_group_id="no_valid_answer_block",
        is_regular_flow=False,
        decision_rule_id="na.no_valid_answer_block",
        decision_reason="na:no_valid_answer_block",
        hybrid_rule_id="",
        hybrid_reason="",
        flow_tags=(),
    )


def _single_block_classification(context: ContextAssessment) -> FlowClassification:
    tags: tuple[str, ...] = ()
    if context.context_status == "context_preserved":
        tags = ("context_preserved",)
    elif context.context_status == "context_interrupted":
        tags = ("context_present",)
    return FlowClassification(
        flow_structure="Single-flow",
        flow_group="Single-block flow",
        flow_group_id="single_block",
        is_regular_flow=True,
        decision_rule_id="single.single_block",
        decision_reason="single:single_block",
        hybrid_rule_id="",
        hybrid_reason="",
        flow_tags=tags,
    )


def _single_flow_classification(context: ContextAssessment, *, decision_rule_id: str) -> FlowClassification:
    if context.context_status == "context_preserved":
        return FlowClassification(
            flow_structure="Single-flow",
            flow_group="Context-preserved single flow",
            flow_group_id="context_preserved_single",
            is_regular_flow=True,
            decision_rule_id=decision_rule_id,
            decision_reason=decision_rule_id.replace(".", ":"),
            hybrid_rule_id="",
            hybrid_reason="",
            flow_tags=("context_preserved", "sequential_multi_block"),
        )
    return FlowClassification(
        flow_structure="Single-flow",
        flow_group="Sequential multi-block flow",
        flow_group_id="sequential_multi_block",
        is_regular_flow=True,
        decision_rule_id=decision_rule_id,
        decision_reason=decision_rule_id.replace(".", ":"),
        hybrid_rule_id="",
        hybrid_reason="",
        flow_tags=("sequential_multi_block",),
    )


def _columnar_flow_classification(
    skeleton: SkeletonAssessment,
    context: ContextAssessment,
) -> FlowClassification:
    if context.context_status == "context_preserved":
        return FlowClassification(
            flow_structure="Columnar-flow",
            flow_group="Context-preserved columnar flow",
            flow_group_id="context_preserved_columnar",
            is_regular_flow=True,
            decision_rule_id="columnar.context_preserved",
            decision_reason="columnar:context_preserved",
            hybrid_rule_id="",
            hybrid_reason="",
            flow_tags=("context_preserved", "columnar"),
        )
    if skeleton.num_detected_columns >= 3:
        return FlowClassification(
            flow_structure="Columnar-flow",
            flow_group="Multi-column flow",
            flow_group_id="multi_column",
            is_regular_flow=True,
            decision_rule_id="columnar.multi_column",
            decision_reason="columnar:multi_column",
            hybrid_rule_id="",
            hybrid_reason="",
            flow_tags=("columnar", "multi_column"),
        )
    return FlowClassification(
        flow_structure="Columnar-flow",
        flow_group="Two-column flow",
        flow_group_id="two_column",
        is_regular_flow=True,
        decision_rule_id="columnar.two_column",
        decision_reason="columnar:two_column",
        hybrid_rule_id="",
        hybrid_reason="",
        flow_tags=("columnar",),
    )


def _hybrid_classification(hybrid_rule_id: str) -> FlowClassification:
    group_by_rule = {
        "cross_column": ("Cross-column hybrid", "cross_column_hybrid", "hybrid.cross_column"),
        "interrupted_context": ("Interrupted-context hybrid", "interrupted_context_hybrid", "hybrid.interrupted_context"),
        "inserted_block": ("Inserted-block hybrid", "inserted_block_hybrid", "hybrid.inserted_block"),
        "irregular_layout": ("Irregular-layout hybrid", "irregular_layout_hybrid", "hybrid.irregular_layout"),
    }
    flow_group, flow_group_id, decision_rule_id = group_by_rule[hybrid_rule_id]
    return FlowClassification(
        flow_structure="Hybrid-flow",
        flow_group=flow_group,
        flow_group_id=flow_group_id,
        is_regular_flow=False,
        decision_rule_id=decision_rule_id,
        decision_reason=f"hybrid:{hybrid_rule_id}",
        hybrid_rule_id=hybrid_rule_id,
        hybrid_reason=derive_legacy_hybrid_reason(hybrid_rule_id),
        flow_tags=derive_legacy_tags(
            flow_structure="Hybrid-flow",
            skeleton_type="unstable",
            context_status="no_context",
            hybrid_rule_id=hybrid_rule_id,
            num_txtBlock=0,
            num_columns=0,
        ),
    )


def _resolve_residual_branch(
    txt_blocks: list[TxtBlockGeometry],
    metrics: FlowStructureMetrics,
    skeleton: SkeletonAssessment,
    context: ContextAssessment,
    *,
    page_width: int,
    page_height: int,
    column_layout_exists: bool,
    single_flow_exists: bool,
) -> FlowClassification:
    residual_ids = detect_residual_block_ids(
        txt_blocks,
        metrics,
        page_height=page_height,
        column_layout_exists=column_layout_exists,
        single_flow_exists=single_flow_exists,
    )
    recovered = recover_skeleton_after_residual_removal(
        txt_blocks,
        residual_ids,
        page_width=page_width,
        page_height=page_height,
    )

    if recovered is not None:
        if residual_blocks_satisfy_inserted(
            txt_blocks,
            residual_ids,
            metrics=metrics,
            page_height=page_height,
            page_width=page_width,
        ):
            return _hybrid_classification("inserted_block")
        if context.context_status == "context_interrupted":
            return _hybrid_classification("interrupted_context")
        return _hybrid_classification("irregular_layout")

    if context.context_status == "context_interrupted":
        return _hybrid_classification("interrupted_context")
    return _hybrid_classification("irregular_layout")


def classify_flow_structure(
    txt_blocks: list[TxtBlockGeometry],
    metrics: FlowStructureMetrics,
    skeleton: SkeletonAssessment,
    context: ContextAssessment,
    *,
    page_width: int,
    page_height: int,
    true_cross_column_bridge: bool,
) -> FlowClassification:
    num_txt = len(txt_blocks)
    if num_txt == 0:
        return _na_classification()
    if num_txt == 1:
        return _single_block_classification(context)

    single_exists = skeleton.single_flow_exists
    column_exists = skeleton.column_layout_exists

    if single_exists:
        if context.context_status == "context_interrupted":
            return _hybrid_classification("interrupted_context")
        return _single_flow_classification(context, decision_rule_id="single.sequential_multi_block")

    if column_exists:
        if true_cross_column_bridge:
            return _hybrid_classification("cross_column")
        if all_txtblocks_fit_column_flow(txt_blocks, metrics, page_height=page_height):
            return _columnar_flow_classification(skeleton, context)
        return _resolve_residual_branch(
            txt_blocks,
            metrics,
            skeleton,
            context,
            page_width=page_width,
            page_height=page_height,
            column_layout_exists=True,
            single_flow_exists=False,
        )

    return _resolve_residual_branch(
        txt_blocks,
        metrics,
        skeleton,
        context,
        page_width=page_width,
        page_height=page_height,
        column_layout_exists=False,
        single_flow_exists=False,
    )
