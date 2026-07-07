"""Main answer skeleton assessment."""

from __future__ import annotations

from benchmark_design.vision.flow_structure.models import (
    FlowStructureMetrics,
    SkeletonAssessment,
    SkeletonType,
    TxtBlockGeometry,
)
from benchmark_design.vision.flow_structure.rules import (
    collect_diagnostic_tags,
    evaluate_column_layout_confidence_rules,
    evaluate_column_layout_existence_rules,
    evaluate_single_flow_rules,
)
from benchmark_design.vision.flow_structure.single_flow import single_flow_exists


def assess_skeleton(
    txt_blocks: list[TxtBlockGeometry],
    metrics: FlowStructureMetrics,
    *,
    page_height: int,
    page_width: int,
) -> SkeletonAssessment:
    num_txt = len(txt_blocks)
    cluster = metrics.column_cluster
    diagnostic_tags = collect_diagnostic_tags(metrics)

    if num_txt == 0:
        return SkeletonAssessment(
            skeleton_type=None,
            stable_column_layout=False,
            stable_vertical_flow=False,
            column_layout_exists=False,
            column_layout_confidence=0.0,
            single_flow_exists=False,
            num_detected_columns=0,
            decision_rule_ids=("skeleton.na",),
            diagnostic_tags=diagnostic_tags,
        )

    if num_txt == 1:
        return SkeletonAssessment(
            skeleton_type="single",
            stable_column_layout=False,
            stable_vertical_flow=False,
            column_layout_exists=False,
            column_layout_confidence=0.0,
            single_flow_exists=True,
            num_detected_columns=1,
            decision_rule_ids=("skeleton.single",),
            diagnostic_tags=diagnostic_tags,
        )

    column_exists = cluster.column_layout_exists if cluster else False
    column_confidence = cluster.column_layout_confidence if cluster else 0.0
    single_exists = single_flow_exists(
        txt_blocks,
        page_width=page_width,
        page_height=page_height,
        column_cluster=cluster,
    )

    existence_rules = evaluate_column_layout_existence_rules(metrics, txt_blocks, page_height=page_height)
    confidence_rules = evaluate_column_layout_confidence_rules(metrics, txt_blocks, page_height=page_height)
    single_rules = evaluate_single_flow_rules(metrics, single_flow_exists=single_exists)

    existence_rule_ids = tuple(rule.name for rule in existence_rules if rule.passed)
    confidence_rule_ids = tuple(rule.name for rule in confidence_rules if rule.passed)
    single_rule_ids = tuple(rule.name for rule in single_rules if rule.passed)

    if column_exists and not single_exists:
        skeleton_type: SkeletonType = "columnar"
        rule_ids = ("skeleton.columnar", *existence_rule_ids)
    elif single_exists:
        skeleton_type = "vertical_single_flow"
        rule_ids = ("skeleton.vertical_single_flow", *single_rule_ids)
    else:
        skeleton_type = "unstable"
        rule_ids = ("skeleton.unstable",)
        if cluster and cluster.column_layout_confidence > 0 and not column_exists:
            diagnostic_tags = (*diagnostic_tags, "weak_column_signal")

    if column_confidence < 0.4 and column_exists:
        diagnostic_tags = (*diagnostic_tags, "low_column_confidence")

    return SkeletonAssessment(
        skeleton_type=skeleton_type,
        stable_column_layout=column_exists,
        stable_vertical_flow=single_exists,
        column_layout_exists=column_exists,
        column_layout_confidence=column_confidence,
        single_flow_exists=single_exists,
        num_detected_columns=metrics.num_detected_columns,
        decision_rule_ids=(*rule_ids, *confidence_rule_ids),
        diagnostic_tags=diagnostic_tags,
    )
