"""Decision engine for Answer-Block Flow Structure."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from benchmark_design.block_level.flow_structure.clustering import true_cross_column_bridge_ids
from benchmark_design.block_level.flow_structure.context import assess_page_context
from benchmark_design.block_level.flow_structure.layout import build_page_layout_context
from benchmark_design.block_level.flow_structure.metrics import (
    build_block_records,
    compute_flow_metrics,
    extract_txt_blocks,
    has_inserted_answer_block,
)
from benchmark_design.block_level.flow_structure.models import (
    FlowConfidenceLabel,
    FlowStructureLabel,
    PageAnnotation,
    PageFlowStructureResult,
    RuleOutcome,
)
from benchmark_design.block_level.flow_structure.classify_flow import classify_flow_structure
from benchmark_design.block_level.flow_structure.rules import (
    collect_diagnostic_tags,
    evaluate_column_layout_rules,
    evaluate_single_span_rule,
    evaluate_vertical_flow_rules,
    format_rules,
)
from benchmark_design.block_level.flow_structure.skeleton import assess_skeleton
from benchmark_design.block_level.flow_structure.thresholds import (
    BOUNDARY_MARGIN_CLUSTER_STABILITY,
    BOUNDARY_MARGIN_VERTICAL_SCORE,
    CLUSTER_STABILITY_MIN_AGREEMENT,
    CONFIDENCE_MARGIN_CENTER_DISTANCE,
    CONFIDENCE_MARGIN_VERTICAL_SCORE,
    CONFIDENCE_MARGIN_X_SPAN,
    COLUMN_CENTER_DISTANCE_NORM,
    COLUMN_Y_OVERLAP_NORM,
    SINGLE_FLOW_X_CENTER_SPAN_NORM,
    VERTICAL_SEQUENTIAL_SCORE_MIN,
)


def _merge_tags(*tag_groups: tuple[str, ...]) -> str:
    merged: list[str] = []
    for group in tag_groups:
        for tag in group:
            if tag and tag not in merged:
                merged.append(tag)
    return ";".join(merged)


def _confidence_for_label(
    *,
    flow_structure: FlowStructureLabel,
    metrics,
    needs_boundary_review: bool,
) -> FlowConfidenceLabel:
    if flow_structure == "NA":
        return "high"
    if flow_structure == "Single-flow" and metrics.num_detected_columns == 1:
        if metrics.vertical_sequential_score >= 1.0 or metrics.x_center_span_norm == 0.0:
            return "high"
        if metrics.vertical_sequential_score >= VERTICAL_SEQUENTIAL_SCORE_MIN + CONFIDENCE_MARGIN_VERTICAL_SCORE:
            return "high"
        if metrics.x_center_span_norm <= SINGLE_FLOW_X_CENTER_SPAN_NORM - CONFIDENCE_MARGIN_X_SPAN:
            return "high"
        return "medium"
    if flow_structure == "Columnar-flow":
        if (
            metrics.column_center_distance_norm >= COLUMN_CENTER_DISTANCE_NORM + CONFIDENCE_MARGIN_CENTER_DISTANCE
            and metrics.column_y_overlap_norm >= COLUMN_Y_OVERLAP_NORM + 0.08
        ):
            return "high"
        return "medium"
    if flow_structure == "Hybrid-flow":
        return "low" if needs_boundary_review else "medium"
    return "medium"


def _needs_boundary_review(
    metrics,
    skeleton,
    *,
    rule_outcomes: tuple[RuleOutcome, ...],
) -> bool:
    tags: list[str] = []
    low = VERTICAL_SEQUENTIAL_SCORE_MIN - BOUNDARY_MARGIN_VERTICAL_SCORE
    high = VERTICAL_SEQUENTIAL_SCORE_MIN
    if low <= metrics.vertical_sequential_score < high:
        tags.append("boundary_case")

    cluster = metrics.column_cluster
    if cluster is not None:
        stability_floor = CLUSTER_STABILITY_MIN_AGREEMENT
        if stability_floor <= 0.67 < stability_floor + BOUNDARY_MARGIN_CLUSTER_STABILITY and not cluster.is_stable:
            tags.append("boundary_case")

    failed_count = sum(1 for rule in rule_outcomes if not rule.passed)
    passed_count = sum(1 for rule in rule_outcomes if rule.passed)
    if failed_count == 1 and passed_count >= 2 and skeleton.skeleton_type in {"columnar", "vertical_single_flow"}:
        tags.append("boundary_case")

    return bool(tags)


def decide_flow_structure(
    page: PageAnnotation,
    *,
    input_dir: Path | None = None,
) -> PageFlowStructureResult:
    layout = build_page_layout_context(page)
    txt_page = PageAnnotation(
        page_id=page.page_id,
        image_name=page.image_name,
        source_file=page.source_file,
        image_width=page.image_width,
        image_height=page.image_height,
        blocks=layout.answer_blocks,
    )
    txt_blocks = extract_txt_blocks(txt_page)
    num_txt = len(txt_blocks)
    review_path = str((input_dir or Path(page.source_file).parent) / page.image_name)
    qa_tags = layout.qa_tags

    if num_txt == 0:
        metrics = compute_flow_metrics(page, txt_blocks)
        skeleton = assess_skeleton(txt_blocks, metrics, page_height=page.image_height, page_width=page.image_width)
        classification = classify_flow_structure(
            txt_blocks,
            metrics,
            skeleton,
            assess_page_context((), txt_blocks, page_width=page.image_width, page_height=page.image_height, skeleton=skeleton),
            page_width=page.image_width,
            page_height=page.image_height,
            true_cross_column_bridge=False,
        )
        return _build_result(
            page,
            txt_blocks=txt_blocks,
            metrics=metrics,
            skeleton=skeleton,
            context=assess_page_context((), txt_blocks, page_width=page.image_width, page_height=page.image_height, skeleton=skeleton),
            classification=classification,
            rule_outcomes=(),
            review_image_path=review_path,
            qa_tags=qa_tags,
            diagnostic_tags=(),
            num_context_blocks=layout.num_context_blocks,
            needs_boundary_review=False,
        )

    metrics_pass1 = compute_flow_metrics(page, txt_blocks)
    skeleton = assess_skeleton(txt_blocks, metrics_pass1, page_height=page.image_height, page_width=page.image_width)

    cluster = metrics_pass1.column_cluster
    column_ids = list(cluster.column_ids) if cluster else [0] * num_txt
    bridge_ids = (
        true_cross_column_bridge_ids(
            txt_blocks,
            column_ids,
            page_width=page.image_width,
            inter_band_gaps=cluster.inter_band_gaps if cluster else None,
        )
        if skeleton.column_layout_exists
        else set()
    )
    inserted = has_inserted_answer_block(
        txt_blocks,
        page_height=page.image_height,
        skeleton_type=skeleton.skeleton_type,
        column_ids=column_ids,
    )
    metrics = replace(
        metrics_pass1,
        has_cross_column_block=bool(bridge_ids),
        true_cross_column_bridge=bool(bridge_ids),
        has_inserted_block=inserted,
        inserted_answer_block=inserted,
    )

    context = assess_page_context(
        layout.context_blocks,
        txt_blocks,
        page_width=page.image_width,
        page_height=page.image_height,
        skeleton=skeleton,
    )
    classification = classify_flow_structure(
        txt_blocks,
        metrics,
        skeleton,
        context,
        page_width=page.image_width,
        page_height=page.image_height,
        true_cross_column_bridge=bool(bridge_ids),
    )

    column_rules = evaluate_column_layout_rules(metrics, txt_blocks, page_height=page.image_height)
    vertical_rules = evaluate_vertical_flow_rules(metrics, stable_column_layout=skeleton.column_layout_exists)
    span_rule = evaluate_single_span_rule(metrics)
    rule_outcomes = (*column_rules, *vertical_rules, span_rule)

    diagnostic_tags = (
        *skeleton.diagnostic_tags,
        *context.diagnostic_tags,
        *collect_diagnostic_tags(metrics),
    )
    boundary_review = _needs_boundary_review(metrics, skeleton, rule_outcomes=rule_outcomes)
    if boundary_review and "boundary_case" not in diagnostic_tags:
        diagnostic_tags = (*diagnostic_tags, "boundary_case")

    return _build_result(
        page,
        txt_blocks=txt_blocks,
        metrics=metrics,
        skeleton=skeleton,
        context=context,
        classification=classification,
        rule_outcomes=rule_outcomes,
        review_image_path=review_path,
        qa_tags=qa_tags,
        diagnostic_tags=diagnostic_tags,
        num_context_blocks=layout.num_context_blocks,
        needs_boundary_review=boundary_review,
        bridge_ids=bridge_ids,
    )


def _build_result(
    page: PageAnnotation,
    *,
    txt_blocks,
    metrics,
    skeleton,
    context,
    classification,
    rule_outcomes: tuple[RuleOutcome, ...],
    review_image_path: str,
    qa_tags: tuple[str, ...],
    diagnostic_tags: tuple[str, ...],
    num_context_blocks: int,
    needs_boundary_review: bool,
    bridge_ids: set[str] | None = None,
) -> PageFlowStructureResult:
    cluster = metrics.column_cluster
    column_ids = list(cluster.column_ids) if cluster else [0] * len(txt_blocks)
    block_records = build_block_records(
        page,
        txt_blocks,
        column_ids,
        page_width=page.image_width,
        bridge_ids=bridge_ids,
    )

    triggered = format_rules(list(rule_outcomes), passed=True) if rule_outcomes else ""
    failed = format_rules(list(rule_outcomes), passed=False) if rule_outcomes else ""

    confidence = _confidence_for_label(
        flow_structure=classification.flow_structure,
        metrics=metrics,
        needs_boundary_review=needs_boundary_review,
    )
    merged_tags = _merge_tags(classification.flow_tags, diagnostic_tags)

    return PageFlowStructureResult(
        page_id=page.page_id,
        image_name=page.image_name,
        image_width=page.image_width,
        image_height=page.image_height,
        num_txtBlock=len(txt_blocks),
        flow_group=classification.flow_group,
        flow_group_id=classification.flow_group_id,
        is_regular_flow=classification.is_regular_flow,
        flow_structure=classification.flow_structure,
        flow_confidence=confidence,
        flow_reason=classification.decision_reason,
        flow_tags=merged_tags,
        needs_manual_review=needs_boundary_review,
        decision_reason=classification.decision_reason,
        decision_rule_id=classification.decision_rule_id,
        failed_rules=failed,
        triggered_rules=triggered,
        hybrid_reason=classification.hybrid_reason,
        review_image_path=review_image_path,
        skeleton_type=skeleton.skeleton_type or "",
        num_context_blocks=num_context_blocks,
        stable_column_layout=skeleton.stable_column_layout,
        stable_vertical_flow=skeleton.stable_vertical_flow,
        column_layout_confidence=skeleton.column_layout_confidence,
        true_cross_column_bridge=metrics.true_cross_column_bridge,
        context_status=context.context_status,
        context_impact_reason=context.context_impact_reason,
        inserted_answer_block=metrics.inserted_answer_block,
        diagnostic_tags=_merge_tags(diagnostic_tags),
        qa_tags=_merge_tags(qa_tags),
        vertical_sequential_score=metrics.vertical_sequential_score,
        max_adjacent_y_overlap_norm=metrics.max_adjacent_y_overlap_norm,
        x_center_span_norm=metrics.x_center_span_norm,
        x_center_std_norm=metrics.x_center_std_norm,
        num_detected_columns=metrics.num_detected_columns,
        column_center_distance_norm=metrics.column_center_distance_norm,
        max_column_gap_norm=metrics.max_column_gap_norm,
        column_y_overlap_norm=metrics.column_y_overlap_norm,
        column_area_balance=metrics.column_area_balance,
        x_cluster_separation_norm=metrics.x_cluster_separation_norm,
        largest_column_area_ratio=metrics.largest_column_area_ratio,
        second_largest_column_area_ratio=metrics.second_largest_column_area_ratio,
        has_cross_column_block=metrics.has_cross_column_block,
        has_interrupting_chart_or_figure=context.context_status == "context_interrupted",
        block_records=block_records,
        rule_outcomes=rule_outcomes,
    )
