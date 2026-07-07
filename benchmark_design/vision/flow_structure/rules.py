"""Rule evaluation for Answer-Block Flow Structure."""

from __future__ import annotations

from benchmark_design.vision.flow_structure.metrics import vertical_sequential_score
from benchmark_design.vision.flow_structure.models import FlowStructureMetrics, RuleOutcome, TxtBlockGeometry
from benchmark_design.vision.flow_structure.thresholds import (
    COLUMN_CENTER_DISTANCE_NORM,
    COLUMN_GAP_NORM,
    COLUMN_Y_OVERLAP_NORM,
    MIN_SECOND_COLUMN_AREA_RATIO,
    SINGLE_FLOW_X_CENTER_SPAN_NORM,
    VERTICAL_SEQUENTIAL_SCORE_MIN,
)


def _rule(
    name: str,
    passed: bool,
    *,
    value: float | None = None,
    threshold: float | None = None,
    message: str = "",
) -> RuleOutcome:
    return RuleOutcome(name=name, passed=passed, value=value, threshold=threshold, message=message)


def evaluate_column_layout_existence_rules(
    metrics: FlowStructureMetrics,
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_height: int,
) -> list[RuleOutcome]:
    cluster = metrics.column_cluster
    center_distance = metrics.column_center_distance_norm
    horizontal_separation = (
        center_distance > COLUMN_CENTER_DISTANCE_NORM or metrics.max_column_gap_norm >= COLUMN_GAP_NORM
    )
    return [
        _rule(
            "column:min_two_columns",
            metrics.num_detected_columns >= 2,
            value=float(metrics.num_detected_columns),
            threshold=2.0,
        ),
        _rule(
            "column:horizontal_separation",
            horizontal_separation,
            value=max(center_distance, metrics.max_column_gap_norm),
            threshold=COLUMN_CENTER_DISTANCE_NORM,
        ),
        _rule(
            "column:layout_exists",
            cluster.column_layout_exists if cluster else False,
            value=1.0 if (cluster and cluster.column_layout_exists) else 0.0,
            threshold=1.0,
        ),
        _rule(
            "column:inner_reading_order",
            _all_columns_vertical(txt_blocks, metrics, page_height=page_height, min_score=0.50),
            value=metrics.vertical_sequential_score,
            threshold=0.50,
        ),
    ]


def evaluate_column_layout_confidence_rules(
    metrics: FlowStructureMetrics,
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_height: int,
) -> list[RuleOutcome]:
    cluster = metrics.column_cluster
    y_overlap = metrics.column_y_overlap_norm
    second_area = metrics.second_largest_column_area_ratio
    return [
        _rule(
            "confidence:cluster_stable",
            cluster.is_stable if cluster else False,
            value=1.0 if (cluster and cluster.is_stable) else 0.0,
            threshold=0.67,
        ),
        _rule(
            "confidence:second_column_area",
            second_area > MIN_SECOND_COLUMN_AREA_RATIO,
            value=second_area,
            threshold=MIN_SECOND_COLUMN_AREA_RATIO,
        ),
        _rule(
            "confidence:parallel_overlap",
            y_overlap > COLUMN_Y_OVERLAP_NORM,
            value=y_overlap,
            threshold=COLUMN_Y_OVERLAP_NORM,
        ),
        _rule(
            "confidence:inner_vertical_order",
            _all_columns_vertical(txt_blocks, metrics, page_height=page_height),
            value=metrics.vertical_sequential_score,
            threshold=VERTICAL_SEQUENTIAL_SCORE_MIN,
        ),
        _rule(
            "confidence:layout_score",
            (cluster.column_layout_confidence if cluster else 0.0) >= 0.5,
            value=cluster.column_layout_confidence if cluster else 0.0,
            threshold=0.5,
        ),
    ]


def evaluate_column_layout_rules(
    metrics: FlowStructureMetrics,
    txt_blocks: list[TxtBlockGeometry],
    *,
    page_height: int,
) -> list[RuleOutcome]:
    """Backward-compatible combined rule list for export/diagnostics."""
    return [
        *evaluate_column_layout_existence_rules(metrics, txt_blocks, page_height=page_height),
        *evaluate_column_layout_confidence_rules(metrics, txt_blocks, page_height=page_height),
    ]


def _all_columns_vertical(
    txt_blocks: list[TxtBlockGeometry],
    metrics: FlowStructureMetrics,
    *,
    page_height: int,
    min_score: float = VERTICAL_SEQUENTIAL_SCORE_MIN,
) -> bool:
    cluster = metrics.column_cluster
    if cluster is None:
        return False
    grouped: dict[int, list[TxtBlockGeometry]] = {}
    for block, column_id in zip(txt_blocks, list(cluster.column_ids), strict=True):
        grouped.setdefault(column_id, []).append(block)
    for blocks in grouped.values():
        if len(blocks) <= 1:
            continue
        score, _ = vertical_sequential_score(blocks, page_height=page_height)
        if score < min_score:
            return False
    return True


def evaluate_single_flow_rules(
    metrics: FlowStructureMetrics,
    *,
    single_flow_exists: bool,
) -> list[RuleOutcome]:
    return [
        _rule(
            "single:sequential_score",
            metrics.vertical_sequential_score >= VERTICAL_SEQUENTIAL_SCORE_MIN,
            value=metrics.vertical_sequential_score,
            threshold=VERTICAL_SEQUENTIAL_SCORE_MIN,
        ),
        _rule(
            "single:flow_exists",
            single_flow_exists,
            value=1.0 if single_flow_exists else 0.0,
            threshold=1.0,
        ),
    ]


def evaluate_vertical_flow_rules(
    metrics: FlowStructureMetrics,
    *,
    stable_column_layout: bool,
) -> list[RuleOutcome]:
    """Backward-compatible alias."""
    return [
        _rule(
            "vertical:sequential_score",
            metrics.vertical_sequential_score >= VERTICAL_SEQUENTIAL_SCORE_MIN,
            value=metrics.vertical_sequential_score,
            threshold=VERTICAL_SEQUENTIAL_SCORE_MIN,
        ),
        _rule(
            "vertical:no_stable_column",
            not stable_column_layout,
            value=1.0 if stable_column_layout else 0.0,
            threshold=0.0,
        ),
    ]


def evaluate_single_span_rule(metrics: FlowStructureMetrics) -> RuleOutcome:
    return _rule(
        "diagnostic:x_center_span",
        metrics.x_center_span_norm < SINGLE_FLOW_X_CENTER_SPAN_NORM,
        value=metrics.x_center_span_norm,
        threshold=SINGLE_FLOW_X_CENTER_SPAN_NORM,
    )


def collect_diagnostic_tags(metrics: FlowStructureMetrics) -> tuple[str, ...]:
    tags: list[str] = []
    if metrics.x_center_span_norm >= SINGLE_FLOW_X_CENTER_SPAN_NORM:
        tags.append("wide_shift")
    if metrics.vertical_sequential_score < VERTICAL_SEQUENTIAL_SCORE_MIN:
        tags.append("irregular_vertical_order")
    cluster = metrics.column_cluster
    if cluster and cluster.column_layout_confidence < 0.4 and cluster.column_layout_exists:
        tags.append("low_column_confidence")
    return tuple(tags)


def format_rules(rules: list[RuleOutcome], *, passed: bool) -> str:
    names = [rule.name for rule in rules if rule.passed == passed]
    return ";".join(names)
