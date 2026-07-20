"""Data models for Answer-Block Flow Structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FlowStructureLabel = Literal["Single-flow", "Columnar-flow", "Hybrid-flow", "NA"]
FlowConfidenceLabel = Literal["high", "medium", "low"]
SkeletonType = Literal["single", "vertical_single_flow", "columnar", "unstable"]
ContextStatus = Literal["no_context", "context_preserved", "context_interrupted"]
HybridRuleId = Literal[
    "cross_column",
    "interrupted_context",
    "inserted_block",
    "irregular_layout",
]
HybridReason = Literal[
    "cross_column_block",
    "unstable_txtblock_flow",
    "interrupting_deleted_block",
    "interrupting_visual_block",
    "interrupted_context",
    "inserted_block",
]


@dataclass(frozen=True, slots=True)
class PageBlockAnnotation:
    page_id: str
    block_id: str
    block_type: str
    block_order: int
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class PageAnnotation:
    page_id: str
    image_name: str
    source_file: str
    image_width: int
    image_height: int
    blocks: tuple[PageBlockAnnotation, ...]


@dataclass(frozen=True, slots=True)
class TxtBlockGeometry:
    page_id: str
    block_id: str
    block_order: int
    mask_area: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    width: float
    height: float
    center_x: float
    center_y: float
    x_interval: tuple[float, float]
    y_interval: tuple[float, float]
    core_x_interval: tuple[float, float]
    core_center_x: float
    norm_center_x: float
    norm_center_y: float
    norm_bbox_x1: float
    norm_bbox_y1: float
    norm_bbox_x2: float
    norm_bbox_y2: float
    norm_mask_area: float


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class ColumnClusterResult:
    column_ids: tuple[int, ...]
    num_columns: int
    column_center_distance_norm: float
    column_y_overlap_norm: float
    x_cluster_separation_norm: float
    max_column_gap_norm: float
    largest_column_area_ratio: float
    second_largest_column_area_ratio: float
    column_area_balance: float
    is_stable: bool
    band_x_ranges: tuple[tuple[float, float], ...] = ()
    inter_band_gaps: tuple[tuple[float, float], ...] = ()
    column_layout_exists: bool = False
    column_layout_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SkeletonAssessment:
    skeleton_type: SkeletonType | None
    stable_column_layout: bool
    stable_vertical_flow: bool
    column_layout_exists: bool
    column_layout_confidence: float
    single_flow_exists: bool
    num_detected_columns: int
    decision_rule_ids: tuple[str, ...]
    diagnostic_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextAssessment:
    context_status: ContextStatus
    context_impact_reason: str
    diagnostic_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FlowClassification:
    flow_structure: FlowStructureLabel
    flow_group: str
    flow_group_id: str
    is_regular_flow: bool
    decision_rule_id: str
    decision_reason: str
    hybrid_rule_id: str
    hybrid_reason: str
    flow_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlowStructureMetrics:
    vertical_sequential_score: float
    max_adjacent_y_overlap_norm: float
    x_center_span_norm: float
    x_center_std_norm: float
    num_detected_columns: int
    column_center_distance_norm: float
    max_column_gap_norm: float
    column_y_overlap_norm: float
    column_area_balance: float
    x_cluster_separation_norm: float
    largest_column_area_ratio: float
    second_largest_column_area_ratio: float
    has_cross_column_block: bool
    true_cross_column_bridge: bool
    has_interrupting_chart_or_figure: bool
    has_inserted_block: bool
    inserted_answer_block: bool
    column_cluster: ColumnClusterResult | None = None


@dataclass(frozen=True, slots=True)
class BlockGeometryRecord:
    page_id: str
    block_id: str
    block_type: str
    mask_area: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    center_x: float
    center_y: float
    norm_center_x: float
    norm_center_y: float
    sort_index: int
    assigned_column_id: int | None
    is_cross_column_block: bool
    polygon: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class PageFlowStructureResult:
    page_id: str
    image_name: str
    image_width: int
    image_height: int
    num_txtBlock: int
    flow_group: str
    flow_group_id: str
    is_regular_flow: bool
    flow_structure: FlowStructureLabel
    flow_confidence: FlowConfidenceLabel
    flow_reason: str
    flow_tags: str
    needs_manual_review: bool
    decision_reason: str
    decision_rule_id: str
    failed_rules: str
    triggered_rules: str
    hybrid_reason: str
    review_image_path: str
    skeleton_type: str
    num_context_blocks: int
    stable_column_layout: bool
    stable_vertical_flow: bool
    column_layout_confidence: float
    true_cross_column_bridge: bool
    context_status: str
    context_impact_reason: str
    inserted_answer_block: bool
    diagnostic_tags: str
    qa_tags: str
    vertical_sequential_score: float
    max_adjacent_y_overlap_norm: float
    x_center_span_norm: float
    x_center_std_norm: float
    num_detected_columns: int
    column_center_distance_norm: float
    max_column_gap_norm: float
    column_y_overlap_norm: float
    column_area_balance: float
    x_cluster_separation_norm: float
    largest_column_area_ratio: float
    second_largest_column_area_ratio: float
    has_cross_column_block: bool
    has_interrupting_chart_or_figure: bool
    block_records: tuple[BlockGeometryRecord, ...]
    rule_outcomes: tuple[RuleOutcome, ...] = field(default_factory=tuple)
