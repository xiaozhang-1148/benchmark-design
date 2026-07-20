"""Legacy resolve entry point delegating to the flow-structure decision tree."""

from __future__ import annotations

from benchmark_design.block_level.flow_structure.classify_flow import classify_flow_structure
from benchmark_design.block_level.flow_structure.models import (
    ContextAssessment,
    FlowClassification,
    FlowStructureMetrics,
    SkeletonAssessment,
    TxtBlockGeometry,
)


def resolve_flow_classification(
    skeleton: SkeletonAssessment,
    context: ContextAssessment,
    metrics: FlowStructureMetrics,
    *,
    num_txt: int,
    txt_blocks: list[TxtBlockGeometry] | None = None,
    page_width: int = 1,
    page_height: int = 1,
    true_cross_column_bridge: bool | None = None,
) -> FlowClassification:
    blocks = txt_blocks if txt_blocks is not None else []
    bridge = (
        metrics.true_cross_column_bridge
        if true_cross_column_bridge is None
        else true_cross_column_bridge
    )
    return classify_flow_structure(
        blocks,
        metrics,
        skeleton,
        context,
        page_width=page_width,
        page_height=page_height,
        true_cross_column_bridge=bridge if num_txt else False,
    )
