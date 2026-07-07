"""Flow structure heuristic thresholds."""

from __future__ import annotations

# Columnar-flow (center-cluster evidence; gutter is auxiliary).
COLUMN_CENTER_DISTANCE_NORM = 0.25
COLUMN_Y_OVERLAP_NORM = 0.25
MIN_SECOND_COLUMN_AREA_RATIO = 0.15
COLUMN_GAP_NORM = 0.08

# Vertical sequential single-flow.
VERTICAL_SEQUENTIAL_SCORE_MIN = 0.80
ADJACENT_Y_OVERLAP_MAX_NORM = 0.15
IRREGULAR_VERTICAL_SCORE_MAX = 0.50

# Single-flow auxiliary.
SINGLE_FLOW_X_CENTER_SPAN_NORM = 0.35

# Confidence margins.
CONFIDENCE_MARGIN_CENTER_DISTANCE = 0.05
CONFIDENCE_MARGIN_OVERLAP = 0.08
CONFIDENCE_MARGIN_X_SPAN = 0.08
CONFIDENCE_MARGIN_VERTICAL_SCORE = 0.10

# Context / hybrid detection.
INTERRUPT_OVERLAP_RATIO = 0.05
CONTEXT_MIN_AREA_RATIO = 0.01
INSERTED_BLOCK_Y_BACKTRACK_NORM = 0.03
INSERTED_BLOCK_X_OFFSET_NORM = 0.08
CLUSTER_STABILITY_MIN_AGREEMENT = 0.67
COLUMN_ASSIGNMENT_MIN_RATIO = 0.80
BRIDGE_COLUMN_OVERLAP_RATIO = 0.20
BOUNDARY_MARGIN_VERTICAL_SCORE = 0.05
BOUNDARY_MARGIN_CLUSTER_STABILITY = 0.08

# Column band existence (soft gates).
COLUMN_BAND_GAP_NORM = 0.04
COLUMN_BAND_CENTER_SEP_NORM = 0.12
COLUMN_BAND_MIN_INNER_VERTICAL = 0.50
MASK_CORE_X_LOWER_PERCENTILE = 0.15
MASK_CORE_X_UPPER_PERCENTILE = 0.85

# Vertical stack detection (priority over column clustering).
VERTICAL_STACK_MIN_SEQUENTIAL_SCORE = 0.80

# Parallel horizontal band detection (single-flow veto).
PARALLEL_BAND_Y_OVERLAP_NORM = 0.20
PARALLEL_BAND_X_SEPARATION_NORM = 0.10

# Confidence weights for column layout (non-veto).
CONFIDENCE_WEIGHT_SECOND_COLUMN_AREA = 0.20
CONFIDENCE_WEIGHT_Y_OVERLAP = 0.20
CONFIDENCE_WEIGHT_MULTI_BLOCK_COLUMNS = 0.20
CONFIDENCE_WEIGHT_INNER_VERTICAL = 0.25
CONFIDENCE_WEIGHT_CLUSTER_STABILITY = 0.15

# Re-export role helpers for backward compatibility.
from benchmark_design.vision.flow_structure.block_roles import (  # noqa: E402
    is_auxiliary_non_txt,
    is_deleted_text_block,
    is_interrupting_block,
    is_txt_block,
    is_visual_structural_block,
    normalize_block_type,
)
