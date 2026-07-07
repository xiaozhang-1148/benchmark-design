"""Tests for thread-safe matplotlib figure export."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_design.report.vision.flow_structure_figures import _draw_overlay
from benchmark_design.vision.flow_structure.models import BlockGeometryRecord, PageFlowStructureResult


def _sample_result(page_id: str = "p_lock") -> PageFlowStructureResult:
    return PageFlowStructureResult(
        page_id=page_id,
        image_name=f"{page_id}.jpg",
        image_width=200,
        image_height=200,
        num_txtBlock=1,
        flow_group="Single-block flow",
        flow_group_id="single_block",
        is_regular_flow=True,
        flow_structure="Single-flow",
        flow_confidence="high",
        flow_reason="single:single_block",
        flow_tags="",
        needs_manual_review=False,
        decision_reason="single:single_block",
        decision_rule_id="single.single_block",
        failed_rules="",
        triggered_rules="",
        hybrid_reason="",
        review_image_path="",
        skeleton_type="single",
        num_context_blocks=0,
        stable_column_layout=False,
        stable_vertical_flow=False,
        column_layout_confidence=0.0,
        true_cross_column_bridge=False,
        context_status="no_context",
        context_impact_reason="",
        inserted_answer_block=False,
        diagnostic_tags="",
        qa_tags="",
        vertical_sequential_score=1.0,
        max_adjacent_y_overlap_norm=0.0,
        x_center_span_norm=0.0,
        x_center_std_norm=0.0,
        num_detected_columns=1,
        column_center_distance_norm=0.0,
        max_column_gap_norm=0.0,
        column_y_overlap_norm=0.0,
        column_area_balance=0.0,
        x_cluster_separation_norm=0.0,
        largest_column_area_ratio=1.0,
        second_largest_column_area_ratio=0.0,
        has_cross_column_block=False,
        has_interrupting_chart_or_figure=False,
        block_records=(
            BlockGeometryRecord(
                page_id=page_id,
                block_id=f"{page_id}:block_0",
                block_type="Txtblock",
                mask_area=10000.0,
                bbox_x1=20.0,
                bbox_y1=20.0,
                bbox_x2=180.0,
                bbox_y2=180.0,
                center_x=100.0,
                center_y=100.0,
                norm_center_x=0.5,
                norm_center_y=0.5,
                sort_index=0,
                assigned_column_id=0,
                is_cross_column_block=False,
                polygon=((20.0, 20.0), (180.0, 20.0), (180.0, 180.0), (20.0, 180.0)),
            ),
        ),
    )


def test_parallel_pyplot_export_no_open_figure_warning(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image

    from benchmark_design.progress import parallel_map

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (200, 200), color="white").save(input_dir / "p_lock.jpg")

    tasks = [
        (
            replace(_sample_result(f"p_lock_{index}"), image_name="p_lock.jpg"),
            input_dir,
            tmp_path / f"out_{index}.png",
        )
        for index in range(30)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcomes = parallel_map(
            lambda args: _draw_overlay(args[0], input_dir=args[1], output_path=args[2]),
            tasks,
            description="Locked overlay export",
            show_progress=False,
            workers=8,
        )
        figure_warnings = [
            item
            for item in caught
            if issubclass(item.category, RuntimeWarning) and "figures have been opened" in str(item.message)
        ]

    assert sum(1 for ok in outcomes if ok) == 30
    assert figure_warnings == []
