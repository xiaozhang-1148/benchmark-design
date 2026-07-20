"""Tests for block-level foreground density figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.block_level.block_foreground_density import BlockForegroundDensityRow
from benchmark_design.block_level.flow_structure.models import (
    PageAnnotation,
    PageBlockAnnotation,
    PageFlowStructureResult,
)
from benchmark_design.page_level.models import CalibrationResult
from benchmark_design.report.block_level.block_density_plotting import (
    BLOCK_FOREGROUND_DENSITY_BIN_LABELS,
    _block_foreground_density_bin_counts,
    _block_foreground_density_xlabel,
    export_block_foreground_density_figure,
)
from benchmark_design.report.block_level.export_pipeline import write_block_density_exports
from benchmark_design.report.block_level.output_layout import BlockLevelOutputLayout


def _sample_flow_result(page_id: str = "page.json") -> PageFlowStructureResult:
    return PageFlowStructureResult(
        page_id=page_id,
        image_name=f"{page_id.removesuffix('.json')}.png",
        image_width=10,
        image_height=10,
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
        column_area_balance=1.0,
        x_cluster_separation_norm=0.0,
        largest_column_area_ratio=1.0,
        second_largest_column_area_ratio=0.0,
        has_cross_column_block=False,
        has_interrupting_chart_or_figure=False,
        block_records=(),
    )


def test_block_foreground_density_bins_use_block_level_intervals() -> None:
    densities = np.array([0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.20], dtype=np.float64)
    counts = _block_foreground_density_bin_counts(densities)
    assert list(BLOCK_FOREGROUND_DENSITY_BIN_LABELS) == [
        "<4%",
        "4–6%",
        "6–8%",
        "8–10%",
        "10–12%",
        "12–15%",
        "≥15%",
    ]
    assert counts.tolist() == [1, 2, 1, 1, 0, 1, 1]


def test_block_foreground_density_xlabel_uses_pooled_otsu_threshold() -> None:
    label = _block_foreground_density_xlabel(gray_threshold=157.0)
    assert "pooled Otsu" in label
    assert "t_I = 157" in label
    assert "valley" not in label.lower()
    assert "τ_D" not in label


def test_export_block_foreground_density_figure(tmp_path: Path) -> None:
    rows = [
        BlockForegroundDensityRow(
            page_id="p1",
            block_id="b1",
            block_type="Txtblock",
            block_order=0,
            foreground_density=0.03,
            annotation_pixel_count=100,
            foreground_pixel_count=3,
        ),
        BlockForegroundDensityRow(
            page_id="p1",
            block_id="b2",
            block_type="Txtblock",
            block_order=1,
            foreground_density=0.11,
            annotation_pixel_count=80,
            foreground_pixel_count=9,
        ),
    ]
    layout = BlockLevelOutputLayout(tmp_path)
    figure_path = export_block_foreground_density_figure(rows, layout.figures)
    assert figure_path.is_file()
    assert figure_path.name == "block_foreground_density_distribution.png"


def test_write_block_density_exports_writes_figure(tmp_path: Path) -> None:
    image_path = tmp_path / "input" / "page.png"
    image_path.parent.mkdir(parents=True)
    gray = np.full((10, 10), 220, dtype=np.uint8)
    gray[2:6, 2:6] = 40
    Image.fromarray(gray, mode="L").save(image_path)

    calibration = CalibrationResult(
        dark_reference=40.0,
        light_reference=220.0,
        gray_threshold=130.0,
        tau_d=130.0 / 255.0,
        dark_percentile=5.0,
        light_percentile=95.0,
        threshold_method="test",
        image_count=1,
    )
    page = PageAnnotation(
        page_id="page.json",
        image_name="page.png",
        source_file="page.json",
        image_width=10,
        image_height=10,
        blocks=(
            PageBlockAnnotation(
                page_id="page.json",
                block_id="b1",
                block_type="Txtblock",
                block_order=0,
                polygon=((2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)),
            ),
        ),
    )
    flow_result = _sample_flow_result("page.json")

    structure_dir = tmp_path / "structure_layout"
    hybrid_dir = tmp_path / "hybrid_layout"
    block_level_dir = tmp_path / "block_level"
    manifest = write_block_density_exports(
        pages=[page],
        flow_results=[flow_result],
        input_dir=tmp_path / "input",
        calibration=calibration,
        structure_layout_dir=structure_dir,
        hybrid_layout_dir=hybrid_dir,
        block_level_dir=block_level_dir,
    )
    assert "block_foreground_density_figure" in manifest
    assert (block_level_dir / "block_foreground_density_distribution.png").is_file()
