"""Answer-Block Flow Structure classifier tests."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from benchmark_design.report.vision.flow_structure_export import (
    PAGE_METRICS_COLUMNS,
    write_flow_group_summary_csv,
    write_flow_structure_page_metrics_csv,
)
from benchmark_design.report.vision.flow_structure_figures import export_flow_structure_figures
from benchmark_design.vision.flow_structure.classifier import classify_page_flow_structure
from benchmark_design.vision.flow_structure.flow_group import derive_flow_group_fields
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.vision.flow_structure.page_loader import _load_page_annotation


def _page(
    page_id: str,
    blocks: list[tuple[str, list[list[float]]]],
    *,
    width: int = 1000,
    height: int = 1000,
) -> PageAnnotation:
    page_blocks = tuple(
        PageBlockAnnotation(
            page_id=page_id,
            block_id=f"{page_id}:block_{index}",
            block_type=block_type,
            block_order=index,
            polygon=tuple((float(x), float(y)) for x, y in polygon),
        )
        for index, (block_type, polygon) in enumerate(blocks)
    )
    return PageAnnotation(
        page_id=page_id,
        image_name=f"{page_id}.jpg",
        source_file=f"/tmp/{page_id}.json",
        image_width=width,
        image_height=height,
        blocks=page_blocks,
    )


def test_flow_structure_na_when_no_txtblock() -> None:
    page = _page("p0", [("figure", [[0, 0], [100, 0], [100, 100], [0, 100]])])
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "NA"
    assert result.flow_group == "no_valid_answer_block"
    assert result.flow_group_id == "no_valid_answer_block"
    assert result.is_regular_flow is False
    assert result.num_txtBlock == 0
    assert result.flow_confidence == "high"
    assert result.decision_rule_id == "na.no_valid_answer_block"
    assert result.context_status == "no_context"


def test_flow_structure_single_for_one_txtblock() -> None:
    page = _page("p1", [("Txtblock", [[100, 100], [900, 100], [900, 900], [100, 900]])])
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Single-block flow"
    assert result.flow_group_id == "single_block"
    assert result.is_regular_flow is True
    assert result.skeleton_type == "single"
    assert result.num_txtBlock == 1
    assert result.decision_rule_id == "single.single_block"
    assert result.needs_manual_review is False


def test_flow_structure_single_for_vertical_stack() -> None:
    page = _page(
        "p2",
        [
            ("Txtblock", [[120, 80], [880, 80], [880, 300], [120, 300]]),
            ("Txtblock", [[130, 350], [870, 350], [870, 700], [130, 700]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Sequential multi-block flow"
    assert result.is_regular_flow is True
    assert result.skeleton_type == "vertical_single_flow"
    assert result.stable_vertical_flow is True
    assert result.vertical_sequential_score >= 0.8
    assert result.decision_rule_id == "single.sequential_multi_block"


def test_flow_structure_single_for_vertical_stack_with_x_offset() -> None:
    page = _page(
        "p2b",
        [
            ("Txtblock", [[80, 80], [880, 80], [880, 300], [80, 300]]),
            ("Txtblock", [[200, 350], [900, 350], [900, 700], [200, 700]]),
            ("Txtblock", [[120, 750], [860, 750], [860, 920], [120, 920]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Sequential multi-block flow"
    assert result.flow_structure != "Hybrid-flow"
    assert result.vertical_sequential_score >= 0.8
    assert result.stable_vertical_flow is True


def test_flow_structure_columnar_for_two_columns() -> None:
    page = _page(
        "p3",
        [
            ("Txtblock", [[40, 80], [420, 80], [420, 700], [40, 700]]),
            ("Txtblock", [[50, 720], [410, 720], [410, 900], [50, 900]]),
            ("Txtblock", [[560, 100], [940, 100], [940, 650], [560, 650]]),
            ("Txtblock", [[570, 680], [930, 680], [930, 880], [570, 880]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Columnar-flow"
    assert result.flow_group == "Two-column flow"
    assert result.flow_group_id == "two_column"
    assert result.is_regular_flow is True
    assert result.skeleton_type == "columnar"
    assert result.stable_column_layout is True
    assert result.num_detected_columns >= 2
    assert result.column_center_distance_norm > 0.25
    assert result.column_y_overlap_norm > 0.25
    assert result.true_cross_column_bridge is False
    assert result.hybrid_reason == ""


def test_flow_structure_columnar_without_gutter() -> None:
    page = _page(
        "p3b",
        [
            ("Txtblock", [[20, 80], [480, 80], [480, 700], [20, 700]]),
            ("Txtblock", [[480, 80], [980, 80], [980, 700], [480, 700]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Columnar-flow"
    assert result.flow_group == "Two-column flow"
    assert result.true_cross_column_bridge is False
    assert result.inserted_answer_block is False


def test_flow_structure_three_column_three_blocks() -> None:
    page = _page(
        "p3col",
        [
            ("Txtblock", [[40, 200], [280, 200], [280, 700], [40, 700]]),
            ("Txtblock", [[360, 200], [600, 200], [600, 700], [360, 700]]),
            ("Txtblock", [[680, 200], [920, 200], [920, 700], [680, 700]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Columnar-flow"
    assert result.flow_group == "Multi-column flow"
    assert result.flow_group_id == "multi_column"
    assert result.stable_column_layout is True
    assert result.num_detected_columns == 3
    assert result.column_layout_confidence > 0.0
    column_ids = {record.assigned_column_id for record in result.block_records}
    assert len(column_ids) == 3
    assert result.flow_structure != "Hybrid-flow"


def test_flow_structure_column_layout_confidence_in_csv(tmp_path: Path) -> None:
    page = _page(
        "p3col_csv",
        [
            ("Txtblock", [[40, 200], [280, 200], [280, 700], [40, 700]]),
            ("Txtblock", [[360, 200], [600, 200], [600, 700], [360, 700]]),
            ("Txtblock", [[680, 200], [920, 200], [920, 700], [680, 700]]),
        ],
    )
    result = classify_page_flow_structure(page)
    csv_path = tmp_path / "page_metrics.csv"
    write_flow_structure_page_metrics_csv([result], csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row = next(reader)
    assert "column_layout_confidence" in header
    confidence = float(row[header.index("column_layout_confidence")])
    assert confidence > 0.0
    assert row[header.index("stable_column_layout")] == "true"


def test_flow_structure_columnar_for_jagged_side_by_side_masks() -> None:
    page = _page(
        "p_jagged_cols",
        [
            ("Txtblock", [[40, 100], [400, 100], [520, 400], [400, 700], [40, 700]]),
            ("Txtblock", [[560, 100], [920, 100], [920, 700], [560, 700]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Columnar-flow"
    assert result.flow_group == "Two-column flow"
    assert result.stable_column_layout is True


def test_flow_structure_single_for_stacked_sections_with_x_shift() -> None:
    page = _page(
        "p_stack_shift",
        [
            ("Txtblock", [[120, 80], [750, 80], [750, 280], [120, 280]]),
            ("Txtblock", [[80, 400], [480, 400], [480, 720], [80, 720]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Sequential multi-block flow"
    assert result.stable_vertical_flow is True


def test_flow_structure_hybrid_for_unstable_overlapping_txtblocks() -> None:
    page = _page(
        "p4",
        [
            ("Txtblock", [[200, 100], [600, 100], [600, 400], [200, 400]]),
            ("Txtblock", [[200, 200], [600, 200], [600, 500], [200, 500]]),
            ("Txtblock", [[200, 400], [600, 400], [600, 700], [200, 700]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Hybrid-flow"
    assert result.flow_group == "Irregular-layout hybrid"
    assert result.skeleton_type == "unstable"
    assert result.is_regular_flow is False
    assert result.hybrid_reason == "unstable_txtblock_flow"
    assert result.decision_rule_id == "hybrid.irregular_layout"
    assert result.needs_manual_review is False


def test_flow_structure_hybrid_for_interrupting_figure() -> None:
    page = _page(
        "p5",
        [
            ("Txtblock", [[120, 80], [880, 80], [880, 300], [120, 300]]),
            ("figure", [[200, 320], [800, 320], [800, 650], [200, 650]]),
            ("Txtblock", [[130, 700], [870, 700], [870, 900], [130, 900]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Hybrid-flow"
    assert result.flow_group == "Interrupted-context hybrid"
    assert result.context_status == "context_interrupted"
    assert result.is_regular_flow is False
    assert result.hybrid_reason == "interrupted_context"
    assert result.decision_rule_id == "hybrid.interrupted_context"
    assert result.needs_manual_review is False


def test_flow_group_mapping_cross_column() -> None:
    group, group_id, reason, tags, is_regular = derive_flow_group_fields(
        flow_structure="Hybrid-flow",
        num_txtBlock=2,
        hybrid_reason="cross_column_block",
        decision_reason="hybrid:cross_column",
        x_center_span_norm=0.5,
        has_cross_column_block=True,
        has_interrupting_visual_block=False,
        has_interrupting_deleted_block=False,
        num_detected_columns=2,
        has_preserved_context=False,
        has_inserted_block=False,
    )
    assert group == "Cross-column hybrid"
    assert group_id == "cross_column_hybrid"
    assert is_regular is False


def test_flow_structure_csv_columns(tmp_path: Path) -> None:
    page = _page("p_csv", [("Txtblock", [[100, 100], [900, 100], [900, 900], [100, 900]])])
    result = classify_page_flow_structure(page)
    csv_path = tmp_path / "page_metrics.csv"
    write_flow_structure_page_metrics_csv([result], csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row = next(reader)
    assert header == list(PAGE_METRICS_COLUMNS)
    assert "decision_rule_id" in header
    assert "skeleton_type" in header
    assert "context_status" in header
    assert row[header.index("flow_group")] == "Single-block flow"


def test_flow_group_summary_csv(tmp_path: Path) -> None:
    page = _page("p_csv", [("Txtblock", [[100, 100], [900, 100], [900, 900], [100, 900]])])
    result = classify_page_flow_structure(page)
    csv_path = tmp_path / "flow_group_summary.csv"
    write_flow_group_summary_csv([result], csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    single_row = next(row for row in rows[1:] if row[0] == "Single-block flow")
    assert single_row[1] == "1"


def test_load_page_annotation_from_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "flow_structure_page.json"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(fixture, input_dir / "sample.jpg.json")
    page = _load_page_annotation(input_dir / "sample.jpg.json", input_dir=input_dir, dataset="ours")
    assert page.image_width > 0
    assert len(page.blocks) == 3


def test_flow_structure_fixture_page(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "flow_structure_page.json"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(fixture, input_dir / "sample.jpg.json")
    page = _load_page_annotation(input_dir / "sample.jpg.json", input_dir=input_dir, dataset="ours")
    result = classify_page_flow_structure(page)
    assert result.num_txtBlock == 2
    assert result.flow_structure in {"Single-flow", "Columnar-flow", "Hybrid-flow"}
    assert result.flow_group != ""
    assert result.decision_rule_id != ""


def test_flow_structure_figure_smoke(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    from PIL import Image

    page = _page(
        "p_fig",
        [
            ("Txtblock", [[100, 100], [900, 100], [920, 300], [80, 320]]),
        ],
    )
    result = classify_page_flow_structure(page, input_dir=tmp_path)
    image_path = tmp_path / page.image_name
    Image.new("RGB", (page.image_width, page.image_height), color="white").save(image_path)
    counts = export_flow_structure_figures(
        [result],
        input_dir=tmp_path,
        figures_root=tmp_path / "figures" / "flow_structure",
    )
    assert counts.get("single_flow/single_block", 0) >= 1


def test_flow_structure_context_preserved_single_with_side_figure() -> None:
    page = _page(
        "p6",
        [
            ("Txtblock", [[120, 80], [880, 80], [880, 300], [120, 300]]),
            ("Txtblock", [[130, 350], [870, 350], [870, 700], [130, 700]]),
            ("figure", [[50, 720], [200, 720], [200, 900], [50, 900]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Context-preserved single flow"
    assert result.context_status == "context_preserved"
    assert "context_preserved" in result.flow_tags


def test_flow_structure_single_block_ignores_context_for_main_class() -> None:
    page = _page(
        "p1_ctx",
        [
            ("Txtblock", [[120, 80], [880, 80], [880, 700], [120, 700]]),
            ("figure", [[50, 720], [200, 720], [200, 900], [50, 900]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Single-block flow"
    assert result.flow_group_id == "single_block"
    assert result.context_status == "context_preserved"


def test_flow_structure_table_is_not_flow_context() -> None:
    page = _page(
        "p_table",
        [
            ("Txtblock", [[120, 80], [880, 80], [880, 300], [120, 300]]),
            ("Txtblock", [[130, 350], [870, 350], [870, 700], [130, 700]]),
            ("table", [[200, 320], [800, 320], [800, 650], [200, 650]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.context_status == "no_context"
    page = _page(
        "p6",
        [
            ("Txtblock", [[120, 80], [880, 80], [880, 300], [120, 300]]),
            ("Txtblock", [[130, 350], [870, 350], [870, 700], [130, 700]]),
            ("figure", [[50, 720], [200, 720], [200, 900], [50, 900]]),
        ],
    )
    result = classify_page_flow_structure(page)
    assert result.flow_structure == "Single-flow"
    assert result.flow_group == "Context-preserved single flow"
    assert result.context_status == "context_preserved"
    assert "context_preserved" in result.flow_tags
