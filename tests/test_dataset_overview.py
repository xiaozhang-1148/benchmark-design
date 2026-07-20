"""Tests for dataset overview export."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.report.dataset_overview import (
    compute_dataset_overview,
    run_dataset_overview_export,
    write_dataset_overview,
)
from benchmark_design.block_level.processing_options import VisionProcessingOptions

SAMPLE_PAGE = Path(__file__).parent / "fixtures" / "sample_page.json"
FLOW_PAGE = Path(__file__).parent / "fixtures" / "flow_structure_page.json"


def test_compute_dataset_overview(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")
    shutil.copy(FLOW_PAGE, input_dir / "flow.jpg.json")

    metrics = compute_dataset_overview(
        input_dir,
        processing=ProcessingOptions(show_progress=False),
        vision_processing=VisionProcessingOptions(
            show_progress=False,
            read_image_dimensions=False,
        ),
    )

    assert metrics.hmer.page_count == 2
    assert metrics.hmer.expression_count == 5
    assert metrics.hmer.total_characters == sum(
        len(record.ocr)
        for record in __import__(
            "benchmark_design.io.benchmark_loader",
            fromlist=["load_expressions"],
        ).load_expressions(input_dir, show_progress=False)
    )
    assert metrics.hmer.avg_expressions_per_page == 2.5
    assert metrics.hmer.max_expressions_per_page == 3

    assert metrics.vision.page_count == 2
    assert len(metrics.vision.aspect_ratios) == 2
    assert metrics.vision.sample_ids == ("sample.jpg", "sample.jpg")
    assert metrics.vision.portrait_count == 0
    assert metrics.vision.landscape_count == 1

    assert metrics.block.txtblock_count == 3
    assert metrics.block.total_block_count == 4
    assert metrics.block.figure_count == 1


def test_run_dataset_overview_export_layout(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")

    output_root = tmp_path / "output"
    overview_path = run_dataset_overview_export(
        input_dir,
        output_root,
        processing=ProcessingOptions(show_progress=False),
        vision_processing=VisionProcessingOptions(
            show_progress=False,
            read_image_dimensions=False,
        ),
    )

    assert overview_path == output_root / "dataset_overview.md"
    assert overview_path.is_file()
    overview_text = overview_path.read_text(encoding="utf-8")
    assert "# 数据集总纲" in overview_text
    assert "| 图像总数 | 1 |" in overview_text
    assert "| 表达式实例总数 | 3 |" in overview_text
    assert "## Block（标注块）" in overview_text
    assert "| Txtblock | 1 |" in overview_text
    assert "| 总数 | 1 |" in overview_text
    assert "| 单页最多表达式数量 | 3 |" in overview_text
    assert "| 纵向图像数量 |" in overview_text
    assert "| 横向图像数量 |" in overview_text
    assert "## 明细" in overview_text
    assert "block_level/overview.md" in overview_text

    hmer_overview = output_root / "HMER" / "overview.md"
    block_level_overview = output_root / "block_level" / "overview.md"
    block_overview = output_root / "block" / "overview.md"
    assert hmer_overview.is_file()
    assert block_level_overview.is_file()
    assert block_overview.is_file()
    assert (output_root / "HMER" / "tables" / "hmer_scale.csv").is_file()
    assert (output_root / "block" / "tables" / "block_counts.csv").is_file()
    assert (output_root / "block_level" / "tables" / "aspect_ratio_distribution.csv").is_file()
    assert (output_root / "block_level" / "tables" / "resolution_distribution.csv").is_file()
    assert (output_root / "block_level" / "tables" / "orientation_distribution.csv").is_file()


def test_write_dataset_overview_empty_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    metrics = compute_dataset_overview(
        input_dir,
        processing=ProcessingOptions(show_progress=False),
        vision_processing=VisionProcessingOptions(show_progress=False),
    )
    output_root = tmp_path / "output"
    write_dataset_overview(metrics, output_root)
    assert (output_root / "dataset_overview.md").is_file()
    assert "0" in (output_root / "dataset_overview.md").read_text(encoding="utf-8")
