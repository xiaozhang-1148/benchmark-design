"""Smoke tests for unified / project benchmark export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.ocr.processing import ProcessingOptions
from benchmark_design.project.export import run_project_export
from benchmark_design.export_layout import BenchmarkExportLayout
from benchmark_design.report.unified_export import run_unified_benchmark_export
from benchmark_design.block_level.processing_options import BlockLevelProcessingOptions


FIXTURE_JSON = Path(__file__).parent / "fixtures" / "sample_page.json"


def _make_fixture_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "sample.jpg.json")
    array = np.full((16, 20), 220, dtype=np.uint8)
    array[4:10, 6:14] = 50
    Image.fromarray(array, mode="L").save(input_dir / "sample.png")
    return input_dir


def test_run_unified_benchmark_export_smoke(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "benchmark_export"
    result = run_unified_benchmark_export(
        input_dir,
        output_root,
        processing=ProcessingOptions(show_progress=False),
        block_level_processing=BlockLevelProcessingOptions(show_progress=False, read_image_dimensions=False),
        skip_hmer_figures=True,
        skip_cross_benchmark=True,
        skip_flow_figures=True,
        skip_page_level_figures=True,
        skip_line_level_figures=True,
        skip_page_level_hmer_figures=True,
        skip_page_level_latex_split=True,
    )

    layout = BenchmarkExportLayout(output_root)
    assert result.output_root == output_root
    assert result.hmer_output == layout.hmer
    assert result.structure_layout_output == layout.structure_layout
    assert result.hybrid_layout_output == layout.hybrid_layout
    assert result.page_level_output == layout.page_level
    assert result.density_output == layout.page_level
    assert result.line_level_output == layout.line_level
    assert result.dataset_overview == output_root / "dataset_overview.md"
    assert result.summary_json == output_root / "summary.json"
    assert result.pipeline_doc.is_file()

    assert (result.hmer_output / "summary.md").is_file()
    assert (result.structure_layout_output / "block_level_summary.md").is_file()
    assert result.dataset_overview.is_file()
    assert result.summary_json.is_file()
    assert not (output_root / "HMER" / "overview.md").exists()
    assert not (layout.structure_layout / "overview.md").exists()
    assert not (output_root / "benchmark_output").exists()
    assert not (output_root / "vision_benchmark_output").exists()
    overview_text = result.dataset_overview.read_text(encoding="utf-8")
    assert "summary.json" in overview_text
    assert "HMER/summary.md" in overview_text
    assert "block_level/structure_layout/block_level_summary.md" in overview_text
    assert (layout.page_level / "report" / "image_analysis_report.md").is_file()
    assert "page_level/report/image_analysis_report.md" in overview_text
    assert (layout.line_level / "report" / "line_analysis_report.md").is_file()
    assert "line_level/report/line_analysis_report.md" in overview_text

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["page_count"] == 1
    assert "pipelines" in summary
    assert "HMER" in summary["pipelines"]
    assert "page_level" in summary["pipelines"]
    assert "block_level/structure_layout" in summary["pipelines"]
    assert "block_level/hybrid_layout" in summary["pipelines"]
    assert "line_level" in summary["pipelines"]
    assert "page_level_HMER" in summary["pipelines"]


def test_run_unified_benchmark_export_skip_line_level(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "benchmark_export_skip_line_level"
    result = run_unified_benchmark_export(
        input_dir,
        output_root,
        processing=ProcessingOptions(show_progress=False),
        block_level_processing=BlockLevelProcessingOptions(show_progress=False, read_image_dimensions=False),
        skip_hmer_figures=True,
        skip_cross_benchmark=True,
        skip_flow_figures=True,
        skip_page_level_figures=True,
        skip_line_level=True,
        skip_line_level_figures=True,
        skip_page_level_hmer_figures=True,
        skip_page_level_latex_split=True,
    )

    assert result.line_level_output is None
    assert result.line_level_manifest is None
    overview_text = result.dataset_overview.read_text(encoding="utf-8")
    assert "line_level/report/line_analysis_report.md" not in overview_text


def test_run_project_export_smoke(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "project_export"
    result = run_project_export(
        input_dir,
        output_root,
        processing=ProcessingOptions(show_progress=False),
        block_level_processing=BlockLevelProcessingOptions(show_progress=False, read_image_dimensions=False),
        skip_hmer_figures=True,
        skip_cross_benchmark=True,
        skip_flow_figures=True,
        skip_page_level_figures=True,
        skip_line_level_figures=True,
        skip_page_level_hmer_figures=True,
        skip_page_level_latex_split=True,
    )
    assert result.summary_json.is_file()
    assert (output_root / "dataset_overview.md").is_file()
    assert (output_root / "block_level" / "structure_layout" / "metadata.json").is_file()
