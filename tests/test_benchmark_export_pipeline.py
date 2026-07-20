"""Orchestration wiring for the unified benchmark export pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.export_pipeline import (
    BENCHMARK_EXPORT_DEPENDENCIES,
    BENCHMARK_EXPORT_PRIMARY_DIRS,
    BenchmarkExportLayout,
    run_benchmark_export_pipeline,
)
from benchmark_design.ocr.processing import ProcessingOptions
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


def test_benchmark_export_dependencies_cover_primary_outputs() -> None:
    assert set(BENCHMARK_EXPORT_PRIMARY_DIRS) == {
        "page_level",
        "block_level",
        "line_level",
        "HMER",
        "page_level_HMER",
        "page_level_latex_split",
    }
    assert "page_level" in BENCHMARK_EXPORT_DEPENDENCIES
    assert "block_level/structure_layout" in BENCHMARK_EXPORT_DEPENDENCIES
    assert "block_level/hybrid_layout" in BENCHMARK_EXPORT_DEPENDENCIES
    assert "block_level/hybrid_layout" in BENCHMARK_EXPORT_DEPENDENCIES["page_level_latex_split"]


def test_benchmark_export_layout_resolves_nested_page_level_paths(tmp_path: Path) -> None:
    layout = BenchmarkExportLayout(tmp_path / "benchmark_export")
    assert layout.page_level == layout.export_root / "page_level"
    assert layout.block_level == layout.export_root / "block_level"
    assert layout.structure_layout == layout.block_level / "structure_layout"
    assert layout.hybrid_layout == layout.block_level / "hybrid_layout"
    assert layout.split_inputs == layout.page_level_latex_split / "inputs"


def test_run_benchmark_export_pipeline_smoke(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "benchmark_export"
    result = run_benchmark_export_pipeline(
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
    assert result.hmer_output == layout.hmer
    assert result.structure_layout_output == layout.structure_layout
    assert result.hybrid_layout_output == layout.hybrid_layout
    assert result.page_level_output == layout.page_level
    assert result.density_output == layout.page_level
    assert result.line_level_output == layout.line_level
    assert result.page_level_hmer_output == layout.page_level_hmer
    assert result.page_level_latex_split_output is None
    assert result.pipeline_doc == output_root / "PIPELINE.md"
    assert result.pipeline_doc.is_file()
    assert (output_root / "pipeline_manifest.json").is_file()
    assert (layout.page_level / "tables" / "image_features.csv").is_file()
    assert (layout.block_level / "structure_layout" / "metadata.json").is_file()
    assert (layout.block_level / "hybrid_layout" / "metadata.json").is_file()
    density_csv = layout.structure_layout / "tables" / "block_foreground_density.csv"
    assert density_csv.is_file()
    if len(density_csv.read_text(encoding="utf-8").strip().splitlines()) > 1:
        assert (layout.block_level / "block_foreground_density_distribution.png").is_file()
        assert (layout.block_level / "block_level_summary.md").is_file()
        assert (layout.block_level / "tables" / "block_density_statistics.csv").is_file()
