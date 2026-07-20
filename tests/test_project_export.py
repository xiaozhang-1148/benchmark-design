"""Project export smoke tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.ocr.processing import ProcessingOptions
from benchmark_design.project.export import run_project_export
from benchmark_design.export_layout import BenchmarkExportLayout
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


def test_run_project_export_root_artifacts(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "benchmark_export"
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

    layout = BenchmarkExportLayout(output_root)
    assert result.summary_json == output_root / "summary.json"
    assert result.dataset_overview == output_root / "dataset_overview.md"
    assert result.pipeline_doc == output_root / "PIPELINE.md"
    assert layout.hmer.is_dir()
    assert layout.page_level.is_dir()
    assert layout.block_level.is_dir()
    assert layout.structure_layout.is_dir()
    assert layout.hybrid_layout.is_dir()
    assert layout.line_level.is_dir()
    assert layout.page_level_hmer.is_dir()

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["generated_at"]
    assert summary["input_root"]
    assert summary["output_root"]
    assert summary["page_count"] == 1
    assert summary["pipelines"]["HMER"]["manifest"] == "HMER/metadata.json"
    assert "page_level" in summary["pipelines"]
    assert "block_level/structure_layout" in summary["pipelines"]
    assert "block_level/hybrid_layout" in summary["pipelines"]
    assert "page_level_HMER" in summary["pipelines"]
    assert "overview" in summary
