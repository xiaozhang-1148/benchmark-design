"""Smoke tests for unified HMER + Vision export."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_design.ocr.processing import ProcessingOptions
from benchmark_design.report.unified_export import run_unified_benchmark_export
from benchmark_design.vision.processing_options import VisionProcessingOptions

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_run_unified_benchmark_export_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "sample.jpg.json")
    (input_dir / "sample.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    output_root = tmp_path / "benchmark_export"
    result = run_unified_benchmark_export(
        input_dir,
        output_root,
        processing=ProcessingOptions(show_progress=False),
        vision_processing=VisionProcessingOptions(show_progress=False, read_image_dimensions=False),
        skip_hmer_figures=True,
        skip_cross_benchmark=True,
        skip_flow_figures=True,
        skip_foreground_load_figures=True,
        skip_deleted_block_scale_figures=True,
    )

    assert result.output_root == output_root
    assert result.hmer_output == output_root / "HMER"
    assert result.vision_output == output_root / "vision"
    assert result.dataset_overview == output_root / "dataset_overview.md"

    assert (result.hmer_output / "summary.md").is_file()
    assert (result.vision_output / "vision_benchmark_summary.md").is_file()
    assert result.dataset_overview.is_file()
    assert not (output_root / "HMER" / "overview.md").exists()
    assert not (output_root / "vision" / "overview.md").exists()
    assert not (output_root / "benchmark_output").exists()
    assert not (output_root / "vision_benchmark_output").exists()
    overview_text = result.dataset_overview.read_text(encoding="utf-8")
    assert "HMER/summary.md" in overview_text
    assert "vision/vision_benchmark_summary.md" in overview_text


def test_run_unified_benchmark_export_with_progress(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "sample.jpg.json")
    (input_dir / "sample.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    result = run_unified_benchmark_export(
        input_dir,
        tmp_path / "benchmark_export_progress",
        processing=ProcessingOptions(show_progress=True),
        vision_processing=VisionProcessingOptions(show_progress=True, read_image_dimensions=False),
        skip_hmer_figures=True,
        skip_cross_benchmark=True,
        skip_flow_figures=True,
        skip_foreground_load_figures=True,
        skip_deleted_block_scale_figures=True,
    )

    assert result.dataset_overview.is_file()
