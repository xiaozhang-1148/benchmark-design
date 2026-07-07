"""Vision export pipeline tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchmark_design.report.vision.export_pipeline import run_vision_benchmark_export
from benchmark_design.vision.processing_options import VisionProcessingOptions

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_run_vision_export_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "sample.jpg.json")
    (input_dir / "sample.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    output_root = tmp_path / "vision_out"
    manifest = run_vision_benchmark_export(
        input_dir,
        output_root,
        processing=VisionProcessingOptions(show_progress=False, read_image_dimensions=False),
    )

    assert (output_root / "vision_benchmark_summary.md").is_file()
    assert (output_root / "tables" / "sample_index.csv").is_file()
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["domain"] == "vision"
    assert metadata["sample_count"] == 1
    assert metadata["page_count"] == 1
    assert (output_root / "tables" / "flow_structure_page_metrics.csv").is_file()
    assert (output_root / "tables" / "flow_structure_block_geometry.csv").is_file()
    assert (output_root / "tables" / "deleted_block_scale_page_metrics.csv").is_file()
    assert (output_root / "deleted_block_scale_summary.md").is_file()
    assert "deleted_block_scale" in metadata
    assert metadata["sample_count"] == metadata["page_count"] == metadata["foreground_pixel_density_page_count"]
    assert metadata["page_count"] == metadata["deleted_block_scale_page_count"]
    assert "summary" in manifest
