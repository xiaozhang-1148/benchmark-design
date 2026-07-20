"""Block-level export pipeline tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark_design.report.block_level.export_pipeline import run_block_level_export
from benchmark_design.block_level.processing_options import VisionProcessingOptions

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_run_vision_export_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "sample.jpg.json")
    image = Image.new("RGB", (400, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 360, 360), fill=(0, 0, 0))
    image.save(input_dir / "sample.jpg")

    output_root = tmp_path / "vision_out"
    manifest = run_block_level_export(
        input_dir,
        output_root,
        processing=VisionProcessingOptions(show_progress=False, read_image_dimensions=False),
        skip_flow_figures=True,
    )

    assert (output_root / "block_level_summary.md").is_file()
    assert (output_root / "tables" / "sample_index.csv").is_file()
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["domain"] == "block_level"
    assert metadata["sample_count"] == 1
    assert metadata["page_count"] == 1
    assert (output_root / "tables" / "flow_structure_page_metrics.csv").is_file()
    assert (output_root / "tables" / "flow_structure_block_geometry.csv").is_file()
    assert (output_root / "flow_structure_summary.md").is_file()
    assert "summary" in manifest
    assert "flow_structure_page_metrics" in manifest
    assert "foreground_pixel_density_page_count" not in metadata
    assert "deleted_block_scale" not in metadata
    assert not (output_root / "tables" / "page_intrinsic").exists()
