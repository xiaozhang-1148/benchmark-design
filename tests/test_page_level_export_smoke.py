"""Smoke test for page-level export pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.report.page_level.export_pipeline import run_page_level_export

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "sample_page.json"


def _make_fixture_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "sample.jpg.json")
    array = np.full((16, 20), 220, dtype=np.uint8)
    array[4:10, 6:14] = 50
    Image.fromarray(array, mode="L").save(input_dir / "sample.png")
    return input_dir


def test_run_page_level_export_smoke(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "page_level"
    manifest = run_page_level_export(
        input_dir,
        output_root,
        workers=1,
        show_progress=False,
        skip_figures=True,
    )

    assert (output_root / "tables" / "image_inventory.parquet").is_file()
    assert (output_root / "tables" / "image_features.parquet").is_file()
    assert (output_root / "calibration" / "calibration.json").is_file()
    assert (output_root / "report" / "image_analysis_report.md").is_file()
    assert (output_root / "report" / "run_manifest.json").is_file()
    assert not (output_root / "heatmaps").exists()
    assert not (output_root / "tables" / "scan_quality_metrics.parquet").exists()
    assert not (output_root / "report" / "scan_quality_report.md").exists()

    calibration = json.loads((output_root / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    assert "global_threshold" in calibration
    assert manifest["report"] == "report/image_analysis_report.md"

    features = (output_root / "tables" / "image_features.csv").read_text(encoding="utf-8")
    assert "foreground_density" in features
    assert "aspect_ratio" in features
    assert "foreground_background_contrast" not in features
