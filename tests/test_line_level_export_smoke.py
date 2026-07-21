"""Smoke test for line-level export pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.report.line_level.export_pipeline import run_line_level_export

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "line_page.json"


def _make_fixture_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(FIXTURE_JSON, input_dir / "line_page.jpg.json")
    array = np.full((160, 200), 230, dtype=np.uint8)
    for box in (
        (30, 20, 170, 32),
        (40, 45, 95, 70),
        (100, 46, 152, 68),
        (110, 51, 141, 62),
        (2, 2, 90, 14),
    ):
        x1, y1, x2, y2 = box
        array[y1:y2, x1:x2] = 40
    Image.fromarray(array, mode="L").save(input_dir / "line_page.png")
    return input_dir


def test_run_line_level_export_smoke(tmp_path: Path) -> None:
    input_dir = _make_fixture_input(tmp_path)
    output_root = tmp_path / "line_level"
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "dark_reference": 10.0,
                "light_reference": 255.0,
                "global_threshold": 128.0,
                "dark_percentile": 1.0,
                "light_percentile": 99.5,
                "threshold_method": "test",
                "image_count": 1,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "line_analysis.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_root: {input_dir}",
                f"output_root: {output_root}",
                "workers: 1",
                f"calibration_path: {calibration_path}",
                "bbox_outside_ink_enabled: true",
                "external_dataset_aspect_enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = run_line_level_export(
        input_dir,
        output_root,
        config_path=config_path,
        workers=1,
        show_progress=False,
        skip_figures=True,
    )

    assert (output_root / "line_metrics.csv").is_file()
    assert not (output_root / "line_metrics.parquet").exists()
    assert (output_root / "line_bbox_outside_ink.csv").is_file()
    assert (output_root / "page_metrics.csv").is_file()
    assert not (output_root / "page_metrics.parquet").exists()
    assert (output_root / "dataset_summary.json").is_file()
    assert (output_root / "report" / "line_analysis_report.md").is_file()
    assert (output_root / "report" / "run_manifest.json").is_file()
    assert not (output_root / "invalid_annotations.csv").exists()
    assert not (output_root / "processing_errors.csv").exists()
    assert not (output_root / "logs").exists()

    summary = json.loads((output_root / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["line_count"] == 5
    assert summary["image_count"] == 1
    assert "overlap_iou_tiers" not in summary
    assert "heatmap_bins" not in summary.get("config", {})
    assert "target_pair_relations" in summary
    assert summary["target_pair_relations"]["thresholds"] == {
        "height_similarity": 0.7,
        "vertical_overlap_ratio": 0.7,
        "horizontal_gap_px": 50.0,
    }
    assert "orientation_validity" in summary
    assert "bbox_outside_ink_natural_states" in summary
    assert (output_root / "tables" / "orientation_validity.csv").is_file()
    assert (output_root / "tables" / "spatial_relations.csv").is_file()
    assert (output_root / "tables" / "bbox_outside_ink_natural_states.csv").is_file()
    assert (output_root / "tables" / "bbox_outside_ink_calibration_threshold.json").is_file()
    assert manifest["report"] == "report/line_analysis_report.md"
    assert manifest["line_metrics"] == "line_metrics.csv"
    assert (output_root / "tables" / "target_pairs.csv").is_file()
    assert manifest["target_pairs"] == "tables/target_pairs.csv"

    import pandas as pd

    pairs = pd.read_csv(output_root / "tables" / "target_pairs.csv")
    for col in (
        "image_id",
        "line_id_a",
        "line_id_b",
        "intersection_area",
        "ioa",
        "horizontal_gap_px",
        "height_similarity",
        "vertical_overlap_px",
        "vertical_overlap_ratio",
        "ioa_positive",
        "horizontal_adjacent",
    ):
        assert col in pairs.columns
    assert pairs["line_id_a"].le(pairs["line_id_b"]).all()

    metrics = pd.read_csv(output_root / "line_metrics.csv")
    for col in (
        "center_x_norm",
        "center_y_norm",
        "nearest_line_id",
        "nearest_distance_px",
        "nearest_direction",
        "max_overlap_iou",
    ):
        assert col not in metrics.columns

    ink = pd.read_csv(output_root / "line_bbox_outside_ink.csv")
    assert "bbox_outside_ink_ratio" in ink.columns
    assert ink["bbox_outside_ink_ratio"].notna().any()
