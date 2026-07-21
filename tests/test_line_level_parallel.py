"""Parallel processing determinism for line-level analysis."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.line_level.config import load_line_level_config
from benchmark_design.line_level.pipeline import run_line_level_analysis

FIXTURE_JSON = Path(__file__).parent / "fixtures" / "line_page.json"


def _make_fixture_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
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


def _run_with_workers(tmp_path: Path, workers: int) -> list[tuple[str, str, float | None]]:
    input_dir = _make_fixture_input(tmp_path / f"workers_{workers}")
    config = load_line_level_config(
        None,
        input_root=input_dir,
        output_root=tmp_path / f"out_{workers}",
        workers=workers,
        show_progress=False,
    )
    result = run_line_level_analysis(config)
    return [
        (
            row.image_id,
            row.line_id,
            row.bbox_height_px if row.is_valid else None,
            row.aspect_ratio if row.is_valid else None,
            row.orientation_deg if row.is_valid else None,
        )
        for row in result.line_metrics
    ]


def test_worker_counts_produce_identical_sorted_metrics(tmp_path: Path) -> None:
    baseline = _run_with_workers(tmp_path, 1)
    for workers in (2, 8):
        assert _run_with_workers(tmp_path, workers) == baseline
