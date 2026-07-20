"""Tests for line validation error visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.line_level.models import LineAnnotation, LineLevelConfig, PageTask
from benchmark_design.report.line_level.error_visualization import (
    REASON_LABELS,
    collect_line_validation_errors,
    export_line_validation_error_figures,
)


def _line(
    *,
    line_id: str,
    polygon: tuple[tuple[float, float], ...],
    ocr: str = "sample",
) -> LineAnnotation:
    return LineAnnotation(
        image_id="error_page",
        image_name="error_page.png",
        line_id=line_id,
        block_order=0,
        line_order=0,
        block_type="Txtblock",
        polygon=polygon,
        ocr=ocr,
        source_file="error_page.png.json",
    )


def _make_page(tmp_path: Path, lines: tuple[LineAnnotation, ...]) -> PageTask:
    image_path = tmp_path / "error_page.png"
    array = np.full((120, 160), 230, dtype=np.uint8)
    array[20:40, 20:140] = 40
    Image.fromarray(array, mode="L").save(image_path)
    return PageTask(
        image_id="error_page",
        image_name="error_page.png",
        json_path=tmp_path / "error_page.png.json",
        image_path=image_path,
        width=160,
        height=120,
        dpi=None,
        lines=lines,
    )


def test_collect_line_validation_errors_detects_reasons(tmp_path: Path) -> None:
    page = _make_page(
        tmp_path,
        (
            _line(line_id="dup-a", polygon=((20, 20), (60, 20), (60, 40), (20, 40))),
            _line(line_id="dup-a", polygon=((70, 20), (110, 20), (110, 40), (70, 40))),
            _line(line_id="too-few", polygon=((10, 10), (20, 10))),
            _line(line_id="nan", polygon=((10, 10), (float("nan"), 20), (30, 30))),
            _line(line_id="oob", polygon=((150, 10), (170, 10), (170, 30), (150, 30))),
        ),
    )

    cases = collect_line_validation_errors([page])
    reasons = {case.reason for case in cases}

    assert "duplicate_line_id" in reasons
    assert "insufficient_points" in reasons
    assert "non_finite_coordinates" in reasons
    assert "out_of_bounds" in reasons
    assert len(cases) == 4


def test_export_line_validation_error_figures_writes_outputs(tmp_path: Path) -> None:
    page = _make_page(
        tmp_path,
        (
            _line(line_id="dup-a", polygon=((20, 20), (60, 20), (60, 40), (20, 40))),
            _line(line_id="dup-a", polygon=((70, 20), (110, 20), (110, 40), (70, 40))),
            _line(line_id="too-few", polygon=((10, 10), (20, 10))),
        ),
    )
    input_root = tmp_path / "input"
    input_root.mkdir()
    output_root = tmp_path / "line_error"
    config = LineLevelConfig(
        input_root=input_root,
        output_root=output_root,
        workers=1,
        show_progress=False,
    )

    from benchmark_design.report.line_level import error_visualization as module

    original_discover = module.discover_pages_from_benchmark
    try:
        module.discover_pages_from_benchmark = lambda _config: [page]
        summary = export_line_validation_error_figures(config, output_root)
    finally:
        module.discover_pages_from_benchmark = original_discover

    assert summary["total_error_count"] == 2
    assert (output_root / "summary.json").is_file()
    assert (output_root / "error_index.csv").is_file()
    for reason in REASON_LABELS:
        assert (output_root / reason).is_dir()

    written = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert written["counts_by_reason"]["duplicate_line_id"] == 1
    assert written["counts_by_reason"]["insufficient_points"] == 1
    assert (output_root / "duplicate_line_id").glob("*.png")
    assert list((output_root / "duplicate_line_id").glob("*.png"))
    assert list((output_root / "insufficient_points").glob("*.png"))
