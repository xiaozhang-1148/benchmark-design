"""Tests for STC high-complexity figure exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.report.stc_figures import export_stc_high_complexity_figures


def _record(expression_id: str, image_name: str, polygon: list[list[float]], ocr: str) -> ExpressionRecord:
    return ExpressionRecord(
        image_name=image_name,
        block_order=0,
        line_order=0,
        block_type="Txtblock",
        ocr=ocr,
        dataset="ours",
        source_file="sample.json",
        expression_id=expression_id,
        line_id="0:0",
        line_polygon=tuple((float(x), float(y)) for x, y in polygon),
    )


def _feature(expression_id: str, latex: str, tokens: tuple[str, ...]) -> ExpressionFeatures:
    return ExpressionFeatures(
        expression_id=expression_id,
        dataset="ours",
        source_file="sample.json",
        line_id="0:0",
        normalized_latex=latex,
        token_sequence=tokens,
        token_length=len(tokens),
        length_bin="short",
        is_duplicate=False,
        duplicate_group_id=0,
        duplicate_count=1,
        token_type_counts={},
        has_rare_1=False,
        has_rare_5=False,
        has_rare_10=False,
        structure_types=("分式", "上标", "根式"),
        structure_type_count=3,
        structure_max_depths={},
        ast_depth=4,
        mean_token_nested_level=0.0,
        parse_status="ok",
    )


def test_export_stc_high_complexity_figures_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image_path = tmp_path / "sample.jpg"
    image = Image.new("RGB", (200, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 90), fill=(0, 0, 0))
    image.save(image_path)

    tokens = (
        r"\begin",
        "{",
        "cases",
        "}",
        r"\frac",
        "{",
        "^",
        "{",
        r"\sqrt",
        "{",
        "x",
        "}",
        "}",
        "}",
        "{",
        "y",
        "}",
        r"\\",
        "z",
        r"\end",
        "{",
        "cases",
        "}",
    )
    record = _record(
        "ours:sample:0:0",
        "sample.jpg",
        [[20, 20], [180, 20], [180, 90], [20, 90]],
        r"\begin{cases} \frac{^{\sqrt{x}}}{y} \\ z \end{cases}",
    )
    feature = _feature(
        "ours:sample:0:0",
        r"\begin{cases} \frac{^{\sqrt{x}}}{y} \\ z \end{cases}",
        tokens,
    )

    counts = export_stc_high_complexity_figures(
        [record],
        [feature],
        input_dir=tmp_path,
        figures_root=tmp_path / "stc_high_complexity",
    )
    assert counts["stc_high_complexity"] == 1
    assert (tmp_path / "stc_high_complexity" / "01_ours_sample_0_0.png").exists()
    assert (tmp_path / "stc_high_complexity" / "index.csv").exists()
