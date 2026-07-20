"""Tests for confusable token figure exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.report.confusable_token_figures import export_confusable_token_example_figures


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
        structure_types=(),
        structure_type_count=0,
        structure_max_depths={},
        ast_depth=0,
        mean_token_nested_level=0.0,
        parse_status="ok",
    )


def test_export_confusable_token_example_figures_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image_path = tmp_path / "sample.jpg"
    image = Image.new("RGB", (200, 120), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 90), fill=(0, 0, 0))
    image.save(image_path)

    record_four = _record(
        "ours:sample:0:0",
        "sample.jpg",
        [[20, 20], [180, 20], [180, 90], [20, 90]],
        r"x + 4 + y",
    )
    record_varphi = _record(
        "ours:sample:0:1",
        "sample.jpg",
        [[20, 20], [180, 20], [180, 90], [20, 90]],
        r"\varphi + 1",
    )
    feature_four = _feature("ours:sample:0:0", r"x + 4 + y", ("x", "+", "4", "+", "y"))
    feature_varphi = _feature("ours:sample:0:1", r"\varphi + 1", (r"\varphi", "+", "1"))
    counts = export_confusable_token_example_figures(
        [record_four, record_varphi],
        [feature_four, feature_varphi],
        input_dir=tmp_path,
        figures_root=tmp_path / "figures",
        per_token=1,
    )
    assert counts["greek-variant/4"] == 1
    assert counts["greek-variant/varphi"] == 1
    assert list((tmp_path / "figures" / "greek-variant" / "4").glob("example_*.png"))
    assert list((tmp_path / "figures" / "greek-variant" / "varphi").glob("example_*.png"))
