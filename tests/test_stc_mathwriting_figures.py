"""Tests for MathWriting STC high-complexity figure exports."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.io.dataset_loaders import load_mathwriting
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.report.stc_figures import (
    _correct_mathwriting_display_orientation,
    _load_expression_image,
    _resolve_expression_image_path,
    export_stc_high_complexity_figures,
)

FIXTURES = Path(__file__).parent / "fixtures" / "datasets" / "mathwriting"


def _ensure_mathwriting_fixture_image() -> Path:
    png_path = FIXTURES / "train" / "shard-000" / "expr001.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not png_path.exists():
        image = Image.new("RGB", (120, 40), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((8, 10), "sqrt(x)", fill=(0, 0, 0))
        image.save(png_path)
    return png_path


def test_resolve_mathwriting_image_path() -> None:
    _ensure_mathwriting_fixture_image()
    records = load_mathwriting(FIXTURES, dataset="MathWriting")
    record = records[0]
    path = _resolve_expression_image_path(record, input_dir=FIXTURES)
    assert path is not None
    assert path.name == "expr001.png"


def test_export_mathwriting_stc_figure_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _ensure_mathwriting_fixture_image()
    records = load_mathwriting(FIXTURES, dataset="MathWriting")
    record = records[0]
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
    feature = ExpressionFeatures(
        expression_id=record.expression_id,
        dataset=record.dataset,
        source_file=record.source_file,
        line_id=record.line_id,
        normalized_latex=record.ocr,
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
    assert _load_expression_image(record, input_dir=FIXTURES) is not None

    counts = export_stc_high_complexity_figures(
        [record],
        [feature],
        input_dir=FIXTURES,
        figures_root=tmp_path / "stc_high_complexity_mathwriting",
    )
    assert counts["stc_high_complexity"] == 1
    assert (tmp_path / "stc_high_complexity_mathwriting" / "01_MathWriting_train_shard-000_expr001.png").exists()


def test_mathwriting_display_orientation_flips_vertical_only() -> None:
    pytest.importorskip("numpy")
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 12, 12), fill=(0, 0, 0))
    corrected = _correct_mathwriting_display_orientation(image)
    corrected_arr = np.array(corrected)
    assert (corrected_arr[-5:, :5] < 128).any()
    assert not (corrected_arr[:5, :5] < 128).any()
