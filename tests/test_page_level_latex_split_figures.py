"""Tests for Chapter 7 cross-domain tables and figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmark_design.page_level_latex_split.config import load_split_config
from benchmark_design.page_level_latex_split.cross_domain_tables import (
    assign_numeric_bin,
    write_cross_domain_tables,
    write_split_category_table,
)
from benchmark_design.page_level_latex_split.figures_ch7 import (
    export_ch7_figures,
    figure7_3_foreground_density,
    figure7_4_layout_composition,
    figure7_5_line_geometry,
    figure7_6_expression_difficulty,
)


def test_assign_numeric_bin_uses_half_open_intervals() -> None:
    config = load_split_config(Path("config/page_level_latex_split.yaml"))
    assert assign_numeric_bin(0.019, config.foreground_density_bins) == "density_lt_2pct"
    assert assign_numeric_bin(0.02, config.foreground_density_bins) == "density_2_4pct"
    assert assign_numeric_bin(0.08, config.foreground_density_bins) == "density_8_10pct"
    assert assign_numeric_bin(0.10, config.foreground_density_bins) == "density_ge_10pct"


def test_write_split_category_table_overall_and_splits(tmp_path: Path) -> None:
    split_counts = {
        "train": {"a": 8, "b": 2},
        "val": {"a": 1, "b": 1},
        "test": {"a": 1, "b": 0},
    }
    path = tmp_path / "sample.csv"
    write_split_category_table(path, ("a", "b"), split_counts)
    frame = pd.read_csv(path)
    overall_a = frame[(frame["split"] == "overall") & (frame["category"] == "a")].iloc[0]
    train_a = frame[(frame["split"] == "train") & (frame["category"] == "a")].iloc[0]
    assert overall_a["count"] == 10
    assert overall_a["ratio"] == pytest.approx(10 / 13)
    assert train_a["count"] == 8
    assert train_a["ratio"] == 0.8


def test_cross_domain_tables_and_figures_smoke(tmp_path: Path) -> None:
    config = load_split_config(Path("config/page_level_latex_split.yaml"))
    export_root = tmp_path / "benchmark_export"
    split_root = export_root / "page_level_latex_split"
    tables = split_root / "tables"
    figures = split_root / "figures"
    tables.mkdir(parents=True)

    manifest = pd.DataFrame(
        [
            {"page_id": "page_a", "split": "train", "annotation_path": ""},
            {"page_id": "page_b", "split": "val", "annotation_path": ""},
            {"page_id": "page_c", "split": "test", "annotation_path": ""},
        ]
    )
    ann_dir = tmp_path / "anns"
    ann_dir.mkdir()
    ann_paths = []
    for page_id, blocks in (
        ("page_a", [{"type": "Txtblock"}, {"type": "figure"}]),
        ("page_b", [{"type": "Txtblock"}]),
        ("page_c", [{"type": "chart"}]),
    ):
        ann_path = ann_dir / f"{page_id}.json"
        ann_path.write_text(json.dumps({"blocks": blocks}), encoding="utf-8")
        ann_paths.append(str(ann_path))
    manifest["annotation_path"] = ann_paths

    page_level = export_root / "page_level" / "tables"
    page_level.mkdir(parents=True)
    pd.DataFrame(
        {
            "image_id": manifest["page_id"],
            "foreground_density": [0.01, 0.03, 0.07],
        }
    ).to_csv(page_level / "image_features.csv", index=False)

    structure_layout = export_root / "block_level" / "structure_layout" / "tables"
    structure_layout.mkdir(parents=True)
    pd.DataFrame(
        {
            "page_id": ["page_a.jpg", "page_c.jpg"],
            "flow_structure": ["Single-flow", "Columnar-flow"],
        }
    ).to_csv(structure_layout / "flow_structure_page_metrics.csv", index=False)
    hybrid_layout = export_root / "block_level" / "hybrid_layout" / "tables"
    hybrid_layout.mkdir(parents=True)
    pd.DataFrame(
        {
            "page_id": ["page_b.jpg"],
            "flow_structure": ["Hybrid-layout"],
        }
    ).to_csv(hybrid_layout / "flow_structure_page_metrics.csv", index=False)
    pd.DataFrame(
        {
            "page_id": ["page_a.jpg", "page_a.jpg", "page_b.jpg", "page_c.jpg"],
            "block_type": ["Txtblock", "figure", "Txtblock", "chart"],
        }
    ).to_csv(structure_layout / "flow_structure_block_geometry.csv", index=False)
    pd.DataFrame(
        {
            "page_id": ["page_a.jpg", "page_a.jpg", "page_b.jpg", "page_c.jpg"],
            "block_id": ["a0", "a1", "b0", "c0"],
            "block_type": ["Txtblock", "figure", "Txtblock", "chart"],
            "block_order": [0, 1, 0, 0],
            "foreground_density": [0.5, 0.5, 0.5, 0.5],
            "annotation_pixel_count": [10, 10, 10, 10],
            "foreground_pixel_count": [5, 5, 5, 5],
        }
    ).to_csv(structure_layout / "block_foreground_density.csv", index=False)
    pd.DataFrame(
        {
            "page_id": ["page_b.jpg"],
            "block_id": ["b0"],
            "block_type": ["deleted_text_block"],
            "block_order": [0],
            "foreground_density": [0.5],
            "annotation_pixel_count": [10],
            "foreground_pixel_count": [5],
        }
    ).to_csv(hybrid_layout / "block_foreground_density.csv", index=False)

    line_level = export_root / "line_level"
    line_level.mkdir(parents=True)
    pd.DataFrame(
        {
            "image_id": ["page_a", "page_a", "page_b", "page_c"],
            "aspect_ratio": [2.5, 9.0, 4.0, 13.0],
            "interference_status": ["zero_no_foreground", "computed", "computed", "computed"],
            "interference_ratio": [0.0, 0.01, 0.04, 0.08],
            "is_valid": [True, True, True, True],
        }
    ).to_csv(line_level / "line_metrics.csv", index=False)

    hmer = export_root / "HMER" / "details"
    hmer.mkdir(parents=True)
    pd.DataFrame(
        {
            "expression_id": [
                "ours:page_a.jpg:0:0",
                "ours:page_b.jpg:0:0",
                "ours:page_c.jpg:0:0",
            ],
            "token_length": [10, 25, 45],
            "structure_type_count": [1, 2, 3],
            "ast_depth": [1, 2, 3],
        }
    ).to_csv(hmer / "expression_level_statistics.csv", index=False)

    outputs = write_cross_domain_tables(manifest, export_root, config, tables)
    assert "table9_foreground_density_bins.csv" in outputs
    assert (tables / "table12_expression_difficulty.csv").is_file()

    flow_table = pd.read_csv(tables / "table10_flow_structure.csv")
    assert set(flow_table["category"].unique()) <= {"structure_layout", "hybrid_layout"}
    block_table = pd.read_csv(tables / "table10_block_type_composition.csv")
    train_figure = block_table[
        (block_table["split"] == "train") & (block_table["category"] == "figure")
    ].iloc[0]["count"]
    assert train_figure > 0
    assert block_table[
        (block_table["split"] == "test") & (block_table["category"] == "chart")
    ].iloc[0]["count"] > 0

    figures.mkdir(parents=True, exist_ok=True)
    for fn in (
        figure7_3_foreground_density,
        figure7_4_layout_composition,
        figure7_5_line_geometry,
        figure7_6_expression_difficulty,
    ):
        fn(tables, figures)
    assert (figures / "figure7_3_page_foreground_density.png").is_file()
    assert (figures / "figure7_4_layout_block_composition.png").is_file()
    assert (figures / "figure7_6_expression_difficulty.png").is_file()
