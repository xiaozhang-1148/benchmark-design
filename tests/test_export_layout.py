"""Tests for unified benchmark export layout and path resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark_design.export_layout import (
    BenchmarkExportLayout,
    cross_domain_inputs_available,
    flow_structure_page_metrics_csv,
    image_features_csv,
    load_flow_structure_page_metrics,
    write_export_pipeline_doc,
)


def test_benchmark_export_layout_paths(tmp_path: Path) -> None:
    layout = BenchmarkExportLayout(tmp_path / "benchmark_export")
    assert layout.page_level == layout.export_root / "page_level"
    assert layout.block_level == layout.export_root / "block_level"
    assert layout.structure_layout == layout.export_root / "block_level" / "structure_layout"
    assert layout.hybrid_layout == layout.export_root / "block_level" / "hybrid_layout"
    assert layout.page_level_hmer == layout.export_root / "page_level_HMER"
    assert layout.split_inputs == layout.export_root / "page_level_latex_split" / "inputs"


def test_cross_domain_path_resolution_prefers_new_layout(tmp_path: Path) -> None:
    export_root = tmp_path / "benchmark_export"
    page_level_tables = export_root / "page_level" / "tables"
    structure_tables = export_root / "block_level" / "structure_layout" / "tables"
    hybrid_tables = export_root / "block_level" / "hybrid_layout" / "tables"
    page_level_tables.mkdir(parents=True)
    structure_tables.mkdir(parents=True)
    hybrid_tables.mkdir(parents=True)
    pd.DataFrame({"image_id": ["p1"], "foreground_density": [0.01]}).to_csv(
        page_level_tables / "image_features.csv",
        index=False,
    )
    pd.DataFrame({"page_id": ["p1.jpg"], "flow_structure": ["Single-flow"]}).to_csv(
        structure_tables / "flow_structure_page_metrics.csv",
        index=False,
    )
    pd.DataFrame({"page_id": ["p2.jpg"], "flow_structure": ["Hybrid-layout"]}).to_csv(
        hybrid_tables / "flow_structure_page_metrics.csv",
        index=False,
    )
    line_level = export_root / "line_level"
    line_level.mkdir(parents=True)
    pd.DataFrame(
        {
            "image_id": ["p1"],
            "aspect_ratio": [2.0],
            "interference_status": ["computed"],
            "interference_ratio": [0.01],
            "is_valid": [True],
        }
    ).to_csv(line_level / "line_metrics.csv", index=False)
    hmer = export_root / "HMER" / "details"
    hmer.mkdir(parents=True)
    pd.DataFrame(
        {
            "expression_id": ["ours:p1.jpg:0:0"],
            "token_length": [5],
            "structure_type_count": [1],
            "ast_depth": [1],
        }
    ).to_csv(hmer / "expression_level_statistics.csv", index=False)

    assert image_features_csv(export_root).name == "image_features.csv"
    assert "block_level/structure_layout" in str(flow_structure_page_metrics_csv(export_root))
    merged = load_flow_structure_page_metrics(export_root)
    assert len(merged) == 2
    assert cross_domain_inputs_available(export_root)


def test_cross_domain_path_resolution_legacy_fallback(tmp_path: Path) -> None:
    export_root = tmp_path / "legacy_export"
    page_level = export_root / "page_level" / "tables"
    block_level = export_root / "block_level" / "tables"
    page_level.mkdir(parents=True)
    block_level.mkdir(parents=True)
    pd.DataFrame({"image_id": ["p1"], "foreground_density": [0.01]}).to_csv(
        page_level / "image_features.csv",
        index=False,
    )
    pd.DataFrame({"page_id": ["p1.jpg"], "flow_structure": ["Single-flow"]}).to_csv(
        block_level / "flow_structure_page_metrics.csv",
        index=False,
    )
    assert cross_domain_inputs_available(export_root) is False
    assert image_features_csv(export_root).is_file()
    assert flow_structure_page_metrics_csv(export_root).is_file()


def test_write_export_pipeline_doc(tmp_path: Path) -> None:
    export_root = tmp_path / "benchmark_export"
    path = write_export_pipeline_doc(export_root)
    text = path.read_text(encoding="utf-8")
    assert "block_level/structure_layout" in text
    assert "block_level/hybrid_layout" in text
    assert "page_id" in text
    assert path == export_root / "PIPELINE.md"
