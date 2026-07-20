"""Shared pytest fixtures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from heatmap_analysis.config import ClusteringConfig, GpuConfig, OutputConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config.example.yaml"


@pytest.fixture
def synthetic_heatmap_cfg(tmp_path: Path):
    """Example config with output redirected to a temp directory."""
    if not EXAMPLE_CONFIG.exists():
        pytest.skip("config.example.yaml missing")
    cfg = load_config(EXAMPLE_CONFIG)
    return replace(
        cfg,
        output=OutputConfig(output_dir=tmp_path / "out", resume=False),
        clustering=replace(
            cfg.clustering,
            k_values=[2, 3],
            gmm_components=[2, 3],
            closest_samples=1,
            boundary_samples=1,
            min_samples_for_clustering=5,
        ),
        gpu=replace(cfg.gpu, enabled=False, clustering=False),
    )
