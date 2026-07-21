"""Tests for spatial metrics."""

from __future__ import annotations

import numpy as np

from heatmap_analysis.config import HeatmapConfig
from heatmap_analysis.heatmap import HeatmapResult, compute_relative_heatmap, ink_to_normalized_grid
from heatmap_analysis.metrics import compute_metrics


def _make_hm(grid: np.ndarray) -> HeatmapResult:
    d_rel, is_blank = compute_relative_heatmap(grid)
    return HeatmapResult(
        d_abs=grid,
        d_rel=d_rel,
        d_abs_smooth=grid,
        d_rel_smooth=d_rel,
        is_blank=is_blank,
        grid_size=grid.shape[0],
    )


def test_centroid_centered_stroke():
    gs = 32
    grid = np.zeros((gs, gs))
    grid[8, 16] = 1.0
    d_rel, _ = compute_relative_heatmap(grid)
    hm = HeatmapResult(grid, d_rel, grid, d_rel, False, gs)
    ink = np.zeros((100, 100))
    ink[30, 50] = 1
    m = compute_metrics("t", hm, ink, HeatmapConfig(grid_size=gs))
    assert 0.45 < m.centroid_x < 0.55
    assert 0.20 < m.centroid_y < 0.35


def test_quarter_shares_sum_to_one():
    gs = 32
    rng = np.random.default_rng(0)
    grid = rng.random((gs, gs))
    hm = _make_hm(grid)
    ink = grid
    m = compute_metrics("t", hm, ink, HeatmapConfig(grid_size=gs))
    v_sum = (
        m.top_quarter_share
        + m.upper_middle_share
        + m.lower_middle_share
        + m.bottom_quarter_share
    )
    assert abs(v_sum - 1.0) < 1e-6


def test_spatial_entropy_in_unit_interval():
    gs = 16
    grid = np.ones((gs, gs))
    hm = _make_hm(grid)
    m = compute_metrics("t", hm, grid, HeatmapConfig(grid_size=gs))
    assert 0.0 <= m.spatial_entropy <= 1.0
