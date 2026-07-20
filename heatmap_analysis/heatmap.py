"""Normalized coordinate heatmap generation (CPU + GPU vectorized)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from heatmap_analysis.config import HeatmapConfig
from heatmap_analysis.gpu import get_xp, to_numpy
from heatmap_analysis.utils import gaussian_smooth_grid


@dataclass
class HeatmapResult:
    d_abs: np.ndarray
    d_rel: np.ndarray
    d_abs_smooth: np.ndarray
    d_rel_smooth: np.ndarray
    is_blank: bool
    grid_size: int


def _cell_pixel_counts(h: int, w: int, grid_size: int, xp: Any) -> Any:
    """Vectorized per-cell pixel counts for a h×w image split into grid_size² cells."""
    row_bounds = xp.linspace(0, h, grid_size + 1).astype(xp.int64)
    col_bounds = xp.linspace(0, w, grid_size + 1).astype(xp.int64)
    row_h = xp.diff(row_bounds)
    col_w = xp.diff(col_bounds)
    row_h = xp.maximum(row_h, 1)
    col_w = xp.maximum(col_w, 1)
    return xp.outer(row_h, col_w).astype(xp.float64)


def ink_to_normalized_grid(
    ink_weights: np.ndarray,
    grid_size: int,
    *,
    use_gpu: bool = False,
    xp: Any | None = None,
) -> np.ndarray:
    """
    Map ink pixel weights to normalized [0,1]×[0,1] grid (fully vectorized).
    D_abs[i,j] = sum(weights in cell) / cell_pixel_count
    """
    if xp is None:
        xp = get_xp(use_gpu)

    h, w = ink_weights.shape[:2]
    ink = xp.asarray(ink_weights, dtype=xp.float64)
    cell_counts = _cell_pixel_counts(h, w, grid_size, xp)

    ys, xs = xp.nonzero(ink > 0)
    if int(ys.size) == 0:
        return to_numpy(xp.zeros((grid_size, grid_size), dtype=xp.float64))

    # Map pixel centers to grid indices via normalized coordinates
    gi = xp.minimum((ys * grid_size // max(h, 1)).astype(xp.int32), grid_size - 1)
    gj = xp.minimum((xs * grid_size // max(w, 1)).astype(xp.int32), grid_size - 1)
    wvals = ink[ys, xs]
    flat_idx = gi.astype(xp.int64) * grid_size + gj.astype(xp.int64)

    grid_flat = xp.bincount(flat_idx, weights=wvals, minlength=grid_size * grid_size).astype(xp.float64)
    grid = grid_flat.reshape(grid_size, grid_size) / cell_counts
    return to_numpy(grid)


def compute_relative_heatmap(d_abs: np.ndarray, blank_threshold: float = 1e-12) -> tuple[np.ndarray, bool]:
    total = float(np.sum(d_abs))
    if total <= blank_threshold:
        return np.zeros_like(d_abs), True
    return d_abs / total, False


def build_heatmaps(
    ink_weights: np.ndarray,
    config: HeatmapConfig,
    blank_threshold: float = 1e-12,
    *,
    use_gpu: bool = False,
    xp: Any | None = None,
) -> HeatmapResult:
    d_abs = ink_to_normalized_grid(
        ink_weights, config.grid_size, use_gpu=use_gpu, xp=xp
    )
    d_rel, is_blank = compute_relative_heatmap(d_abs, blank_threshold)

    d_abs_smooth = (
        gaussian_smooth_grid(d_abs, config.gaussian_sigma, use_gpu=use_gpu, xp=xp)
        if config.save_smoothed_grid
        else d_abs.copy()
    )
    if is_blank:
        d_rel_smooth = d_rel.copy()
    else:
        d_rel_smooth_raw = gaussian_smooth_grid(d_rel, config.gaussian_sigma, use_gpu=use_gpu, xp=xp)
        s = float(np.sum(d_rel_smooth_raw))
        d_rel_smooth = d_rel_smooth_raw / s if s > blank_threshold else d_rel_smooth_raw

    return HeatmapResult(
        d_abs=d_abs,
        d_rel=d_rel,
        d_abs_smooth=d_abs_smooth,
        d_rel_smooth=d_rel_smooth,
        is_blank=is_blank,
        grid_size=config.grid_size,
    )


def flatten_heatmap(grid: np.ndarray) -> np.ndarray:
    return grid.ravel().astype(np.float64)
