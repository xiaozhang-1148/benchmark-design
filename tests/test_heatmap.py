"""Tests for normalized heatmap generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from heatmap_analysis.config import HeatmapConfig, PreprocessingConfig
from heatmap_analysis.handwriting import extract_ink_mask
from heatmap_analysis.heatmap import build_heatmaps, ink_to_normalized_grid


def _stroke_image(w: int, h: int, x_frac: float, y_frac: float) -> np.ndarray:
    img = np.full((h, w), 240, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), 200, 2)
    x = int(x_frac * w)
    y = int(y_frac * h)
    cv2.line(img, (x - 30, y), (x + 30, y), 30, 4)
    cv2.line(img, (x - 10, y - 15), (x + 10, y + 15), 30, 3)
    return img


def test_same_relative_position_similar_heatmap():
    """Same relative stroke location yields similar normalized heatmaps across resolutions."""
    cfg = HeatmapConfig(grid_size=32, gaussian_sigma=0)

    def ink_at(size: tuple[int, int]) -> np.ndarray:
        w, h = size
        ink = np.zeros((h, w), dtype=np.float32)
        x0, x1 = int(0.35 * w), int(0.65 * w)
        y0, y1 = int(0.25 * h), int(0.35 * h)
        ink[y0:y1, x0:x1] = 1.0
        return ink

    refs = [build_heatmaps(ink_at(s), cfg).d_rel for s in [(400, 600), (800, 1200), (200, 300)]]
    for r in refs:
        assert abs(float(np.sum(r)) - 1.0) < 1e-6
    corr = np.corrcoef(refs[0].ravel(), refs[1].ravel())[0, 1]
    assert corr > 0.85


def test_grid_dimension_independent_of_image_size():
    cfg = HeatmapConfig(grid_size=64, gaussian_sigma=0)
    for size in [(200, 300), (1000, 1500)]:
        gray = _stroke_image(*size, 0.4, 0.6)
        ink, _, _ = extract_ink_mask(gray, PreprocessingConfig())
        hm = build_heatmaps(ink, cfg)
        assert hm.d_abs.shape == (64, 64)
        assert hm.d_rel.shape == (64, 64)


def test_relative_heatmap_sums_to_one():
    gray = _stroke_image(500, 700, 0.5, 0.5)
    ink, _, _ = extract_ink_mask(gray, PreprocessingConfig())
    hm = build_heatmaps(ink, HeatmapConfig(grid_size=16))
    assert abs(float(np.sum(hm.d_rel)) - 1.0) < 1e-6


def test_blank_image_no_division_by_zero():
    blank = np.full((400, 600), 255, dtype=np.uint8)
    ink, _, _ = extract_ink_mask(blank, PreprocessingConfig())
    hm = build_heatmaps(ink, HeatmapConfig())
    assert hm.is_blank
    assert float(np.sum(hm.d_rel)) == 0.0
