"""Tests for GPU image operations."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from heatmap_analysis.gpu import is_gpu_available
from heatmap_analysis import image_ops as iops


def _stroke_page(w: int = 400, h: int = 600) -> np.ndarray:
    img = np.full((h, w), 240, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), 200, 2)
    # Diagonal and curved strokes avoid H/V line suppression
    cv2.line(img, (w // 5, h // 4), (4 * w // 5, h // 3), 30, 3)
    cv2.line(img, (w // 3, h // 2), (2 * w // 3, 3 * h // 4), 30, 3)
    for i in range(5):
        cv2.circle(img, (w // 2 + i * 8, h // 2 + i * 5), 4, 25, -1)
    return img


@pytest.mark.skipif(not is_gpu_available(), reason="CUDA not available")
def test_gpu_otsu_matches_cpu_shape():
    import cupy as cp

    gray = _stroke_page()
    cpu = iops.otsu_binary_inv(gray, xp=np)
    gpu = iops.otsu_binary_inv(gray, xp=cp)
    assert gpu.shape == cpu.shape
    assert int(cp.asnumpy(gpu).sum()) > 0


@pytest.mark.skipif(not is_gpu_available(), reason="CUDA not available")
def test_gpu_pipeline_produces_ink():
    import cupy as cp
    from heatmap_analysis.config import PreprocessingConfig
    from heatmap_analysis.handwriting import extract_ink_mask

    gray = _stroke_page()
    weights, _, info = extract_ink_mask(gray, PreprocessingConfig(), use_gpu=True, xp=cp)
    assert info["backend"] == "gpu"
    assert float(cp.asnumpy(weights).sum()) > 0
