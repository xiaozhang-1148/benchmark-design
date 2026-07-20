"""Tests for page alignment and normalization."""

from __future__ import annotations

import cv2
import numpy as np

from heatmap_analysis.alignment import detect_page_region, normalize_to_canvas, pixel_to_normalized


def test_detect_page_region_finds_content():
    img = np.full((500, 400), 255, dtype=np.uint8)
    cv2.rectangle(img, (50, 60), (350, 440), 0, -1)
    region = detect_page_region(img)
    assert region.width > 200
    assert region.height > 200


def test_normalize_preserves_aspect_ratio():
    mask = np.zeros((200, 400), dtype=np.float32)
    mask[50:150, 100:300] = 1.0
    canvas, meta = normalize_to_canvas(mask, preserve_aspect_ratio=True, target_size=512)
    assert canvas.shape == (512, 512)
    assert meta["content_size"][1] > meta["content_size"][0]


def test_pixel_to_normalized_range():
    nx, ny = pixel_to_normalized(np.array([0, 99]), np.array([0, 199]), 100, 200)
    assert nx[0] == 0.0
    assert abs(nx[1] - 1.0) < 0.02
