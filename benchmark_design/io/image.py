"""Shared grayscale image I/O and histogram helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_grayscale_image(image_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            return gray.astype(np.uint8, copy=False)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for grayscale image loading. Install with: pip install Pillow"
        ) from exc
    with Image.open(image_path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


def gray_histogram(values: np.ndarray, *, bins: int = 256) -> np.ndarray:
    """Histogram of uint8 grayscale values on [0, 255]."""
    flat = np.asarray(values, dtype=np.uint8).ravel()
    if flat.size == 0:
        return np.zeros(bins, dtype=np.int64)
    return np.bincount(flat, minlength=bins).astype(np.int64, copy=False)


def otsu_from_histogram(hist: np.ndarray) -> float:
    counts = hist.astype(np.float64).ravel()
    if counts.size != 256:
        raise ValueError("histogram must have 256 bins")
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot compute Otsu on empty histogram")

    bin_indices = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(counts, bin_indices))
    weight_background = np.cumsum(counts)
    sum_background = np.cumsum(counts * bin_indices)
    weight_foreground = total - weight_background

    valid = (weight_background > 0) & (weight_foreground > 0)
    mean_background = np.divide(
        sum_background,
        weight_background,
        out=np.zeros_like(sum_background),
        where=weight_background > 0,
    )
    mean_foreground = np.divide(
        sum_total - sum_background,
        weight_foreground,
        out=np.zeros_like(weight_foreground),
        where=weight_foreground > 0,
    )
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    variance[~valid] = -1.0
    return float(np.argmax(variance))
