"""Fast Otsu threshold from grayscale histograms."""

from __future__ import annotations

import numpy as np


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


def histogram_uint8(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint8).ravel()
    if flat.size == 0:
        return np.zeros(256, dtype=np.int64)
    return np.bincount(flat, minlength=256)


def otsu_threshold(gray_values: np.ndarray) -> float:
    return otsu_from_histogram(histogram_uint8(gray_values))


def foreground_count(values: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(np.asarray(values, dtype=np.uint8) <= threshold))


def otsu_from_darkness_histogram(hist: np.ndarray) -> float:
    """Return Otsu threshold in [0, 1] for darkness histogram with *bins* entries."""
    counts = hist.astype(np.float64).ravel()
    bins = counts.size
    if bins <= 0:
        raise ValueError("histogram must be non-empty")
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot compute Otsu on empty histogram")

    # Bin centers on (0, 1]; index i maps to (i + 0.5) / bins.
    bin_indices = (np.arange(bins, dtype=np.float64) + 0.5) / bins
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
    best_index = int(np.argmax(variance))
    return float((best_index + 0.5) / bins)


def foreground_count_darkness(darkness: np.ndarray, mask: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero((darkness >= threshold) & mask))


def ink_mass(darkness: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(darkness[mask].mean())
