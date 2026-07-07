"""Robust grayscale normalization and page-level darkness maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DARKNESS_BINS = 256
DEFAULT_Q_LOW = 1.0
DEFAULT_Q_HIGH = 99.0


@dataclass(frozen=True, slots=True)
class PercentileNormConfig:
    q_low: float = DEFAULT_Q_LOW
    q_high: float = DEFAULT_Q_HIGH


def robust_normalize_gray(
    gray: np.ndarray,
    *,
    q_low: float = DEFAULT_Q_LOW,
    q_high: float = DEFAULT_Q_HIGH,
) -> np.ndarray:
    """Map grayscale to G_tilde in [0, 1] via robust percentile normalization."""
    values = np.asarray(gray, dtype=np.float32)
    lo = float(np.percentile(values, q_low))
    hi = float(np.percentile(values, q_high))
    if hi <= lo:
        hi = lo + 1.0
    normalized = (values - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def compute_darkness_from_gray(
    gray: np.ndarray,
    *,
    q_low: float = DEFAULT_Q_LOW,
    q_high: float = DEFAULT_Q_HIGH,
) -> np.ndarray:
    """S(x) = 1 - G_tilde(x); higher values indicate darker ink."""
    normalized = robust_normalize_gray(gray, q_low=q_low, q_high=q_high)
    return (1.0 - normalized).astype(np.float32, copy=False)


def gray_histogram(values: np.ndarray, *, bins: int = 256) -> np.ndarray:
    """Histogram of uint8 grayscale values on [0, 255]."""
    flat = np.asarray(values, dtype=np.uint8).ravel()
    if flat.size == 0:
        return np.zeros(bins, dtype=np.int64)
    return np.bincount(flat, minlength=bins).astype(np.int64, copy=False)


def normalized_gray_histogram(
    gray: np.ndarray,
    *,
    q_low: float = DEFAULT_Q_LOW,
    q_high: float = DEFAULT_Q_HIGH,
    bins: int = DARKNESS_BINS,
) -> np.ndarray:
    """Histogram of robust-normalized grayscale inside [0, 1]."""
    normalized = robust_normalize_gray(gray, q_low=q_low, q_high=q_high)
    return darkness_histogram_in_mask(normalized, np.ones(normalized.shape, dtype=bool), bins=bins)


def darkness_histogram_in_mask(
    darkness: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int = DARKNESS_BINS,
) -> np.ndarray:
    """Build histogram of darkness values inside *mask* on [0, 1]."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    values = darkness[mask]
    if values.size == 0:
        return np.zeros(bins, dtype=np.int64)
    clipped = np.clip(values, 0.0, 1.0)
    bin_indices = np.minimum((clipped * bins).astype(np.int64), bins - 1)
    return np.bincount(bin_indices, minlength=bins).astype(np.int64, copy=False)


def merge_histograms(histograms: list[np.ndarray]) -> np.ndarray:
    if not histograms:
        return np.zeros(DARKNESS_BINS, dtype=np.int64)
    merged = np.zeros_like(histograms[0], dtype=np.int64)
    for hist in histograms:
        merged += hist.astype(np.int64, copy=False)
    return merged
