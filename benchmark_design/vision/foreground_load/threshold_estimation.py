"""Dataset-level darkness threshold estimation."""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from benchmark_design.vision.foreground_load.otsu import otsu_from_darkness_histogram

SENSITIVITY_DELTA = 0.03
DEFAULT_TAU_D = 0.5


class ThresholdMethod(StrEnum):
    BIMODAL_VALLEY = "bimodal_valley"
    GMM_INTERSECTION = "gmm_intersection"
    POOLED_OTSU = "pooled_otsu"


def _bin_centers(bins: int) -> np.ndarray:
    return (np.arange(bins, dtype=np.float64) + 0.5) / bins


def _smooth_histogram(hist: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return hist.astype(np.float64, copy=False)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(hist.astype(np.float64), kernel, mode="same")


def _local_maxima(values: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for index in range(1, len(values) - 1):
        if values[index] >= values[index - 1] and values[index] >= values[index + 1] and values[index] > 0:
            peaks.append(index)
    return peaks


def bimodal_valley_threshold(hist: np.ndarray) -> float | None:
    """Return darkness threshold at the valley between the two strongest histogram peaks."""
    counts = hist.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return None

    smoothed = _smooth_histogram(counts)
    peaks = _local_maxima(smoothed)
    if len(peaks) < 2:
        return None

    midpoint = len(hist) // 2
    lower_peaks = [index for index in peaks if index < midpoint]
    upper_peaks = [index for index in peaks if index >= midpoint]
    if not lower_peaks or not upper_peaks:
        ranked = sorted(peaks, key=lambda index: smoothed[index], reverse=True)
        left, right = sorted(ranked[:2])
    else:
        left = max(lower_peaks, key=lambda index: smoothed[index])
        right = max(upper_peaks, key=lambda index: smoothed[index])
    if right <= left:
        return None

    valley_index = left + int(np.argmin(smoothed[left : right + 1]))
    return float((valley_index + 0.5) / len(hist))


def _gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(std, 1e-4)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z)


def gmm_intersection_threshold(hist: np.ndarray, *, max_iter: int = 80) -> float | None:
    """Estimate threshold at the intersection of a two-component Gaussian mixture."""
    counts = hist.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return None

    centers = _bin_centers(len(hist))
    weights = counts / total
    cdf = np.cumsum(weights)

    def weighted_quantile(probability: float) -> float:
        index = int(np.searchsorted(cdf, probability, side="left"))
        index = min(max(index, 0), len(centers) - 1)
        return float(centers[index])

    mean = float(np.average(centers, weights=counts))
    variance = float(np.average((centers - mean) ** 2, weights=counts))
    std = max(variance**0.5, 0.05)

    mu1 = weighted_quantile(0.25)
    mu2 = weighted_quantile(0.75)
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
    sigma1 = sigma2 = std
    pi1 = 0.5

    for _ in range(max_iter):
        resp1 = pi1 * _gaussian_pdf(centers, mu1, sigma1)
        resp2 = (1.0 - pi1) * _gaussian_pdf(centers, mu2, sigma2)
        denom = resp1 + resp2 + 1e-12
        r1 = resp1 / denom
        r2 = resp2 / denom

        w1 = weights * r1
        w2 = weights * r2
        if w1.sum() <= 0 or w2.sum() <= 0:
            return None

        mu1 = float(np.average(centers, weights=w1))
        mu2 = float(np.average(centers, weights=w2))
        sigma1 = max(float(np.sqrt(np.average((centers - mu1) ** 2, weights=w1))), 1e-4)
        sigma2 = max(float(np.sqrt(np.average((centers - mu2) ** 2, weights=w2))), 1e-4)
        pi1 = float(w1.sum())

    if abs(sigma1 - sigma2) < 1e-6:
        return float((mu1 + mu2) / 2.0)

    a = 1.0 / (2.0 * sigma2 * sigma2) - 1.0 / (2.0 * sigma1 * sigma1)
    b = mu1 / (sigma1 * sigma1) - mu2 / (sigma2 * sigma2)
    c = (mu2 * mu2) / (2.0 * sigma2 * sigma2) - (mu1 * mu1) / (2.0 * sigma1 * sigma1) + np.log(sigma2 / sigma1)
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return float((mu1 + mu2) / 2.0)
        return float(np.clip(-c / b, 0.0, 1.0))

    roots = np.roots([a, b, c])
    real_roots = [float(root.real) for root in roots if abs(root.imag) < 1e-8]
    between = [root for root in real_roots if min(mu1, mu2) <= root <= max(mu1, mu2)]
    if between:
        return float(np.clip(between[0], 0.0, 1.0))
    if real_roots:
        return float(np.clip(real_roots[0], 0.0, 1.0))
    return float(np.clip((mu1 + mu2) / 2.0, 0.0, 1.0))


def estimate_dataset_threshold(
    hist: np.ndarray,
    *,
    default_tau_D: float = DEFAULT_TAU_D,
) -> tuple[float, ThresholdMethod]:
    """Estimate fixed dataset threshold tau_D from pooled calibration histogram."""
    if hist.sum() <= 0:
        return default_tau_D, ThresholdMethod.POOLED_OTSU

    valley = bimodal_valley_threshold(hist)
    if valley is not None and 0.0 < valley < 1.0:
        return valley, ThresholdMethod.BIMODAL_VALLEY

    gmm = gmm_intersection_threshold(hist)
    if gmm is not None and 0.0 < gmm < 1.0:
        return gmm, ThresholdMethod.GMM_INTERSECTION

    return otsu_from_darkness_histogram(hist), ThresholdMethod.POOLED_OTSU
