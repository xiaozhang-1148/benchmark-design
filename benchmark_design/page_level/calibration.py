"""Dataset-level grayscale calibration with pooled Otsu threshold."""

from __future__ import annotations

import numpy as np

from benchmark_design.foreground.normalize import compute_g_tilde
from benchmark_design.foreground.threshold import (
    accumulate_grayscale_histogram,
    global_pooled_otsu_gray_threshold,
    gray_threshold_to_tau_d,
)
from benchmark_design.io.image import gray_histogram, load_grayscale_image
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.models import CalibrationResult, ImageRecord, PageLevelConfig
from benchmark_design.progress import parallel_map


def density_gray_histogram(gray: np.ndarray) -> np.ndarray:
    hist = gray_histogram(gray).astype(np.float64)
    area = gray.size
    if area == 0:
        return np.zeros(256, dtype=np.float64)
    return hist / area


def average_equal_image_histograms(histograms: list[np.ndarray]) -> np.ndarray:
    if not histograms:
        return np.zeros(256, dtype=np.float64)
    stacked = np.stack(histograms, axis=0)
    return stacked.mean(axis=0)


def percentile_from_density_histogram(hist: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(hist)
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    target = total * (percentile / 100.0)
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(np.clip(index, 0, 255))


def normalize_gray_to_uint8(
    gray: np.ndarray,
    *,
    dark_reference: float,
    light_reference: float,
) -> np.ndarray:
    """Legacy uint8 view of G~ for histogram exports only."""
    g_tilde = compute_g_tilde(
        gray,
        dark_reference=dark_reference,
        light_reference=light_reference,
    )
    return (g_tilde * 255.0).astype(np.uint8)


def collect_density_histogram(record: ImageRecord) -> np.ndarray:
    gray = load_grayscale_image(record.absolute_path)
    return density_gray_histogram(gray)


def collect_normalized_density_histogram(
    args: tuple[ImageRecord, float, float, PageGrayCache | None],
) -> np.ndarray:
    record, dark_reference, light_reference, gray_cache = args
    if gray_cache is not None:
        gray = gray_cache.try_load(record.image_id)
        if gray is None:
            gray = load_grayscale_image(record.absolute_path)
            gray_cache.store(record.image_id, gray)
    else:
        gray = load_grayscale_image(record.absolute_path)
    normalized = normalize_gray_to_uint8(
        gray,
        dark_reference=dark_reference,
        light_reference=light_reference,
    )
    return density_gray_histogram(normalized)


def collect_pooled_grayscale_histogram(
    args: tuple[ImageRecord, PageGrayCache | None],
) -> np.ndarray:
    record, gray_cache = args
    if gray_cache is not None:
        gray = gray_cache.try_load(record.image_id)
        if gray is None:
            gray = load_grayscale_image(record.absolute_path)
            gray_cache.store(record.image_id, gray)
    else:
        gray = load_grayscale_image(record.absolute_path)
    return accumulate_grayscale_histogram(gray)


def compute_calibration_from_histograms(
    histograms: list[np.ndarray],
    records: list[ImageRecord],
    config: PageLevelConfig,
    *,
    gray_cache: PageGrayCache | None = None,
) -> CalibrationResult:
    """Finish calibration when raw per-image histograms were collected elsewhere."""
    average_hist = average_equal_image_histograms(histograms)
    dark_reference = percentile_from_density_histogram(average_hist, config.dark_percentile)
    light_reference = percentile_from_density_histogram(average_hist, config.light_percentile)
    if light_reference <= dark_reference:
        light_reference = dark_reference + 1.0

    normalized_args = [
        (record, dark_reference, light_reference, gray_cache) for record in records
    ]
    normalized_histograms = parallel_map(
        collect_normalized_density_histogram,
        normalized_args,
        description="Building normalized grayscale histograms",
        show_progress=config.show_progress,
        workers=config.workers,
    )
    normalized_average = average_equal_image_histograms(normalized_histograms)

    gray_args = [(record, gray_cache) for record in records]
    page_gray_histograms = parallel_map(
        collect_pooled_grayscale_histogram,
        gray_args,
        description="Building pooled grayscale histograms",
        show_progress=config.show_progress,
        workers=config.workers,
    )
    pooled_gray = np.zeros(256, dtype=np.int64)
    for page_hist in page_gray_histograms:
        pooled_gray += page_hist

    gray_threshold = global_pooled_otsu_gray_threshold(pooled_gray)
    tau_d = gray_threshold_to_tau_d(
        gray_threshold,
        dark_reference=dark_reference,
        light_reference=light_reference,
    )

    return CalibrationResult(
        dark_reference=dark_reference,
        light_reference=light_reference,
        gray_threshold=gray_threshold,
        tau_d=tau_d,
        dark_percentile=config.dark_percentile,
        light_percentile=config.light_percentile,
        threshold_method="global_pooled_otsu",
        image_count=len(records),
        gray_histogram=tuple(int(v) for v in pooled_gray.tolist()),
        average_histogram=tuple(int(round(v * 1_000_000)) for v in average_hist),
        normalized_average_histogram=tuple(int(round(v * 1_000_000)) for v in normalized_average),
    )


def compute_calibration(
    records: list[ImageRecord],
    config: PageLevelConfig,
) -> CalibrationResult:
    histograms = parallel_map(
        collect_density_histogram,
        records,
        description="Building equal-image-weighted grayscale histograms",
        show_progress=config.show_progress,
        workers=config.workers,
    )
    return compute_calibration_from_histograms(histograms, records, config)
