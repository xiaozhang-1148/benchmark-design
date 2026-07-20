"""Core page-level image analysis pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark_design.page_level.calibration import compute_calibration_from_histograms
from benchmark_design.page_level.features import compute_all_page_image_metrics
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.inventory import (
    build_image_inventory_and_histograms,
    discover_images_from_benchmark,
)
from benchmark_design.page_level.models import PageLevelAnalysisResult, PageLevelConfig


def run_page_level_analysis(
    config: PageLevelConfig,
    *,
    gray_cache_root: Path | None = None,
    cleanup_gray_cache: bool = True,
) -> PageLevelAnalysisResult:
    """Discover images, build inventory, calibrate, and compute per-image features."""
    records = discover_images_from_benchmark(
        config.input_root,
        show_progress=config.show_progress,
        workers=config.workers,
    )
    if not records:
        raise ValueError(f"No benchmark images discovered under {config.input_root}")

    inventory, raw_histograms = build_image_inventory_and_histograms(
        records,
        show_progress=config.show_progress,
        workers=config.workers,
    )
    if gray_cache_root is None:
        cache_root = Path(tempfile.mkdtemp(prefix="page_level_gray_"))
        owns_cache = True
    else:
        cache_root = gray_cache_root
        owns_cache = cleanup_gray_cache
    gray_cache = PageGrayCache(cache_root)
    try:
        calibration = compute_calibration_from_histograms(
            raw_histograms,
            records,
            config,
            gray_cache=gray_cache,
        )
        page_metrics = compute_all_page_image_metrics(
            records,
            inventory,
            calibration,
            config,
            show_progress=config.show_progress,
            workers=config.workers,
            gray_cache=gray_cache,
        )
    finally:
        if owns_cache:
            gray_cache.cleanup()
    features = [item.features for item in page_metrics]

    return PageLevelAnalysisResult(
        config=config,
        image_records=tuple(records),
        inventory=tuple(inventory),
        calibration=calibration,
        features=tuple(features),
    )
