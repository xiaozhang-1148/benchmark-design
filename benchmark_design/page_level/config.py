"""Load YAML configuration for page-level image analysis."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmark_design.config.hmer import DEFAULT_BENCHMARK_INPUT
from benchmark_design.config.page_level import (
    DEFAULT_ASPECT_RATIO_BINS,
    DEFAULT_PAGE_LEVEL_OUTPUT,
)
from benchmark_design.page_level.models import AspectRatioBin, PageLevelConfig


def load_page_level_config(
    config_path: Path | None = None,
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    workers: int | None = None,
    show_progress: bool = True,
) -> PageLevelConfig:
    payload: dict = {}
    if config_path is not None and config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    normalization = payload.get("normalization", {})
    aspect_ratio_groups = payload.get("aspect_ratio_groups", {})

    bins: list[AspectRatioBin] = []
    for item in aspect_ratio_groups.get("bins", []):
        bins.append(
            AspectRatioBin(
                name=str(item["name"]),
                min_ratio=float(item["min"]),
                max_ratio=float(item["max"]),
            )
        )
    aspect_enabled = bool(aspect_ratio_groups.get("enabled", True))
    if aspect_enabled and not bins:
        bins = list(DEFAULT_ASPECT_RATIO_BINS)

    resolved_input = input_root or Path(payload.get("input_root", DEFAULT_BENCHMARK_INPUT))
    resolved_output = output_root or Path(payload.get("output_root", DEFAULT_PAGE_LEVEL_OUTPUT))

    return PageLevelConfig(
        input_root=resolved_input,
        output_root=resolved_output,
        random_seed=int(payload.get("random_seed", 42)),
        dark_percentile=float(normalization.get("dark_percentile", 1.0)),
        light_percentile=float(normalization.get("light_percentile", 99.5)),
        threshold_method=str(normalization.get("threshold_method", "equal_image_weighted_otsu")),
        aspect_ratio_groups_enabled=aspect_enabled,
        aspect_ratio_bins=tuple(bins),
        workers=workers,
        show_progress=show_progress,
    )
