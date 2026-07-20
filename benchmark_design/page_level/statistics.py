"""Dataset-level descriptive statistics for aspect ratio and foreground density."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd

from benchmark_design.page_level.models import ImageFeatureRow, ImageInventoryRow

CONTINUOUS_METRICS: tuple[str, ...] = (
    "width",
    "height",
    "aspect_ratio",
    "foreground_density",
)


def _continuous_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def compute_continuous_statistics(features: list[ImageFeatureRow]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(row) for row in features])
    rows: list[dict[str, float | int | str]] = []
    for metric in CONTINUOUS_METRICS:
        stats = _continuous_stats(frame[metric].to_numpy(dtype=np.float64))
        rows.append({"metric": metric, **stats})
    return pd.DataFrame(rows)


def compute_categorical_statistics(
    inventory: list[ImageInventoryRow],
    features: list[ImageFeatureRow],
) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []

    def _append_counts(column: str, values: list[str]) -> None:
        total = len(values)
        for label, count in Counter(values).most_common():
            rows.append(
                {
                    "category": column,
                    "label": label,
                    "count": count,
                    "ratio": count / total if total else 0.0,
                }
            )

    _append_counts("file_format", [row.file_format for row in inventory])
    _append_counts("stored_color_mode", [row.stored_color_mode for row in inventory])
    _append_counts("effective_color_type", [row.effective_color_type for row in features])
    _append_counts("aspect_ratio_group", [row.aspect_ratio_group for row in features])
    _append_counts("bits_per_channel", [str(row.bits_per_channel) for row in features])
    _append_counts(
        "alpha_used",
        ["yes" if "transparency_used" in row.effective_color_type else "no" for row in features],
    )
    return pd.DataFrame(rows)


def compute_aspect_group_summary(features: list[ImageFeatureRow]) -> dict[str, dict[str, float | int]]:
    total = len(features)
    grouped: dict[str, int] = {}
    for row in features:
        grouped[row.aspect_ratio_group] = grouped.get(row.aspect_ratio_group, 0) + 1
    return {
        name: {
            "count": count,
            "ratio": count / total if total else 0.0,
        }
        for name, count in sorted(grouped.items())
    }


def compute_dataset_highlights(features: list[ImageFeatureRow]) -> dict[str, object]:
    densities = np.array([row.foreground_density for row in features], dtype=np.float64)
    return {
        "density_below_0_03_ratio": float(np.mean(densities < 0.03)) if densities.size else 0.0,
        "density_above_0_08_ratio": float(np.mean(densities > 0.08)) if densities.size else 0.0,
        "aspect_groups": compute_aspect_group_summary(features),
    }


def build_dataset_summary(
    features: list[ImageFeatureRow],
    inventory: list[ImageInventoryRow],
    *,
    calibration: dict,
) -> dict:
    continuous = compute_continuous_statistics(features)
    categorical = compute_categorical_statistics(inventory, features)
    highlights = compute_dataset_highlights(features)
    return {
        "image_count": len(features),
        "calibration": calibration,
        **highlights,
        "continuous_statistics": continuous.to_dict(orient="records"),
        "categorical_statistics": categorical.to_dict(orient="records"),
    }
