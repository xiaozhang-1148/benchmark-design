"""Table writers for page-level image analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from benchmark_design.page_level.models import (
    CalibrationResult,
    ImageFeatureRow,
    ImageInventoryRow,
)
from benchmark_design.page_level.statistics import (
    compute_categorical_statistics,
    compute_continuous_statistics,
)


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)
    frame.to_csv(stem.with_suffix(".csv"), index=False)


def write_inventory_tables(inventory: list[ImageInventoryRow], tables_dir: Path) -> None:
    frame = pd.DataFrame([asdict(row) for row in inventory])
    _write_frame(frame, tables_dir / "image_inventory")
    summary = (
        frame.groupby(["file_format", "stored_color_mode", "effective_color_type"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    _write_frame(summary, tables_dir / "format_mode_summary")


def write_feature_tables(features: list[ImageFeatureRow], tables_dir: Path) -> None:
    _write_frame(pd.DataFrame([asdict(row) for row in features]), tables_dir / "image_features")


def write_calibration_outputs(calibration: CalibrationResult, layout_calibration: Path) -> None:
    layout_calibration.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(
        {
            "gray_level": list(range(256)),
            "average_density_micro": list(calibration.average_histogram),
            "normalized_average_density_micro": list(calibration.normalized_average_histogram),
        }
    )
    hist.to_csv(layout_calibration / "grayscale_histogram.csv", index=False)
    payload = {
        "dark_reference": calibration.dark_reference,
        "light_reference": calibration.light_reference,
        "gray_threshold": calibration.gray_threshold,
        "darkness_threshold": calibration.tau_d,
        "tau_D": calibration.tau_d,
        "tau_d": calibration.tau_d,
        "global_threshold": calibration.global_threshold,
        "foreground_valley_threshold": calibration.foreground_valley_threshold,
        "dark_percentile": calibration.dark_percentile,
        "light_percentile": calibration.light_percentile,
        "threshold_method": calibration.threshold_method,
        "num_pages": calibration.image_count,
        "image_count": calibration.image_count,
        "foreground_rule": "gray <= gray_threshold",
        "histogram_weighting": "pooled_pixels",
        "gray_histogram": list(calibration.gray_histogram),
    }
    (layout_calibration / "calibration.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_statistics_tables(
    inventory: list[ImageInventoryRow],
    features: list[ImageFeatureRow],
    tables_dir: Path,
    report_dir: Path,
    calibration_payload: dict,
) -> dict:
    continuous = compute_continuous_statistics(features)
    categorical = compute_categorical_statistics(inventory, features)
    _write_frame(continuous, tables_dir / "continuous_statistics")
    _write_frame(categorical, tables_dir / "categorical_statistics")
    from benchmark_design.page_level.statistics import build_dataset_summary

    summary = build_dataset_summary(features, inventory, calibration=calibration_payload)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def write_aspect_ratio_groups(features: list[ImageFeatureRow], tables_dir: Path) -> None:
    frame = (
        pd.DataFrame([asdict(row) for row in features])
        .groupby("aspect_ratio_group", as_index=False)
        .agg(image_count=("image_id", "count"), aspect_ratio_median=("aspect_ratio", "median"))
    )
    _write_frame(frame, tables_dir / "aspect_ratio_groups")
