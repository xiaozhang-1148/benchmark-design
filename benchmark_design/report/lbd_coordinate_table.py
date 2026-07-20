"""CSV export for L/B/D coordinate counts and structural difficulty tiers."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from benchmark_design.ocr.lbd_coordinates import (
    LbdCoordinateMetrics,
    compute_lbd_coordinate_metrics,
    validate_lbd_coordinates,
)

POSITION_COUNT_COLUMNS: tuple[str, ...] = (
    "position_id",
    "L_bin",
    "B_bin",
    "D_bin",
    "L_range",
    "B_range",
    "D_range",
    "structural_difficulty",
    "count",
    "ratio",
)

STRUCTURAL_DIFFICULTY_COLUMNS: tuple[str, ...] = (
    "structural_difficulty",
    "count",
    "ratio",
)


def write_expression_lbd_coordinate_counts_csv(
    features: Sequence,
    output_path: Path,
) -> LbdCoordinateMetrics:
    metrics = compute_lbd_coordinate_metrics(features)
    violations = validate_lbd_coordinates(metrics)
    if violations:
        joined = "; ".join(violations[:5])
        msg = f"L/B/D coordinate validation failed: {joined}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(POSITION_COUNT_COLUMNS)
        for row in metrics.position_counts:
            writer.writerow(
                [
                    row.position_id,
                    row.l_bin,
                    row.b_bin,
                    row.d_bin,
                    row.l_range,
                    row.b_range,
                    row.d_range,
                    row.structural_difficulty,
                    row.count,
                    f"{row.ratio * 100:.2f}%",
                ]
            )
    return metrics


def write_expression_structural_difficulty_counts_csv(
    features: Sequence,
    output_path: Path,
    *,
    metrics: LbdCoordinateMetrics | None = None,
) -> LbdCoordinateMetrics:
    resolved = metrics or compute_lbd_coordinate_metrics(features)
    violations = validate_lbd_coordinates(resolved)
    if violations:
        joined = "; ".join(violations[:5])
        msg = f"L/B/D coordinate validation failed: {joined}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STRUCTURAL_DIFFICULTY_COLUMNS)
        for row in resolved.structural_difficulty_counts:
            writer.writerow(
                [
                    row.structural_difficulty,
                    row.count,
                    f"{row.ratio * 100:.2f}%",
                ]
            )
    return resolved
