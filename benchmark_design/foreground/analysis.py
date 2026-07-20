"""Export foreground EDA artifacts (histogram, threshold, sensitivity)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from benchmark_design.block_level.block_foreground_density import BlockForegroundDensityRow
from benchmark_design.foreground.models import ForegroundThresholdConfig
from benchmark_design.foreground.threshold import GRAY_BIN_COUNT
from benchmark_design.page_level.models import CalibrationResult, ImageFeatureRow
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot


def write_gray_histogram_csv(
    histogram: tuple[int, ...] | list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gray_level", "pixel_count"])
        for level, count in enumerate(histogram):
            writer.writerow([level, int(count)])


def write_threshold_candidates_csv(
    config: ForegroundThresholdConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            "global_pooled_otsu",
            f"{config.gray_threshold:.6f}",
            "1",
            "Otsu on pooled uint8 grayscale histogram (t_I)",
        ),
        (
            "darkness_equivalent",
            f"{config.tau_d:.6f}",
            "1",
            "Equivalent darkness threshold tau_D",
        ),
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "threshold", "selected", "notes"])
        writer.writerows(rows)


@with_locked_pyplot
def plot_gray_histogram_log(
    histogram: tuple[int, ...] | list[int],
    gray_threshold: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    counts = np.asarray(histogram, dtype=np.float64)
    levels = np.arange(GRAY_BIN_COUNT, dtype=np.float64)
    log_counts = np.log10(counts + 1.0)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.plot(levels, log_counts, color="#4472C4", linewidth=1.2)
    ax.axvline(
        gray_threshold,
        color="#C00000",
        linestyle="--",
        linewidth=1.2,
        label=f"t_I = {gray_threshold:.1f}",
    )
    ax.set_xlabel("Raw grayscale level I")
    ax.set_ylabel("log10(pooled pixel count + 1)")
    ax.set_title("Pooled grayscale histogram (all pages, pixel-weighted)")
    ax.set_xlim(0.0, 255.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_density_sensitivity_page_csv(
    features: list[ImageFeatureRow],
    gray_threshold: float,
    output_path: Path,
) -> None:
    from benchmark_design.report.page_level.plotting import (
        _foreground_density_bin_counts as page_density_bin_counts,
    )

    densities = np.array([row.foreground_density for row in features], dtype=np.float64)
    counts = page_density_bin_counts(densities)
    total = int(counts.sum())
    ratios = counts / total if total else counts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gray_threshold",
                "page_count",
                "page_mean_density",
                "page_median_density",
                "page_band_extremely_sparse_ratio",
                "page_band_sparse_ratio",
                "page_band_low_medium_ratio",
                "page_band_medium_ratio",
                "page_band_medium_high_ratio",
                "page_band_dense_ratio",
                "page_band_very_dense_ratio",
            ]
        )
        writer.writerow(
            [
                f"{gray_threshold:.4f}",
                len(features),
                f"{float(np.mean(densities)):.6f}",
                f"{float(np.median(densities)):.6f}",
                f"{float(ratios[0]):.6f}",
                f"{float(ratios[1]):.6f}",
                f"{float(ratios[2]):.6f}",
                f"{float(ratios[3]):.6f}",
                f"{float(ratios[4]):.6f}",
                f"{float(ratios[5]):.6f}",
                f"{float(ratios[5]):.6f}",
            ]
        )


def write_density_sensitivity_block_csv(
    rows: list[BlockForegroundDensityRow],
    gray_threshold: float,
    output_path: Path,
) -> None:
    from benchmark_design.report.block_level.block_density_plotting import (
        BLOCK_FOREGROUND_DENSITY_BIN_LABELS,
        _block_foreground_density_bin_counts,
    )

    densities = np.array([row.foreground_density for row in rows], dtype=np.float64)
    counts = _block_foreground_density_bin_counts(densities)
    total = int(counts.sum())
    ratios = counts / total if total else counts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "gray_threshold",
            "block_count",
            "block_mean_density",
            "block_median_density",
        ]
        header.extend(f"block_band_{index}" for index, _ in enumerate(BLOCK_FOREGROUND_DENSITY_BIN_LABELS))
        writer.writerow(header)
        writer.writerow(
            [
                f"{gray_threshold:.4f}",
                len(rows),
                f"{float(np.mean(densities)):.6f}",
                f"{float(np.median(densities)):.6f}",
                *[f"{ratio:.6f}" for ratio in ratios],
            ]
        )


def export_foreground_analysis(
    *,
    output_dir: Path,
    calibration: CalibrationResult,
    threshold_config: ForegroundThresholdConfig,
    features: list[ImageFeatureRow] | None = None,
    block_rows: list[BlockForegroundDensityRow] | None = None,
) -> dict[str, Path]:
    """Write foreground analysis directory for the frozen global threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    hist_csv = output_dir / "gray_histogram.csv"
    write_gray_histogram_csv(calibration.gray_histogram, hist_csv)
    outputs["gray_histogram"] = hist_csv

    hist_png = output_dir / "gray_hist_log.png"
    plot_gray_histogram_log(
        calibration.gray_histogram,
        calibration.gray_threshold,
        hist_png,
    )
    outputs["gray_hist_log"] = hist_png

    candidates = output_dir / "threshold_candidates.csv"
    write_threshold_candidates_csv(threshold_config, candidates)
    outputs["threshold_candidates"] = candidates

    if features:
        page_sensitivity = output_dir / "density_sensitivity_page.csv"
        write_density_sensitivity_page_csv(
            features,
            calibration.gray_threshold,
            page_sensitivity,
        )
        outputs["density_sensitivity_page"] = page_sensitivity

    if block_rows:
        block_sensitivity = output_dir / "density_sensitivity_block.csv"
        write_density_sensitivity_block_csv(
            block_rows,
            calibration.gray_threshold,
            block_sensitivity,
        )
        outputs["density_sensitivity_block"] = block_sensitivity

    return outputs
