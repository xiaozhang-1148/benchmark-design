"""Markdown summary for foreground pixel density exports."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.report.vision.foreground_pixel_density_stats import (
    BlockDensityBand,
    DensityDistributionStats,
    DiagnosticSignal,
    FlowStructureDensityStats,
    compute_block_density_bands,
    compute_block_diagnostic_signals,
    compute_page_density_bands,
    format_area_weighted_pct,
    format_density_pct,
    format_ratio_pct,
)
from benchmark_design.vision.foreground_load.models import ForegroundLoadThresholds, PageForegroundLoadResult


def _format_overall_row(
    *,
    level_label: str,
    unit: str,
    stats: DensityDistributionStats,
    analysis_role: str,
) -> str:
    if stats.n == 0:
        return f"| {level_label} | {unit} | 0 | | | | {analysis_role} |"
    return (
        f"| {level_label} | {unit} | {stats.n:,} | "
        f"{format_density_pct(stats.mean)} | {format_density_pct(stats.median)} | "
        f"{format_area_weighted_pct(stats.area_weighted_density)} | {analysis_role} |"
    )


def _format_band_row(band: BlockDensityBand) -> str:
    return (
        f"| {band.label} | `{band.range_label}` | {band.interpretation} | "
        f"{band.count:,} | {format_ratio_pct(band.ratio)} |"
    )


def _format_signal_row(signal: DiagnosticSignal) -> str:
    return (
        f"| {signal.label} | {signal.count:,} | {format_ratio_pct(signal.ratio)} | "
        f"{signal.interpretation} |"
    )


def write_foreground_pixel_density_summary_md(
    results: list[PageForegroundLoadResult],
    output_path: Path,
    *,
    page_stats: DensityDistributionStats,
    block_stats: DensityDistributionStats,
    flow_stats: list[FlowStructureDensityStats],
    thresholds: ForegroundLoadThresholds,
    diagnostics_json_path: str = "metadata/foreground_pixel_density_diagnostics.json",
) -> None:
    del flow_stats

    page_band_summary = compute_page_density_bands(results)
    block_band_summary = compute_block_density_bands(results)
    diagnostic_signals = compute_block_diagnostic_signals(results)

    lines = [
        "# Foreground Pixel Density Summary",
        "",
        "This summary reports **foreground pixel density** in annotated regions "
        "at page level and answer-block level. The pipeline converts each page to "
        "grayscale, applies robust percentile normalization, builds a darkness map "
        "`S = 1 - G_tilde`, estimates a **dataset-level fixed threshold** (`tau_D`) "
        "from pooled darkness values inside `R_eff`, and binarizes foreground with "
        "`S >= tau_D`. No per-page or per-block threshold tuning is used for the primary metric.",
        "",
        "Supplementary metric: **mean darkness** (`mean(S)` over each region). "
        "Baseline (non-primary): **Raw Otsu Density** using per-region Otsu on grayscale.",
        "",
        f"Dataset `tau_D`: {_format_optional(thresholds.tau_D)} "
        f"(method: {thresholds.threshold_method or 'NA'}; "
        f"normalization: P{_format_percentile(thresholds.q_low)}/P{_format_percentile(thresholds.q_high)}). "
        f"See `{diagnostics_json_path}` for calibration, sensitivity, and QA details.",
        "",
        "## Overall Profile",
        "",
        "| metric level | unit | n | mean density | median density | area-weighted density | analysis role |",
        "|---|---|---:|---:|---:|---:|---|",
        _format_overall_row(
            level_label="Page-level D_page",
            unit="page",
            stats=page_stats,
            analysis_role="Primary foreground pixel density in annotated page regions (R_eff)",
        ),
        _format_overall_row(
            level_label="Block-level D_block",
            unit="txtBlock",
            stats=block_stats,
            analysis_role="Primary foreground pixel density in txtBlock regions",
        ),
        "",
        "![Page-level and block-level foreground density comparison](figures/density_level_comparison.png)",
        "",
        "## Page-level Density Distribution",
        "",
        "![Page-level foreground pixel density bands](figures/d_page_density_bands.png)",
        "",
        "| density group | D range | interpretation | count | ratio |",
        "|---|---|---|---:|---:|",
    ]
    for band in page_band_summary.bands:
        lines.append(_format_band_row(band))

    lines.extend(
        [
            "",
            "## Block-level Density Distribution",
            "",
            "![Block-level foreground pixel density bands](figures/d_block_density_bands.png)",
            "",
            "| density group | D range | interpretation | count | ratio |",
            "|---|---|---|---:|---:|",
        ]
    )
    for band in block_band_summary.bands:
        lines.append(_format_band_row(band))

    lines.extend(
        [
            "",
            "## High-density Comparison Sheets",
            "",
            "Pages and txtBlocks with **foreground density > 15%** export 5-panel "
            "comparison sheets under "
            "`figures/foreground_pixel_density/high_density_comparisons/` "
            "(original, raw Otsu, dataset-threshold foreground, difference overlay, darkness heatmap).",
            "",
            "Quality-check figures are written to "
            "`figures/foreground_pixel_density/quality_checks/` "
            "(calibration histogram and sample page visualizations).",
            "",
            "## Diagnostic Signals",
            "",
            "| diagnostic signal | count | ratio | interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    for signal in diagnostic_signals:
        lines.append(_format_signal_row(signal))

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def _format_percentile(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:g}"
