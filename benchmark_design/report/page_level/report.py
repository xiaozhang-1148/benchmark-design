"""Markdown report and manifest writers for page-level image analysis."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from benchmark_design import __version__
from benchmark_design.page_level.models import (
    CalibrationResult,
    ImageFeatureRow,
    ImageInventoryRow,
    PageLevelConfig,
)
from benchmark_design.report.page_level.output_layout import PageLevelOutputLayout


def configure_analysis_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark_design.page_level")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _config_file_hash(config_path: Path | None) -> str | None:
    if config_path is None or not config_path.is_file():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _metric_row(continuous: pd.DataFrame, metric: str) -> str:
    row = continuous.loc[continuous["metric"] == metric]
    if row.empty:
        return f"| {metric} | — |"
    values = row.iloc[0]
    return (
        f"| {metric} | "
        f"p05={values['p05']:.3g}, median={values['median']:.3g}, p95={values['p95']:.3g} |"
    )


def _lookup_continuous_median(dataset_summary: dict, metric: str) -> float:
    for row in dataset_summary.get("continuous_statistics", []):
        if row.get("metric") == metric:
            return float(row.get("median", 0.0))
    return 0.0


def _analysis_conclusions(dataset_summary: dict) -> list[str]:
    aspect_groups = dataset_summary.get("aspect_groups", {})
    dominant_group = None
    dominant_ratio = 0.0
    if isinstance(aspect_groups, dict):
        for name, payload in aspect_groups.items():
            if name in {"all", "other"}:
                continue
            ratio = float(payload.get("ratio", 0.0))
            if ratio > dominant_ratio:
                dominant_ratio = ratio
                dominant_group = name
    lines = [
        "## 7. Analysis Conclusions",
        "",
        (
            f"- **版式结构**：数据集呈多峰宽高比分布；"
            f"{'主导版式为 ' + str(dominant_group) + f'（{dominant_ratio * 100:.1f}%）' if dominant_group else '需结合 aspect_groups 表'}。"
        ),
        (
            f"- **前景密度**：全页墨迹密度中位数约为 "
            f"{_lookup_continuous_median(dataset_summary, 'foreground_density') * 100:.2f}%；"
            f"{dataset_summary.get('density_below_0_03_ratio', 0) * 100:.1f}% 页面低于 3%，"
            f"{dataset_summary.get('density_above_0_08_ratio', 0) * 100:.1f}% 高于 8%。"
        ),
        "",
    ]
    return lines


def write_image_analysis_report(
    *,
    layout: PageLevelOutputLayout,
    config: PageLevelConfig,
    inventory: list[ImageInventoryRow],
    features: list[ImageFeatureRow],
    calibration: CalibrationResult,
    dataset_summary: dict,
    figure_manifest: dict[str, str],
) -> Path:
    continuous = pd.DataFrame(dataset_summary["continuous_statistics"])
    categorical = pd.DataFrame(dataset_summary["categorical_statistics"])

    lines = [
        "# HMER Page-Level Image Analysis Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## 1. Scope",
        "",
        "Pure image-level analysis of HMER answer-page images, limited to aspect ratio",
        "and page foreground density. OCR text and polygon annotations are not used",
        "beyond enumerating image paths from benchmark JSON.",
        "",
        "## 2. Input and Output",
        "",
        f"- Input root: `{config.input_root}`",
        f"- Output root: `{config.output_root}`",
        f"- Images analyzed: {len(features):,}",
        "",
        "## 3. Inventory Summary",
        "",
        f"- Unique images discovered: {len(inventory):,}",
        "- Machine-readable features: `tables/image_features.parquet`",
        "- Human review copy: `tables/image_features.csv`",
        "",
        "## 4. Dataset Calibration",
        "",
        f"- Equal-image-weighted grayscale histogram over {calibration.image_count:,} images",
        f"- `I_dark` (P{calibration.dark_percentile:g}): {calibration.dark_reference:.3f}",
        f"- `I_light` (P{calibration.light_percentile:g}): {calibration.light_reference:.3f}",
        f"- Global pooled Otsu gray threshold `t_I`: {calibration.gray_threshold:.3f}",
        f"- Equivalent darkness threshold `τ_D`: {calibration.tau_d:.6f}",
        f"- Foreground rule: `I <= t_I` (equivalently `S >= τ_D`)",
        "",
        "## 5. Foreground Extraction",
        "",
        "Full-page binary mask on normalized grayscale: foreground iff pixel < `T_global`.",
        "Foreground density = foreground pixels / page pixels.",
        "",
        "## 6. Continuous Feature Statistics",
        "",
        "| Metric | P5 / median / P95 |",
        "| --- | --- |",
        _metric_row(continuous, "width"),
        _metric_row(continuous, "height"),
        _metric_row(continuous, "aspect_ratio"),
        _metric_row(continuous, "foreground_density"),
        "",
    ]

    lines.extend(_analysis_conclusions(dataset_summary))

    lines.extend(
        [
            "## 8. Categorical Statistics",
            "",
            "| Category | Label | Count | Ratio |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, row in categorical.iterrows():
        lines.append(
            f"| {row['category']} | {row['label']} | {int(row['count'])} | {row['ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 9. Result Interpretation Boundaries",
            "",
            "- Metrics describe stored image pixels, not OCR quality or annotation coverage.",
            "- Foreground density is page-wide ink occupancy, not effective annotation union.",
            "",
            "## Figures",
            "",
            "| Key | Path |",
            "| --- | --- |",
        ]
    )
    for key, rel_path in sorted(figure_manifest.items()):
        lines.append(f"| {key} | `{rel_path}` |")

    report_path = layout.report / "image_analysis_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_figure_manifest(figure_manifest: dict[str, str], report_dir: Path) -> Path:
    frame = pd.DataFrame(
        [{"figure_key": key, "relative_path": path} for key, path in sorted(figure_manifest.items())]
    )
    path = report_dir / "figure_manifest.csv"
    frame.to_csv(path, index=False)
    return path


def write_run_manifest(
    *,
    layout: PageLevelOutputLayout,
    config: PageLevelConfig,
    image_count: int,
    calibration: CalibrationResult,
    figure_manifest: dict[str, str],
    config_path: Path | None = None,
) -> Path:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "config_path": str(config_path) if config_path else None,
        "config_file_sha256": _config_file_hash(config_path),
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "image_count": image_count,
        "workers": config.workers,
        "random_seed": config.random_seed,
        "scope": ["aspect_ratio", "foreground_density"],
        "calibration": {
            "dark_reference": calibration.dark_reference,
            "light_reference": calibration.light_reference,
            "gray_threshold": calibration.gray_threshold,
            "darkness_threshold": calibration.tau_d,
            "tau_D": calibration.tau_d,
            "global_threshold": calibration.global_threshold,
            "foreground_valley_threshold": calibration.foreground_valley_threshold,
            "dark_percentile": calibration.dark_percentile,
            "light_percentile": calibration.light_percentile,
            "threshold_method": calibration.threshold_method,
        },
        "figure_count": len(figure_manifest),
        "figures": figure_manifest,
    }
    path = layout.report / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
