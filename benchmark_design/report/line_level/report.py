"""Markdown report and manifests for line-level geometry analysis."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from benchmark_design import __version__
from benchmark_design.line_level.models import LineLevelAnalysisResult
from benchmark_design.report.line_level.output_layout import LineLevelOutputLayout


def configure_analysis_logger() -> logging.Logger:
    logger = logging.getLogger("benchmark_design.line_level")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _config_hash(config_path: Path | None) -> str | None:
    if config_path is None or not config_path.is_file():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def write_line_analysis_report(
    result: LineLevelAnalysisResult,
    dataset_summary: dict,
    layout: LineLevelOutputLayout,
    figure_manifest: dict[str, str],
) -> Path:
    relations = dataset_summary.get("target_pair_relations", {})
    thresholds = relations.get("thresholds", {})
    orientation_validity = dataset_summary.get("orientation_validity", {})
    ink_states = dataset_summary.get("bbox_outside_ink_natural_states", [])
    pair_scope = relations.get("pair_scope", [])
    lines = [
        "# HMER Line-Level Geometry Analysis Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Chapter deliverables",
        "",
        "Bodies for chapter text use the tables and figures below only.",
        "Generic continuous-summary quantiles and angle-threshold ratios are retained in CSVs for audit, not for chapter claims.",
        "",
        "## Scale",
        "",
        f"- Pages: {dataset_summary.get('image_count', dataset_summary.get('page_count', 0)):,}",
        f"- Lines: {dataset_summary.get('line_count', 0):,}",
        f"- Valid lines: {dataset_summary.get('valid_line_count', 0):,}",
        "",
        "## Table 1 — Orientation validity",
        "",
        f"- All lines: {int(orientation_validity.get('all_lines', 0)):,}",
        f"- Orientation valid: {int(orientation_validity.get('orientation_valid', 0)):,}",
        f"- Orientation excluded: {int(orientation_validity.get('orientation_excluded', 0)):,}",
        "",
        "## Table 2 — Horizontal adjacent line relations (same page)",
        "",
        "| 统计项 | 数量 | 比例 |",
        "| --- | ---: | ---: |",
    ]
    for row in pair_scope:
        lines.append(
            f"| {row.get('item')} | {int(row.get('count', 0)):,} | "
            f"{float(row.get('ratio', 0.0)) * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"- IoA > 0 pairs (reference): {int(relations.get('ioa_positive_pair_count', 0)):,}",
            f"- All qualifying nearby pairs (reference): {int(relations.get('pair_count', 0)):,}",
            (
                f"- Adjacent thresholds: S_h≥{thresholds.get('height_similarity', 0.7)}, "
                f"R_v≥{thresholds.get('vertical_overlap_ratio', 0.7)}, "
                f"G_x≤{thresholds.get('horizontal_gap_px', 50)} px"
            ),
            "",
            "比例说明：水平相邻 line 与无序 pair 以 valid line 数为分母；涉及页面以总页数为分母。",
            "",
            "## Table 3 — Neighboring context pixels (bbox \\ mask)",
            "",
        ]
    )
    for state in ink_states:
        lines.append(
            f"- {state.get('state_label')}: {int(state.get('count', 0)):,} "
            f"({float(state.get('ratio_of_all_lines', 0.0)) * 100:.2f}%)"
        )
    lines.extend(
        [
            "",
            "Notes: ratio is undefined when no countable region exists; "
            "ratio distribution uses positive-ink rows only. "
            "Threshold values are exported to `tables/bbox_outside_ink_calibration_threshold.json`.",
            "",
            "## Figures",
            "",
            "| Key | Path |",
            "| --- | --- |",
        ]
    )
    for key, rel in sorted(figure_manifest.items()):
        lines.append(f"| {key} | `{rel}` |")

    validation_errors = dataset_summary.get("export_validation_errors", [])
    if validation_errors:
        lines.extend(["", "## Validation warnings", ""])
        lines.extend(f"- {error}" for error in validation_errors)

    layout.report.mkdir(parents=True, exist_ok=True)
    path = layout.report / "line_analysis_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_figure_manifest(figure_manifest: dict[str, str], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{"figure_key": k, "relative_path": v} for k, v in sorted(figure_manifest.items())])
    path = report_dir / "figure_manifest.csv"
    frame.to_csv(path, index=False)
    return path


def write_run_manifest(
    result: LineLevelAnalysisResult,
    layout: LineLevelOutputLayout,
    figure_manifest: dict[str, str],
    *,
    config_path: Path | None = None,
) -> Path:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "code_version": __version__,
        "config_path": str(config_path) if config_path else None,
        "config_file_sha256": _config_hash(config_path),
        "input_root": str(result.config.input_root),
        "output_root": str(result.config.output_root),
        "image_count": len(result.page_metrics),
        "line_count": len(result.line_metrics),
        "workers": result.config.workers,
        "processing_time_ms": result.processing_time_ms,
        "processing_error_count": len(result.processing_errors),
        "figure_count": len(figure_manifest),
        "figures": figure_manifest,
    }
    layout.report.mkdir(parents=True, exist_ok=True)
    path = layout.report / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
