"""CSV/JSON writers for line-level export."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from benchmark_design.line_level.models import (
    InvalidAnnotationRow,
    LineLevelAnalysisResult,
    LineLevelConfig,
    LineMetricsRow,
    PageMetricsRow,
    ProcessingErrorRow,
    TargetPairRow,
)
from benchmark_design.line_level.statistics import build_dataset_summary


def _write_frame(frame: pd.DataFrame, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path_stem.with_suffix(".csv"), index=False)


def write_line_metrics(rows: list[LineMetricsRow], output_root: Path) -> None:
    frame = pd.DataFrame([asdict(row) for row in rows])
    _write_frame(frame, output_root / "line_metrics")
    ink_cols = [
        "image_id",
        "line_id",
        "mask_area",
        "bbox_pixel_count",
        "bbox_outside_pixel_count",
        "bbox_outside_ink_count",
        "has_interference",
        "bbox_outside_ink_ratio",
        "is_valid",
    ]
    present = [col for col in ink_cols if col in frame.columns]
    if present and "is_valid" in frame.columns:
        ink_frame = frame.loc[frame["is_valid"].astype(bool), present]
        ink_path = output_root / "line_bbox_outside_ink.csv"
        ink_frame.to_csv(ink_path, index=False)
    elif present:
        ink_path = output_root / "line_bbox_outside_ink.csv"
        frame[present].to_csv(ink_path, index=False)


def write_external_dataset_aspect_tables(rows: list, output_root: Path) -> dict[str, Path]:
    from benchmark_design.line_level.dataset_aspect import (
        DatasetLineGeometryRow,
        dataset_size_summary_frame,
    )

    typed_rows: list[DatasetLineGeometryRow] = list(rows)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame([asdict(row) for row in typed_rows])
    detail_path = tables_dir / "external_dataset_line_geometry"
    _write_frame(detail, detail_path)
    summary = dataset_size_summary_frame(typed_rows)
    summary_path = tables_dir / "external_dataset_line_geometry_summary"
    _write_frame(summary, summary_path)
    return {
        "external_dataset_line_geometry": detail_path.with_suffix(".csv"),
        "external_dataset_line_geometry_summary": summary_path.with_suffix(".csv"),
    }


def write_page_metrics(rows: list[PageMetricsRow], output_root: Path) -> None:
    frame = pd.DataFrame([asdict(row) for row in rows])
    _write_frame(frame, output_root / "page_metrics")


def write_target_pairs(rows: list[TargetPairRow], output_root: Path) -> Path | None:
    if not rows:
        return None
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    path_stem = tables_dir / "target_pairs"
    _write_frame(frame, path_stem)
    return path_stem.with_suffix(".csv")


def write_invalid_annotations(rows: list[InvalidAnnotationRow], output_root: Path) -> Path | None:
    if not rows:
        return None
    frame = pd.DataFrame([asdict(row) for row in rows])
    path = output_root / "invalid_annotations.csv"
    frame.to_csv(path, index=False)
    return path


def write_processing_errors(rows: list[ProcessingErrorRow], output_root: Path) -> Path | None:
    if not rows:
        return None
    frame = pd.DataFrame([asdict(row) for row in rows])
    path = output_root / "processing_errors.csv"
    frame.to_csv(path, index=False)
    return path


def write_dataset_summary(result: LineLevelAnalysisResult, output_root: Path) -> dict:
    summary = build_dataset_summary(result)
    path = output_root / "dataset_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def write_config_snapshot(config: LineLevelConfig, config_path: Path | None, output_root: Path) -> Path:
    dest = output_root / "config_snapshot.yaml"
    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, dest)
    else:
        dest.write_text(
            json.dumps(
                {
                    "input_root": str(config.input_root),
                    "output_root": str(config.output_root),
                    "workers": config.workers,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return dest
