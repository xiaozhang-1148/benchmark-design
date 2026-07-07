"""Write OCR scale tables to CSV and Markdown."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.report.output_layout import relative_input_path

from benchmark_design.ocr.scale import OcrScaleMetrics


def _format_value(metric: str, value: float | int) -> str:
    if metric == "duplicate rate":
        return f"{float(value):.6f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def write_scale_table_csv(metrics: OcrScaleMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in metrics.as_rows():
            writer.writerow([metric, _format_value(metric, value)])


def write_scale_table_markdown(metrics: OcrScaleMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Data Scale",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for metric, value in metrics.as_rows():
        lines.append(f"| {metric} | {_format_value(metric, value)} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(
    metrics: OcrScaleMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "json_file_count": metrics.json_file_count,
        "metrics": {metric: value for metric, value in metrics.as_rows()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_scale_report(
    metrics: OcrScaleMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    """Write CSV, Markdown, and metadata files; return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scale_table.csv"
    md_path = output_dir / "scale_table.md"
    meta_path = output_dir / "metadata.json"

    write_scale_table_csv(metrics, csv_path)
    write_scale_table_markdown(metrics, md_path)
    write_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "metadata": meta_path,
    }
