"""Write OCR expression length distribution tables."""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.ocr.length_distribution import OcrLengthDistributionMetrics
from benchmark_design.report.output_layout import relative_input_path


def _format_value(metric: str, value: float | int) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if metric in {"p50", "p90", "p95", "p99", "max"} and isinstance(value, float) and value.is_integer():
        return str(int(value))
    if metric in {"mean length", "std", "cv", "p50", "p90", "p95", "p99"}:
        return f"{float(value):.6f}"
    return str(value)


def write_length_table_csv(metrics: OcrLengthDistributionMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in metrics.as_rows():
            writer.writerow([metric, _format_value(metric, value)])


def write_length_table_markdown(
    metrics: OcrLengthDistributionMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Expression Length Distribution",
        "",
        "Length is the per-expression token count after `LATEX_DICT` greedy tokenization.",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for metric, value in metrics.as_rows():
        lines.append(f"| {metric} | {_format_value(metric, value)} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_length_metadata(
    metrics: OcrLengthDistributionMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "expression_count": metrics.expression_count,
        "metrics": {metric: value for metric, value in metrics.as_rows()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_length_report(
    metrics: OcrLengthDistributionMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "length_distribution_table.csv"
    md_path = output_dir / "length_distribution_table.md"
    meta_path = output_dir / "length_distribution_metadata.json"

    write_length_table_csv(metrics, csv_path)
    write_length_table_markdown(metrics, md_path)
    write_length_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "metadata": meta_path,
    }
