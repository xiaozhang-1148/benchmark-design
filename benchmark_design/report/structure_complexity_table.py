"""Write OCR structure combination complexity tables."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.report.output_layout import relative_input_path

from benchmark_design.ocr.structure_complexity import OcrStructureComplexityMetrics


def write_structure_complexity_csv(
    metrics: OcrStructureComplexityMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "definition", "value"])
        for metric, definition, value in metrics.as_rows():
            if isinstance(value, float) and metric != "Max structure type count":
                writer.writerow([metric, definition, f"{value:.6f}"])
            else:
                writer.writerow([metric, definition, value])


def write_structure_complexity_markdown(
    metrics: OcrStructureComplexityMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Structure Combination Complexity",
        "",
        "Counts distinct table-6 structure types co-occurring within one expression.",
        "",
        "| Metric | Definition | Value |",
        "| --- | --- | ---: |",
    ]
    for metric, definition, value in metrics.as_rows():
        if isinstance(value, float) and metric != "Max structure type count":
            formatted = f"{value:.6f}"
        else:
            formatted = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        lines.append(f"| {metric} | {definition} | {formatted} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structure_complexity_metadata(
    metrics: OcrStructureComplexityMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "expression_count": metrics.expression_count,
        "metrics": [
            {"metric": metric, "definition": definition, "value": value}
            for metric, definition, value in metrics.as_rows()
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_structure_complexity_report(
    metrics: OcrStructureComplexityMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "structure_complexity_table.csv"
    md_path = output_dir / "structure_complexity_table.md"
    meta_path = output_dir / "structure_complexity_metadata.json"

    write_structure_complexity_csv(metrics, csv_path)
    write_structure_complexity_markdown(metrics, md_path)
    write_structure_complexity_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "metadata": meta_path,
    }
