"""Write fixed OCR length-bin tables."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.report.output_layout import relative_input_path

from benchmark_design.ocr.length_bins import DEFAULT_LENGTH_BINS, OcrLengthBinMetrics


def write_length_bins_csv(metrics: OcrLengthBinMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["length_bin", "count", "share"])
        for label, count, share in metrics.as_rows():
            writer.writerow([label, count, f"{share:.6f}"])


def write_length_bins_markdown(metrics: OcrLengthBinMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Fixed Length Bins",
        "",
        "Cross-benchmark comparable token-length intervals "
        "(per-expression `LATEX_DICT` token count).",
        "",
        "| length_bin | count | share |",
        "| --- | ---: | ---: |",
    ]
    for label, count, share in metrics.as_rows():
        lines.append(f"| {label} | {count:,} | {share:.6f} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_length_bins_metadata(
    metrics: OcrLengthBinMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "expression_count": metrics.expression_count,
        "bins": [
            {"length_bin": label, "count": count, "share": share}
            for label, count, share in metrics.as_rows()
        ],
        "bin_definitions": [
            {
                "label": spec.label,
                "min_tokens": spec.min_tokens,
                "max_tokens": spec.max_tokens,
            }
            for spec in DEFAULT_LENGTH_BINS
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_length_bins_report(
    metrics: OcrLengthBinMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "length_bin_table.csv"
    md_path = output_dir / "length_bin_table.md"
    meta_path = output_dir / "length_bin_metadata.json"

    write_length_bins_csv(metrics, csv_path)
    write_length_bins_markdown(metrics, md_path)
    write_length_bins_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "metadata": meta_path,
    }
