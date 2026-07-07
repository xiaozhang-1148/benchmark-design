"""Write OCR token long-tail summary and frequency distribution tables."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.report.output_layout import relative_input_path

from benchmark_design.ocr.token_longtail import OcrTokenLongtailMetrics


def _format_summary_value(metric: str, value: float | int) -> str:
    if metric == "gini" or "coverage" in metric or "ratio" in metric:
        return f"{float(value):.6f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def write_longtail_summary_csv(metrics: OcrTokenLongtailMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in metrics.summary_rows():
            writer.writerow([metric, _format_summary_value(metric, value)])


def write_longtail_summary_markdown(metrics: OcrTokenLongtailMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Token Long-Tail Summary",
        "",
        "Symbol long-tail statistics over corpus tokens (`LATEX_DICT` greedy tokenization).",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for metric, value in metrics.summary_rows():
        lines.append(f"| {metric} | {_format_summary_value(metric, value)} |")
    lines.extend(
        [
            "",
            "Gini close to 1 means a few tokens dominate; close to 0 means a more uniform tail.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_token_frequency_distribution_csv(
    metrics: OcrTokenLongtailMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "token", "count", "frequency_share", "cumulative_share"])
        for row in metrics.frequency_distribution:
            writer.writerow(
                [
                    row.rank,
                    row.token,
                    row.count,
                    f"{row.frequency_share:.6f}",
                    f"{row.cumulative_share:.6f}",
                ]
            )


def write_longtail_metadata(
    metrics: OcrTokenLongtailMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "summary": {metric: value for metric, value in metrics.summary_rows()},
        "frequency_distribution_rows": len(metrics.frequency_distribution),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_token_longtail_report(
    metrics: OcrTokenLongtailMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "token_longtail_table.csv"
    summary_md = output_dir / "token_longtail_table.md"
    frequency_csv = output_dir / "token_frequency_distribution.csv"
    meta_path = output_dir / "token_longtail_metadata.json"

    write_longtail_summary_csv(metrics, summary_csv)
    write_longtail_summary_markdown(metrics, summary_md)
    write_token_frequency_distribution_csv(metrics, frequency_csv)
    write_longtail_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "summary_csv": summary_csv,
        "summary_markdown": summary_md,
        "frequency_csv": frequency_csv,
        "metadata": meta_path,
    }
