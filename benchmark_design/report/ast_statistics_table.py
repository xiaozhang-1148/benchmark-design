"""Write structure-forest AST depth statistics tables."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.ocr.ast_statistics import OcrAstStatisticsMetrics
from benchmark_design.report.output_layout import relative_input_path


def _format_summary_value(metric: str, value: float | int) -> str:
    if metric == "Max max nested level":
        return str(int(value))
    if isinstance(value, float) and value.is_integer() and metric.startswith(("P50", "P90")):
        return f"{value:.6f}"
    return f"{float(value):.6f}" if isinstance(value, float) else str(value)


def write_ast_statistics_summary_csv(
    metrics: OcrAstStatisticsMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "definition", "value"])
        for metric, definition, value in metrics.as_summary_rows():
            writer.writerow([metric, definition, _format_summary_value(metric, value)])


def write_ast_statistics_metadata(
    metrics: OcrAstStatisticsMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "reference": "benchmark_design.ocr.structure_forest (unified LaTeX structure forest)",
        "expression_count": metrics.expression_count,
        "summary": [
            {"metric": metric, "definition": definition, "value": value}
            for metric, definition, value in metrics.as_summary_rows()
        ],
        "bins": [
            {
                "label": item.label,
                "min_level": item.min_level,
                "max_level": item.max_level,
                "count": item.count,
                "share": item.share,
            }
            for item in metrics.bins
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_ast_statistics_report(
    metrics: OcrAstStatisticsMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
    output_root: Path | None = None,
    metadata_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_root or output_dir
    metadata_dir = metadata_dir or (root / "docs" / "metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "ast_depth_summary.csv"
    metadata = metadata_dir / "ast_depth_metadata.json"

    write_ast_statistics_summary_csv(metrics, summary_csv)
    write_ast_statistics_metadata(metrics, input_dir=input_dir, output_path=metadata)

    return {
        "summary_csv": summary_csv,
        "metadata": metadata,
    }
