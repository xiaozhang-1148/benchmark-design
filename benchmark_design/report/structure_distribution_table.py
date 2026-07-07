"""Write OCR structure-type distribution tables."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.report.output_layout import relative_input_path

from benchmark_design.ocr.structure_distribution import OcrStructureDistributionMetrics


def write_structure_distribution_csv(
    metrics: OcrStructureDistributionMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "structure_type",
                "trigger_tokens",
                "expr_ratio",
                "occ_ratio",
                "max_depth",
                "expression_count",
                "occurrence_count",
            ]
        )
        for (
            structure_type,
            trigger_tokens,
            expression_ratio,
            occurrence_ratio,
            max_depth,
            expression_count,
            occurrence_count,
        ) in metrics.as_rows():
            writer.writerow(
                [
                    structure_type,
                    trigger_tokens,
                    f"{expression_ratio:.6f}",
                    f"{occurrence_ratio:.6f}",
                    max_depth,
                    expression_count,
                    occurrence_count,
                ]
            )


def write_structure_distribution_markdown(
    metrics: OcrStructureDistributionMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Structure Type Distribution",
        "",
        "Expr. Ratio = expressions containing the structure / all expressions.",
        "Occ. Ratio = structure trigger occurrences / all structural token occurrences.",
        "",
        "| Structure Type | Trigger tokens | Expr. Ratio | Occ. Ratio | Max Depth |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in metrics.rows:
        lines.append(
            "| "
            f"{row.structure_type} | {row.trigger_tokens} | "
            f"{row.expression_ratio:.6f} | {row.occurrence_ratio:.6f} | {row.max_depth} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_structure_distribution_metadata(
    metrics: OcrStructureDistributionMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "expression_count": metrics.expression_count,
        "structural_token_count": metrics.structural_token_count,
        "rows": [
            {
                "structure_type": row.structure_type,
                "trigger_tokens": row.trigger_tokens,
                "expr_ratio": row.expression_ratio,
                "occ_ratio": row.occurrence_ratio,
                "max_depth": row.max_depth,
                "expression_count": row.expression_count,
                "occurrence_count": row.occurrence_count,
            }
            for row in metrics.rows
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_structure_distribution_report(
    metrics: OcrStructureDistributionMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "structure_type_table.csv"
    md_path = output_dir / "structure_type_table.md"
    meta_path = output_dir / "structure_type_metadata.json"

    write_structure_distribution_csv(metrics, csv_path)
    write_structure_distribution_markdown(metrics, md_path)
    write_structure_distribution_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "metadata": meta_path,
    }
