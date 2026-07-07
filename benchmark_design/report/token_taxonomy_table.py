"""Write OCR token taxonomy composition tables."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.report.output_layout import relative_input_path

from benchmark_design.ocr.token_taxonomy import OcrTokenTaxonomyMetrics, TOKEN_CATEGORY_ORDER


def write_token_taxonomy_csv(metrics: OcrTokenTaxonomyMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token_type", "count", "share"])
        for token_type, count, share in metrics.as_rows():
            writer.writerow([token_type, count, f"{share:.6f}"])
        writer.writerow(["other / unknown token ratio", "", f"{metrics.other_unknown_ratio:.6f}"])


def write_token_taxonomy_markdown(
    metrics: OcrTokenTaxonomyMetrics,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Token Taxonomy Composition",
        "",
        "Mutually exclusive token categories over corpus tokens "
        "(`LATEX_DICT` greedy tokenization).",
        "",
        "| token_type | count | share |",
        "| --- | ---: | ---: |",
    ]
    for token_type, count, share in metrics.as_rows():
        lines.append(f"| {token_type} | {count:,} | {share:.6f} |")
    lines.extend(
        [
            "",
            f"**other / unknown token ratio:** {metrics.other_unknown_ratio:.6f}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_token_taxonomy_metadata(
    metrics: OcrTokenTaxonomyMetrics,
    *,
    input_dir: Path,
    output_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(input_dir),
        "total_token_count": metrics.total_token_count,
        "other_unknown_ratio": metrics.other_unknown_ratio,
        "categories": [
            {"token_type": token_type, "count": count, "share": share}
            for token_type, count, share in metrics.as_rows()
        ],
        "category_order": [category.value for category in TOKEN_CATEGORY_ORDER],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_token_taxonomy_report(
    metrics: OcrTokenTaxonomyMetrics,
    output_dir: Path,
    *,
    input_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "token_taxonomy_table.csv"
    md_path = output_dir / "token_taxonomy_table.md"
    meta_path = output_dir / "token_taxonomy_metadata.json"

    write_token_taxonomy_csv(metrics, csv_path)
    write_token_taxonomy_markdown(metrics, md_path)
    write_token_taxonomy_metadata(metrics, input_dir=input_dir, output_path=meta_path)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "metadata": meta_path,
    }
