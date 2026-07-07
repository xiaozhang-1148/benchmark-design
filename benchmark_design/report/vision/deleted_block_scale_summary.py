"""Markdown summary for Deleted-Block Scale exports."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.report.vision.deleted_block_scale_stats import (
    DeletedBlockScaleSummaryStats,
    compute_deleted_block_scale_summary_stats,
)
from benchmark_design.vision.deleted_block_scale.models import PageDeletedBlockScaleResult


def _fmt_count(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{value:,}"


def _summary_table_rows(summary: DeletedBlockScaleSummaryStats) -> list[str]:
    prevalence_pct = summary.deleted_block_prevalence * 100.0
    without_pct = (summary.pages_without_deleted / summary.pages_analyzed * 100.0) if summary.pages_analyzed else 0.0
    rows = [
        ("Pages analyzed", _fmt_count(summary.pages_analyzed)),
        (
            "Pages with deleted_text_block",
            f"{_fmt_count(summary.pages_with_deleted)} ({prevalence_pct:.1f}%)",
        ),
        (
            "Pages without deleted_text_block",
            f"{_fmt_count(summary.pages_without_deleted)} ({without_pct:.1f}%)",
        ),
        ("Total deleted_text_block instances", _fmt_count(summary.total_deleted_instances)),
        ("Mean instances per affected page", f"{summary.mean_deleted_count_affected:.2f}"),
        ("Max instances per affected page", _fmt_count(summary.max_deleted_count_affected)),
        ("Total deleted area", _fmt_count(summary.total_deleted_area)),
        ("Total answer-related area", _fmt_count(summary.total_answer_related_area)),
        ("Dataset-level deleted area ratio", f"{summary.dataset_level_deleted_area_ratio:.6f}"),
        ("Mean R_del among affected pages", f"{summary.mean_r_del:.6f}"),
        ("Max R_del among affected pages", f"{summary.max_r_del:.6f}"),
        ("Pages with R_del >= 0.2", _fmt_count(summary.pages_r_del_ge_0_2)),
        ("Pages with R_del >= 0.3", _fmt_count(summary.pages_r_del_ge_0_3)),
        ("Pages with R_del >= 0.5", _fmt_count(summary.pages_r_del_ge_0_5)),
    ]
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {metric} | {value} |" for metric, value in rows)
    return lines


def write_deleted_block_scale_summary_md(
    results: list[PageDeletedBlockScaleResult],
    output_path: Path,
    *,
    stats: DeletedBlockScaleSummaryStats | None = None,
) -> None:
    summary = stats or compute_deleted_block_scale_summary_stats(results)

    lines = [
        "## Deleted-Block Scale Summary",
        "",
        "Area definitions (mask union, no double counting within each class):",
        "",
        "- `A_valid` = Txtblock ∪ chart ∪ figure",
        "- `A_deleted` = deleted_text_block",
        "- `A_ans` = A_valid ∪ A_deleted",
        "- `R_del` = |A_deleted| / |A_ans| (page level; dataset ratio uses summed areas)",
        "",
        "### 2.3.2 Overview",
        "",
        *_summary_table_rows(summary),
        "",
        "### Distribution figures",
        "",
        "- `figures/deleted_block_scale/r_del_histogram.png` — R_del distribution among affected pages",
        "- `figures/deleted_block_scale/deleted_instance_histogram.png` — deleted_text_block count "
        "per affected page (optional)",
        "- `figures/deleted_block_scale/high_r_del_examples/` — highest-burden page overlays",
        "",
        "### Annotation review summary",
        "",
        f"- Manual review pages: **{summary.manual_review_pages:,}**",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    if summary.review_counts:
        for reason, count in summary.review_counts.most_common():
            lines.append(f"| {reason} | {count:,} |")
    else:
        lines.append("| *(none)* | 0 |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
