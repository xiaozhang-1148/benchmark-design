"""Markdown summary for Answer-Block Flow Structure exports."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from benchmark_design.vision.flow_structure.flow_group import FLOW_GROUP_LABELS
from benchmark_design.vision.flow_structure.models import PageFlowStructureResult

FLOW_STRUCTURE_LABELS: tuple[str, ...] = ("Single-flow", "Columnar-flow", "Hybrid-flow", "NA")
CONTEXT_STATUS_LABELS: tuple[str, ...] = ("no_context", "context_preserved", "context_interrupted")
SKELETON_LABELS: tuple[str, ...] = ("single", "vertical_single_flow", "columnar", "unstable")


def write_flow_structure_summary_md(
    results: list[PageFlowStructureResult],
    output_path: Path,
) -> None:
    total = len(results)
    group_counts = Counter(result.flow_group for result in results)
    structure_counts = Counter(result.flow_structure for result in results)
    skeleton_counts = Counter(result.skeleton_type for result in results if result.skeleton_type)
    context_counts = Counter(result.context_status for result in results if result.context_status)
    manual_review = sum(1 for result in results if result.needs_manual_review)

    by_num_txt: dict[int, Counter[str]] = defaultdict(Counter)
    for result in results:
        by_num_txt[result.num_txtBlock][result.flow_group] += 1

    lines = [
        "## Flow Structure Summary (top level)",
        "",
    ]
    for label in FLOW_STRUCTURE_LABELS:
        count = structure_counts.get(label, 0)
        pct = (100.0 * count / total) if total else 0.0
        lines.append(f"- {label}: {count:,} ({pct:.1f}%)")

    review_pct = (100.0 * manual_review / total) if total else 0.0
    lines.extend(
        [
            f"- Manual review (boundary only): {manual_review:,} ({review_pct:.1f}%)",
            "",
            "## Skeleton Summary",
            "",
        ]
    )
    for label in SKELETON_LABELS:
        count = skeleton_counts.get(label, 0)
        pct = (100.0 * count / total) if total else 0.0
        lines.append(f"- {label}: {count:,} ({pct:.1f}%)")

    lines.extend(["", "## Context Status Summary", ""])
    for label in CONTEXT_STATUS_LABELS:
        count = context_counts.get(label, 0)
        pct = (100.0 * count / total) if total else 0.0
        lines.append(f"- {label}: {count:,} ({pct:.1f}%)")

    lines.extend(["", "## Flow Group Summary (hierarchical)", ""])
    for flow_structure, groups in (
        ("Single-flow", FLOW_GROUP_LABELS[:3]),
        ("Columnar-flow", FLOW_GROUP_LABELS[3:6]),
        ("Hybrid-flow", FLOW_GROUP_LABELS[6:10]),
    ):
        structure_total = structure_counts.get(flow_structure, 0)
        structure_pct = (100.0 * structure_total / total) if total else 0.0
        lines.append(f"### {flow_structure} ({structure_total:,}, {structure_pct:.1f}%)")
        lines.append("")
        for label in groups:
            count = group_counts.get(label, 0)
            pct = (100.0 * count / total) if total else 0.0
            subgroup_pct = (100.0 * count / structure_total) if structure_total else 0.0
            lines.append(
                f"- {label}: {count:,} ({pct:.1f}% of all; {subgroup_pct:.1f}% of {flow_structure})"
            )
        lines.append("")

    na_count = group_counts.get("no_valid_answer_block", 0)
    if na_count:
        pct = (100.0 * na_count / total) if total else 0.0
        lines.extend([f"### NA ({na_count:,}, {pct:.1f}%)", ""])

    all_subgroup_labels = [*FLOW_GROUP_LABELS, "no_valid_answer_block"]
    lines.extend(
        [
            "### By num_txtBlock (flow_group)",
            "",
            "| num_txtBlock | " + " | ".join(all_subgroup_labels) + " |",
            "| ---: | " + " | ".join(["---:"] * len(all_subgroup_labels)) + " |",
        ]
    )
    for num_txt in sorted(by_num_txt):
        counts = by_num_txt[num_txt]
        row = " | ".join(str(counts.get(label, 0)) for label in all_subgroup_labels)
        lines.append(f"| {num_txt} | {row} |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
