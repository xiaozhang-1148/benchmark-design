"""CSV and JSONL export for Answer-Block Flow Structure metrics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from benchmark_design.block_level.flow_structure.flow_group import FLOW_GROUP_LABELS
from benchmark_design.block_level.flow_structure.models import BlockGeometryRecord, PageFlowStructureResult
from benchmark_design.export_layout import export_flow_structure_label

PAGE_METRICS_COLUMNS: tuple[str, ...] = (
    "page_id",
    "image_width",
    "image_height",
    "num_txtBlock",
    "num_context_blocks",
    "flow_group",
    "flow_group_id",
    "is_regular_flow",
    "flow_structure",
    "flow_confidence",
    "needs_manual_review",
    "flow_reason",
    "flow_tags",
    "decision_reason",
    "decision_rule_id",
    "skeleton_type",
    "stable_column_layout",
    "stable_vertical_flow",
    "column_layout_confidence",
    "true_cross_column_bridge",
    "context_status",
    "context_impact_reason",
    "inserted_answer_block",
    "diagnostic_tags",
    "qa_tags",
    "failed_rules",
    "triggered_rules",
    "hybrid_reason",
    "review_image_path",
    "vertical_sequential_score",
    "max_adjacent_y_overlap_norm",
    "x_center_span_norm",
    "x_center_std_norm",
    "num_detected_columns",
    "column_center_distance_norm",
    "max_column_gap_norm",
    "column_y_overlap_norm",
    "column_area_balance",
    "x_cluster_separation_norm",
)


def _page_metrics_row(result: PageFlowStructureResult) -> list[str]:
    return [
        result.page_id,
        str(result.image_width),
        str(result.image_height),
        str(result.num_txtBlock),
        str(result.num_context_blocks),
        result.flow_group,
        result.flow_group_id,
        str(result.is_regular_flow).lower(),
        export_flow_structure_label(result.flow_structure),
        result.flow_confidence,
        str(result.needs_manual_review).lower(),
        result.flow_reason,
        result.flow_tags,
        result.decision_reason,
        result.decision_rule_id,
        result.skeleton_type,
        str(result.stable_column_layout).lower(),
        str(result.stable_vertical_flow).lower(),
        f"{result.column_layout_confidence:.6f}",
        str(result.true_cross_column_bridge).lower(),
        result.context_status,
        result.context_impact_reason,
        str(result.inserted_answer_block).lower(),
        result.diagnostic_tags,
        result.qa_tags,
        result.failed_rules,
        result.triggered_rules,
        result.hybrid_reason,
        result.review_image_path,
        f"{result.vertical_sequential_score:.6f}",
        f"{result.max_adjacent_y_overlap_norm:.6f}",
        f"{result.x_center_span_norm:.6f}",
        f"{result.x_center_std_norm:.6f}",
        str(result.num_detected_columns),
        f"{result.column_center_distance_norm:.6f}",
        f"{result.max_column_gap_norm:.6f}",
        f"{result.column_y_overlap_norm:.6f}",
        f"{result.column_area_balance:.6f}",
        f"{result.x_cluster_separation_norm:.6f}",
    ]


def write_flow_structure_page_metrics_csv(
    results: list[PageFlowStructureResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAGE_METRICS_COLUMNS)
        for result in results:
            writer.writerow(_page_metrics_row(result))


def write_flow_group_summary_csv(
    results: list[PageFlowStructureResult],
    output_path: Path,
) -> None:
    total = len(results)
    grouped: dict[str, list[PageFlowStructureResult]] = defaultdict(list)
    for result in results:
        grouped[result.flow_group].append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["flow_group", "page_count", "page_ratio", "mean_num_txtBlock", "manual_review_count"]
        )
        labels = [*FLOW_GROUP_LABELS, "no_valid_answer_block"]
        for label in labels:
            bucket = grouped.get(label, [])
            count = len(bucket)
            ratio = count / total if total else 0.0
            mean_txt = (sum(item.num_txtBlock for item in bucket) / count) if count else 0.0
            manual = sum(1 for item in bucket if item.needs_manual_review)
            writer.writerow(
                [
                    label,
                    count,
                    f"{ratio:.6f}",
                    f"{mean_txt:.6f}",
                    manual,
                ]
            )


def write_flow_structure_block_geometry_csv(
    block_records: list[BlockGeometryRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "page_id",
                "block_id",
                "block_type",
                "mask_area",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "center_x",
                "center_y",
                "norm_center_x",
                "norm_center_y",
                "sort_index",
                "assigned_column_id",
                "is_cross_column_block",
            ]
        )
        for record in block_records:
            writer.writerow(
                [
                    record.page_id,
                    record.block_id,
                    record.block_type,
                    f"{record.mask_area:.3f}",
                    f"{record.bbox_x1:.3f}",
                    f"{record.bbox_y1:.3f}",
                    f"{record.bbox_x2:.3f}",
                    f"{record.bbox_y2:.3f}",
                    f"{record.center_x:.3f}",
                    f"{record.center_y:.3f}",
                    f"{record.norm_center_x:.6f}",
                    f"{record.norm_center_y:.6f}",
                    record.sort_index,
                    "" if record.assigned_column_id is None else record.assigned_column_id,
                    str(record.is_cross_column_block).lower(),
                ]
            )


def _result_to_decision_record(result: PageFlowStructureResult) -> dict[str, object]:
    blocks = [
        {
            "block_id": record.block_id,
            "sort_index": record.sort_index,
            "assigned_column_id": record.assigned_column_id,
            "is_cross_column_block": record.is_cross_column_block,
            "norm_center_x": record.norm_center_x,
            "norm_center_y": record.norm_center_y,
        }
        for record in sorted(result.block_records, key=lambda item: item.sort_index)
    ]
    return {
        "page_id": result.page_id,
        "image_name": result.image_name,
        "flow_group": result.flow_group,
        "flow_group_id": result.flow_group_id,
        "is_regular_flow": result.is_regular_flow,
        "flow_structure": result.flow_structure,
        "flow_confidence": result.flow_confidence,
        "flow_reason": result.flow_reason,
        "flow_tags": result.flow_tags,
        "needs_manual_review": result.needs_manual_review,
        "decision_reason": result.decision_reason,
        "decision_rule_id": result.decision_rule_id,
        "skeleton_type": result.skeleton_type,
        "num_context_blocks": result.num_context_blocks,
        "stable_column_layout": result.stable_column_layout,
        "stable_vertical_flow": result.stable_vertical_flow,
        "column_layout_confidence": result.column_layout_confidence,
        "true_cross_column_bridge": result.true_cross_column_bridge,
        "context_status": result.context_status,
        "context_impact_reason": result.context_impact_reason,
        "inserted_answer_block": result.inserted_answer_block,
        "diagnostic_tags": result.diagnostic_tags,
        "qa_tags": result.qa_tags,
        "failed_rules": result.failed_rules,
        "triggered_rules": result.triggered_rules,
        "hybrid_reason": result.hybrid_reason,
        "review_image_path": result.review_image_path,
        "metrics": {
            "vertical_sequential_score": result.vertical_sequential_score,
            "max_adjacent_y_overlap_norm": result.max_adjacent_y_overlap_norm,
            "x_center_span_norm": result.x_center_span_norm,
            "x_center_std_norm": result.x_center_std_norm,
            "num_detected_columns": result.num_detected_columns,
            "column_center_distance_norm": result.column_center_distance_norm,
            "max_column_gap_norm": result.max_column_gap_norm,
            "column_y_overlap_norm": result.column_y_overlap_norm,
            "column_area_balance": result.column_area_balance,
            "x_cluster_separation_norm": result.x_cluster_separation_norm,
            "column_layout_confidence": result.column_layout_confidence,
            "largest_column_area_ratio": result.largest_column_area_ratio,
            "second_largest_column_area_ratio": result.second_largest_column_area_ratio,
            "has_cross_column_block": result.has_cross_column_block,
            "true_cross_column_bridge": result.true_cross_column_bridge,
            "has_interrupting_chart_or_figure": result.has_interrupting_chart_or_figure,
            "inserted_answer_block": result.inserted_answer_block,
        },
        "rule_outcomes": [asdict(rule) for rule in result.rule_outcomes],
        "blocks": blocks,
    }


def write_flow_structure_decisions_jsonl(
    results: list[PageFlowStructureResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_result_to_decision_record(result), ensure_ascii=False) + "\n")
