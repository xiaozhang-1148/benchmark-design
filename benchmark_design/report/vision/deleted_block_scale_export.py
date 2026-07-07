"""CSV and JSON export for Deleted-Block Scale metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark_design.report.vision.deleted_block_scale_stats import (
    DeletedBlockScaleSummaryStats,
    R_DEL_TAIL_CUTOFFS,
    compute_deleted_block_scale_summary_stats,
)
from benchmark_design.vision.deleted_block_scale.models import (
    BlockDeletedScaleGeometryRecord,
    PageDeletedBlockScaleResult,
)

PAGE_METRICS_COLUMNS: tuple[str, ...] = (
    "page_id",
    "image_name",
    "image_width",
    "image_height",
    "num_txtBlock",
    "num_deleted_text_block",
    "valid_area",
    "deleted_area",
    "answer_related_area",
    "r_del",
    "has_deleted_text_block",
    "needs_manual_review",
    "review_reason",
    "review_image_path",
)

BLOCK_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "page_id",
    "block_id",
    "block_order",
    "block_type",
    "polygon_area",
    "mask_area",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "is_valid_answer_block",
    "is_deleted_text_block",
    "mask_out_of_bounds",
    "geometry_valid",
)


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_deleted_block_scale_page_metrics_csv(
    results: list[PageDeletedBlockScaleResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAGE_METRICS_COLUMNS)
        for result in results:
            writer.writerow(
                [
                    result.page_id,
                    result.image_name,
                    result.image_width,
                    result.image_height,
                    result.num_txtBlock,
                    result.num_deleted_text_block,
                    result.valid_area,
                    result.deleted_area,
                    result.answer_related_area,
                    _fmt_float(result.r_del),
                    str(result.has_deleted_text_block).lower(),
                    str(result.needs_manual_review).lower(),
                    result.review_reason,
                    result.review_image_path,
                ]
            )


def write_deleted_block_scale_block_geometry_csv(
    block_records: list[BlockDeletedScaleGeometryRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BLOCK_GEOMETRY_COLUMNS)
        for record in block_records:
            writer.writerow(
                [
                    record.page_id,
                    record.block_id,
                    record.block_order,
                    record.block_type,
                    f"{record.polygon_area:.3f}",
                    record.mask_area,
                    f"{record.bbox_x1:.2f}",
                    f"{record.bbox_y1:.2f}",
                    f"{record.bbox_x2:.2f}",
                    f"{record.bbox_y2:.2f}",
                    str(record.is_valid_answer_block).lower(),
                    str(record.is_deleted_text_block).lower(),
                    str(record.mask_out_of_bounds).lower(),
                    str(record.geometry_valid).lower(),
                ]
            )


def write_deleted_block_scale_diagnostics_json(
    stats: DeletedBlockScaleSummaryStats,
    output_path: Path,
) -> None:
    payload = {
        "area_definitions": {
            "A_valid": "Txtblock ∪ chart ∪ figure",
            "A_deleted": "deleted_text_block",
            "A_ans": "A_valid ∪ A_deleted",
            "R_del": "|A_deleted| / |A_ans|",
            "dataset_level_deleted_area_ratio": "Σ|A_deleted| / Σ|A_ans|",
        },
        "total_deleted_area": stats.total_deleted_area,
        "total_answer_related_area": stats.total_answer_related_area,
        "dataset_level_deleted_area_ratio": stats.dataset_level_deleted_area_ratio,
        "r_del_tail_cutoffs": list(R_DEL_TAIL_CUTOFFS),
        "pages_r_del_ge_cutoffs": {
            "0.2": stats.pages_r_del_ge_0_2,
            "0.3": stats.pages_r_del_ge_0_3,
            "0.5": stats.pages_r_del_ge_0_5,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
