"""Chapter-facing line-level summary tables (validity / pairs / ink states)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from benchmark_design.line_level.bbox_ink import load_calibration_result
from benchmark_design.line_level.layout import horizontal_adjacent_scope_rows
from benchmark_design.line_level.models import LineLevelConfig, LineMetricsRow, TargetPairRow


INK_STATE_NO_REGION = "no_countable_region"
INK_STATE_ZERO_INK = "countable_region_zero_ink"
INK_STATE_POSITIVE_INK = "countable_region_positive_ink"


def classify_bbox_outside_ink_state(row: LineMetricsRow) -> str:
    """Two-way natural state: zero ink vs positive ink in (AABB \\ mask).

    When mask fills its AABB (bbox_outside_pixel_count == 0), there is no gap
    region — counted as 区域内墨=0.
    """
    outside = int(row.bbox_outside_pixel_count or 0)
    ink = int(row.bbox_outside_ink_count or 0)
    if outside <= 0 or ink <= 0:
        return INK_STATE_ZERO_INK
    return INK_STATE_POSITIVE_INK


def orientation_validity_rows(line_rows: list[LineMetricsRow]) -> list[dict[str, object]]:
    total = sum(1 for row in line_rows if row.is_valid)
    valid_orient = sum(1 for row in line_rows if row.is_valid and row.orientation_direction_valid)
    return [
        {"item": "all_lines", "count": total},
        {"item": "orientation_valid", "count": valid_orient},
        {"item": "orientation_excluded", "count": total - valid_orient},
    ]


def spatial_relation_rows(
    pair_rows: list[TargetPairRow],
    config: LineLevelConfig,
) -> list[dict[str, object]]:
    ioa_positive = sum(1 for row in pair_rows if row.ioa_positive)
    adjacent = sum(1 for row in pair_rows if row.horizontal_adjacent)
    return [
        {
            "relation": "target_region_overlap",
            "criterion": "IoA > 0",
            "unique_same_page_pair_count": ioa_positive,
        },
        {
            "relation": "horizontal_adjacent",
            "criterion": (
                f"IoA = 0; S_h >= {config.height_similarity_threshold:g}; "
                f"R_v >= {config.vertical_overlap_ratio_threshold:g}; "
                f"G_x <= {config.horizontal_gap_px_threshold:g} px"
            ),
            "unique_same_page_pair_count": adjacent,
        },
    ]


def ink_natural_state_rows(line_rows: list[LineMetricsRow]) -> list[dict[str, object]]:
    valid = [row for row in line_rows if row.is_valid]
    total = len(valid)
    counts = {
        INK_STATE_ZERO_INK: 0,
        INK_STATE_POSITIVE_INK: 0,
    }
    for row in valid:
        counts[classify_bbox_outside_ink_state(row)] += 1
    labels = {
        INK_STATE_ZERO_INK: "bbox\\mask 内墨迹为0（含无缝隙区域）",
        INK_STATE_POSITIVE_INK: "bbox\\mask 内存在黑色像素",
    }
    return [
        {
            "state": state,
            "state_label": labels[state],
            "count": counts[state],
            "ratio_of_all_lines": (counts[state] / total) if total else 0.0,
        }
        for state in (INK_STATE_ZERO_INK, INK_STATE_POSITIVE_INK)
    ]


def write_chapter_tables(
    line_rows: list[LineMetricsRow],
    pair_rows: list[TargetPairRow],
    config: LineLevelConfig,
    output_root: Path,
) -> dict[str, Path]:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    orient_path = tables_dir / "orientation_validity.csv"
    pd.DataFrame(orientation_validity_rows(line_rows)).to_csv(orient_path, index=False)
    outputs["orientation_validity"] = orient_path

    pair_path = tables_dir / "spatial_relations.csv"
    pair_frame = pd.DataFrame(spatial_relation_rows(pair_rows, config))
    pair_frame.to_csv(pair_path, index=False)
    outputs["spatial_relations"] = pair_path

    scope_path = tables_dir / "horizontal_adjacent_scope.csv"
    valid_line_count = sum(1 for row in line_rows if row.is_valid)
    page_count = len({row.image_id for row in line_rows})
    scope_frame = pd.DataFrame(
        horizontal_adjacent_scope_rows(
            pair_rows,
            valid_line_count=valid_line_count,
            page_count=page_count,
        )
    )
    scope_frame.to_csv(scope_path, index=False)
    outputs["horizontal_adjacent_scope"] = scope_path

    note_path = tables_dir / "spatial_relations_note.txt"
    note_path.write_text(
        "\n".join(
            [
                "Pairs are same-page, unordered, and unique.",
                "Horizontal adjacency thresholds:",
                f"- height similarity S_h >= {config.height_similarity_threshold:g}",
                f"- vertical projection overlap R_v >= {config.vertical_overlap_ratio_threshold:g}",
                f"- horizontal gap G_x <= {config.horizontal_gap_px_threshold:g} px",
                "No coverage ratios or IoA grade bins are reported for chapter tables.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["spatial_relations_note"] = note_path

    ink_path = tables_dir / "bbox_outside_ink_natural_states.csv"
    ink_frame = pd.DataFrame(ink_natural_state_rows(line_rows))
    ink_frame.to_csv(ink_path, index=False)
    outputs["bbox_outside_ink_natural_states"] = ink_path

    ink_note = tables_dir / "bbox_outside_ink_natural_states_note.txt"
    ink_note.write_text(
        "\n".join(
            [
                "Notes:",
                "- Empty (bbox \\ mask) (bbox_outside_pixel_count == 0) is counted as zero-ink.",
                "- Ratio distribution plots include only records with bbox_outside_ink_count > 0.",
                "- Natural states are binary: 区域内墨=0 vs 区域内墨>0.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["bbox_outside_ink_natural_states_note"] = ink_note

    calib_path = tables_dir / "bbox_outside_ink_calibration_threshold.json"
    calib_payload: dict[str, object] = {
        "calibration_path": str(config.calibration_path) if config.calibration_path else None,
        "bbox_outside_ink_enabled": config.bbox_outside_ink_enabled,
    }
    calibration = config.calibration
    if calibration is None and config.calibration_path is not None and config.calibration_path.is_file():
        calibration = load_calibration_result(config.calibration_path)
    if calibration is not None:
        calib_payload.update(asdict(calibration))
        calib_payload["threshold_used_for_ink"] = float(calibration.gray_threshold)
        calib_payload["ink_definition"] = "gray <= gray_threshold (global pooled Otsu)"
    calib_path.write_text(json.dumps(calib_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs["bbox_outside_ink_calibration_threshold"] = calib_path

    return outputs
