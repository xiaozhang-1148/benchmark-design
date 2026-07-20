"""Visual review figures for expression-level structural difficulty tiers."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.lbd_coordinates import (
    EXPRESSION_STRUCTURAL_DIFFICULTY_LABEL,
    ExpressionLbdCoordinate,
    LbdCoordinateMetrics,
    STRUCTURAL_DIFFICULTY_TIERS,
    assign_lbd_from_feature,
    compute_lbd_coordinate_metrics,
)
from benchmark_design.report.confusable_token_figures import (
    _wrap_latex,
    build_expression_record_index,
)
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot
from benchmark_design.report.stc_figures import (
    FIGURE_DPI,
    _load_expression_image,
    _safe_filename,
)

SAMPLES_PER_TIER = 20

INDEX_COLUMNS: tuple[str, ...] = (
    "structural_difficulty",
    "rank",
    "position_id",
    "expression_id",
    "l_bin",
    "b_bin",
    "d_bin",
    "token_length",
    "structure_types",
    "structure_type_count",
    "ast_depth",
    "figure_path",
)


def _structure_types_label(structure_types: tuple[str, ...]) -> str:
    if not structure_types:
        return ""
    return "|".join(structure_types)


def _build_lbd_caption(feature: ExpressionFeatures, coordinate: ExpressionLbdCoordinate) -> str:
    wrapped_latex = _wrap_latex(feature.normalized_latex)
    return (
        f"{EXPRESSION_STRUCTURAL_DIFFICULTY_LABEL}: {coordinate.structural_difficulty}\n"
        f"position_id: {coordinate.position_id}\n"
        f"L_bin: {coordinate.l_bin} | B_bin: {coordinate.b_bin} | D_bin: {coordinate.d_bin}\n"
        f"token_length: {coordinate.token_length}\n"
        f"structure_types: {coordinate.structure_type_count} "
        f"({_structure_types_label(coordinate.structure_types)})\n"
        f"ast_depth: {coordinate.ast_depth}\n\n"
        f"expression_id: {feature.expression_id}\n\n"
        f"{wrapped_latex}"
    )


@with_locked_pyplot
def _draw_lbd_example_figure(
    *,
    structural_difficulty: str,
    rank: int,
    feature: ExpressionFeatures,
    coordinate: ExpressionLbdCoordinate,
    crop_image,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    image_array = np.asarray(crop_image)
    image_height, image_width = image_array.shape[:2]
    text_block_inches = 2.8
    image_inches = max(image_height / FIGURE_DPI, 1.6)
    fig_height = image_inches + text_block_inches
    fig_width = max(image_width / FIGURE_DPI, 6.0)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
    grid = fig.add_gridspec(2, 1, height_ratios=[image_inches, text_block_inches], hspace=0.08)

    image_axis = fig.add_subplot(grid[0, 0])
    image_axis.imshow(image_array)
    image_axis.axis("off")
    image_axis.set_title(
        (
            f"{structural_difficulty} | {coordinate.position_id} | "
            f"L={coordinate.l_bin} B={coordinate.b_bin} D={coordinate.d_bin} | example {rank:02d}"
        ),
        fontsize=11,
        loc="left",
        pad=8,
    )

    text_axis = fig.add_subplot(grid[1, 0])
    text_axis.axis("off")
    text_axis.text(
        0.0,
        1.0,
        _build_lbd_caption(feature, coordinate),
        va="top",
        ha="left",
        fontsize=8,
        wrap=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _group_features_by_structural_difficulty(
    features: Sequence[ExpressionFeatures],
) -> dict[str, list[tuple[ExpressionFeatures, ExpressionLbdCoordinate]]]:
    grouped: dict[str, list[tuple[ExpressionFeatures, ExpressionLbdCoordinate]]] = defaultdict(list)
    for feature in features:
        coordinate = assign_lbd_from_feature(feature)
        grouped[coordinate.structural_difficulty].append((feature, coordinate))
    for tier in grouped:
        grouped[tier].sort(key=lambda item: item[0].expression_id)
    return grouped


def export_lbd_coordinate_example_figures(
    records: Sequence[ExpressionRecord],
    features: Sequence[ExpressionFeatures],
    *,
    input_dir: Path,
    figures_root: Path,
    per_tier: int = SAMPLES_PER_TIER,
    metrics: LbdCoordinateMetrics | None = None,
) -> dict[str, int]:
    """Export up to *per_tier* OCR crop figures for each structural difficulty tier."""
    record_index = build_expression_record_index(records)
    resolved_metrics = metrics or compute_lbd_coordinate_metrics(features)
    non_empty_tiers = {
        row.structural_difficulty
        for row in resolved_metrics.structural_difficulty_counts
        if row.count > 0
    }
    grouped = _group_features_by_structural_difficulty(features)

    if figures_root.exists():
        allowed = set(STRUCTURAL_DIFFICULTY_TIERS)
        for child in figures_root.iterdir():
            if child.is_dir() and child.name not in allowed:
                shutil.rmtree(child)

    counts: dict[str, int] = {}
    index_rows: list[dict[str, str | int]] = []

    for structural_difficulty in STRUCTURAL_DIFFICULTY_TIERS:
        if structural_difficulty not in non_empty_tiers:
            continue
        candidates = grouped.get(structural_difficulty, [])
        tier_dir = figures_root / structural_difficulty
        exported = 0

        for feature, coordinate in candidates:
            if exported >= per_tier:
                break
            record = record_index.get(feature.expression_id)
            if record is None:
                continue
            crop_image = _load_expression_image(record, input_dir=input_dir)
            if crop_image is None:
                continue

            exported += 1
            figure_name = f"example_{exported:02d}_{_safe_filename(feature.expression_id)}.png"
            output_path = tier_dir / figure_name
            _draw_lbd_example_figure(
                structural_difficulty=structural_difficulty,
                rank=exported,
                feature=feature,
                coordinate=coordinate,
                crop_image=crop_image,
                output_path=output_path,
            )
            index_rows.append(
                {
                    "structural_difficulty": structural_difficulty,
                    "rank": exported,
                    "position_id": coordinate.position_id,
                    "expression_id": feature.expression_id,
                    "l_bin": coordinate.l_bin,
                    "b_bin": coordinate.b_bin,
                    "d_bin": coordinate.d_bin,
                    "token_length": coordinate.token_length,
                    "structure_types": _structure_types_label(coordinate.structure_types),
                    "structure_type_count": coordinate.structure_type_count,
                    "ast_depth": coordinate.ast_depth,
                    "figure_path": f"{structural_difficulty}/{figure_name}",
                }
            )

        counts[structural_difficulty] = exported

    figures_root.mkdir(parents=True, exist_ok=True)
    with (figures_root / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_rows)

    return counts
