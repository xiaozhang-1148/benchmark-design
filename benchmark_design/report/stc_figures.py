"""Visual review figures for high-complexity NTC/CBC cohort expressions."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.line_crop import crop_line_polygon, resolve_page_image_path
from benchmark_design.ocr.structure_stc import (
    NtcCbcResult,
    StructurePath,
    compute_ntc_cbc,
    format_chain_segments,
    format_single_node,
    format_structure_path,
    is_stc_export_cohort,
    ntc_cbc_sort_key,
)
from benchmark_design.report.confusable_token_figures import (
    _safe_filename,
    _wrap_latex,
    build_expression_record_index,
)
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot

FIGURE_DPI = 120
INDEX_COLUMNS: tuple[str, ...] = (
    "rank",
    "expression_id",
    "ast_depth",
    "structure_type_count",
    "structure_types",
    "ntc",
    "cbc",
    "max_recursive_len",
    "nested_path_count",
    "top_nested_path",
    "single_nodes",
    "figure_path",
)


def _ranked_figure_filename(rank: int, expression_id: str, *, total: int) -> str:
    width = max(2, len(str(total)))
    return f"{rank:0{width}d}_{_safe_filename(expression_id)}.png"


def _resolve_expression_image_path(record: ExpressionRecord, *, input_dir: Path) -> Path | None:
    if record.dataset == "MathWriting" and record.source_file:
        png_path = Path(record.source_file).with_suffix(".png")
        if png_path.is_file():
            return png_path
    return resolve_page_image_path(record.image_name, input_dir)


def _correct_mathwriting_display_orientation(image):
    """Flip MathWriting PNGs vertically to match normal reading orientation.

    Processed MathWriting rasters were saved with a vertically inverted Y axis.
    A full 180° rotation over-corrects and leaves the ink horizontally mirrored.
    """
    from PIL import Image

    return image.transpose(Image.FLIP_TOP_BOTTOM)


def _load_expression_image(record: ExpressionRecord, *, input_dir: Path):
    if len(record.line_polygon) >= 3:
        image_path = resolve_page_image_path(record.image_name, input_dir)
        if image_path is None:
            return None
        try:
            return crop_line_polygon(image_path, record.line_polygon, margin_px=6)
        except OSError:
            return None

    image_path = _resolve_expression_image_path(record, input_dir=input_dir)
    if image_path is None:
        return None
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if record.dataset == "MathWriting":
            image = _correct_mathwriting_display_orientation(image)
        return image
    except OSError:
        return None


def _format_nested_path_line(path: StructurePath) -> str:
    line = f"  {format_structure_path(path)}  cost={path.cost}"
    if path.raw_repeat_lens:
        repeat_details = ", ".join(str(length) for length in path.raw_repeat_lens)
        line = f"{line}  [raw_repeat={repeat_details}]"
    return line


def _format_single_nodes(result: NtcCbcResult) -> str:
    if not result.single_nodes:
        return "  (none)"
    return "\n".join(f"  {format_single_node(node)}" for node in result.single_nodes)


def _build_ntc_cbc_caption(feature: ExpressionFeatures, result: NtcCbcResult) -> str:
    nested_lines = [_format_nested_path_line(path) for path in result.nested_paths]
    if not nested_lines:
        nested_lines = ["  (none)"]

    wrapped_latex = _wrap_latex(feature.normalized_latex)
    return (
        f"depth: {feature.ast_depth}\n"
        f"types: {feature.structure_type_count} ({feature.structure_types_str()})\n\n"
        f"nested_paths:\n"
        + "\n".join(nested_lines)
        + "\n\nsingle_nodes:\n"
        + _format_single_nodes(result)
        + f"\n\nNTC = {result.ntc}\n"
        f"CBC = {result.cbc}\n"
        f"max_recursive_len = {result.max_recursive_len}\n\n"
        f"expression_id: {feature.expression_id}\n\n"
        f"{wrapped_latex}"
    )


@with_locked_pyplot
def _draw_ntc_cbc_figure(
    *,
    feature: ExpressionFeatures,
    result: NtcCbcResult,
    crop_image,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    image_array = np.asarray(crop_image)
    image_height, image_width = image_array.shape[:2]
    text_block_inches = max(
        3.6,
        2.0 + 0.35 * len(result.nested_paths) + 0.25 * len(result.single_nodes),
    )
    image_inches = max(image_height / FIGURE_DPI, 1.6)
    fig_height = image_inches + text_block_inches
    fig_width = max(image_width / FIGURE_DPI, 7.0)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=FIGURE_DPI)
    grid = fig.add_gridspec(2, 1, height_ratios=[image_inches, text_block_inches], hspace=0.08)

    image_axis = fig.add_subplot(grid[0, 0])
    image_axis.imshow(image_array)
    image_axis.axis("off")
    image_axis.set_title(
        (
            f"NTC high-complexity | NTC={result.ntc} | CBC={result.cbc} | "
            f"depth={feature.ast_depth} | types={feature.structure_type_count}"
        ),
        fontsize=11,
        loc="left",
        pad=8,
    )

    text_axis = fig.add_subplot(grid[1, 0])
    text_axis.axis("off")
    caption = _build_ntc_cbc_caption(feature, result)
    text_axis.text(
        0.0,
        1.0,
        caption,
        va="top",
        ha="left",
        fontsize=8,
        wrap=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _single_nodes_summary(result: NtcCbcResult) -> str:
    if not result.single_nodes:
        return ""
    return "|".join(format_single_node(node) for node in result.single_nodes)


def _build_index_row(
    feature: ExpressionFeatures,
    result: NtcCbcResult,
    *,
    rank: int,
    total: int,
) -> dict[str, str | int]:
    figure_name = _ranked_figure_filename(rank, feature.expression_id, total=total)
    top_nested_path = (
        format_chain_segments(result.nested_paths[0].segments) if result.nested_paths else ""
    )
    return {
        "rank": rank,
        "expression_id": feature.expression_id,
        "ast_depth": feature.ast_depth,
        "structure_type_count": feature.structure_type_count,
        "structure_types": feature.structure_types_str(),
        "ntc": result.ntc,
        "cbc": result.cbc,
        "max_recursive_len": result.max_recursive_len,
        "nested_path_count": len(result.nested_paths),
        "top_nested_path": top_nested_path,
        "single_nodes": _single_nodes_summary(result),
        "figure_path": figure_name,
    }


def sort_ntc_cbc_index_rows(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    return sorted(
        rows,
        key=lambda row: ntc_cbc_sort_key(
            ast_depth=int(row["ast_depth"]),
            structure_type_count=int(row["structure_type_count"]),
            ntc=int(row["ntc"]),
            cbc=int(row["cbc"]),
            max_recursive_len=int(row["max_recursive_len"]),
            expression_id=str(row["expression_id"]),
        ),
    )


def _write_index_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_ntc_cbc_index_rows(rows)
    total = len(sorted_rows)
    ranked_rows: list[dict[str, str | int]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = rank
        ranked_row["figure_path"] = _ranked_figure_filename(rank, str(row["expression_id"]), total=total)
        ranked_rows.append(ranked_row)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(ranked_rows)


def build_ntc_cbc_cohort_metrics(
    features: Sequence[ExpressionFeatures],
) -> list[tuple[ExpressionFeatures, NtcCbcResult]]:
    prepared: list[tuple[ExpressionFeatures, NtcCbcResult]] = []
    for feature in features:
        if not is_stc_export_cohort(feature):
            continue
        result = compute_ntc_cbc(list(feature.token_sequence))
        prepared.append((feature, result))
    prepared.sort(
        key=lambda item: ntc_cbc_sort_key(
            ast_depth=item[0].ast_depth,
            structure_type_count=item[0].structure_type_count,
            ntc=item[1].ntc,
            cbc=item[1].cbc,
            max_recursive_len=item[1].max_recursive_len,
            expression_id=item[0].expression_id,
        )
    )
    return prepared


def export_stc_high_complexity_figures(
    records: Sequence[ExpressionRecord],
    features: Sequence[ExpressionFeatures],
    *,
    input_dir: Path,
    figures_root: Path,
    max_exports: int | None = None,
) -> dict[str, int]:
    record_index = build_expression_record_index(records)
    prepared = build_ntc_cbc_cohort_metrics(features)
    if max_exports is not None:
        prepared = prepared[:max_exports]
    total = len(prepared)

    index_rows = [
        _build_index_row(feature, result, rank=rank, total=total)
        for rank, (feature, result) in enumerate(prepared, start=1)
    ]
    _write_index_csv(index_rows, figures_root / "index.csv")

    figures_exported = 0
    for rank, (feature, result) in enumerate(prepared, start=1):
        record = record_index.get(feature.expression_id)
        if record is None:
            continue
        crop_image = _load_expression_image(record, input_dir=input_dir)
        if crop_image is None:
            continue
        output_path = figures_root / _ranked_figure_filename(rank, feature.expression_id, total=total)
        _draw_ntc_cbc_figure(
            feature=feature,
            result=result,
            crop_image=crop_image,
            output_path=output_path,
        )
        figures_exported += 1

    return {"stc_high_complexity": total, "figures_exported": figures_exported}


def export_stc_high_complexity_figures_for_dataset(
    dataset_name: str,
    input_dir: Path,
    figures_root: Path,
    *,
    processing=None,
    max_exports: int | None = None,
) -> dict[str, int]:
    """Load a benchmark dataset and export ranked NTC/CBC high-complexity figures."""
    from benchmark_design.ocr.processing import build_enriched_corpus
    from benchmark_design.ocr.processing_options import ProcessingOptions

    enriched = build_enriched_corpus(dataset_name, input_dir, processing or ProcessingOptions())
    return export_stc_high_complexity_figures(
        list(enriched.expressions),
        list(enriched.features),
        input_dir=input_dir,
        figures_root=figures_root,
        max_exports=max_exports,
    )
