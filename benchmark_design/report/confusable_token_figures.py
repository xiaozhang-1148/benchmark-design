"""Visual review figures for confusable-token example expressions."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.confusable_tokens import (
    CONFUSABLE_EXAMPLE_COUNT_PER_TOKEN,
    CONFUSABLE_EXAMPLE_MIN_OCR_CHARS,
    select_confusable_token_examples,
)
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.line_crop import crop_line_polygon, resolve_page_image_path
from benchmark_design.report.confusable_token_examples import GREEK_VARIANT_EXAMPLE_TOKENS
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot

GREEK_VARIANT_GROUP_NAME = "greek-variant"
FIGURE_DPI = 120
TEXT_WRAP_WIDTH = 96


def build_expression_record_index(
    records: Sequence[ExpressionRecord],
) -> dict[str, ExpressionRecord]:
    return {record.expression_id: record for record in records}


def _token_directory_name(token: str) -> str:
    cleaned = token.strip().strip("\\")
    return cleaned or "token"


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _wrap_latex(text: str, *, width: int = TEXT_WRAP_WIDTH) -> str:
    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        paragraph = paragraph.strip()
        if not paragraph:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False))
    return "\n".join(wrapped_lines)


def _load_crop_image(record: ExpressionRecord, *, input_dir: Path):
    if len(record.line_polygon) < 3:
        return None
    image_path = resolve_page_image_path(record.image_name, input_dir)
    if image_path is None:
        return None
    try:
        return crop_line_polygon(image_path, record.line_polygon, margin_px=6)
    except OSError:
        return None


@with_locked_pyplot
def _draw_confusable_example_figure(
    *,
    rank: int,
    token: str,
    group_name: str,
    record: ExpressionRecord,
    feature: ExpressionFeatures,
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
        f"{group_name} | token `{token}` | example {rank:02d}",
        fontsize=11,
        loc="left",
        pad=8,
    )

    text_axis = fig.add_subplot(grid[1, 0])
    text_axis.axis("off")
    wrapped = _wrap_latex(feature.normalized_latex)
    caption = (
        f"expression_id: {feature.expression_id}\n"
        f"ocr_char_count: {sum(1 for char in feature.normalized_latex if not char.isspace())}\n"
        f"token_length: {feature.token_length}\n\n"
        f"{wrapped}"
    )
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


def export_confusable_token_example_figures(
    records: Sequence[ExpressionRecord],
    features: Sequence[ExpressionFeatures],
    *,
    input_dir: Path,
    figures_root: Path,
    tokens: tuple[str, ...] = GREEK_VARIANT_EXAMPLE_TOKENS,
    group_name: str = GREEK_VARIANT_GROUP_NAME,
    min_ocr_chars: int = CONFUSABLE_EXAMPLE_MIN_OCR_CHARS,
    per_token: int = CONFUSABLE_EXAMPLE_COUNT_PER_TOKEN,
) -> dict[str, int]:
    record_index = build_expression_record_index(records)
    selected = select_confusable_token_examples(
        features,
        tokens=tokens,
        min_ocr_chars=min_ocr_chars,
        per_token=per_token,
    )

    counts: dict[str, int] = {f"{group_name}/{_token_directory_name(token)}": 0 for token in tokens}
    token_rank: dict[str, int] = {token: 0 for token in tokens}

    for token, feature in selected:
        record = record_index.get(feature.expression_id)
        if record is None:
            continue
        crop_image = _load_crop_image(record, input_dir=input_dir)
        if crop_image is None:
            continue

        token_rank[token] += 1
        rank = token_rank[token]
        token_dir = figures_root / group_name / _token_directory_name(token)
        output_path = token_dir / f"example_{rank:02d}_{_safe_filename(feature.expression_id)}.png"
        _draw_confusable_example_figure(
            rank=rank,
            token=token,
            group_name=group_name,
            record=record,
            feature=feature,
            crop_image=crop_image,
            output_path=output_path,
        )
        counts[f"{group_name}/{_token_directory_name(token)}"] += 1

    return counts
