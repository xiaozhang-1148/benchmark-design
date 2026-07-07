"""Figure: line-level examples of potentially confusable tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.confusable_tokens import (
    PRIMARY_CONFUSABLE_GROUPS,
    ConfusableGroupSpec,
    group_tokens_present,
    subgroup_co_occurrence_count,
)
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.line_crop import crop_line_polygon, resolve_page_image_path
from benchmark_design.report.pyplot_lock import with_locked_pyplot

EXAMPLES_PER_GROUP = 1


@dataclass(frozen=True, slots=True)
class ConfusableLineExample:
    group: ConfusableGroupSpec
    feature: ExpressionFeatures
    record: ExpressionRecord
    highlight_tokens: tuple[str, ...]

    @property
    def token_type_label(self) -> str:
        present = set(self.highlight_tokens)
        best_subgroup: tuple[str, ...] | None = None
        best_score = -1
        for subgroup in self.group.subgroups:
            overlap = subgroup_co_occurrence_count(self.feature.token_sequence, subgroup)
            if overlap > best_score:
                best_score = overlap
                best_subgroup = subgroup.tokens
        if best_subgroup is not None and best_score > 0:
            shown = [token for token in best_subgroup if token in present]
            if shown:
                return " / ".join(shown)
        return " / ".join(self.highlight_tokens)


def _require_matplotlib():
    import matplotlib.pyplot as plt

    from benchmark_design.report.export_figures import _configure_matplotlib_fonts

    _configure_matplotlib_fonts(plt)
    return plt


def _feature_index(
    features: list[ExpressionFeatures],
    expressions: list[ExpressionRecord],
) -> dict[str, tuple[ExpressionFeatures, ExpressionRecord]]:
    return {
        feature.expression_id: (feature, record)
        for feature, record in zip(features, expressions, strict=True)
    }


def _example_score(
    feature: ExpressionFeatures,
    record: ExpressionRecord,
    group: ConfusableGroupSpec,
) -> tuple[int, int, int]:
    present = group_tokens_present(feature.token_sequence, group)
    has_polygon = 1 if len(record.line_polygon) >= 3 else 0
    co_occur = len(set(present))
    return (has_polygon, -feature.token_length, co_occur)


def select_confusable_line_examples(
    features: list[ExpressionFeatures],
    expressions: list[ExpressionRecord],
    *,
    examples_per_group: int = EXAMPLES_PER_GROUP,
) -> list[ConfusableLineExample]:
    by_id = _feature_index(features, expressions)
    selected: list[ConfusableLineExample] = []

    for group in PRIMARY_CONFUSABLE_GROUPS:
        candidates: list[tuple[tuple[int, int, int], ConfusableLineExample]] = []
        for feature in features:
            present = group_tokens_present(feature.token_sequence, group)
            if not present:
                continue
            record = by_id[feature.expression_id][1]
            example = ConfusableLineExample(
                group=group,
                feature=feature,
                record=record,
                highlight_tokens=present,
            )
            candidates.append((_example_score(feature, record, group), example))

        candidates.sort(key=lambda item: item[0], reverse=True)
        seen_latex: set[str] = set()
        group_examples: list[ConfusableLineExample] = []
        for _, example in candidates:
            if example.feature.normalized_latex in seen_latex:
                continue
            seen_latex.add(example.feature.normalized_latex)
            group_examples.append(example)
            if len(group_examples) >= examples_per_group:
                break
        selected.extend(group_examples)

    return selected


def _load_example_image(example: ConfusableLineExample, input_dir: Path) -> np.ndarray | None:
    if len(example.record.line_polygon) < 3:
        return None
    image_path = resolve_page_image_path(example.record.image_name, input_dir)
    if image_path is None:
        return None
    try:
        cropped = crop_line_polygon(image_path, example.record.line_polygon)
    except (OSError, ValueError):
        return None
    return np.asarray(cropped)


@with_locked_pyplot
def write_confusable_token_examples_figure(
    features: list[ExpressionFeatures],
    expressions: list[ExpressionRecord],
    input_dir: Path,
    output_path: Path,
    *,
    examples_per_group: int = EXAMPLES_PER_GROUP,
) -> Path | None:
    examples = select_confusable_line_examples(
        features,
        expressions,
        examples_per_group=examples_per_group,
    )
    renderable = [example for example in examples if _load_example_image(example, input_dir) is not None]
    if not renderable:
        return None

    plt = _require_matplotlib()
    row_count = len(renderable)
    fig_height = max(3.0, row_count * 1.1)
    fig, axes = plt.subplots(row_count, 2, figsize=(8.5, fig_height))
    if row_count == 1:
        axes = np.array([axes])

    for row_index, example in enumerate(renderable):
        label_ax, image_ax = axes[row_index]
        label_ax.axis("off")
        label_ax.text(
            0.0,
            0.62,
            example.group.name,
            transform=label_ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="left",
        )
        label_ax.text(
            0.0,
            0.28,
            example.token_type_label,
            transform=label_ax.transAxes,
            fontsize=11,
            color="#333333",
            va="center",
            ha="left",
        )

        image = _load_example_image(example, input_dir)
        image_ax.axis("off")
        if image is not None:
            image_ax.imshow(image)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
