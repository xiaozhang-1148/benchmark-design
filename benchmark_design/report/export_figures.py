"""Generate benchmark figures with matplotlib."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from matplotlib.patches import Patch

from benchmark_design.report.confusable_token_figures import write_confusable_token_examples_figure
from benchmark_design.report.pyplot_lock import with_locked_pyplot

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.expression_features import ExpressionFeatures, resolve_token_counter
from benchmark_design.ocr.length_bin_specs import DEFAULT_LENGTH_BINS, assign_length_bin
from benchmark_design.ocr.length_distribution import percentile
from benchmark_design.ocr.structure_distribution import STRUCTURE_TYPES
from benchmark_design.ocr.token_longtail import DEFAULT_TOP_K, top_k_coverage
from benchmark_design.ocr.token_taxonomy import TOKEN_CATEGORY_ORDER, TokenCategory, classify_token

CJK_FONT_CANDIDATES: tuple[str, ...] = ("SimSun", "WenQuanYi Micro Hei")
_MATPLOTLIB_CONFIGURED = False

BIN_DISPLAY_LABELS: tuple[str, ...] = ("1-10", "11-20", "21-40", "41-80", ">80")

CATEGORY_COLORS: dict[TokenCategory, str] = {
    TokenCategory.LATIN_VARIABLE: "#4472C4",
    TokenCategory.DIGIT: "#70AD47",
    TokenCategory.SPECIAL_SYMBOL: "#FFC000",
    TokenCategory.OPERATOR: "#ED7D31",
    TokenCategory.GROUPING: "#5B9BD5",
    TokenCategory.STRUCTURAL: "#7030A0",
    TokenCategory.CJK: "#C00000",
    TokenCategory.PUNCTUATION: "#808080",
    TokenCategory.LAYOUT_ALIGNMENT: "#00B0F0",
    TokenCategory.OTHER: "#A5A5A5",
}


def _resolve_cjk_font_families() -> list[str]:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    resolved: list[str] = []
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in installed:
            resolved.append(candidate)
            continue
        match = next((name for name in installed if candidate.casefold() in name.casefold()), None)
        if match and match not in resolved:
            resolved.append(match)
    if not resolved:
        resolved.append("DejaVu Sans")
    resolved.append("sans-serif")
    return resolved


def _configure_matplotlib_fonts(plt) -> None:
    global _MATPLOTLIB_CONFIGURED
    if _MATPLOTLIB_CONFIGURED:
        return
    font_families = _resolve_cjk_font_families()
    plt.rcParams["font.family"] = font_families
    plt.rcParams["font.sans-serif"] = font_families
    plt.rcParams["axes.unicode_minus"] = False
    _MATPLOTLIB_CONFIGURED = True


def _require_matplotlib():
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    return plt


HEATMAP_EXCLUDED_STRUCTURE_TYPES: frozenset[str] = frozenset({"求和", "积分", "极限"})

HEATMAP_TICK_FONTSIZE = 12
HEATMAP_TITLE_FONTSIZE = 14
HEATMAP_CELL_FONTSIZE = 10
HEATMAP_CAPTION_FONTSIZE = 10


def _expression_lengths(features: list[ExpressionFeatures]) -> list[int]:
    return [feature.token_length for feature in features]


def _structure_cooccurrence_matrix(features: list[ExpressionFeatures]) -> tuple[list[str], np.ndarray]:
    specs = [
        spec for spec in STRUCTURE_TYPES if spec.structure_type not in HEATMAP_EXCLUDED_STRUCTURE_TYPES
    ]
    type_names = [spec.structure_type for spec in specs]
    size = len(type_names)
    matrix = np.zeros((size, size), dtype=int)
    index = {name: idx for idx, name in enumerate(type_names)}
    for feature in features:
        present = [name for name in feature.structure_types if name in index]
        for left in present:
            for right in present:
                matrix[index[left], index[right]] += 1
    return type_names, matrix


@with_locked_pyplot
def write_length_histogram(features: list[ExpressionFeatures], output_path: Path) -> None:
    plt = _require_matplotlib()
    total = len(features)
    counts = {spec.label: 0 for spec in DEFAULT_LENGTH_BINS}
    for feature in features:
        counts[assign_length_bin(feature.token_length)] += 1

    labels = list(BIN_DISPLAY_LABELS)
    values = [counts[spec.label] for spec in DEFAULT_LENGTH_BINS]
    shares = [value / total if total else 0.0 for value in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#4472C4", edgecolor="white")
    ax.set_xlabel("Token length bin")
    ax.set_ylabel("Expression count")
    ax.set_title("Expression Length Distribution by Fixed Bins")
    ymax = max(values) if values else 1
    ax.set_ylim(0, ymax * 1.18)
    for bar, count, share in zip(bars, values, shares, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n{share * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@with_locked_pyplot
def write_length_rank_curve(features: list[ExpressionFeatures], output_path: Path) -> None:
    plt = _require_matplotlib()
    lengths = _expression_lengths(features)
    ranked = sorted(lengths, reverse=True)
    expression_count = len(ranked)
    ranks = list(range(1, expression_count + 1))

    p50 = percentile(lengths, 50)
    p90 = percentile(lengths, 90)
    p95 = percentile(lengths, 95)
    p99 = percentile(lengths, 99)
    max_length = max(lengths) if lengths else 0
    long_share = sum(1 for length in lengths if length > 80) / expression_count if expression_count else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ranks, ranked, color="#ED7D31", linewidth=1.2)
    ax.set_xlabel("Expression rank (sorted by token length, descending)")
    ax.set_ylabel("Token length")
    ax.set_title("Expression Length Rank Curve")
    ax.grid(True, linestyle="--", alpha=0.35)

    percentile_lines = [
        (p50, "P50"),
        (p90, "P90"),
        (p95, "P95"),
        (p99, "P99"),
    ]
    for value, label in percentile_lines:
        ax.axhline(value, color="#666666", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(
            expression_count * 0.02,
            value,
            f"{label}={int(value) if float(value).is_integer() else value:.1f}",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    ax.axhline(max_length, color="#C00000", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.text(
        expression_count * 0.65,
        max_length,
        f"max={max_length}",
        va="bottom",
        fontsize=8,
        color="#C00000",
    )
    ax.text(
        0.98,
        0.05,
        f"Only {long_share * 100:.2f}% expressions are >80 tokens",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.95},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@with_locked_pyplot
def write_token_top50_histogram(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    plt = _require_matplotlib()
    counter = resolve_token_counter(features, token_counter)
    total = sum(counter.values())
    top = counter.most_common(50)
    tokens = [item[0] for item in top]
    counts = [item[1] for item in top]
    categories = [classify_token(token) for token in tokens]
    colors = [CATEGORY_COLORS[category] for category in categories]

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(tokens))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top-50 Token Frequency by Category")
    xmax = max(counts) if counts else 1
    ax.set_xlim(0, xmax * 1.22)
    for bar, count in zip(bars, counts, strict=True):
        share = count / total if total else 0.0
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {count:,} ({share * 100:.2f}%)",
            va="center",
            ha="left",
            fontsize=7,
        )

    legend_handles = [
        Patch(color=CATEGORY_COLORS[category], label=category.value)
        for category in TOKEN_CATEGORY_ORDER
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@with_locked_pyplot
def write_token_rank_frequency_log(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    plt = _require_matplotlib()
    counter = resolve_token_counter(features, token_counter)
    vocab_size = len(counter)
    ranked_items = counter.most_common()
    frequencies = [count for _, count in ranked_items]
    ranks = list(range(1, len(frequencies) + 1))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(ranks, frequencies, color="#5B9BD5", linewidth=1.4)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Frequency Rank Curve (log-log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    ax.text(
        0.02,
        0.97,
        f"vocab size = {vocab_size:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.95},
    )

    rank_marks = (10, 50, 100, 500)
    ymin, ymax = ax.get_ylim()
    for rank in rank_marks:
        if rank <= len(ranks):
            ax.axvline(rank, color="#999999", linestyle=":", linewidth=0.9, alpha=0.8)
            ax.text(rank, ymin, f"rank {rank}", rotation=90, va="bottom", ha="right", fontsize=7)

    coverage_lines = []
    for k in DEFAULT_TOP_K:
        coverage = top_k_coverage(counter, k)
        coverage_lines.append(f"top-{k} {coverage * 100:.2f}%")
    ax.text(
        0.98,
        0.97,
        "\n".join(coverage_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F7F7F7", "edgecolor": "#CCCCCC", "alpha": 0.95},
    )

    rare_thresholds = (1, 5, 10)
    rare_labels = []
    for threshold in rare_thresholds:
        rare_count = sum(1 for count in counter.values() if count <= threshold)
        rare_labels.append(f"freq ≤ {threshold}: {rare_count:,} types")
    ax.text(
        0.02,
        0.02,
        "Rare token types\n" + "\n".join(rare_labels),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF2CC", "edgecolor": "#E0C080", "alpha": 0.95},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _annotate_structure_heatmap(
    ax,
    matrix: np.ndarray,
    expression_count: int,
    *,
    fontsize: float = HEATMAP_CELL_FONTSIZE,
    lower_triangle_only: bool = False,
    color_scale_max: int | None = None,
) -> None:
    scale_max = color_scale_max if color_scale_max is not None else int(matrix.max())
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if lower_triangle_only and col >= row:
                continue
            count = int(matrix[row, col])
            share = count / expression_count if expression_count else 0.0
            text_color = "white" if count > scale_max * 0.55 else "black"
            ax.text(
                col,
                row,
                f"{count:,}\n{share * 100:.2f}%",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )


def _pairwise_lower_triangle_matrix(matrix: np.ndarray) -> np.ma.MaskedArray:
    """Mask diagonal and upper triangle; keep lower triangle for symmetric co-occurrence."""
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=0)
    return np.ma.masked_array(matrix, mask=mask)


@with_locked_pyplot
def write_structure_cooccurrence_heatmap(features: list[ExpressionFeatures], output_path: Path) -> None:
    plt = _require_matplotlib()
    type_names, matrix = _structure_cooccurrence_matrix(features)
    expression_count = len(features)
    display_matrix = _pairwise_lower_triangle_matrix(matrix)
    off_diagonal = matrix[np.triu(np.ones(matrix.shape, dtype=bool), k=1)]
    vmax = int(off_diagonal.max()) if off_diagonal.size else 1

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    ax.imshow(display_matrix, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(type_names)))
    ax.set_yticks(range(len(type_names)))
    ax.set_xticklabels(type_names, rotation=45, ha="right", fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_yticklabels(type_names, fontsize=HEATMAP_TICK_FONTSIZE)
    ax.set_title(
        "Pairwise Co-occurrence of Structure Types in Ours",
        fontsize=HEATMAP_TITLE_FONTSIZE,
    )
    _annotate_structure_heatmap(
        ax,
        matrix,
        expression_count,
        lower_triangle_only=True,
        color_scale_max=vmax,
    )
    caption = (
        "Each cell denotes expressions containing both structure types. "
        f"Percentages are calculated over all {expression_count:,} expression instances. "
        "Diagonal cells are omitted."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=HEATMAP_CAPTION_FONTSIZE, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@with_locked_pyplot
def write_ast_depth_histogram(features: list[ExpressionFeatures], output_path: Path) -> None:
    plt = _require_matplotlib()
    counter = Counter(feature.ast_depth for feature in features)
    depths = sorted(counter)
    counts = [counter[depth] for depth in depths]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(depths, counts, width=0.85, color="#A5A5A5", edgecolor="white")
    ax.set_xlabel("AST depth (PosFormer max nested level)")
    ax.set_ylabel("Expression count")
    ax.set_title("AST Depth Histogram")
    ax.set_xticks(depths)
    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax * 1.18)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_all_figures(
    features: list[ExpressionFeatures],
    figures_dir: Path,
    *,
    token_counter: Counter[str] | None = None,
    input_dir: Path | None = None,
    expressions: list[ExpressionRecord] | None = None,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    resolved_counter = resolve_token_counter(features, token_counter)
    paths: dict[str, Path] = {
        "length_histogram": figures_dir / "length_histogram.png",
        "length_rank_curve": figures_dir / "length_rank_curve.png",
        "token_top50_histogram": figures_dir / "token_top50_histogram.png",
        "token_rank_frequency_log": figures_dir / "token_rank_frequency_log.png",
        "structure_cooccurrence_heatmap": figures_dir / "structure_cooccurrence_heatmap.png",
        "ast_depth_histogram": figures_dir / "ast_depth_histogram.png",
        "confusable_token_examples": figures_dir / "confusable_token_examples.png",
    }
    write_length_histogram(features, paths["length_histogram"])
    write_length_rank_curve(features, paths["length_rank_curve"])
    write_token_top50_histogram(features, paths["token_top50_histogram"], token_counter=resolved_counter)
    write_token_rank_frequency_log(features, paths["token_rank_frequency_log"], token_counter=resolved_counter)
    write_structure_cooccurrence_heatmap(features, paths["structure_cooccurrence_heatmap"])
    write_ast_depth_histogram(features, paths["ast_depth_histogram"])
    if input_dir is not None and expressions is not None:
        confusable_path = write_confusable_token_examples_figure(
            features,
            expressions,
            input_dir,
            paths["confusable_token_examples"],
        )
        if confusable_path is None:
            paths.pop("confusable_token_examples")
    else:
        paths.pop("confusable_token_examples")
    return paths
