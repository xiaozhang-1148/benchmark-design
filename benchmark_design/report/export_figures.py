"""Generate benchmark figures with matplotlib."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from matplotlib.patches import Patch

from benchmark_design.report.pyplot_lock import with_locked_pyplot

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
    TokenCategory.ENGLISH: "#4472C4",
    TokenCategory.DIGIT: "#70AD47",
    TokenCategory.GREEK: "#996633",
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


def _structure_count_row_index(structure_type_count: int) -> int | None:
    if structure_type_count <= 0:
        return None
    if structure_type_count >= 4:
        return 3
    return structure_type_count - 1


def _ast_depth_column_index(ast_depth: int) -> int | None:
    if ast_depth <= 0:
        return None
    return min(ast_depth, 5) - 1


STRUCTURE_DEPTH_HEATMAP_TITLE = "Joint Distribution of Structure Type Count and AST Depth in Ours"
STRUCTURE_DEPTH_HEATMAP_CAPTION = "Percentages are computed over all expression instances in Ours."
STRUCTURE_DEPTH_HEATMAP_TICK_FONTSIZE = 9
STRUCTURE_DEPTH_HEATMAP_TITLE_FONTSIZE = 11
STRUCTURE_DEPTH_HEATMAP_CELL_FONTSIZE = 7.5
STRUCTURE_DEPTH_HEATMAP_CAPTION_FONTSIZE = 8

STRUCTURE_COUNT_ROW_LABELS: tuple[str, ...] = (
    "1 Structure Type",
    "2 Structure Types",
    "3 Structure Types",
    "≥4 Structure Types",
)
AST_DEPTH_COLUMN_LABELS: tuple[str, ...] = tuple(f"depth {depth}" for depth in range(1, 6))


def _structure_count_vs_ast_depth_matrix(features: list[ExpressionFeatures]) -> np.ndarray:
    matrix = np.zeros((len(STRUCTURE_COUNT_ROW_LABELS), len(AST_DEPTH_COLUMN_LABELS)), dtype=int)
    for feature in features:
        row = _structure_count_row_index(feature.structure_type_count)
        col = _ast_depth_column_index(feature.ast_depth)
        if row is None or col is None:
            continue
        matrix[row, col] += 1
    return matrix


@with_locked_pyplot
def write_structure_cooccurrence_heatmap(features: list[ExpressionFeatures], output_path: Path) -> None:
    from matplotlib.colors import LogNorm

    plt = _require_matplotlib()
    matrix = _structure_count_vs_ast_depth_matrix(features)
    expression_count = len(features)
    if expression_count == 0:
        return

    positive = matrix[matrix > 0]
    vmax = int(positive.max()) if positive.size else 1
    masked = np.ma.array(matrix.astype(float), mask=matrix <= 0)

    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.imshow(
        masked,
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(vmax, 1)),
        aspect="auto",
    )
    ax.set_xticks(range(len(AST_DEPTH_COLUMN_LABELS)))
    ax.set_yticks(range(len(STRUCTURE_COUNT_ROW_LABELS)))
    ax.set_xticklabels(
        AST_DEPTH_COLUMN_LABELS,
        rotation=30,
        ha="right",
        fontsize=STRUCTURE_DEPTH_HEATMAP_TICK_FONTSIZE,
    )
    ax.set_yticklabels(STRUCTURE_COUNT_ROW_LABELS, fontsize=STRUCTURE_DEPTH_HEATMAP_TICK_FONTSIZE)
    ax.set_xlabel("Maximum AST nesting depth", fontsize=STRUCTURE_DEPTH_HEATMAP_TICK_FONTSIZE)
    ax.set_ylabel("Structure type count", fontsize=STRUCTURE_DEPTH_HEATMAP_TICK_FONTSIZE)
    ax.set_title(STRUCTURE_DEPTH_HEATMAP_TITLE, fontsize=STRUCTURE_DEPTH_HEATMAP_TITLE_FONTSIZE, pad=10)

    log_vmax = np.log(max(vmax, 2))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            count = int(matrix[row, col])
            if count == 0:
                continue
            share = count / expression_count
            log_pos = np.log(count) / log_vmax if count > 1 else 0.0
            text_color = "white" if log_pos > 0.55 else "black"
            ax.text(
                col,
                row,
                f"{count:,}\n{share * 100:.2f}%",
                ha="center",
                va="center",
                fontsize=STRUCTURE_DEPTH_HEATMAP_CELL_FONTSIZE,
                color=text_color,
            )

    fig.text(
        0.5,
        0.015,
        STRUCTURE_DEPTH_HEATMAP_CAPTION,
        ha="center",
        va="bottom",
        fontsize=STRUCTURE_DEPTH_HEATMAP_CAPTION_FONTSIZE,
        style="italic",
    )
    fig.subplots_adjust(bottom=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@with_locked_pyplot
def write_ast_depth_histogram(features: list[ExpressionFeatures], output_path: Path) -> None:
    plt = _require_matplotlib()
    counter = Counter(feature.ast_depth for feature in features)
    depths = sorted(counter)
    counts = [counter[depth] for depth in depths]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(depths, counts, width=0.85, color="#4472C4", edgecolor="white")
    ax.set_xlabel("AST depth (structure-forest max depth)")
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


DIFFICULTY_REGION_COLORS: dict[str, str] = {
    "L1": "#70AD47",
    "L2": "#4472C4",
    "L3": "#FFC000",
    "L4": "#C00000",
}


@with_locked_pyplot
def write_expression_lbd_coordinate_distribution(
    features: list[ExpressionFeatures],
    output_path: Path,
) -> None:
    from benchmark_design.ocr.lbd_coordinates import (
        L_BINS,
        B_BINS,
        D_BINS,
        compute_lbd_coordinate_metrics,
        lbd_bin_index,
    )

    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    metrics = compute_lbd_coordinate_metrics(features)
    count_lookup = {row.position_id: row for row in metrics.position_counts}
    max_count = max((row.count for row in metrics.position_counts), default=1)

    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    dx = dy = 0.75
    for l_bin in L_BINS:
        for b_bin in B_BINS:
            for d_bin in D_BINS:
                position_id = f"{l_bin}{b_bin}{d_bin}"
                row = count_lookup[position_id]
                x = lbd_bin_index(l_bin)
                y = lbd_bin_index(b_bin)
                z = lbd_bin_index(d_bin)
                height = 0.35 + (row.count / max_count) * 2.5 if max_count else 0.35
                color = DIFFICULTY_REGION_COLORS[row.structural_difficulty]
                ax.bar3d(x, y, z, dx, dy, height, color=color, alpha=0.88, shade=True)
                label = f"{row.count:,}" if row.count else "0"
                ax.text(x + dx / 2, y + dy / 2, z + height + 0.05, label, ha="center", va="bottom", fontsize=7)

    ax.set_xticks([0.375, 1.375, 2.375])
    ax.set_xticklabels(L_BINS)
    ax.set_yticks([0.375, 1.375, 2.375])
    ax.set_yticklabels(B_BINS)
    ax.set_zticks([0.375, 1.375, 2.375])
    ax.set_zticklabels(D_BINS)
    ax.set_xlabel("L: token length")
    ax.set_ylabel("B: structure breadth")
    ax.set_zlabel("D: AST depth")
    ax.set_title("Expression L/B/D Coordinate Distribution")
    ax.view_init(elev=24, azim=-135)

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=region)
        for region, color in DIFFICULTY_REGION_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_all_figures(
    features: list[ExpressionFeatures],
    figures_dir: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    resolved_counter = resolve_token_counter(features, token_counter)
    paths: dict[str, Path] = {
        "length_histogram": figures_dir / "length_histogram.png",
        "token_top50_histogram": figures_dir / "token_top50_histogram.png",
        "structure_cooccurrence_heatmap": figures_dir / "structure_cooccurrence_heatmap.png",
        "ast_depth_histogram": figures_dir / "ast_depth_histogram.png",
    }
    write_length_histogram(features, paths["length_histogram"])
    write_token_top50_histogram(features, paths["token_top50_histogram"], token_counter=resolved_counter)
    write_structure_cooccurrence_heatmap(features, paths["structure_cooccurrence_heatmap"])
    write_ast_depth_histogram(features, paths["ast_depth_histogram"])
    return paths
