"""Chapter-6 page-level LaTeX figures (English labels, PNG + plot-data CSVs)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.page_latex_metrics import PageLatexMetricsRow
from benchmark_design.page_level_latex.plot_data import (
    FIG6_1_BIN_SPECS,
    build_fig6_1_plot_data,
    build_fig6_3_plot_data,
    build_fig6_4_plot_data,
    build_fig6_5_joint_grouped,
    build_fig6_6_category_plot_data,
    build_fig6_6_distinct_token_plot_data,
    build_fig6_7_plot_data,
    build_fig6_8_plot_data,
    build_fig6_9_plot_data,
    build_fig6_10_plot_data,
    total_pages,
)
from benchmark_design.page_level_latex.plot_style import (
    COLOR_PAGE_COVERAGE,
    COLOR_PAGE_COUNT,
    COLOR_PAGE_MAX,
    COLOR_RARE10,
    COLOR_SIMILAR,
    FONT_ANNOT,
    HEATMAP_CMAP,
    add_stats_box,
    annotate_bars_dual,
    apply_chapter6_style,
    assert_page_partition,
    dual_label,
    enable_horizontal_grid_only,
    save_figure_outputs,
    write_plot_csv,
)
from benchmark_design.report.pyplot_lock import with_locked_pyplot

STRUCTURE_GROUP_LABELS = {
    "0": "0 Structure Types",
    "1": "1 Structure Type",
    "2": "2 Structure Types",
    "3": "3 Structure Types",
    "at_least_4": "≥4 Structure Types",
}


def _finalize_figure(fig, *, figure_stem: Path, csv_path: Path, plot_data) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    write_plot_csv(plot_data, csv_path)
    paths = save_figure_outputs(fig, figure_stem)
    plt.close(fig)
    paths["csv"] = csv_path
    return paths


@with_locked_pyplot
def _save_fig6_1(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    plot_data = build_fig6_1_plot_data(page_rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    metrics = (
        ("ast_tree_count", "AST trees per page", COLOR_PAGE_COUNT),
        ("total_ast_node_count", "AST nodes per page", COLOR_PAGE_COVERAGE),
        ("max_ast_depth", "Max AST depth per page", COLOR_PAGE_MAX),
    )
    for ax, (field, title, color) in zip(axes, metrics, strict=True):
        values = np.array([getattr(row, field) for row in page_rows], dtype=np.float64)
        frame = plot_data[plot_data["metric"] == field].reset_index(drop=True)
        bars = ax.bar(frame["bin_label"], frame["page_count"], color=color, edgecolor="white")
        annotate_bars_dual(ax, bars, frame["page_count"], frame["page_ratio"], fontsize=FONT_ANNOT)
        add_stats_box(ax, vmin=float(values.min()), vmean=float(values.mean()), vmax=float(values.max()))
        # Mean marker on categorical axis: place at the bin containing the mean.
        specs = FIG6_1_BIN_SPECS[field]
        mean_val = float(values.mean())
        mean_label = None
        for start, end, label in specs:
            if end is None and start is not None and mean_val >= start:
                mean_label = label
            elif start is not None and end is not None and start <= mean_val <= end:
                mean_label = label
        if mean_label is not None and mean_label in set(frame["bin_label"]):
            idx = frame["bin_label"].tolist().index(mean_label)
            ax.axvline(idx, color="#333333", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(field)
        ax.set_ylabel("Pages")
        ax.tick_params(axis="x", rotation=25 if field == "total_ast_node_count" else 20)
        enable_horizontal_grid_only(ax)

    fig.suptitle("Figure 6-1 Page AST scale distribution")
    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


def _grouped_depth_bars(ax, frame, depths: list[int], *, title: str, show_bar_labels: bool = True) -> None:
    labels = [str(d) for d in depths]
    x = np.arange(len(depths))
    width = 0.36
    sub = frame[frame["ast_depth"].isin(depths)].set_index("ast_depth")
    cov = [sub.loc[d, "coverage_page_ratio"] * 100 for d in depths]
    mx = [sub.loc[d, "max_depth_page_ratio"] * 100 for d in depths]
    bars_cov = ax.bar(x - width / 2, cov, width, label="Page coverage", color=COLOR_PAGE_COVERAGE)
    bars_max = ax.bar(x + width / 2, mx, width, label="Page max AST depth", color=COLOR_PAGE_MAX)
    if show_bar_labels:
        annotate_bars_dual(
            ax,
            bars_cov,
            [int(sub.loc[d, "coverage_page_count"]) for d in depths],
            [float(sub.loc[d, "coverage_page_ratio"]) for d in depths],
        )
        annotate_bars_dual(
            ax,
            bars_max,
            [int(sub.loc[d, "max_depth_page_count"]) for d in depths],
            [float(sub.loc[d, "max_depth_page_ratio"]) for d in depths],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("AST depth")
    ax.set_ylabel("Page ratio (%)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    enable_horizontal_grid_only(ax)


@with_locked_pyplot
def _save_fig6_3(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    plot_data = build_fig6_3_plot_data(expression_rows, page_rows)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    _grouped_depth_bars(
        ax,
        plot_data,
        [0, 1, 2, 3, 4, 5],
        title="Figure 6-3 Page coverage and page max AST depth by depth",
        show_bar_labels=False,
    )
    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


@with_locked_pyplot
def _save_fig6_4(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    plot_data = build_fig6_4_plot_data(page_rows)
    cov = plot_data[plot_data["data_type"] == "structure_coverage"].sort_values(
        "page_ratio", ascending=False
    )
    typ = plot_data[
        (plot_data["data_type"] == "structure_type_count") & (plot_data["category"] != "9")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    bars = axes[0].bar(cov["category"], cov["page_ratio"] * 100, color=COLOR_PAGE_COVERAGE)
    annotate_bars_dual(axes[0], bars, cov["page_count"], cov["page_ratio"])
    axes[0].set_title("Page coverage by structure type")
    axes[0].set_xlabel("Structure type")
    axes[0].set_ylabel("Page ratio (%)")
    enable_horizontal_grid_only(axes[0])

    bars = axes[1].bar(typ["category"], typ["page_ratio"] * 100, color=COLOR_PAGE_COUNT)
    annotate_bars_dual(axes[1], bars, typ["page_count"], typ["page_ratio"])
    axes[1].set_title("Distinct structure types per page")
    axes[1].set_xlabel("Distinct structure type count")
    axes[1].set_ylabel("Page ratio (%)")
    enable_horizontal_grid_only(axes[1])

    fig.suptitle("Figure 6-4 Structure-type page coverage and per-page type count")
    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


@with_locked_pyplot
def _save_fig6_5(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    plot_data = build_fig6_5_joint_grouped(page_rows)
    assert_page_partition(plot_data["page_count"], total_pages(page_rows), label="fig6_5 joint")

    # Display layout matches the sample: rows 1/2/3/≥4, columns depth 1–5.
    row_keys = ["1", "2", "3", "at_least_4"]
    col_depths = list(range(1, 6))
    row_labels = [STRUCTURE_GROUP_LABELS[key] for key in row_keys]
    col_labels = [f"depth {d}" for d in col_depths]
    matrix = np.full((len(row_keys), len(col_depths)), np.nan, dtype=float)
    count_matrix = np.zeros((len(row_keys), len(col_depths)), dtype=int)
    lookup = {
        (str(rec.structure_type_count_group), int(rec.max_ast_depth)): rec
        for rec in plot_data.itertuples(index=False)
    }
    for r, key in enumerate(row_keys):
        for c, depth in enumerate(col_depths):
            rec = lookup.get((key, depth))
            if rec is None:
                continue
            count = int(rec.page_count)
            count_matrix[r, c] = count
            if count > 0:
                matrix[r, c] = float(rec.page_ratio) * 100

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=max(1.0, vmax),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(col_depths)))
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Maximum AST nesting depth")
    ax.set_ylabel("Structure type count")
    ax.set_title("Joint Distribution of Structure Type Count and AST Depth")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.8)

    for r in range(len(row_keys)):
        for c in range(len(col_depths)):
            count = int(count_matrix[r, c])
            if count <= 0:
                continue
            ratio_pct = float(matrix[r, c])
            bg = im.cmap(im.norm(ratio_pct))
            luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            color = "white" if luminance < 0.55 else "black"
            ax.text(
                c,
                r,
                dual_label(count, ratio_pct / 100.0),
                ha="center",
                va="center",
                fontsize=FONT_ANNOT,
                color=color,
            )

    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


@with_locked_pyplot
def _save_fig6_6(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
    category_csv_path: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    distinct_frame = build_fig6_6_distinct_token_plot_data(page_rows)
    category_frame = build_fig6_6_category_plot_data(expression_rows, page_rows)
    values = np.array([page.distinct_token_count for page in page_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    bars = axes[0].bar(
        distinct_frame["bin_label"],
        distinct_frame["page_count"],
        color=COLOR_PAGE_COVERAGE,
        edgecolor="white",
    )
    annotate_bars_dual(axes[0], bars, distinct_frame["page_count"], distinct_frame["page_ratio"])
    add_stats_box(axes[0], vmin=float(values.min()), vmean=float(values.mean()), vmax=float(values.max()))
    axes[0].set_title("Distinct tokens per page")
    axes[0].set_xlabel("distinct_token_count")
    axes[0].set_ylabel("Pages")
    axes[0].tick_params(axis="x", rotation=20)
    enable_horizontal_grid_only(axes[0])

    bars = axes[1].bar(
        category_frame["token_category"],
        category_frame["page_ratio"] * 100,
        color=COLOR_PAGE_COVERAGE,
        edgecolor="white",
    )
    annotate_bars_dual(axes[1], bars, category_frame["page_count"], category_frame["page_ratio"])
    axes[1].set_ylabel("Page coverage (%)")
    axes[1].set_xlabel("Token category")
    axes[1].set_title("Token-category page coverage")
    axes[1].tick_params(axis="x", rotation=30)
    enable_horizontal_grid_only(axes[1])

    fig.suptitle("Figure 6-6 Distinct tokens per page and token-category page coverage")
    fig.tight_layout()
    write_plot_csv(distinct_frame, csv_path)
    write_plot_csv(category_frame, category_csv_path)
    paths = save_figure_outputs(fig, figure_stem)
    plt.close(fig)
    paths["csv"] = csv_path
    paths["category_csv"] = category_csv_path
    return paths


@with_locked_pyplot
def _save_fig6_7(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    *,
    rare_tokens: set[str],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    plot_data = build_fig6_7_plot_data(expression_rows, page_rows, rare_tokens)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    if len(plot_data):
        bars = ax.bar(
            plot_data["occurrence_bin"],
            plot_data["page_count"],
            color=COLOR_RARE10,
        )
        annotate_bars_dual(ax, bars, plot_data["page_count"], plot_data["page_ratio"])
    ax.set_xlabel("Rare-token instances per page (corpus frequency ≤ 8)")
    ax.set_ylabel("Pages")
    ax.set_title(
        "Figure 6-7 Rare-vocabulary token load per page\n"
        "(token instances from types that appear at most 8 times in the corpus)"
    )
    enable_horizontal_grid_only(ax)
    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


@with_locked_pyplot
def _save_fig6_8(group_summary, figure_stem: Path, csv_path: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    plot_data = build_fig6_8_plot_data(group_summary)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    labels = plot_data["group_display"].tolist()
    bars = ax.bar(
        labels,
        plot_data["cooccurrence_event_count"],
        color=COLOR_SIMILAR,
        edgecolor="white",
    )
    annotate_bars_dual(
        ax,
        bars,
        plot_data["cooccurrence_event_count"],
        plot_data["cooccurrence_page_ratio"],
    )
    ax.set_ylabel("Same-page co-occurrence events")
    ax.set_xlabel("Similar-token group")
    ax.set_title("Figure 6-8 Same-page co-occurrence counts of similar-token groups")
    ax.tick_params(axis="x", rotation=20)
    enable_horizontal_grid_only(ax)
    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


def _save_page_token_count_figure(
    page_rows: Sequence[PageLatexMetricsRow],
    *,
    field: str,
    plot_data,
    title: str,
    suptitle: str,
    xlabel: str,
    color: str,
    figure_stem: Path,
    csv_path: Path,
    x_rotation: int = 20,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    values = np.array([getattr(row, field) for row in page_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(plot_data["bin_label"], plot_data["page_count"], color=color, edgecolor="white")
    annotate_bars_dual(ax, bars, plot_data["page_count"], plot_data["page_ratio"])
    if values.size:
        add_stats_box(ax, vmin=float(values.min()), vmean=float(values.mean()), vmax=float(values.max()))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pages")
    ax.tick_params(axis="x", rotation=x_rotation)
    enable_horizontal_grid_only(ax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


@with_locked_pyplot
def _save_fig6_9(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    plot_data = build_fig6_9_plot_data(page_rows)
    return _save_page_token_count_figure(
        page_rows,
        field="total_token_count",
        plot_data=plot_data,
        title="Total tokens per page",
        suptitle="Figure 6-9 Total token count per page",
        xlabel="total_token_count",
        color=COLOR_PAGE_COUNT,
        figure_stem=figure_stem,
        csv_path=csv_path,
        x_rotation=25,
    )


@with_locked_pyplot
def _save_fig6_10(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    plot_data = build_fig6_10_plot_data(page_rows)
    return _save_page_token_count_figure(
        page_rows,
        field="distinct_token_count",
        plot_data=plot_data,
        title="Distinct tokens per page",
        suptitle="Figure 6-10 Distinct token count per page",
        xlabel="distinct_token_count",
        color=COLOR_PAGE_COVERAGE,
        figure_stem=figure_stem,
        csv_path=csv_path,
    )


def export_page_latex_figures(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    figures_dir: Path,
    *,
    rare_tokens: set[str] | None = None,
    similar_group_summary=None,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir = figures_dir.parent / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    def record(key: str, result: dict[str, Path]) -> None:
        outputs[f"{key}.png"] = result["png"]
        outputs[f"{key}.csv"] = result["csv"]

    record(
        "fig6_1_page_ast_scale",
        _save_fig6_1(page_rows, figures_dir / "fig6_1_page_ast_scale", plot_data_dir / "fig6_1_page_ast_scale_plot_data.csv"),
    )
    record(
        "fig6_3_ast_depth_coverage",
        _save_fig6_3(
            expression_rows,
            page_rows,
            figures_dir / "fig6_3_ast_depth_coverage",
            plot_data_dir / "fig6_3_page_ast_depth_distribution.csv",
        ),
    )
    record(
        "fig6_4_structure_coverage",
        _save_fig6_4(page_rows, figures_dir / "fig6_4_structure_coverage", plot_data_dir / "fig6_4_page_structure_distribution.csv"),
    )
    record(
        "fig6_5_structure_depth_joint",
        _save_fig6_5(page_rows, figures_dir / "fig6_5_structure_depth_joint", plot_data_dir / "fig6_5_structure_depth_joint_distribution.csv"),
    )
    result6 = _save_fig6_6(
        expression_rows,
        page_rows,
        figures_dir / "fig6_6_token_category",
        plot_data_dir / "fig6_6_distinct_token_distribution.csv",
        plot_data_dir / "fig6_6_token_category_coverage.csv",
    )
    record("fig6_6_token_category", result6)
    outputs["fig6_6_token_category.category_csv"] = result6["category_csv"]

    record(
        "fig6_7_rare10",
        _save_fig6_7(
            expression_rows,
            page_rows,
            rare_tokens=rare_tokens or set(),
            figure_stem=figures_dir / "fig6_7_rare10",
            csv_path=plot_data_dir / "fig6_7_rare10_occurrence_distribution.csv",
        ),
    )
    record(
        "fig6_9_page_total_tokens",
        _save_fig6_9(
            page_rows,
            figures_dir / "fig6_9_page_total_tokens",
            plot_data_dir / "fig6_9_page_total_token_distribution.csv",
        ),
    )
    record(
        "fig6_10_page_distinct_tokens",
        _save_fig6_10(
            page_rows,
            figures_dir / "fig6_10_page_distinct_tokens",
            plot_data_dir / "fig6_10_page_distinct_token_distribution.csv",
        ),
    )

    if similar_group_summary is not None and len(similar_group_summary):
        record(
            "fig6_8_similar_token",
            _save_fig6_8(
                similar_group_summary,
                figure_stem=figures_dir / "fig6_8_similar_token",
                csv_path=plot_data_dir / "fig6_8_similar_token_group_distribution.csv",
            ),
        )

    return outputs
