"""Chapter 7 split figures — plain page-count ratios from CSV tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS = ("overall", "train", "val", "test")
PLOT_SPLITS = ("train", "val", "test")
SPLIT_LABELS = {
    "overall": "Overall",
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
# Basic matplotlib colors
SPLIT_COLORS = {
    "overall": "gray",
    "train": "tab:blue",
    "val": "tab:orange",
    "test": "tab:green",
}

STRUCTURE_NAMES = {
    "has_frac": "Fraction",
    "has_sup": "Superscript",
    "has_sub": "Subscript",
    "has_sqrt": "Radical",
    "has_env": "Environment",
    "has_bigop": "Big operators",
    "has_accent": "Accents",
    "has_stackrel": "Stackrel",
    "has_textcircled": "Textcircled",
}

EXPR_BIN_ORDER = [
    ("expr_1_10", "1-10"),
    ("expr_11_20", "11-20"),
    ("expr_21_40", "21-40"),
    ("expr_gt_40", ">40"),
]
TOKEN_BIN_ORDER = [
    ("page_token_bin_1", "<200"),
    ("page_token_bin_2", "200-399"),
    ("page_token_bin_3", "400-699"),
    ("page_token_bin_4", "700-1199"),
    ("page_token_tail", r"$\geq 1200$"),
]
MAXLEN_BIN_ORDER = [
    ("maxlen_le_20", r"$\leq 20$"),
    ("maxlen_21_40", "21-40"),
    ("maxlen_41_80", "41-80"),
    ("maxlen_gt_80", ">80"),
]
MAXLEN_MERGE = {
    "maxlen_le_20": ("maxlen_1_10", "maxlen_11_20"),
    "maxlen_21_40": ("maxlen_21_40",),
    "maxlen_41_80": ("maxlen_41_80",),
    "maxlen_gt_80": ("maxlen_gt_80",),
}
FOREGROUND_DENSITY_ORDER = [
    ("density_lt_2pct", "<2%"),
    ("density_2_4pct", "2–4%"),
    ("density_4_6pct", "4–6%"),
    ("density_6_8pct", "6–8%"),
    ("density_8_10pct", "8–10%"),
    ("density_ge_10pct", "≥10%"),
]
LAYOUT_DOMAIN_ORDER = [
    ("structure_layout", "structure_layout"),
    ("hybrid_layout", "hybrid_layout"),
]
BLOCK_TYPE_ORDER = [
    ("Txtblock", "Text block"),
    ("deleted_text_block", "Deleted text"),
    ("figure", "Figure"),
    ("chart", "Chart"),
]
LINE_ASPECT_RATIO_ORDER = [
    ("aspect_lt_3", "<3"),
    ("aspect_3_5", "3–5"),
    ("aspect_5_8", "5–8"),
    ("aspect_8_12", "8–12"),
    ("aspect_gt_12", ">12"),
]
LINE_INTERFERENCE_ORDER = [
    ("interference_zero", "Zero"),
    ("interference_low", "Low"),
    ("interference_mid", "Medium"),
    ("interference_high", "High"),
]
EXPRESSION_DIFFICULTY_ORDER = [
    ("L1", "L1"),
    ("L2", "L2"),
    ("L3", "L3"),
    ("L4", "L4"),
]


def _pct(df: pd.DataFrame, split: str, key: str, *, key_col: str = "bin") -> float:
    row = df[(df["split"] == split) & (df[key_col] == key)]
    if row.empty:
        return 0.0
    col = "page_ratio" if "page_ratio" in df.columns else "ratio"
    return float(row.iloc[0][col]) * 100.0


def _abs_count(df: pd.DataFrame, split: str, key: str, *, key_col: str = "bin", count_col: str = "count") -> float:
    row = df[(df["split"] == split) & (df[key_col] == key)]
    if row.empty:
        return 0.0
    return float(row.iloc[0][count_col])


def _merged_maxlen_table(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    split_totals = raw.groupby("split")["count"].sum().to_dict()
    for split in SPLITS:
        sub = raw[raw["split"] == split]
        total = split_totals.get(split, 1) or 1
        for merged_key, parts in MAXLEN_MERGE.items():
            count = int(sub[sub["bin"].isin(parts)]["count"].sum())
            rows.append({"bin": merged_key, "split": split, "count": count, "ratio": count / total})
    return pd.DataFrame(rows)


def _format_pct_bar_label(val: float) -> str:
    """Format percentage bar labels; preserve non-zero rare bins that round to 0.0."""
    if val <= 0:
        return "0"
    if val < 0.1:
        return f"{val:.2f}"
    return f"{val:.1f}"


LOW_SUPPORT_BOX_FONT = 8


def _format_low_support_box_text(label: str, counts: dict[str, int]) -> str:
    return "\n".join(
        [
            "Train / Val / Test",
            label,
            f"Train: {counts.get('train', 0)}",
            f"Val: {counts.get('val', 0)}",
            f"Test: {counts.get('test', 0)}",
        ]
    )


def _add_low_support_box(ax, label: str, counts: dict[str, int]) -> None:
    ax.text(
        0.98,
        0.98,
        _format_low_support_box_text(label, counts),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=LOW_SUPPORT_BOX_FONT,
        linespacing=1.2,
        clip_on=False,
        bbox={
            "boxstyle": "square,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.8,
        },
    )


def _grouped_bars(
    ax,
    categories: list[str],
    labels: list[str],
    df: pd.DataFrame,
    *,
    key_col: str = "bin",
    splits: tuple[str, ...] = PLOT_SPLITS,
    show_values: bool = False,
    use_counts: bool = False,
    count_col: str = "count",
    ylabel: str | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper right",
) -> None:
    x = np.arange(len(categories))
    n = len(splits)
    width = 0.8 / n
    for i, split in enumerate(splits):
        if use_counts:
            vals = [_abs_count(df, split, cat, key_col=key_col, count_col=count_col) for cat in categories]
        else:
            vals = [_pct(df, split, cat, key_col=key_col) for cat in categories]
        bars = ax.bar(
            x + (i - (n - 1) / 2) * width,
            vals,
            width=width,
            label=SPLIT_LABELS[split],
            color=SPLIT_COLORS[split],
        )
        if show_values:
            for bar, val in zip(bars, vals):
                label = f"{int(val)}" if use_counts else _format_pct_bar_label(val)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel or ("Pages" if use_counts else "Pages (%)"))
    if show_legend:
        ax.legend(fontsize=8, loc=legend_loc)
    if show_values:
        ymax = ax.get_ylim()[1]
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.12)


def figure7_1_output_scale(tables_dir: Path, figures_dir: Path) -> Path:
    expr = pd.read_csv(tables_dir / "table2_expression_count_bins.csv")
    token = pd.read_csv(tables_dir / "table2_page_token_bins.csv")
    maxlen = _merged_maxlen_table(tables_dir / "table2_max_expression_length_bins.csv")

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    panels = [
        (axes[0], expr, EXPR_BIN_ORDER, "Expressions per page"),
        (axes[1], token, TOKEN_BIN_ORDER, "Tokens per page"),
        (axes[2], maxlen, MAXLEN_BIN_ORDER, "Maximum expression length"),
    ]
    for i, (ax, df, order, title) in enumerate(panels):
        cats = [k for k, _ in order]
        labels = [l for _, l in order]
        _grouped_bars(ax, cats, labels, df, show_values=True, show_legend=i == 0)
        ax.set_title(title)

    fig.suptitle("Page-level output scale distributions")
    fig.tight_layout(pad=1.2)
    png = figures_dir / "figure7_1_output_scale_distributions.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def figure7_2_structural(tables_dir: Path, figures_dir: Path) -> Path:
    ast = pd.read_csv(tables_dir / "table4_ast_depth.csv")
    structure = pd.read_csv(tables_dir / "table5_structure.csv")

    fig, axes = plt.subplots(1, 3, figsize=(22, 4.5))
    ax_depth, ax_presence, ax_count = axes

    depths = [str(d) for d in range(5)]
    ast_rows = []
    ast_depth5_counts: dict[str, int] = {}
    for split in PLOT_SPLITS:
        for depth in range(5):
            row = ast[(ast["split"] == split) & (ast["ast_depth"] == depth)]
            ratio = float(row.iloc[0]["max_depth_ratio"]) if len(row) else 0.0
            ast_rows.append({"split": split, "bin": str(depth), "ratio": ratio})
        row5 = ast[(ast["split"] == split) & (ast["ast_depth"] == 5)]
        ast_depth5_counts[split] = int(row5.iloc[0]["pages_with_max_depth"]) if len(row5) else 0
    ast_df = pd.DataFrame(ast_rows)
    _grouped_bars(ax_depth, depths, depths, ast_df, show_values=True, show_legend=True, legend_loc="upper left")
    _add_low_support_box(ax_depth, "AST depth 5", ast_depth5_counts)
    ax_depth.set_xlabel("Maximum AST depth")
    ax_depth.set_title("Maximum AST depth")

    presence_cats = list(STRUCTURE_NAMES)
    presence_labels = list(STRUCTURE_NAMES.values())
    _grouped_bars(
        ax_presence,
        presence_cats,
        presence_labels,
        structure,
        key_col="structure",
        show_values=True,
        show_legend=False,
    )
    ax_presence.set_xlabel("Structure type")
    ax_presence.set_title("Structure-type presence")
    ax_presence.tick_params(axis="x", labelsize=7)

    sc_cats = [f"structure_count_{i}" for i in range(6)]
    sc_labels = [str(i) for i in range(6)]
    structure_count6: dict[str, int] = {}
    for split in PLOT_SPLITS:
        row = structure[(structure["split"] == split) & (structure["structure"] == "structure_count_6")]
        structure_count6[split] = int(row.iloc[0]["page_count"]) if len(row) else 0
    _grouped_bars(
        ax_count,
        sc_cats,
        sc_labels,
        structure,
        key_col="structure",
        show_values=True,
        show_legend=False,
    )
    _add_low_support_box(ax_count, "Structure type count 6", structure_count6)
    ax_count.set_xlabel("Number of structure types per page")
    ax_count.set_title("Number of structure types per page")

    fig.suptitle("Page-level structural distributions")
    fig.tight_layout(pad=1.2)
    png = figures_dir / "figure7_2_structural_distributions.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def _figure_from_category_table(
    tables_dir: Path,
    figures_dir: Path,
    *,
    table_name: str,
    figure_name: str,
    title: str,
    category_order: list[tuple[str, str]],
    ylabel: str,
    ncols: int = 1,
    panel_titles: list[str] | None = None,
) -> Path:
    df = pd.read_csv(tables_dir / table_name)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5), squeeze=False)
    panel_axes = list(axes[0])
    if panel_titles is None:
        panel_titles = [title]
    for ax, panel_title in zip(panel_axes, panel_titles, strict=True):
        cats = [key for key, _ in category_order]
        labels = [label for _, label in category_order]
        _grouped_bars(
            ax,
            cats,
            labels,
            df,
            key_col="category",
            show_values=True,
            show_legend=ax is panel_axes[0],
            ylabel=ylabel,
        )
        ax.set_title(panel_title)
    if ncols == 1:
        fig.suptitle(title)
    else:
        fig.suptitle(title, y=1.02)
    fig.tight_layout(pad=1.2)
    png = figures_dir / figure_name
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def figure7_3_foreground_density(tables_dir: Path, figures_dir: Path) -> Path:
    return _figure_from_category_table(
        tables_dir,
        figures_dir,
        table_name="table9_foreground_density_bins.csv",
        figure_name="figure7_3_page_foreground_density.png",
        title="Page-level foreground density (I ≤ t_I, pooled Otsu)",
        category_order=FOREGROUND_DENSITY_ORDER,
        ylabel="Pages (%)",
    )


def figure7_4_layout_composition(tables_dir: Path, figures_dir: Path) -> Path:
    flow = pd.read_csv(tables_dir / "table10_flow_structure.csv")
    blocks = pd.read_csv(tables_dir / "table10_block_type_composition.csv")
    block_order = list(BLOCK_TYPE_ORDER)
    if "other" in blocks["category"].unique():
        block_order = [*block_order, ("other", "Other")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    panels = [
        (axes[0], flow, LAYOUT_DOMAIN_ORDER, "Layout domain", "Pages (%)"),
        (axes[1], blocks, block_order, "Block type composition", "Blocks (%)"),
    ]
    for i, (ax, df, order, panel_title, ylabel) in enumerate(panels):
        cats = [key for key, _ in order]
        labels = [label for _, label in order]
        _grouped_bars(
            ax,
            cats,
            labels,
            df,
            key_col="category",
            show_values=True,
            show_legend=i == 0,
            ylabel=ylabel,
        )
        ax.set_title(panel_title)
    fig.suptitle("Page layout and block composition", y=1.02)
    fig.tight_layout(pad=1.2)
    png = figures_dir / "figure7_4_layout_block_composition.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def figure7_5_line_geometry(tables_dir: Path, figures_dir: Path) -> Path:
    aspect = pd.read_csv(tables_dir / "table11_line_aspect_ratio_bins.csv")
    interference = pd.read_csv(tables_dir / "table11_line_interference_bins.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    panels = [
        (axes[0], aspect, LINE_ASPECT_RATIO_ORDER, "Line aspect ratio"),
        (axes[1], interference, LINE_INTERFERENCE_ORDER, "Neighboring-stroke interference"),
    ]
    for i, (ax, df, order, panel_title) in enumerate(panels):
        cats = [key for key, _ in order]
        labels = [label for _, label in order]
        _grouped_bars(
            ax,
            cats,
            labels,
            df,
            key_col="category",
            show_values=True,
            show_legend=i == 0,
            ylabel="Lines (%)",
        )
        ax.set_title(panel_title)
    fig.suptitle("Line-level geometry and interference", y=1.02)
    fig.tight_layout(pad=1.2)
    png = figures_dir / "figure7_5_line_geometry_interference.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png


def figure7_6_expression_difficulty(tables_dir: Path, figures_dir: Path) -> Path:
    return _figure_from_category_table(
        tables_dir,
        figures_dir,
        table_name="table12_expression_difficulty.csv",
        figure_name="figure7_6_expression_difficulty.png",
        title="Expression-level difficulty distribution",
        category_order=EXPRESSION_DIFFICULTY_ORDER,
        ylabel="Expressions (%)",
    )


def export_ch7_figures(tables_dir: Path, figures_dir: Path) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for pdf in figures_dir.glob("*.pdf"):
        pdf.unlink()
    for stale in (
        "appendix_figure_a1_joint_structure_depth_fidelity.png",
        "figure7_3_overall_split_fidelity.png",
        "figure7_3_fidelity_summary.csv",
    ):
        path = figures_dir / stale
        if path.is_file():
            path.unlink()
    outputs = {}
    figure_specs = (
        ("figure7_1_output_scale_distributions", figure7_1_output_scale),
        ("figure7_2_structural_distributions", figure7_2_structural),
        ("figure7_3_page_foreground_density", figure7_3_foreground_density),
        ("figure7_4_layout_block_composition", figure7_4_layout_composition),
        ("figure7_5_line_geometry_interference", figure7_5_line_geometry),
        ("figure7_6_expression_difficulty", figure7_6_expression_difficulty),
    )
    for stem, fn in figure_specs:
        table_path = {
            "figure7_3_page_foreground_density": tables_dir / "table9_foreground_density_bins.csv",
            "figure7_4_layout_block_composition": tables_dir / "table10_flow_structure.csv",
            "figure7_5_line_geometry_interference": tables_dir / "table11_line_aspect_ratio_bins.csv",
            "figure7_6_expression_difficulty": tables_dir / "table12_expression_difficulty.csv",
        }.get(stem)
        if table_path is not None and not table_path.is_file():
            continue
        png = fn(tables_dir, figures_dir)
        outputs[f"{stem}.png"] = png.name
    return outputs


def remove_legacy_figures(figures_dir: Path) -> None:
    legacy = [
        "fig1_scale_ecdf.png",
        "fig2_max_length_bins.png",
        "fig3_ast_depth.png",
        "fig4_structure.png",
        "fig5_structure_depth_joint.png",
        "fig6_rare8_similar.png",
        "fig7_deviation_heatmap.png",
        "figure7_3_overall_split_fidelity.png",
        "figure7_3_fidelity_summary.csv",
        "appendix_figure_a1_joint_structure_depth_fidelity.png",
    ]
    for name in legacy:
        path = figures_dir / name
        if path.is_file():
            path.unlink()
