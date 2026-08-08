#!/usr/bin/env python3
"""Render page-level distribution panel (3 KDE + 2 bar charts, SVG only)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
from _common import configure_fonts, save_svg  # noqa: E402
from _page_level_panel import (  # noqa: E402
    DEPTH_LABELS,
    bar_frame,
    count_by_category_and_hard,
    draw_density_count,
    draw_density_percent,
    draw_share_bars,
    legend_handles,
    place_panel_captions,
    split_density_by_hard,
    split_metrics_by_hard,
)

DEFAULT_METRICS = ROOT / "data" / "page_latex_metrics.csv"
DEFAULT_FEATURES = ROOT / "data" / "image_features.csv"
DEFAULT_KEPT = ROOT / "data" / "kept_samples.csv"
DEFAULT_OUTPUT = ROOT / "figure" / "page_level_distribution.svg"
OUTPUT_FILENAME = "page_level_distribution.svg"

COMBINED_FIGSIZE = (18.0, 10.5)
COMBINED_GRIDSPEC = dict(left=0.06, right=0.99, top=0.88, bottom=0.08, hspace=0.55, wspace=0.36)
BOTTOM_BAR_WIDTH = 0.18
BOTTOM_BAR_OFFSETS = (-0.22, 0.0, 0.22)
BOTTOM_WIDE_BAR_OFFSETS = (-0.30, 0.0, 0.30)
BOTTOM_CATEGORY_EDGE_PAD = 0.58
BOTTOM_WIDE_CATEGORY_GAP = 0.38


def resolve_output_path(output_arg: str | None, hard_index: Path | None) -> Path:
    """Resolve the SVG output path.

    - ``--output`` given as a directory (trailing separator, or an existing
      directory) -> writes ``page_level_distribution.svg`` inside it.
    - ``--output`` given as a file path -> used as-is (with ``.svg`` suffix).
    - No ``--output`` -> defaults next to ``--hard-index`` when provided,
      otherwise to the built-in ``figure/`` location.
    """
    if output_arg is not None:
        raw = str(output_arg)
        path = Path(raw)
        if path.is_dir() or raw.endswith(("/", os.sep)):
            return path / OUTPUT_FILENAME
        return path
    if hard_index is not None:
        return hard_index.parent / OUTPUT_FILENAME
    return DEFAULT_OUTPUT


def render(
    metrics_csv: Path,
    features_csv: Path,
    output_path: Path,
    *,
    kept_samples_csv: Path | None = DEFAULT_KEPT,
    hard_index_csv: Path | None = None,
) -> Path:
    metrics = pd.read_csv(metrics_csv)
    features = pd.read_csv(features_csv)
    sample_csv = hard_index_csv if hard_index_csv is not None else kept_samples_csv
    if sample_csv is None or not sample_csv.is_file():
        raise ValueError(f"sample index csv is required: {sample_csv}")

    values_total, total_hard, total_non_hard = split_metrics_by_hard(
        metrics, "total_token_count", sample_csv
    )
    values_distinct, distinct_hard, distinct_non_hard = split_metrics_by_hard(
        metrics, "distinct_token_count", sample_csv
    )
    densities, density_hard, density_non_hard = split_density_by_hard(features, sample_csv)
    n = int(values_total.size)
    if n != int(densities.size):
        raise ValueError(f"metrics pages ({n}) != feature pages ({int(densities.size)})")

    struct_cats = [str(i) for i in range(9)]
    struct_cats, _, _ = bar_frame(metrics["distinct_structure_type_count"], struct_cats, n)
    depth_cats, _, _ = bar_frame(metrics["max_ast_depth"], list(DEPTH_LABELS), n)
    struct_hard_counts, struct_non_hard_counts = count_by_category_and_hard(
        metrics, "distinct_structure_type_count", struct_cats, sample_csv
    )
    depth_hard_counts, depth_non_hard_counts = count_by_category_and_hard(
        metrics, "max_ast_depth", depth_cats, sample_csv
    )

    configure_fonts(plt)
    fig = plt.figure(figsize=COMBINED_FIGSIZE)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], **COMBINED_GRIDSPEC)
    ax_density = fig.add_subplot(gs[0, 0:2])
    ax_total = fig.add_subplot(gs[0, 2:4])
    ax_distinct = fig.add_subplot(gs[0, 4:6])
    ax_struct = fig.add_subplot(gs[1, 0:3])
    ax_depth = fig.add_subplot(gs[1, 3:6])

    fig.suptitle(f"Page-level token and structural distributions (n={n:,} pages)", fontsize=13, y=0.972)
    fig.legend(
        handles=legend_handles(
            n=n,
            n_non_hard=int(total_non_hard.size),
            n_hard=int(total_hard.size),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    draw_density_percent(
        ax_density,
        densities,
        "Foreground density (%)",
        hard_values=density_hard,
        non_hard_values=density_non_hard,
        axis_frame="closed",
        grid="both",
    )
    draw_density_count(
        ax_total,
        values_total,
        "Total token count",
        head_tick=int(values_total.min()),
        tail_tick=int(values_total.max()),
        hard_values=total_hard,
        non_hard_values=total_non_hard,
        round_peak=True,
        axis_frame="closed",
        grid="both",
    )
    draw_density_count(
        ax_distinct,
        values_distinct,
        "Distinct token count",
        head_tick=int(values_distinct.min()),
        tail_tick=int(values_distinct.max()),
        hard_values=distinct_hard,
        non_hard_values=distinct_non_hard,
        round_peak=True,
        axis_frame="closed",
        grid="both",
    )
    draw_share_bars(
        ax_struct,
        struct_cats,
        struct_hard_counts,
        struct_non_hard_counts,
        "Structural command type count",
        n_all=n,
        n_hard=int(total_hard.size),
        n_non_hard=int(total_non_hard.size),
        label_focus_range=(2, 8),
        wide_spacing_categories={"8"},
        bar_width=BOTTOM_BAR_WIDTH,
        category_edge_pad=BOTTOM_CATEGORY_EDGE_PAD,
        bar_offsets=BOTTOM_BAR_OFFSETS,
        wide_bar_offsets=BOTTOM_WIDE_BAR_OFFSETS,
        wide_category_gap=BOTTOM_WIDE_CATEGORY_GAP,
        axis_frame="closed",
        grid="y",
    )
    draw_share_bars(
        ax_depth,
        depth_cats,
        depth_hard_counts,
        depth_non_hard_counts,
        "Maximum nested level",
        n_all=n,
        n_hard=int(total_hard.size),
        n_non_hard=int(total_non_hard.size),
        label_focus_range=(1, 5),
        wide_spacing_categories={"5"},
        bar_width=BOTTOM_BAR_WIDTH,
        category_edge_pad=BOTTOM_CATEGORY_EDGE_PAD,
        bar_offsets=BOTTOM_BAR_OFFSETS,
        wide_bar_offsets=BOTTOM_WIDE_BAR_OFFSETS,
        wide_category_gap=BOTTOM_WIDE_CATEGORY_GAP,
        axis_frame="closed",
        grid="y",
    )

    place_panel_captions(
        fig,
        [ax_density, ax_total, ax_distinct],
        [
            "(a) Foreground density per page",
            "(b) Total token count per page",
            "(c) Distinct token count per page",
        ],
        gap=0.014,
    )
    place_panel_captions(
        fig,
        [ax_struct, ax_depth],
        [
            "(d) Structural command diversity per page",
            "(e) Maximum nested level per page",
        ],
        gap=0.014,
    )

    out = save_svg(fig, output_path)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--kept-samples", type=Path, default=DEFAULT_KEPT,
                        help="sample index CSV (basename + post_selection_source), default kept_samples.csv")
    parser.add_argument("--hard-index", type=Path, default=None,
                        help="optional hard-index CSV; when provided it replaces --kept-samples "
                             "and the SVG defaults to the CSV's directory")
    parser.add_argument("--output", type=str, default=None,
                        help="output path; a directory (or trailing /) writes "
                             "page_level_distribution.svg inside it, default: hard-index dir "
                             "if --hard-index given, else figure/page_level_distribution.svg")
    args = parser.parse_args()
    output_path = resolve_output_path(args.output, args.hard_index)
    print(render(args.metrics, args.features, output_path,
                 kept_samples_csv=args.kept_samples, hard_index_csv=args.hard_index))


if __name__ == "__main__":
    main()
