#!/usr/bin/env python3
"""Render knowledge coverage figure (SVG only)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from _common import configure_fonts, save_svg  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "第二批数据题目_seed21pro_latex.csv"
DEFAULT_OUTPUT = ROOT / "figure" / "知识点结构与题目占比.svg"

COLORS = {
    "预备知识": "#173F73",
    "函数": "#2F78B7",
    "几何与代数": "#5299C5",
    "概率与统计": "#9BC5DB",
}

LEVEL_2_STRUCTURE = {
    "预备知识": [(1, "集合"), (2, "常用逻辑用语"), (3, "等式与不等式"), (4, "一元二次函数、方程与不等式")],
    "函数": [
        (5, "函数概念与性质"), (6, "幂函数"), (7, "指数与指数函数"), (8, "对数与对数函数"),
        (9, "三角函数"), (10, "函数应用"), (11, "数列"), (12, "一元函数导数及其应用"),
    ],
    "几何与代数": [
        (13, "平面向量"), (14, "解三角形"), (15, "复数"), (16, "立体几何初步"),
        (17, "空间向量与立体几何"), (18, "直线与圆"), (19, "圆锥曲线与方程"),
    ],
    "概率与统计": [
        (20, "随机事件与概率"), (21, "随机抽样与统计描述"), (22, "用样本估计总体"),
        (23, "计数原理"), (24, "条件概率"), (25, "随机变量及其分布"),
        (26, "正态分布"), (27, "成对数据的统计分析"),
    ],
}

FONT_MAIN_TITLE = 20
FONT_SECTION_TITLE = 14.5
FONT_PANEL_CAPTION = 13.5
FONT_AXIS_LABEL = 11.5
FONT_TABLE_DOMAIN = 13.5
FONT_TABLE_COL = 10.5
FONT_COUNT_LABEL = 10.2
FONT_TABLE_BODY = 10.0

# Manual label placement overrides keyed by secondary-module number.
# Values: (offset_x, offset_y, va, ha) in points.
COUNT_LABEL_OVERRIDES: dict[int, tuple[int, int, str, str]] = {
    6: (-16, 0, "center", "right"),
    10: (16, 0, "center", "left"),
    16: (16, 0, "center", "left"),
    22: (16, 0, "center", "left"),
    26: (16, 0, "center", "left"),
}

GRID_COLOR = "#D4DAE0"
GRID_STYLE = {"axis": "y", "color": GRID_COLOR, "linestyle": "--", "linewidth": 0.55, "alpha": 0.9}
SPINE_COLOR = "#3A424A"
SPINE_WIDTH = 1.0

FIG_WIDTH = 18.0
MARGIN_LEFT = 0.065
MARGIN_RIGHT = 0.975
MARGIN_BOTTOM_PX = 28
MARGIN_TOP_PX = 12
SECTION_GAP_PX = 45
TITLE_TO_SECTION_GAP_PX = 45
MAIN_TITLE_HEIGHT_PX = 28
SUPTITLE_Y = 0.985
SECTION_HEIGHTS_PX = (205, 300, 425)  # A, B, C
B_DOMAIN_LABEL_OFFSET_PX = 26


def _figure_height_in() -> float:
    total_px = (
        MARGIN_TOP_PX
        + MAIN_TITLE_HEIGHT_PX
        + TITLE_TO_SECTION_GAP_PX
        + sum(SECTION_HEIGHTS_PX)
        + 2 * SECTION_GAP_PX
        + MARGIN_BOTTOM_PX
    )
    return total_px / 72

REMOVED_TITLES = {
    "20250302-2343-271e-0255-bb6b16097802_%e4%ba%8c.18",
    "20250825-0258-3866-a577-42ed1b311756_19",
    "20250925-0232-42f2-b523-a5d15b420018_19",
    "20240530-0703-3456-733b-887290986497_%e4%b8%80.17",
    "20250915-1217-09f4-78c5-af7146901132_%e4%ba%8c.16",
    "20250913-0821-097b-0e63-d8dfb5016574_%e4%b8%80.18",
    "20250917-0224-5414-c8b8-e7e46c157027_19",
    "20231112-0458-5581-6875-e00846727645",
    "20250921-2357-1664-8db6-2748b8427632_13",
    "20250917-0850-557e-a7b5-76c773514305_19",
    "20240422-0602-1838-21fc-94987a345614",
    "20250915-1217-09f4-78c5-af7146901132_%e4%ba%8c.19",
    "20250915-1217-09f4-78c5-af7146901132_%e4%ba%8c.17",
    "20250827-0746-4910-63db-fb026a157065_18",
    "20250912-0723-5526-e880-1638d5001195_%e4%ba%8c.18",
    "20231214-0338-2277-32c3-07db41860810",
    "20240527-1328-161a-c2c0-676cb0382723_19",
    "20250828-0742-3331-8c0e-60f42c401241_18",
    "20250925-0232-42f2-b523-a5d15b420018_18",
    "20250824-0231-5998-1040-afb4c7110116_19",
    "20250915-1217-09f4-78c5-af7146901132_%e4%ba%8c.18",
    "20240527-0257-25e8-3008-c6addc717258_18",
    "20230906-1003-51c0-3907-ad371d134036",
    "20250913-0821-097b-0e63-d8dfb5016574_%e4%b8%80.15",
    "20250302-2343-271e-0255-bb6b16097802_%e4%ba%8c.17",
    "20240315-0656-01c7-dd6c-464899045582",
    "20250225-1109-2036-d5a8-48cb2a607426_18",
    "20250116-1027-1447-a24b-ff520d057443",
    "20231011-0220-2509-8f50-4169a1317033",
    "20240529-0315-024e-6c33-50e02f788585_%e4%ba%8c.18",
    "20250917-1206-23a2-8f6e-972b5b337375_18",
    "20250918-0139-1634-a8b1-30a529526108_18",
    "20250829-0225-35b5-4745-e35fe5638955_18",
    "20250825-0258-3866-a577-42ed1b311756_17",
    "20250228-1139-5613-ccac-210615076857",
    "20241202-0510-234f-c14c-f91d4b301757",
    "20250924-2229-55b2-270a-a4d6f0216854_18",
    "20250924-2229-55b2-270a-a4d6f0216854_16",
    "20250924-0247-5099-e682-6d078a188993_18",
    "20250919-0513-32c0-d6b9-b851f6035002_%e4%b8%80.19",
    "20250919-0513-32c0-d6b9-b851f6035002_%e4%b8%80.17",
    "20250924-0247-5099-e682-6d078a188993_17",
    "20250919-0513-32c0-d6b9-b851f6035002_%e4%b8%80.15",
    "20250924-0149-1937-4298-dae6af069156_16",
    "20250917-1206-23a2-8f6e-972b5b337375_19",
    "20250923-0044-33bb-bcb2-ec539b608487_%e4%b8%80.16",
    "20250917-0850-557e-a7b5-76c773514305_15",
    "20250918-0139-1634-a8b1-30a529526108_17",
    "20250916-0658-4889-b9ca-51548b850223_19",
    "20250917-1206-23a2-8f6e-972b5b337375_17",
}


def pct(count: int, total: int) -> float:
    return count / total * 100


def _fig_y(fig, px: float) -> float:
    return px / (fig.get_figheight() * fig.dpi)


def _section_layout(fig) -> dict:
    """Pixel-precise vertical layout: fixed top title, 45px gaps, equal-width sections."""
    gap = _fig_y(fig, SECTION_GAP_PX)
    title_gap = _fig_y(fig, TITLE_TO_SECTION_GAP_PX)
    h_a = _fig_y(fig, SECTION_HEIGHTS_PX[0])
    h_b = _fig_y(fig, SECTION_HEIGHTS_PX[1])
    h_c = _fig_y(fig, SECTION_HEIGHTS_PX[2])
    bottom = _fig_y(fig, MARGIN_BOTTOM_PX)

    title_bottom = SUPTITLE_Y - _fig_y(fig, MAIN_TITLE_HEIGHT_PX) / 2
    y_a1 = title_bottom - title_gap
    y_a0 = y_a1 - h_a
    y_b1 = y_a0 - gap
    y_b0 = y_b1 - h_b
    y_c1 = y_b0 - gap
    y_c0 = y_c1 - h_c

    if y_c0 < bottom:
        raise ValueError(f"section C overflows bottom margin: {y_c0:.4f} < {bottom:.4f}")

    return {
        "left": MARGIN_LEFT,
        "right": MARGIN_RIGHT,
        "a": (y_a0, y_a1),
        "b": (y_b0, y_b1),
        "c": (y_c0, y_c1),
    }


def _style_axes(ax, *, ylabel: str | None = None, show_left: bool = True) -> None:
    ax.grid(False, axis="x")
    ax.grid(**GRID_STYLE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(SPINE_WIDTH)
    if not show_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL, labelpad=8)


def _place_all_count_labels(
    ax,
    xs: list[int],
    ys: list[float],
    counts: list[int],
) -> None:
    """Place every non-zero count label with trend-aware offsets to reduce overlap."""
    candidate_offsets = (
        (0, 13, "bottom"),
        (0, -14, "top"),
        (8, 13, "bottom"),
        (-8, 13, "bottom"),
        (8, -14, "top"),
        (-8, -14, "top"),
        (0, 19, "bottom"),
        (0, -19, "top"),
        (11, 10, "bottom"),
        (-11, 10, "bottom"),
    )
    placed: list[tuple[float, float, int, int]] = []

    for i, (x, y, count) in enumerate(zip(xs, ys, counts, strict=True)):
        if count <= 0:
            continue
        if x in COUNT_LABEL_OVERRIDES:
            ox, oy, va, ha = COUNT_LABEL_OVERRIDES[x]
        else:
            prev_y = ys[i - 1] if i > 0 else y
            next_y = ys[i + 1] if i < len(ys) - 1 else y
            rising = y >= prev_y and y >= next_y
            falling = y <= prev_y and y <= next_y
            if rising:
                preferred = candidate_offsets[0]
            elif falling:
                preferred = candidate_offsets[1]
            else:
                preferred = candidate_offsets[0] if i % 2 == 0 else candidate_offsets[1]
            ordered = [preferred, *[c for c in candidate_offsets if c != preferred]]

            chosen = ordered[0]
            for ox, oy, va in ordered:
                conflict = False
                for px, py, pox, poy in placed:
                    x_near = abs(x - px) <= 1.15
                    y_near = abs(y - py) < 1.25
                    same_side = (oy > 0) == (poy > 0)
                    if x_near and y_near and same_side:
                        conflict = True
                        break
                if not conflict:
                    chosen = (ox, oy, va)
                    break

            ox, oy, va = chosen
            ha = "center"

        placed.append((x, y, ox, oy))
        ax.annotate(
            f"{count}题",
            (x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=FONT_COUNT_LABEL,
            color="#2E3842",
            clip_on=False,
            zorder=6,
        )


def read_and_count(csv_path: Path) -> tuple[int, list[dict]]:
    title_to_l1: dict[str, set[str]] = defaultdict(set)
    title_to_l2: dict[str, set[str]] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            title = (row["题目名"] or "").strip()
            if not title or title in REMOVED_TITLES:
                continue
            title_to_l1.setdefault(title, set())
            title_to_l2.setdefault(title, set())
            l1 = (row["Label_level_1"] or "").strip()
            l2 = (row["Label_level_2"] or "").strip()
            if l1:
                title_to_l1[title].add(l1)
            if l2:
                title_to_l2[title].add(l2)

    all_titles = set(title_to_l1) | set(title_to_l2)
    total = len(all_titles)
    if total == 0:
        raise ValueError("no questions after filtering")

    groups = []
    for name, modules in LEVEL_2_STRUCTURE.items():
        l1_count = sum(name in title_to_l1[t] for t in all_titles)
        items = [(num, mod, sum(mod in title_to_l2[t] for t in all_titles)) for num, mod in modules]
        groups.append({"name": name, "count": l1_count, "color": COLORS[name], "items": items})
    return total, groups


def draw_table(ax, group: dict, total: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column centers for center-aligned headers and body cells.
    table_columns = (
        ("编号", 0.05),
        ("二级知识模块", 0.36),
        ("占比", 0.72),
        ("数量", 0.91),
    )

    header_h, column_header_h = 0.12, 0.09
    body_top = 1 - header_h - column_header_h
    row_h = body_top / 8
    color = group["color"]
    header_on_color = "white" if group["name"] != "概率与统计" else "#17334D"

    ax.add_patch(Rectangle((0, 1 - header_h), 1, header_h, color=color, lw=0))
    header_y = 1 - header_h / 2
    ax.text(
        0.5,
        header_y,
        f"{group['name']}  |  {pct(group['count'], total):.1f}% · {group['count']}题",
        ha="center",
        va="center",
        fontsize=FONT_TABLE_DOMAIN,
        fontweight="bold",
        color=header_on_color,
    )

    ax.add_patch(Rectangle((0, body_top), 1, column_header_h, facecolor="#EDF2F7", edgecolor="none"))
    header_row_y = body_top + column_header_h / 2
    for label, x in table_columns:
        ax.text(
            x,
            header_row_y,
            label,
            va="center",
            ha="center",
            fontsize=FONT_TABLE_COL,
            fontweight="bold",
            color="#2E3842",
        )

    for i in range(8):
        y0 = body_top - (i + 1) * row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y0), 1, row_h, facecolor="#F7F9FC", edgecolor="none"))
        ax.plot([0, 1], [y0, y0], color="#E4EAF0", lw=0.6)
        if i >= len(group["items"]):
            continue
        number, module, count = group["items"][i]
        y = y0 + row_h / 2
        values = (str(number), module, f"{pct(count, total):.1f}%", str(count))
        for (_, x), value in zip(table_columns, values, strict=True):
            ax.text(
                x,
                y,
                value,
                va="center",
                ha="center",
                fontsize=FONT_TABLE_BODY,
                fontweight="bold",
                color="#20262D",
            )

    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=1.0))


def render(total: int, groups: list[dict], output_path: Path) -> Path:
    configure_fonts(plt)
    fig = plt.figure(figsize=(FIG_WIDTH, _figure_height_in()), facecolor="white", dpi=72)
    layout = _section_layout(fig)
    left, right = layout["left"], layout["right"]
    y_a0, y_a1 = layout["a"]
    y_b0, y_b1 = layout["b"]
    y_c0, y_c1 = layout["c"]

    fig.suptitle(
        f"高中数学知识点题目覆盖率分布（n={total}）",
        fontsize=FONT_MAIN_TITLE,
        y=SUPTITLE_Y,
        color="#17212B",
        fontweight="bold",
    )

    gs_a = fig.add_gridspec(1, 1, left=left, right=right, bottom=y_a0, top=y_a1)
    ax_top = fig.add_subplot(gs_a[0])
    y_pos = list(range(len(groups)))
    values = [pct(g["count"], total) for g in groups]
    bars = ax_top.barh(y_pos, values, color=[g["color"] for g in groups], height=0.55, edgecolor="none")
    ax_top.set_yticks(y_pos, [g["name"] for g in groups], fontsize=12, fontweight="bold")
    ax_top.invert_yaxis()
    ax_top.set_xlim(0, 52)
    ax_top.set_xticks([10, 20, 30, 40, 50])
    ax_top.set_xlabel("题目占比（%）", fontsize=FONT_AXIS_LABEL, labelpad=8)
    ax_top.set_title("（A）一级知识领域覆盖率", loc="left", fontsize=FONT_SECTION_TITLE, pad=18, color="#17212B", fontweight="bold")
    ax_top.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.55, alpha=0.9)
    ax_top.grid(False, axis="y")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(True)
    ax_top.spines["bottom"].set_visible(True)
    ax_top.spines["left"].set_color(SPINE_COLOR)
    ax_top.spines["bottom"].set_color(SPINE_COLOR)
    ax_top.spines["left"].set_linewidth(SPINE_WIDTH)
    ax_top.spines["bottom"].set_linewidth(SPINE_WIDTH)
    for bar, group, value in zip(bars, groups, values):
        ax_top.text(
            bar.get_width() + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}% · {group['count']}题",
            va="center",
            fontsize=10.5,
            color="#252D35",
        )

    gs_b = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.075, 0.925],
        left=left,
        right=right,
        bottom=y_b0,
        top=y_b1,
        hspace=0.03,
    )
    ax_b_title = fig.add_subplot(gs_b[0, 0])
    ax_b_title.axis("off")
    ax_b_title.text(
        0.0,
        0.35,
        "（B）二级知识模块题目分布",
        ha="left",
        va="center",
        fontsize=FONT_SECTION_TITLE,
        color="#17212B",
        fontweight="bold",
        transform=ax_b_title.transAxes,
    )

    mid = gs_b[1, 0].subgridspec(1, 4, wspace=0.10)
    mid_axes = []
    for idx, group in enumerate(groups):
        ax = fig.add_subplot(mid[0, idx], sharey=mid_axes[0] if mid_axes else None)
        mid_axes.append(ax)
        nums = [x[0] for x in group["items"]]
        counts = [x[2] for x in group["items"]]
        coverages = [pct(c, total) for c in counts]
        ax.plot(
            nums,
            coverages,
            color=group["color"],
            lw=3.0,
            marker="o",
            ms=11.2,
            markerfacecolor=group["color"],
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        ax.set_ylim(0, 13.8)
        ax.set_yticks(range(0, 13, 2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
        ax.set_xticks(nums)
        ax.margins(x=0.04)
        _style_axes(ax, ylabel="题目占比（%）" if idx == 0 else None, show_left=idx == 0)
        _place_all_count_labels(ax, nums, coverages, counts)
        ax.annotate(
            group["name"],
            xy=(0.5, 0),
            xycoords="axes fraction",
            xytext=(0, -B_DOMAIN_LABEL_OFFSET_PX),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=FONT_PANEL_CAPTION,
            color=group["color"],
            fontweight="bold",
            clip_on=False,
        )

    fig.canvas.draw()
    label_room = _fig_y(fig, B_DOMAIN_LABEL_OFFSET_PX + 8)
    for ax in mid_axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + label_room, box.width, max(box.height - label_room, 0.01)])

    gs_c = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.065, 0.935],
        left=left,
        right=right,
        bottom=y_c0,
        top=y_c1,
        hspace=0.02,
    )
    ax_c_title = fig.add_subplot(gs_c[0, 0])
    ax_c_title.axis("off")
    ax_c_title.text(
        0.0,
        0.35,
        "（C）二级知识模块统计",
        ha="left",
        va="center",
        fontsize=FONT_SECTION_TITLE,
        color="#17212B",
        fontweight="bold",
        transform=ax_c_title.transAxes,
    )

    table_grid = gs_c[1, 0].subgridspec(1, 4, wspace=0.04)
    for idx, group in enumerate(groups):
        draw_table(fig.add_subplot(table_grid[0, idx]), group, total)

    out = save_svg(fig, output_path, tight=False)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    total, groups = read_and_count(args.csv)
    path = render(total, groups, args.output)
    print(path)


if __name__ == "__main__":
    main()
