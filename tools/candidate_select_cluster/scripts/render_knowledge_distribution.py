#!/usr/bin/env python3
"""从原始 CSV 统计并绘制高中数学知识点题目覆盖率分布。

统计口径：
1. 按题目名剔除指定的 50 道题。
2. 同一道题若在 CSV 中出现多行，只对同一个知识标签计数一次。
3. 覆盖率 = 涉及该知识点的题目数 / 剔除后的题目总数。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_CLUSTER_ROOT = Path(__file__).resolve().parents[1]
if str(_CLUSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLUSTER_ROOT))

from bootstrap import setup

setup()

from paths import FIGURES_DIR, LABELED_CSV, KNOWLEDGE_COVERAGE_PNG

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

DEFAULT_CSV_PATH = LABELED_CSV
DEFAULT_OUTPUT_PATH = KNOWLEDGE_COVERAGE_PNG

COLORS = {
    "预备知识": "#173F73",
    "函数": "#2F78B7",
    "几何与代数": "#5299C5",
    "概率与统计": "#9BC5DB",
}

# 二级知识模块的层级与显示顺序依据标签框架固定。
LEVEL_2_STRUCTURE = {
    "预备知识": [
        (1, "集合"),
        (2, "常用逻辑用语"),
        (3, "等式与不等式"),
        (4, "一元二次函数、方程与不等式"),
    ],
    "函数": [
        (5, "函数概念与性质"),
        (6, "幂函数"),
        (7, "指数与指数函数"),
        (8, "对数与对数函数"),
        (9, "三角函数"),
        (10, "函数应用"),
        (11, "数列"),
        (12, "一元函数导数及其应用"),
    ],
    "几何与代数": [
        (13, "平面向量"),
        (14, "解三角形"),
        (15, "复数"),
        (16, "立体几何初步"),
        (17, "空间向量与立体几何"),
        (18, "直线与圆"),
        (19, "圆锥曲线与方程"),
    ],
    "概率与统计": [
        (20, "随机事件与概率"),
        (21, "随机抽样与统计描述"),
        (22, "用样本估计总体"),
        (23, "计数原理"),
        (24, "条件概率"),
        (25, "随机变量及其分布"),
        (26, "正态分布"),
        (27, "成对数据的统计分析"),
    ],
}

# 前述方案确定剔除的 50 道题。
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制高中数学知识点覆盖率分布图")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="源 CSV 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出 PNG 路径（同时写出同主名 .svg）",
    )
    return parser.parse_args()


def read_and_count(csv_path: Path) -> tuple[int, list[dict]]:
    required_columns = {"题目名", "Label_level_1", "Label_level_2"}
    title_to_level_1: dict[str, set[str]] = defaultdict(set)
    title_to_level_2: dict[str, set[str]] = defaultdict(set)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要字段：{', '.join(sorted(missing))}")

        for row in reader:
            title = (row["题目名"] or "").strip()
            if not title or title in REMOVED_TITLES:
                continue

            # 即便某行标签为空，也要把题目计入过滤后的题目总数。
            title_to_level_1.setdefault(title, set())
            title_to_level_2.setdefault(title, set())

            level_1 = (row["Label_level_1"] or "").strip()
            level_2 = (row["Label_level_2"] or "").strip()
            if level_1:
                title_to_level_1[title].add(level_1)
            if level_2:
                title_to_level_2[title].add(level_2)

    all_titles = set(title_to_level_1) | set(title_to_level_2)
    total = len(all_titles)
    if total == 0:
        raise ValueError("过滤后没有可用于统计的题目。")

    known_level_2 = {
        module
        for modules in LEVEL_2_STRUCTURE.values()
        for _, module in modules
    }
    unexpected_level_2 = sorted(
        {
            label
            for labels in title_to_level_2.values()
            for label in labels
            if label not in known_level_2
        }
    )
    if unexpected_level_2:
        raise ValueError(f"发现标签框架之外的二级标签：{unexpected_level_2}")

    groups = []
    for group_name, modules in LEVEL_2_STRUCTURE.items():
        level_1_count = sum(
            group_name in title_to_level_1[title] for title in all_titles
        )
        items = []
        for number, module in modules:
            module_count = sum(
                module in title_to_level_2[title] for title in all_titles
            )
            items.append((number, module, module_count))
        groups.append(
            {
                "name": group_name,
                "count": level_1_count,
                "color": COLORS[group_name],
                "items": items,
            }
        )

    return total, groups


def percentage(count: int, total: int) -> float:
    return count / total * 100


def draw_table(ax, group: dict, total: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    header_h = 0.135
    column_header_h = 0.085
    body_top = 1 - header_h - column_header_h
    row_h = body_top / 8
    color = group["color"]
    header_text_color = "white" if group["name"] != "概率与统计" else "#17334D"

    ax.add_patch(Rectangle((0, 1 - header_h), 1, header_h, color=color, lw=0))
    ax.text(
        0.5,
        1 - header_h / 2,
        f'{group["name"]}\n'
        f'{percentage(group["count"], total):.1f}%｜{group["count"]}题',
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
        color=header_text_color,
        linespacing=1.05,
    )

    ax.add_patch(
        Rectangle(
            (0, body_top),
            1,
            column_header_h,
            facecolor="#EDF2F7",
            edgecolor="none",
        )
    )
    ax.text(
        0.035, body_top + column_header_h / 2, "编号",
        va="center", ha="left", fontsize=8.6, color="#45515D",
    )
    ax.text(
        0.145, body_top + column_header_h / 2, "二级知识模块",
        va="center", ha="left", fontsize=8.6, color="#45515D",
    )
    ax.text(
        0.855, body_top + column_header_h / 2, "覆盖率",
        va="center", ha="right", fontsize=8.6, color="#45515D",
    )
    ax.text(
        0.965, body_top + column_header_h / 2, "题目数",
        va="center", ha="right", fontsize=8.6, color="#45515D",
    )

    for row_index in range(8):
        y0 = body_top - (row_index + 1) * row_h
        if row_index % 2 == 0:
            ax.add_patch(
                Rectangle((0, y0), 1, row_h, facecolor="#F7F9FC", edgecolor="none")
            )
        ax.plot([0, 1], [y0, y0], color="#E4EAF0", lw=0.6)

        if row_index >= len(group["items"]):
            continue

        number, module, count = group["items"][row_index]
        y = y0 + row_h / 2
        ax.text(
            0.035, y, str(number),
            va="center", ha="left", fontsize=9, color="#303841",
        )
        # 所有二级标签保持单行显示。
        ax.text(
            0.145, y, module,
            va="center", ha="left", fontsize=8.9, color="#20262D",
        )
        ax.text(
            0.855, y, f"{percentage(count, total):.1f}%",
            va="center", ha="right", fontsize=9, color="#303841",
        )
        ax.text(
            0.965, y, str(count),
            va="center", ha="right", fontsize=9, color="#303841",
        )

    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=0.9))


def draw_figure(total: int, groups: list[dict], png_path: Path, svg_path: Path) -> None:
    from benchmark_design.report.export_figures import _configure_matplotlib_fonts

    _configure_matplotlib_fonts(plt)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig = plt.figure(figsize=(18, 10.8), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.55, 2.15],
        left=0.065,
        right=0.975,
        top=0.92,
        bottom=0.045,
        hspace=0.30,
    )

    fig.suptitle(
        f"高中数学知识点题目覆盖率分布（n={total}）",
        fontsize=19,
        fontweight="semibold",
        y=0.972,
        color="#17212B",
    )

    ax_top = fig.add_subplot(outer[0])
    y_positions = list(range(len(groups)))
    values = [percentage(group["count"], total) for group in groups]
    bars = ax_top.barh(
        y_positions,
        values,
        color=[group["color"] for group in groups],
        height=0.55,
        edgecolor="none",
    )
    ax_top.set_yticks(
        y_positions,
        [group["name"] for group in groups],
        fontsize=11.5,
    )
    ax_top.invert_yaxis()
    ax_top.set_xlim(0, 52)
    ax_top.set_xticks(range(0, 51, 10))
    ax_top.tick_params(axis="x", labelsize=9.5, pad=4)
    ax_top.set_xlabel("题目覆盖率（%）", fontsize=11, labelpad=6)
    ax_top.set_title(
        "（A）一级知识领域覆盖率",
        loc="left",
        fontsize=13,
        fontweight="semibold",
        pad=12,
    )
    ax_top.grid(axis="x", color="#DCE2E8", lw=0.8)
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.spines["bottom"].set_color("#5A6570")
    for bar, group, value in zip(bars, groups, values):
        ax_top.text(
            value + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f'{value:.1f}%｜{group["count"]}题',
            va="center",
            ha="left",
            fontsize=10.2,
            color="#252D35",
        )

    middle_grid = outer[1].subgridspec(1, 4, wspace=0.09)
    middle_axes = []
    for index, group in enumerate(groups):
        ax = fig.add_subplot(
            middle_grid[0, index],
            sharey=middle_axes[0] if middle_axes else None,
        )
        middle_axes.append(ax)
        numbers = [item[0] for item in group["items"]]
        counts = [item[2] for item in group["items"]]
        coverages = [percentage(count, total) for count in counts]
        ax.plot(
            numbers,
            coverages,
            color=group["color"],
            lw=2.3,
            marker="o",
            ms=5.6,
            markerfacecolor=group["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.set_ylim(0, 12.7)
        ax.set_yticks(range(0, 13, 2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}%"))
        ax.set_xticks(numbers)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", color="#DCE2E8", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#59636D")
        ax.spines["bottom"].set_color("#59636D")
        ax.set_title(
            group["name"],
            color=group["color"],
            fontsize=11.5,
            fontweight="semibold",
            pad=9,
        )
        if index == 0:
            ax.set_ylabel("题目覆盖率（%）", fontsize=11, labelpad=7)
        else:
            ax.tick_params(labelleft=False)
        for x_value, y_value, count in zip(numbers, coverages, counts):
            ax.annotate(
                f"{count}题",
                (x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.6,
                color="#3A424A",
            )

    middle_y_top = middle_axes[0].get_position().y1
    middle_y_bottom = middle_axes[0].get_position().y0
    fig.text(
        0.065,
        middle_y_top + 0.030,
        "（B）二级知识模块覆盖率（按一级知识领域分组）",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="semibold",
        color="#17212B",
    )
    fig.text(
        0.52,
        middle_y_bottom - 0.035,
        "二级知识模块编号",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#313941",
    )

    bottom_grid = outer[2].subgridspec(1, 4, wspace=0.035)
    for index, group in enumerate(groups):
        draw_table(fig.add_subplot(bottom_grid[0, index]), group, total)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"找不到 CSV：{args.csv}")

    png_path = args.output
    svg_path = png_path.with_suffix(".svg")

    total, groups = read_and_count(args.csv)
    draw_figure(total, groups, png_path, svg_path)

    print(f"源文件：{args.csv}")
    print(f"剔除题目：{len(REMOVED_TITLES)} 道")
    print(f"保留题目：{total} 道")
    for group in groups:
        item_counts = "，".join(
            f"{name}={count}" for _, name, count in group["items"]
        )
        print(f'{group["name"]}={group["count"]}；{item_counts}')
    print(f"PNG：{png_path}")
    print(f"SVG：{svg_path}")


if __name__ == "__main__":
    main()
