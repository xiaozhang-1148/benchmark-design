#!/usr/bin/env python3
"""Render annotation overlays with sequence arrows from group_id and order_id."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon

INPUT_DIR = Path(
    "/mnt/nvme_metadata/private/qitian/labeled_data/Batch01_qitian_9911/"
    "20260630-Batch01_remove_operatorname/Folders-Doc"
)
OUTPUT_DIR = Path("/home/baoquan/ocr-process/benchmark-design/tempt/images")

TARGET_STEMS = (
    "mamatwrik22198rwdlg",
    "mamavb6d7ac4tagmve",
    "mamazs85c60l4l8ndvn",
    "mambdy7izqgcxesn8pm",
    "mb7qovob315cteecewe",
)

GROUP_COLORS = {
    0: ("#BDC3C7", "#7F8C8D"),
    1: ("#3498DB", "#2C3E50"),
    2: ("#2ECC71", "#1E8449"),
    3: ("#F39C12", "#D68910"),
    4: ("#9B59B6", "#6C3483"),
}
DEFAULT_GROUP_COLOR = ("#E74C3C", "#922B21")
ARROW_COLORS = (
    "#E74C3C",
    "#2980B9",
    "#27AE60",
    "#F39C12",
    "#8E44AD",
    "#16A085",
    "#D35400",
    "#C0392B",
)
CROSS_GROUP_ARROW_COLOR = "#FF00FF"
TARGET_ALIAS = "handwritten"


def _annotation_alias(annotation: dict) -> str:
    return str((annotation.get("class") or {}).get("alias") or "")


def _filter_handwritten(annotations: list[dict]) -> list[dict]:
    return [annotation for annotation in annotations if _annotation_alias(annotation) == TARGET_ALIAS]


def _load_points(annotation: dict) -> list[tuple[float, float]]:
    contour = annotation.get("contour") or {}
    raw_points = contour.get("points") or []
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def _polygon_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    xs, ys = zip(*points)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _group_color(group_id: int) -> tuple[str, str]:
    return GROUP_COLORS.get(group_id, DEFAULT_GROUP_COLOR)


def _duplicate_orders(annotations: list[dict]) -> set[str]:
    orders = [str(annotation.get("order")) for annotation in annotations if annotation.get("order") is not None]
    return {order for order, count in Counter(orders).items() if count > 1}


def _duplicate_order_ids_by_group(annotations: list[dict]) -> dict[int, set[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for annotation in annotations:
        grouped[int(annotation.get("group_id", -1))].append(int(annotation.get("order_id", -1)))
    return {
        group_id: {order_id for order_id, count in Counter(order_ids).items() if count > 1}
        for group_id, order_ids in grouped.items()
        if any(count > 1 for count in Counter(order_ids).values())
    }


def _sorted_handwritten_sequence(
    annotations: list[dict],
) -> list[tuple[int, int, int, dict, tuple[float, float]]]:
    sequence: list[tuple[int, int, int, dict, tuple[float, float]]] = []
    for index, annotation in enumerate(annotations):
        points = _load_points(annotation)
        if len(points) < 3:
            continue
        group_id = int(annotation.get("group_id", -1))
        order_id = int(annotation.get("order_id", -1))
        sequence.append((group_id, order_id, index, annotation, _polygon_center(points)))
    sequence.sort(key=lambda item: (item[0], item[1], item[2]))
    return sequence


def _draw_large_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    step_index: int,
    cross_group: bool,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=34 if cross_group else 26,
        linewidth=6.0 if cross_group else 4.5,
        color=color,
        alpha=0.95,
        shrinkA=24,
        shrinkB=24,
        zorder=8,
        connectionstyle="arc3,rad=0.12" if cross_group else "arc3,rad=0.0",
    )
    axis.add_patch(arrow)

    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    axis.text(
        mid_x,
        mid_y,
        str(step_index),
        color="white",
        fontsize=9 if cross_group else 8,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=9,
        bbox={
            "facecolor": color,
            "alpha": 0.95,
            "pad": 2.5,
            "edgecolor": "white",
            "linewidth": 1.2,
            "boxstyle": "circle,pad=0.2",
        },
    )


def render_page(stem: str, *, input_dir: Path, output_dir: Path) -> Path:
    json_path = input_dir / f"{stem}.json"
    image_path = input_dir / f"{stem}.jpg"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    annotations = _filter_handwritten(payload.get("annotations", []))

    duplicate_orders = _duplicate_orders(annotations)
    duplicate_order_ids = _duplicate_order_ids_by_group(annotations)
    sequence = _sorted_handwritten_sequence(annotations)

    fig, axis = plt.subplots(figsize=(12, 16))
    axis.imshow(plt.imread(str(image_path)))

    for step_index, (group_id, order_id, _, annotation, center) in enumerate(sequence, start=1):
        points = _load_points(annotation)
        facecolor, edgecolor = _group_color(group_id)
        order = str(annotation.get("order") or "")
        is_dup_order = order in duplicate_orders
        is_dup_order_id = order_id in duplicate_order_ids.get(group_id, set())

        patch = MplPolygon(
            points,
            closed=True,
            fill=True,
            facecolor="#E74C3C" if is_dup_order else facecolor,
            edgecolor="#C0392B" if is_dup_order else edgecolor,
            alpha=0.34 if is_dup_order else 0.26,
            linewidth=2.8 if is_dup_order or is_dup_order_id else 2.0,
            linestyle="--" if is_dup_order else "-",
            zorder=3,
        )
        axis.add_patch(patch)

        label = f"#{step_index}\ng{group_id}:{order_id}"
        if order:
            label += f"\n{order}"
        axis.text(
            center[0],
            center[1],
            label,
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
            bbox={
                "facecolor": "#C0392B" if is_dup_order else edgecolor,
                "alpha": 0.92,
                "pad": 3,
                "edgecolor": "white",
                "linewidth": 1.2,
            },
        )

    for step_index, (current, nxt) in enumerate(zip(sequence, sequence[1:]), start=1):
        _, _, _, _, start = current
        next_group_id, _, _, _, end = nxt
        current_group_id = current[0]
        cross_group = next_group_id != current_group_id
        color = CROSS_GROUP_ARROW_COLOR if cross_group else ARROW_COLORS[(step_index - 1) % len(ARROW_COLORS)]
        _draw_large_arrow(
            axis,
            start,
            end,
            color=color,
            step_index=step_index,
            cross_group=cross_group,
        )

    dup_order_text = ", ".join(sorted(duplicate_orders)) or "(none)"
    dup_oid_parts = [
        f"group {group_id}: {sorted(order_ids)}"
        for group_id, order_ids in sorted(duplicate_order_ids.items())
    ]
    dup_oid_text = "; ".join(dup_oid_parts) or "(none)"
    group_chain = " -> ".join(str(group_id) for group_id, _ in groupby(item[0] for item in sequence))
    title = (
        f"{stem}.jpg  |  alias={TARGET_ALIAS!r} only ({len(sequence)} regions)\n"
        f"duplicate order: {dup_order_text}\n"
        f"duplicate order_id: {dup_oid_text}\n"
        f"global order: sort by (group_id, order_id); magenta arrows connect groups ({group_chain})"
    )
    axis.set_title(title, fontsize=9, loc="left")
    axis.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_handwritten_order_arrows.png"
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render annotation order arrows for selected pages.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--stems", nargs="*", default=list(TARGET_STEMS))
    args = parser.parse_args()

    written: list[Path] = []
    for stem in args.stems:
        written.append(render_page(stem, input_dir=args.input_dir, output_dir=args.output_dir))

    print(f"Wrote {len(written)} figures to {args.output_dir.resolve()}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
