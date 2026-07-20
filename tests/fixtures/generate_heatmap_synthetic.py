"""Synthetic test images for heatmap analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def draw_strokes(canvas: np.ndarray, strokes: list[tuple[int, int, int, int]], thickness: int = 2) -> None:
    for x0, y0, x1, y1 in strokes:
        cv2.line(canvas, (x0, y0), (x1, y1), 0, thickness)


def make_center_layout(w: int, h: int) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    draw_strokes(
        img,
        [
            (cx - w // 6, cy - h // 8, cx + w // 6, cy - h // 8),
            (cx - w // 8, cy, cx + w // 8, cy + h // 10),
            (cx - w // 10, cy + h // 6, cx + w // 10, cy + h // 5),
        ],
        thickness=max(2, min(w, h) // 100),
    )
    return img


def make_left_right_layout(w: int, h: int) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)
    draw_strokes(
        img,
        [
            (w // 6, h // 4, w // 6, 3 * h // 4),
            (5 * w // 6, h // 4, 5 * w // 6, 3 * h // 4),
        ],
        thickness=max(2, min(w, h) // 80),
    )
    return img


def make_top_layout(w: int, h: int) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)
    draw_strokes(
        img,
        [(w // 4, h // 6, 3 * w // 4, h // 6), (w // 3, h // 5, 2 * w // 3, h // 4)],
        thickness=max(2, min(w, h) // 90),
    )
    return img


def make_blank(w: int, h: int) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def build_synthetic_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)

    specs = [
        ("img_center_lo", 400, 600, "center", "A", 1, 85, "t1"),
        ("img_center_hi", 800, 1200, "center", "A", 1, 90, "t1"),
        ("img_lr_lo", 400, 600, "lr", "B", 0, 70, "t1"),
        ("img_lr_hi", 800, 1200, "lr", "B", 0, 75, "t2"),
        ("img_top_lo", 500, 700, "top", "A", 1, 88, "t2"),
        ("img_top_hi", 1000, 1400, "top", "A", 1, 92, "t2"),
        ("img_blank", 400, 600, "blank", "C", 0, 0, "t1"),
        ("img_center2", 600, 900, "center", "C", 0, 60, "t1"),
        ("img_lr2", 600, 900, "lr", "A", 1, 80, "t2"),
        ("img_top2", 600, 900, "top", "B", 0, 65, "t2"),
    ]

    rows = []
    for name, w, h, layout, school, correct, score, tmpl in specs:
        if layout == "center":
            img = make_center_layout(w, h)
        elif layout == "lr":
            img = make_left_right_layout(w, h)
        elif layout == "top":
            img = make_top_layout(w, h)
        else:
            img = make_blank(w, h)
        path = ds / f"{name}.png"
        cv2.imwrite(str(path), img)
        rows.append(
            {
                "image_id": name,
                "image_path": f"{name}.png",
                "school": school,
                "correct": correct,
                "score": score,
                "template_id": tmpl,
                "solution_type": layout,
            }
        )

    pd.DataFrame(rows).to_csv(ds / "metadata.csv", index=False)


if __name__ == "__main__":
    build_synthetic_dataset(Path(__file__).resolve().parent / "heatmap_synthetic")
