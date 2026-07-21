"""Parse grounding layout into explicit structural features (normalized coords)."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .feature_store import atomic_replace_parquet
from .utils import atomic_write_json, atomic_write_text, ensure_dir

REF_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)

LAYOUT_COLUMNS = [
    "image_id",
    "layout_available",
    "layout_missing_reason",
    # coverage
    "content_area_ratio",
    "blank_ratio_diag",  # diagnostic only; not in PCA
    "mean_center_x",
    "mean_center_y",
    "center_spread_x",
    "center_spread_y",
    "margin_top",
    "margin_bottom",
    "margin_left",
    "margin_right",
    # blocks
    "block_count",
    "text_block_count",
    "formula_block_count",
    "figure_block_count",
    "table_block_count",
    "text_area_ratio",
    "formula_area_ratio",
    "figure_area_ratio",
    "table_area_ratio",
    "median_block_area",
    "iqr_block_area",
    "mean_block_aspect_ratio",
    "std_block_aspect_ratio",
    "max_block_area_ratio",
    # arrangement
    "estimated_row_count",
    "estimated_col_count",
    "mean_horizontal_gap",
    "mean_vertical_gap",
    "left_align_score",
    "center_align_score",
    "two_column_score",
    "block_overlap_ratio",
    # reading order (OCR order vs geometric order)
    "reading_order_inversion_count",
    "reading_order_row_jump_count",
    "reading_order_mean_jump_distance",
] + [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)] + [
    "block_count_log1p",
    "text_block_count_log1p",
    "formula_block_count_log1p",
    "figure_block_count_log1p",
    "table_block_count_log1p",
]


def parse_grounding_blocks(text: str) -> list[dict[str, Any]]:
    """Preserve OCR emission order as list index (reading order)."""
    blocks = []
    for m in REF_PATTERN.finditer(text or ""):
        label = (m.group(1) or "").strip().lower()
        det_raw = (m.group(2) or "").strip()
        try:
            coords = eval(det_raw, {"__builtins__": {}})  # noqa: S307
        except Exception:
            continue
        if not isinstance(coords, list):
            continue
        for box in coords:
            if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                continue
            x1, y1, x2, y2 = [float(v) / 999.0 for v in box[:4]]
            x1, x2 = sorted([_clip01(x1), _clip01(x2)])
            y1, y2 = sorted([_clip01(y1), _clip01(y2)])
            blocks.append(
                {
                    "label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "w": max(x2 - x1, 0.0),
                    "h": max(y2 - y1, 0.0),
                    "area": max(x2 - x1, 0.0) * max(y2 - y1, 0.0),
                    "aspect": (max(x2 - x1, 1e-8) / max(y2 - y1, 1e-8)),
                }
            )
    return blocks


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _label_bucket(label: str) -> str:
    if any(k in label for k in ("formula", "equation", "math", "latex")):
        return "formula"
    if any(k in label for k in ("figure", "image", "picture", "photo")):
        return "figure"
    if "table" in label:
        return "table"
    return "text"


def _reading_order_metrics(blocks: list[dict[str, Any]]) -> dict[str, float]:
    """Compare OCR emission order vs geometric top-to-bottom, left-to-right."""
    n = len(blocks)
    if n < 2:
        return {
            "reading_order_inversion_count": 0.0,
            "reading_order_row_jump_count": 0.0,
            "reading_order_mean_jump_distance": 0.0,
        }
    geo = sorted(range(n), key=lambda i: (blocks[i]["cy"], blocks[i]["cx"]))
    rank = {i: r for r, i in enumerate(geo)}
    # Kendall-style inversions in OCR sequence vs geo ranks
    inversions = 0
    for a in range(n):
        for b in range(a + 1, n):
            if rank[a] > rank[b]:
                inversions += 1
    # Row jumps: consecutive OCR blocks whose geo row bands differ by >1
    # Band by y terciles of geo order
    cys = np.array([blocks[i]["cy"] for i in range(n)])
    # Assign row ids by clustering cy with tolerance
    order_cy = np.argsort(cys)
    row_id = np.zeros(n, dtype=np.int32)
    rid = 0
    prev_cy = cys[order_cy[0]]
    for idx in order_cy:
        if cys[idx] - prev_cy > 0.06:
            rid += 1
            prev_cy = cys[idx]
        row_id[idx] = rid
    jumps = 0
    jump_dists = []
    for a, b in zip(range(n - 1), range(1, n)):
        dy = abs(blocks[b]["cy"] - blocks[a]["cy"])
        dx = abs(blocks[b]["cx"] - blocks[a]["cx"])
        dist = float(math.hypot(dx, dy))
        jump_dists.append(dist)
        if abs(int(row_id[b]) - int(row_id[a])) > 1:
            jumps += 1
    return {
        "reading_order_inversion_count": float(inversions),
        "reading_order_row_jump_count": float(jumps),
        "reading_order_mean_jump_distance": float(np.mean(jump_dists)) if jump_dists else 0.0,
    }


def _empty_features(image_id: str, reason: str) -> dict[str, Any]:
    feat: dict[str, Any] = {c: 0.0 for c in LAYOUT_COLUMNS}
    feat["image_id"] = image_id
    feat["layout_available"] = False
    feat["layout_missing_reason"] = reason
    feat["content_area_ratio"] = 0.0
    feat["blank_ratio_diag"] = 1.0
    return feat


def blocks_to_features(image_id: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        return _empty_features(image_id, "no_parseable_grounding_coordinates")

    feat: dict[str, Any] = {c: 0.0 for c in LAYOUT_COLUMNS}
    feat["image_id"] = image_id
    feat["layout_available"] = True
    feat["layout_missing_reason"] = None

    areas = np.array([b["area"] for b in blocks], dtype=np.float64)
    aspects = np.array([b["aspect"] for b in blocks], dtype=np.float64)
    cxs = np.array([b["cx"] for b in blocks], dtype=np.float64)
    cys = np.array([b["cy"] for b in blocks], dtype=np.float64)
    buckets = [_label_bucket(b["label"]) for b in blocks]

    feat["block_count"] = len(blocks)
    feat["text_block_count"] = sum(1 for b in buckets if b == "text")
    feat["formula_block_count"] = sum(1 for b in buckets if b == "formula")
    feat["figure_block_count"] = sum(1 for b in buckets if b == "figure")
    feat["table_block_count"] = sum(1 for b in buckets if b == "table")

    grid = np.zeros((64, 64), dtype=np.float64)
    for b in blocks:
        x1 = int(b["x1"] * 64)
        x2 = max(int(math.ceil(b["x2"] * 64)), x1 + 1)
        y1 = int(b["y1"] * 64)
        y2 = max(int(math.ceil(b["y2"] * 64)), y1 + 1)
        grid[y1:y2, x1:x2] = 1.0
    content = float(grid.mean())
    feat["content_area_ratio"] = content
    feat["blank_ratio_diag"] = 1.0 - content

    feat["mean_center_x"] = float(cxs.mean())
    feat["mean_center_y"] = float(cys.mean())
    feat["center_spread_x"] = float(cxs.std()) if len(cxs) > 1 else 0.0
    feat["center_spread_y"] = float(cys.std()) if len(cys) > 1 else 0.0

    x1s = np.array([b["x1"] for b in blocks])
    y1s = np.array([b["y1"] for b in blocks])
    x2s = np.array([b["x2"] for b in blocks])
    y2s = np.array([b["y2"] for b in blocks])
    feat["margin_left"] = float(x1s.min())
    feat["margin_right"] = float(1.0 - x2s.max())
    feat["margin_top"] = float(y1s.min())
    feat["margin_bottom"] = float(1.0 - y2s.max())

    area_by = {k: 0.0 for k in ("text", "formula", "figure", "table")}
    for b, buck in zip(blocks, buckets):
        area_by[buck] += b["area"]
    total_area = max(float(areas.sum()), 1e-12)
    feat["text_area_ratio"] = area_by["text"] / total_area
    feat["formula_area_ratio"] = area_by["formula"] / total_area
    feat["figure_area_ratio"] = area_by["figure"] / total_area
    feat["table_area_ratio"] = area_by["table"] / total_area

    q25, q50, q75 = np.percentile(areas, [25, 50, 75])
    feat["median_block_area"] = float(q50)
    feat["iqr_block_area"] = float(q75 - q25)
    feat["mean_block_aspect_ratio"] = float(aspects.mean())
    feat["std_block_aspect_ratio"] = float(aspects.std()) if len(aspects) > 1 else 0.0
    feat["max_block_area_ratio"] = float(areas.max() / max(content, 1e-12)) if content > 0 else float(areas.max())

    # Estimated rows/cols via cy/cx banding
    order_cy = np.argsort(cys)
    rows = 1
    prev = cys[order_cy[0]]
    for idx in order_cy[1:]:
        if cys[idx] - prev > 0.06:
            rows += 1
            prev = cys[idx]
    order_cx = np.argsort(cxs)
    cols = 1
    prev = cxs[order_cx[0]]
    for idx in order_cx[1:]:
        if cxs[idx] - prev > 0.08:
            cols += 1
            prev = cxs[idx]
    feat["estimated_row_count"] = float(rows)
    feat["estimated_col_count"] = float(cols)

    # Gaps between nearest neighbors in x/y
    h_gaps, v_gaps = [], []
    for i, bi in enumerate(blocks):
        hx = [
            bj["x1"] - bi["x2"]
            for j, bj in enumerate(blocks)
            if j != i and abs(bj["cy"] - bi["cy"]) < 0.05 and bj["x1"] >= bi["x2"]
        ]
        vy = [
            bj["y1"] - bi["y2"]
            for j, bj in enumerate(blocks)
            if j != i and abs(bj["cx"] - bi["cx"]) < 0.08 and bj["y1"] >= bi["y2"]
        ]
        if hx:
            h_gaps.append(min(hx))
        if vy:
            v_gaps.append(min(vy))
    feat["mean_horizontal_gap"] = float(np.mean(h_gaps)) if h_gaps else 0.0
    feat["mean_vertical_gap"] = float(np.mean(v_gaps)) if v_gaps else 0.0

    # Alignment scores: fraction of blocks near left edge / page center
    feat["left_align_score"] = float(np.mean(np.abs(x1s - np.median(x1s)) < 0.03))
    feat["center_align_score"] = float(np.mean(np.abs(cxs - 0.5) < 0.08))

    # Two-column heuristic: bimodal cx
    left = cxs < 0.45
    right = cxs > 0.55
    feat["two_column_score"] = float(
        (left.mean() > 0.25) and (right.mean() > 0.25) and (np.abs(cxs[left].mean() - cxs[right].mean()) > 0.25)
    )

    # Pairwise overlap ratio (area intersection / union sum)
    overlap = 0.0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a, b = blocks[i], blocks[j]
            ix1 = max(a["x1"], b["x1"])
            iy1 = max(a["y1"], b["y1"])
            ix2 = min(a["x2"], b["x2"])
            iy2 = min(a["y2"], b["y2"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            overlap += inter
    feat["block_overlap_ratio"] = float(overlap / max(total_area, 1e-12))

    feat.update(_reading_order_metrics(blocks))

    occ = np.zeros((4, 4), dtype=np.float64)
    for b in blocks:
        for i in range(4):
            for j in range(4):
                cell = (j / 4, i / 4, (j + 1) / 4, (i + 1) / 4)
                ix1 = max(b["x1"], cell[0])
                iy1 = max(b["y1"], cell[1])
                ix2 = min(b["x2"], cell[2])
                iy2 = min(b["y2"], cell[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                occ[i, j] += inter / (1 / 16)
    occ = np.clip(occ, 0, 1)
    for i in range(4):
        for j in range(4):
            feat[f"occupancy_{i}_{j}"] = float(occ[i, j])

    for key in [
        "block_count",
        "text_block_count",
        "formula_block_count",
        "figure_block_count",
        "table_block_count",
    ]:
        feat[f"{key}_log1p"] = float(np.log1p(feat[key]))

    # Clamp content ratio
    if feat["content_area_ratio"] > 1.0:
        feat["content_area_ratio"] = 1.0
    return feat


def parse_one(image_id: str, raw_text: str, raw_dir: Path) -> dict[str, Any]:
    ensure_dir(raw_dir)
    atomic_write_text(raw_dir / f"{image_id}.md", raw_text or "")
    blocks = parse_grounding_blocks(raw_text or "")
    atomic_write_json(raw_dir / f"{image_id}.blocks.json", blocks)
    return blocks_to_features(image_id, blocks)


def run_parse_layout(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    raw_ocr = out_dir / "recognition_raw"
    layout_raw = out_dir / "layout_raw"
    ensure_dir(layout_raw)
    ocr_index = out_dir / "ocr_generations.parquet"
    if not ocr_index.exists():
        raise FileNotFoundError(f"Missing {ocr_index}; run vllm_ocr_runner first")

    ocr_df = pd.read_parquet(ocr_index)
    rows = []
    for _, r in ocr_df.iterrows():
        if str(r.get("status")) != "ok":
            continue
        image_id = str(r["image_id"])
        text_path = raw_ocr / f"{image_id}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(r.get("text") or "")
        rows.append(parse_one(image_id, text, layout_raw))

    df = pd.DataFrame(rows)
    for c in LAYOUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[LAYOUT_COLUMNS]
    atomic_replace_parquet(df, out_dir / "layout_features.parquet")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    df = run_parse_layout(cfg)
    n_avail = int(df["layout_available"].sum()) if len(df) else 0
    print(f"[parse_layout] n={len(df)} layout_available={n_avail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
