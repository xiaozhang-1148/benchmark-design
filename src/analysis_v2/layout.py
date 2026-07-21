"""Validate layout boxes from layout_raw/*.blocks.json and recompute layout features v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..feature_store import atomic_replace_parquet
from ..utils import atomic_write_json, ensure_dir
from .paths import analysis_v2_dir, transformers_dir
from .preprocess import apply_log1p, fit_clip_robust_scale

GRID_SIZE = 256
EPS_W = 1e-6
EPS_H = 1e-6
EPS_AREA = 1e-8
DEDUP_IOU = 0.95
DEDUP_CENTER = 0.02
DEDUP_SIZE = 0.02

TEXT_LABELS = {"text", "sub_title", "subtitle", "title", "header", "footer", "caption", "paragraph"}
FORMULA_LABELS = {"equation", "formula", "inline_formula", "math", "latex", "display_formula"}
FIGURE_LABELS = {"image", "figure", "chart", "diagram", "picture", "photo"}
TABLE_LABELS = {"table"}

LAYOUT_PCA_COLS = [
    "content_area_ratio",
    "mean_center_x",
    "mean_center_y",
    "center_spread_x",
    "center_spread_y",
    "margin_top",
    "margin_bottom",
    "margin_left",
    "margin_right",
    "text_area_ratio",
    "formula_area_ratio",
    "figure_area_ratio",
    "table_area_ratio",
    "other_area_ratio",
    "median_block_area",
    "iqr_block_area",
    "mean_log_block_aspect_ratio",
    "std_log_block_aspect_ratio",
    "median_log_block_aspect_ratio",
    "iqr_log_block_aspect_ratio",
    "max_block_area_ratio",
    "estimated_row_count",
    "estimated_col_count",
    "mean_horizontal_gap",
    "mean_vertical_gap",
    "left_align_score",
    "center_align_score",
    "two_column_score",
    "block_overlap_ratio",
    "reading_order_inversion_count_transformed",
    "reading_order_row_jump_count_transformed",
    "reading_order_mean_jump_distance",
    "block_count_log1p",
    "text_block_count_log1p",
    "formula_block_count_log1p",
    "figure_block_count_log1p",
    "table_block_count_log1p",
    "other_block_count_log1p",
    "valid_block_count_log1p",
] + [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)]

LOG1P_RAW = [
    "block_count",
    "text_block_count",
    "formula_block_count",
    "figure_block_count",
    "table_block_count",
    "other_block_count",
    "valid_block_count",
]


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def map_label(label: str) -> str:
    lab = (label or "").strip().lower()
    if lab in TEXT_LABELS or any(k in lab for k in ("title", "header", "footer", "caption", "text")):
        if lab in FORMULA_LABELS or any(k in lab for k in ("formula", "equation", "math", "latex")):
            return "formula"
        if lab in FIGURE_LABELS or any(k in lab for k in ("figure", "image", "chart", "diagram", "picture", "photo")):
            return "figure"
        if "table" in lab:
            return "table"
        return "text"
    if lab in FORMULA_LABELS or any(k in lab for k in ("formula", "equation", "math", "latex")):
        return "formula"
    if lab in FIGURE_LABELS or any(k in lab for k in ("figure", "image", "chart", "diagram", "picture", "photo")):
        return "figure"
    if lab in TABLE_LABELS or "table" in lab:
        return "table"
    return "other"


def box_iou(a: dict[str, float], b: dict[str, float]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(a["x2"] - a["x1"], 0.0) * max(a["y2"] - a["y1"], 0.0)
    area_b = max(b["x2"] - b["x1"], 0.0) * max(b["y2"] - b["y1"], 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def validate_box(raw: dict[str, Any], *, image_id: str, box_index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (valid_box, invalid_record). Exactly one is non-None."""
    try:
        x1 = float(raw.get("x1"))
        y1 = float(raw.get("y1"))
        x2 = float(raw.get("x2"))
        y2 = float(raw.get("y2"))
    except (TypeError, ValueError):
        return None, {
            "image_id": image_id,
            "box_index": box_index,
            "label": str(raw.get("label") or ""),
            "reason": "invalid_coordinate",
            "x1": raw.get("x1"),
            "y1": raw.get("y1"),
            "x2": raw.get("x2"),
            "y2": raw.get("y2"),
        }

    finite = bool(np.isfinite([x1, y1, x2, y2]).all())
    if not finite:
        return None, {
            "image_id": image_id,
            "box_index": box_index,
            "label": str(raw.get("label") or ""),
            "reason": "nonfinite",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    x1, x2 = sorted([_clip01(x1), _clip01(x2)])
    y1, y2 = sorted([_clip01(y1), _clip01(y2)])
    width = x2 - x1
    height = y2 - y1
    area = width * height

    reason = None
    if width <= EPS_W:
        reason = "zero_width"
    elif height <= EPS_H:
        reason = "zero_height"
    elif area <= EPS_AREA:
        reason = "zero_area"
    elif not (width > EPS_W and height > EPS_H and area > EPS_AREA):
        reason = "empty_after_clip"

    if reason:
        return None, {
            "image_id": image_id,
            "box_index": box_index,
            "label": str(raw.get("label") or ""),
            "reason": reason,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "area": area,
        }

    log_ar = float(np.clip(math.log(width / height), -5.0, 5.0))
    label = str(raw.get("label") or "")
    box = {
        "image_id": image_id,
        "box_index": box_index,
        "label": label,
        "label_bucket": map_label(label),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "w": width,
        "h": height,
        "area": area,
        "log_aspect_ratio": log_ar,
        # diagnostic only — not for PCA
        "aspect_ratio_diag": width / max(height, 1e-12),
    }
    return box, None


def deduplicate_boxes(boxes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop near-duplicates with same label, IoU>0.95, similar center/size. Keep emission order."""
    kept: list[dict[str, Any]] = []
    dup = 0
    for b in boxes:
        is_dup = False
        for k in kept:
            if k["label_bucket"] != b["label_bucket"] and k["label"] != b["label"]:
                # only dedup when label identical (string) per spec
                continue
            if k["label"] != b["label"]:
                continue
            if box_iou(k, b) < DEDUP_IOU:
                continue
            if abs(k["cx"] - b["cx"]) > DEDUP_CENTER or abs(k["cy"] - b["cy"]) > DEDUP_CENTER:
                continue
            if abs(k["w"] - b["w"]) > DEDUP_SIZE or abs(k["h"] - b["h"]) > DEDUP_SIZE:
                continue
            is_dup = True
            break
        if is_dup:
            dup += 1
        else:
            kept.append(b)
    return kept, dup


def raster_union(boxes: list[dict[str, Any]], grid: int = GRID_SIZE) -> np.ndarray:
    mask = np.zeros((grid, grid), dtype=np.uint8)
    for b in boxes:
        x1 = int(b["x1"] * grid)
        x2 = max(int(math.ceil(b["x2"] * grid)), x1 + 1)
        y1 = int(b["y1"] * grid)
        y2 = max(int(math.ceil(b["y2"] * grid)), y1 + 1)
        x1 = max(0, min(grid, x1))
        x2 = max(0, min(grid, x2))
        y1 = max(0, min(grid, y1))
        y2 = max(0, min(grid, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def union_area_ratio(boxes: list[dict[str, Any]], grid: int = GRID_SIZE) -> float:
    if not boxes:
        return 0.0
    return float(raster_union(boxes, grid).mean())


def _reading_order_metrics(blocks: list[dict[str, Any]]) -> dict[str, float]:
    n = len(blocks)
    if n < 2:
        return {
            "reading_order_inversion_count": 0.0,
            "reading_order_row_jump_count": 0.0,
            "reading_order_mean_jump_distance": 0.0,
        }
    geo = sorted(range(n), key=lambda i: (blocks[i]["cy"], blocks[i]["cx"]))
    rank = {i: r for r, i in enumerate(geo)}
    inversions = 0
    for a in range(n):
        for b in range(a + 1, n):
            if rank[a] > rank[b]:
                inversions += 1
    cys = np.array([blocks[i]["cy"] for i in range(n)])
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
    jump_dists: list[float] = []
    for a, b in zip(range(n - 1), range(1, n)):
        dy = abs(blocks[b]["cy"] - blocks[a]["cy"])
        dx = abs(blocks[b]["cx"] - blocks[a]["cx"])
        jump_dists.append(float(math.hypot(dx, dy)))
        if abs(int(row_id[b]) - int(row_id[a])) > 1:
            jumps += 1
    return {
        "reading_order_inversion_count": float(inversions),
        "reading_order_row_jump_count": float(jumps),
        "reading_order_mean_jump_distance": float(np.mean(jump_dists)) if jump_dists else 0.0,
    }


def boxes_to_features(image_id: str, valid: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    feat: dict[str, Any] = {
        "image_id": image_id,
        **stats,
    }
    if not valid:
        feat["layout_available"] = False
        feat["layout_missing_reason"] = "no_valid_blocks"
        for k in LAYOUT_PCA_COLS:
            if k not in feat:
                feat[k] = 0.0
        feat["mean_block_aspect_ratio"] = 0.0
        feat["std_block_aspect_ratio"] = 0.0
        feat["blank_ratio_diag"] = 1.0
        return feat

    feat["layout_available"] = True
    feat["layout_missing_reason"] = None

    areas = np.array([b["area"] for b in valid], dtype=np.float64)
    log_ars = np.array([b["log_aspect_ratio"] for b in valid], dtype=np.float64)
    aspects_diag = np.array([b["aspect_ratio_diag"] for b in valid], dtype=np.float64)
    cxs = np.array([b["cx"] for b in valid], dtype=np.float64)
    cys = np.array([b["cy"] for b in valid], dtype=np.float64)
    buckets = [b["label_bucket"] for b in valid]

    feat["block_count"] = len(valid)
    feat["valid_block_count"] = len(valid)
    for name in ("text", "formula", "figure", "table", "other"):
        feat[f"{name}_block_count"] = sum(1 for b in buckets if b == name)

    content = union_area_ratio(valid)
    feat["content_area_ratio"] = content
    feat["blank_ratio_diag"] = 1.0 - content

    for name in ("text", "formula", "figure", "table", "other"):
        subset = [b for b, buck in zip(valid, buckets) if buck == name]
        feat[f"{name}_area_ratio"] = union_area_ratio(subset)

    feat["mean_center_x"] = float(cxs.mean())
    feat["mean_center_y"] = float(cys.mean())
    feat["center_spread_x"] = float(cxs.std()) if len(cxs) > 1 else 0.0
    feat["center_spread_y"] = float(cys.std()) if len(cys) > 1 else 0.0

    x1s = np.array([b["x1"] for b in valid])
    y1s = np.array([b["y1"] for b in valid])
    x2s = np.array([b["x2"] for b in valid])
    y2s = np.array([b["y2"] for b in valid])
    feat["margin_left"] = float(x1s.min())
    feat["margin_right"] = float(1.0 - x2s.max())
    feat["margin_top"] = float(y1s.min())
    feat["margin_bottom"] = float(1.0 - y2s.max())

    q25, q50, q75 = np.percentile(areas, [25, 50, 75])
    feat["median_block_area"] = float(q50)
    feat["iqr_block_area"] = float(q75 - q25)
    feat["max_block_area_ratio"] = float(areas.max() / max(content, 1e-12)) if content > 0 else float(areas.max())

    lq25, lq50, lq75 = np.percentile(log_ars, [25, 50, 75])
    feat["mean_log_block_aspect_ratio"] = float(log_ars.mean())
    feat["std_log_block_aspect_ratio"] = float(log_ars.std()) if len(log_ars) > 1 else 0.0
    feat["median_log_block_aspect_ratio"] = float(lq50)
    feat["iqr_log_block_aspect_ratio"] = float(lq75 - lq25)
    # diagnostic only
    feat["mean_block_aspect_ratio"] = float(aspects_diag.mean())
    feat["std_block_aspect_ratio"] = float(aspects_diag.std()) if len(aspects_diag) > 1 else 0.0

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

    h_gaps, v_gaps = [], []
    for i, bi in enumerate(valid):
        hx = [
            bj["x1"] - bi["x2"]
            for j, bj in enumerate(valid)
            if j != i and abs(bj["cy"] - bi["cy"]) < 0.05 and bj["x1"] >= bi["x2"]
        ]
        vy = [
            bj["y1"] - bi["y2"]
            for j, bj in enumerate(valid)
            if j != i and abs(bj["cx"] - bi["cx"]) < 0.08 and bj["y1"] >= bi["y2"]
        ]
        if hx:
            h_gaps.append(min(hx))
        if vy:
            v_gaps.append(min(vy))
    feat["mean_horizontal_gap"] = float(np.mean(h_gaps)) if h_gaps else 0.0
    feat["mean_vertical_gap"] = float(np.mean(v_gaps)) if v_gaps else 0.0
    feat["left_align_score"] = float(np.mean(np.abs(x1s - np.median(x1s)) < 0.03))
    feat["center_align_score"] = float(np.mean(np.abs(cxs - 0.5) < 0.08))
    left = cxs < 0.45
    right = cxs > 0.55
    feat["two_column_score"] = float(
        (left.mean() > 0.25) and (right.mean() > 0.25) and (np.abs(cxs[left].mean() - cxs[right].mean()) > 0.25)
    )

    overlap = 0.0
    total_area = max(float(areas.sum()), 1e-12)
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            ix1 = max(a["x1"], b["x1"])
            iy1 = max(a["y1"], b["y1"])
            ix2 = min(a["x2"], b["x2"])
            iy2 = min(a["y2"], b["y2"])
            overlap += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    feat["block_overlap_ratio"] = float(overlap / total_area)
    feat.update(_reading_order_metrics(valid))

    occ = np.zeros((4, 4), dtype=np.float64)
    for b in valid:
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

    return feat


def process_image_blocks(image_id: str, raw_blocks: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict[str, Any]]:
    valid_raw: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    counts = {
        "invalid_zero_width_count": 0,
        "invalid_zero_height_count": 0,
        "invalid_coordinate_count": 0,
        "invalid_nonfinite_count": 0,
    }
    for i, raw in enumerate(raw_blocks or []):
        box, inv = validate_box(raw, image_id=image_id, box_index=i)
        if inv is not None:
            invalid.append(inv)
            r = inv["reason"]
            if r == "zero_width":
                counts["invalid_zero_width_count"] += 1
            elif r == "zero_height":
                counts["invalid_zero_height_count"] += 1
            elif r == "nonfinite":
                counts["invalid_nonfinite_count"] += 1
            else:
                counts["invalid_coordinate_count"] += 1
            continue
        assert box is not None
        valid_raw.append(box)

    valid, dup_n = deduplicate_boxes(valid_raw)
    stats = {
        "raw_block_count": len(raw_blocks or []),
        "valid_block_count": len(valid),
        "invalid_block_count": len(invalid),
        "duplicate_block_count": dup_n,
        **counts,
    }
    feat = boxes_to_features(image_id, valid, stats)
    for b in valid:
        b["kept_after_dedup"] = True
    return valid, invalid, feat


def run_layout_v2(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    layout_raw = out_dir / "layout_raw"
    v2 = analysis_v2_dir(cfg)
    tf = transformers_dir(cfg)

    paths = sorted(layout_raw.glob("*.blocks.json"))
    if not paths:
        raise FileNotFoundError(f"No blocks.json under {layout_raw}")

    all_valid: list[dict[str, Any]] = []
    all_invalid: list[dict[str, Any]] = []
    feats: list[dict[str, Any]] = []

    for p in paths:
        image_id = p.stem.replace(".blocks", "") if p.name.endswith(".blocks.json") else p.stem
        # stem of foo.blocks.json is foo.blocks
        if image_id.endswith(".blocks"):
            image_id = image_id[: -len(".blocks")]
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raw = []
        valid, invalid, feat = process_image_blocks(image_id, raw)
        all_valid.extend(valid)
        all_invalid.extend(invalid)
        feats.append(feat)

    valid_df = pd.DataFrame(all_valid) if all_valid else pd.DataFrame()
    invalid_df = pd.DataFrame(all_invalid) if all_invalid else pd.DataFrame(
        columns=["image_id", "box_index", "label", "reason", "x1", "y1", "x2", "y2"]
    )
    feat_df = pd.DataFrame(feats)

    # log1p transforms for analysis
    feat_df = apply_log1p(feat_df, LOG1P_RAW, suffix="_log1p")
    # reading-order counts use _transformed name expected by LAYOUT_PCA_COLS
    if "reading_order_inversion_count" in feat_df.columns:
        feat_df["reading_order_inversion_count_transformed"] = np.log1p(
            pd.to_numeric(feat_df["reading_order_inversion_count"], errors="coerce").fillna(0).clip(lower=0)
        )
    if "reading_order_row_jump_count" in feat_df.columns:
        feat_df["reading_order_row_jump_count_transformed"] = np.log1p(
            pd.to_numeric(feat_df["reading_order_row_jump_count"], errors="coerce").fillna(0).clip(lower=0)
        )

    atomic_replace_parquet(valid_df, v2 / "layout_boxes_validated.parquet")
    atomic_replace_parquet(invalid_df, v2 / "layout_invalid_boxes.parquet")
    atomic_replace_parquet(feat_df, v2 / "layout_features_v2.parquet")

    avail = feat_df[feat_df["layout_available"].astype(bool)].copy() if "layout_available" in feat_df.columns else feat_df
    Xs, cols, meta = fit_clip_robust_scale(
        avail,
        LAYOUT_PCA_COLS,
        out_joblib=tf / "layout_scaler.joblib",
    )
    # also keep a copy at analysis_v2 root for convenience
    import shutil

    shutil.copy2(tf / "layout_scaler.joblib", v2 / "layout_scaler.joblib")
    np.save(v2 / "layout_X_scaled.npy", Xs)
    # index aligned with scaled matrix
    idx = avail[["image_id"]].reset_index(drop=True)
    atomic_replace_parquet(idx, v2 / "layout_index_aligned.parquet")
    atomic_write_json(v2 / "layout_preprocess_meta.json", {"scaler_columns": cols, **{k: meta[k] for k in meta if k != "drop_meta"}, "drop_meta": meta["drop_meta"]})
    print(f"[layout_v2] images={len(feat_df)} available={int(feat_df['layout_available'].sum())} invalid_boxes={len(invalid_df)}")
    return feat_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run_layout_v2(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
