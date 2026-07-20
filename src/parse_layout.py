"""Parse grounding / markdown layout into fixed-schema explicit features."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .feature_store import atomic_replace_parquet, merge_status_parquet
from .utils import atomic_write_json, atomic_write_text, ensure_dir

REF_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)

LAYOUT_COLUMNS = [
    "image_id",
    "layout_available",
    "layout_missing_reason",
    "block_count",
    "text_block_count",
    "formula_block_count",
    "figure_block_count",
    "table_block_count",
    "blank_ratio",
    "content_area_ratio",
    "mean_block_area",
    "std_block_area",
    "mean_block_aspect_ratio",
    "std_block_aspect_ratio",
    "mean_center_x",
    "mean_center_y",
    "upper_region_density",
    "middle_region_density",
    "lower_region_density",
    "left_region_density",
    "center_region_density",
    "right_region_density",
    "reading_order_length",
    "reading_order_vertical_violation_count",
] + [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)] + [
    "block_count_log1p",
    "text_block_count_log1p",
    "formula_block_count_log1p",
    "figure_block_count_log1p",
    "table_block_count_log1p",
    "reading_order_length_log1p",
    "reading_order_vertical_violation_count_log1p",
]


def parse_grounding_blocks(text: str) -> list[dict[str, Any]]:
    blocks = []
    for m in REF_PATTERN.finditer(text or ""):
        label = (m.group(1) or "").strip().lower()
        det_raw = (m.group(2) or "").strip()
        try:
            coords = eval(det_raw, {"__builtins__": {}})  # noqa: S307 — model output boxes
        except Exception:
            continue
        if not isinstance(coords, list):
            continue
        for box in coords:
            if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                continue
            # Official DeepSeek coords are in 0..999
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


def blocks_to_features(image_id: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    feat: dict[str, Any] = {c: None for c in LAYOUT_COLUMNS}
    feat["image_id"] = image_id
    if not blocks:
        feat["layout_available"] = False
        feat["layout_missing_reason"] = "no_parseable_grounding_coordinates"
        # Still fill zeros for schema stability
        for c in LAYOUT_COLUMNS:
            if c.endswith("_count") or c.endswith("_ratio") or c.startswith("occupancy_") or c.endswith("_density") or c.startswith("mean_") or c.startswith("std_") or c.endswith("_log1p") or c in {
                "blank_ratio",
                "content_area_ratio",
                "reading_order_length",
                "reading_order_vertical_violation_count",
            }:
                if feat[c] is None:
                    feat[c] = 0.0 if ("ratio" in c or "density" in c or c.startswith("occupancy_") or c.startswith("mean_") or c.startswith("std_")) else 0
        feat["blank_ratio"] = 1.0
        feat["content_area_ratio"] = 0.0
        return feat

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

    # Union area approx via occupancy grid (more stable than naive sum)
    grid = np.zeros((64, 64), dtype=np.float64)
    for b in blocks:
        x1 = int(b["x1"] * 64)
        x2 = max(int(math.ceil(b["x2"] * 64)), x1 + 1)
        y1 = int(b["y1"] * 64)
        y2 = max(int(math.ceil(b["y2"] * 64)), y1 + 1)
        grid[y1:y2, x1:x2] = 1.0
    content = float(grid.mean())
    feat["content_area_ratio"] = content
    feat["blank_ratio"] = 1.0 - content

    feat["mean_block_area"] = float(areas.mean())
    feat["std_block_area"] = float(areas.std()) if len(areas) > 1 else 0.0
    feat["mean_block_aspect_ratio"] = float(aspects.mean())
    feat["std_block_aspect_ratio"] = float(aspects.std()) if len(aspects) > 1 else 0.0
    feat["mean_center_x"] = float(cxs.mean())
    feat["mean_center_y"] = float(cys.mean())

    # Region densities by block centers
    feat["upper_region_density"] = float((cys < 1 / 3).mean())
    feat["middle_region_density"] = float(((cys >= 1 / 3) & (cys < 2 / 3)).mean())
    feat["lower_region_density"] = float((cys >= 2 / 3).mean())
    feat["left_region_density"] = float((cxs < 1 / 3).mean())
    feat["center_region_density"] = float(((cxs >= 1 / 3) & (cxs < 2 / 3)).mean())
    feat["right_region_density"] = float((cxs >= 2 / 3).mean())

    # Reading order: sort by cy then cx; count vertical violations (later block above earlier)
    order = sorted(range(len(blocks)), key=lambda i: (blocks[i]["cy"], blocks[i]["cx"]))
    feat["reading_order_length"] = len(order)
    violations = 0
    for a, b in zip(order, order[1:]):
        if blocks[b]["cy"] + 1e-8 < blocks[a]["cy"]:
            violations += 1
    feat["reading_order_vertical_violation_count"] = violations

    # 4x4 occupancy
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
        "reading_order_length",
        "reading_order_vertical_violation_count",
    ]:
        feat[f"{key}_log1p"] = float(np.log1p(feat[key]))

    return feat


def markdown_structure_fallback(text: str) -> dict[str, Any]:
    """When no coords: derive limited structure stats (not forged coordinates)."""
    text = text or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return {
        "markdown_line_count": len(lines),
        "markdown_heading_count": sum(1 for ln in lines if ln.lstrip().startswith("#")),
        "markdown_table_row_count": sum(1 for ln in lines if "|" in ln),
        "has_formula_markers": int(("\\(" in text) or ("$$" in text) or ("\\[" in text)),
    }


def parse_one(image_id: str, raw_text: str, raw_dir: Path) -> dict[str, Any]:
    ensure_dir(raw_dir)
    atomic_write_text(raw_dir / f"{image_id}.md", raw_text or "")
    blocks = parse_grounding_blocks(raw_text or "")
    atomic_write_json(raw_dir / f"{image_id}.blocks.json", blocks)
    feat = blocks_to_features(image_id, blocks)
    feat.update({f"fallback_{k}": v for k, v in markdown_structure_fallback(raw_text or "").items()})
    return feat


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
