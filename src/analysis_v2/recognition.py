"""Recognition content features v2: clean_text (no grounding) + continuous metrics only."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..feature_store import atomic_replace_parquet
from ..utils import atomic_write_json
from .paths import analysis_v2_dir, transformers_dir
from .preprocess import apply_log1p, fit_clip_robust_scale

GROUNDING_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.DOTALL)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MATH_OPS = set("+-*/=<>≤≥≠≈±×÷∑∏∫∂∇^_")
BRACKETS = set("()[]{}（）【】")
PUNCT = set(".,;:!?，。；：！？、·…\"'`")

LOG1P_COLS = [
    "output_character_count",
    "line_count",
    "mean_line_length",
    "math_structure_per_line",
    "equals_count",
    "frac_count",
    "sqrt_count",
    "subscript_count",
    "superscript_count",
    "vector_symbol_count",
    "angle_symbol_count",
    "latex_env_count",
    "formula_span_count",
]

# Columns that enter PCA (transformed counts + ratios). No quality, no morphology.
RECOG_PCA_COLS = [
    "chinese_ratio",
    "latin_ratio",
    "digit_ratio",
    "math_operator_ratio",
    "bracket_ratio",
    "punctuation_ratio",
    "whitespace_ratio",
    "formula_character_ratio",
    "formula_line_ratio",
    "non_empty_line_ratio",
    "output_character_count_transformed",
    "line_count_transformed",
    "mean_line_length_transformed",
    "math_structure_per_line_transformed",
    "equals_count_transformed",
    "frac_count_transformed",
    "sqrt_count_transformed",
    "subscript_count_transformed",
    "superscript_count_transformed",
    "vector_symbol_count_transformed",
    "angle_symbol_count_transformed",
    "latex_env_count_transformed",
    "formula_span_count_transformed",
]


def clean_ocr_text(raw: str) -> str:
    """Remove grounding tags/coords; keep body + LaTeX; NFC; normalize newlines; strip controls."""
    text = raw or ""
    text = GROUNDING_RE.sub("", text)
    # stray tags
    text = re.sub(r"<\|/?ref\|>", "", text)
    text = re.sub(r"<\|/?det\|>", "", text)
    text = re.sub(r"\[\[[^\]]*\]\]", "", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    # collapse >2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _visible_nonspace(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _char_ratios(text: str) -> dict[str, float]:
    n_all = max(len(text), 1)
    vis = _visible_nonspace(text)
    n_vis = max(len(vis), 1)
    return {
        "chinese_ratio": sum("\u4e00" <= ch <= "\u9fff" for ch in vis) / n_vis,
        "latin_ratio": sum("a" <= ch.lower() <= "z" for ch in vis) / n_vis,
        "digit_ratio": sum(ch.isdigit() for ch in vis) / n_vis,
        "math_operator_ratio": sum(ch in MATH_OPS for ch in vis) / n_vis,
        "bracket_ratio": sum(ch in BRACKETS for ch in vis) / n_vis,
        "punctuation_ratio": sum(ch in PUNCT for ch in vis) / n_vis,
        "whitespace_ratio": sum(ch.isspace() for ch in text) / n_all,
    }


def _formula_spans(text: str) -> tuple[int, float]:
    patterns = [
        r"\$\$.*?\$\$",
        r"\$[^$]+\$",
        r"\\\(.*?\\\)",
        r"\\\[.*?\\\]",
    ]
    spans: list[list[int]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.DOTALL):
            spans.append([m.start(), m.end()])
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    formula_chars = sum(e - s for s, e in merged)
    vis_n = max(len(_visible_nonspace(text)), 1)
    return len(merged), formula_chars / vis_n


def _math_structure_counts(text: str) -> dict[str, int]:
    return {
        "equals_count": text.count("=") + text.count("＝"),
        "frac_count": len(re.findall(r"\\frac\b|／|/", text)),
        "sqrt_count": len(re.findall(r"\\sqrt\b|√", text)),
        "subscript_count": len(re.findall(r"_[\{a-zA-Z0-9]|_", text)),
        "superscript_count": len(re.findall(r"\^[\{a-zA-Z0-9]|\^", text)),
        "vector_symbol_count": len(re.findall(r"\\vec\b|\\mathbf\b|→|⃗", text)),
        "angle_symbol_count": len(re.findall(r"\\angle\b|∠|°", text)),
        "latex_env_count": len(re.findall(r"\\begin\{[^}]+\}", text)),
    }


def _is_formula_line(ln: str) -> bool:
    s = ln.strip()
    if not s:
        return False
    if s.startswith("$$") or s.startswith("\\[") or s.startswith("\\("):
        return True
    if "$" in s and any(c in s for c in "=+-*/\\^_{}"):
        return True
    return False


def extract_recognition_v2_with_text(image_id: str, raw_text: str) -> tuple[dict[str, Any], str, str]:
    raw_text = raw_text or ""
    clean = clean_ocr_text(raw_text)
    lines = clean.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    n_lines = max(len(lines), 1)
    formula_span_count, formula_character_ratio = _formula_spans(clean)
    formula_lines = sum(1 for ln in lines if _is_formula_line(ln))
    structs = _math_structure_counts(clean)
    struct_total = sum(structs.values())
    math_per_line = struct_total / max(len(non_empty), 1)
    ratios = _char_ratios(clean)
    feat: dict[str, Any] = {
        "image_id": image_id,
        "output_character_count": len(clean),
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
        "non_empty_line_ratio": len(non_empty) / n_lines,
        "mean_line_length": float(np.mean([len(ln) for ln in non_empty])) if non_empty else 0.0,
        "formula_line_ratio": float(formula_lines / n_lines),
        **ratios,
        **structs,
        "math_structure_per_line": float(math_per_line),
        "formula_span_count": int(formula_span_count),
        "formula_character_ratio": float(formula_character_ratio),
        "raw_text_len": len(raw_text),
        "clean_text_len": len(clean),
    }
    return feat, raw_text, clean


def extract_recognition_v2(image_id: str, raw_text: str) -> dict[str, Any]:
    feat, _, _ = extract_recognition_v2_with_text(image_id, raw_text)
    return feat


def run_recognition_v2(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    raw_dir = out_dir / "recognition_raw"
    ocr_index = out_dir / "ocr_generations.parquet"
    v2 = analysis_v2_dir(cfg)
    tf = transformers_dir(cfg)

    if not ocr_index.exists():
        raise FileNotFoundError(f"Missing {ocr_index}")

    q_path = v2 / "ocr_quality_v2.parquet"
    if not q_path.exists():
        raise FileNotFoundError(f"Missing {q_path}; run ocr_quality_v2 first")
    qdf = pd.read_parquet(q_path)
    strict_ids = set(qdf.loc[qdf["ocr_usable_strict"].astype(bool), "image_id"].astype(str))

    ocr_df = pd.read_parquet(ocr_index)
    rows = []
    text_rows = []
    for _, r in ocr_df.iterrows():
        if str(r.get("status")) != "ok":
            continue
        image_id = str(r["image_id"])
        text_path = raw_dir / f"{image_id}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(r.get("text") or "")
        feat, raw_t, clean_t = extract_recognition_v2_with_text(image_id, text)
        rows.append(feat)
        text_rows.append({"image_id": image_id, "raw_text": raw_t, "clean_text": clean_t})

    df = pd.DataFrame(rows)
    df = apply_log1p(df, LOG1P_COLS, suffix="_transformed")
    atomic_replace_parquet(df, v2 / "recognition_features_v2.parquet")
    atomic_replace_parquet(pd.DataFrame(text_rows), v2 / "recognition_text_v2.parquet")

    # Scale only strict-usable subset for PCA matrix
    use = df[df["image_id"].astype(str).isin(strict_ids)].copy()
    Xs, cols, meta = fit_clip_robust_scale(
        use,
        RECOG_PCA_COLS,
        out_joblib=tf / "recognition_scaler.joblib",
    )
    import shutil

    shutil.copy2(tf / "recognition_scaler.joblib", v2 / "recognition_scaler.joblib")
    np.save(v2 / "recognition_X_scaled.npy", Xs)
    atomic_replace_parquet(use[["image_id"]].reset_index(drop=True), v2 / "recognition_index_aligned.parquet")
    atomic_write_json(
        v2 / "recognition_preprocess_meta.json",
        {
            "scaler_columns": cols,
            "n_strict": int(len(use)),
            "n_all": int(len(df)),
            "max_abs_scaled": meta["max_abs_scaled"],
            "drop_meta": meta["drop_meta"],
        },
    )
    print(f"[recognition_v2] n={len(df)} strict_scaled={len(use)} cols={len(cols)}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run_recognition_v2(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
