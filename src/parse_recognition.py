"""Parse OCR text into explicit *content* features (no quality fields)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .feature_store import atomic_replace_parquet
from .utils import atomic_write_json, ensure_dir

# Content-only schema (quality lives in ocr_quality.parquet)
RECOG_COLUMNS = [
    "image_id",
    "output_character_count",
    "line_count",
    "non_empty_line_count",
    "non_empty_line_ratio",
    "mean_line_length",
    "formula_line_ratio",
    "chinese_ratio",
    "latin_ratio",
    "digit_ratio",
    "math_operator_ratio",
    "bracket_ratio",
    "punctuation_ratio",
    "whitespace_ratio",
    "equals_count",
    "frac_count",
    "sqrt_count",
    "subscript_count",
    "superscript_count",
    "vector_symbol_count",
    "angle_symbol_count",
    "latex_env_count",
    "math_structure_per_line",
    "formula_span_count",
    "formula_character_ratio",
    "content_morphology",
]

MATH_OPS = set("+-*/=<>≤≥≠≈±×÷∑∏∫∂∇^_")
BRACKETS = set("()[]{}（）【】")
PUNCT = set(".,;:!?，。；：！？、·…\"'`")


MORPHOLOGY_RULES = {
    "plain_text": "formula_character_ratio < 0.05 and formula_line_ratio < 0.1 and figure-like markers absent",
    "text_formula_mix": "0.05 <= formula_character_ratio < 0.25 or 0.1 <= formula_line_ratio < 0.4",
    "formula_heavy": "formula_character_ratio >= 0.25 or formula_line_ratio >= 0.4 or math_structure_per_line >= 1.5",
    "figure_caption": "contains figure/image grounding labels or low char density with figure markers",
}


def _char_ratios(text: str) -> dict[str, float]:
    if not text:
        return {
            "chinese_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "math_operator_ratio": 0.0,
            "bracket_ratio": 0.0,
            "punctuation_ratio": 0.0,
            "whitespace_ratio": 0.0,
        }
    n = len(text)
    return {
        "chinese_ratio": sum("\u4e00" <= ch <= "\u9fff" for ch in text) / n,
        "latin_ratio": sum("a" <= ch.lower() <= "z" for ch in text) / n,
        "digit_ratio": sum(ch.isdigit() for ch in text) / n,
        "math_operator_ratio": sum(ch in MATH_OPS for ch in text) / n,
        "bracket_ratio": sum(ch in BRACKETS for ch in text) / n,
        "punctuation_ratio": sum(ch in PUNCT for ch in text) / n,
        "whitespace_ratio": sum(ch.isspace() for ch in text) / n,
    }


def _formula_spans(text: str) -> tuple[int, float]:
    patterns = [
        r"\$\$.*?\$\$",
        r"\$[^$]+\$",
        r"\\\(.*?\\\)",
        r"\\\[.*?\\\]",
        r"<\|ref\|>formula<\|/ref\|>",
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
    return len(merged), formula_chars / max(len(text), 1)


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
    if "<|ref|>formula" in s.lower():
        return True
    return False


def _content_morphology(
    *,
    text: str,
    formula_character_ratio: float,
    formula_line_ratio: float,
    math_structure_per_line: float,
) -> str:
    low = text.lower()
    figure_like = any(
        k in low
        for k in (
            "<|ref|>figure",
            "<|ref|>image",
            "<|ref|>picture",
            "![",
        )
    )
    if figure_like and formula_character_ratio < 0.15 and len(text) < 800:
        return "figure_caption"
    if formula_character_ratio >= 0.25 or formula_line_ratio >= 0.4 or math_structure_per_line >= 1.5:
        return "formula_heavy"
    if formula_character_ratio >= 0.05 or formula_line_ratio >= 0.1:
        return "text_formula_mix"
    return "plain_text"


def extract_recognition_features(image_id: str, text: str) -> dict[str, Any]:
    text = text or ""
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    n_lines = max(len(lines), 1)
    formula_span_count, formula_character_ratio = _formula_spans(text)
    formula_lines = sum(1 for ln in lines if _is_formula_line(ln))
    formula_line_ratio = formula_lines / n_lines
    structs = _math_structure_counts(text)
    struct_total = sum(structs.values())
    math_per_line = struct_total / max(len(non_empty), 1)
    ratios = _char_ratios(text)
    morphology = _content_morphology(
        text=text,
        formula_character_ratio=formula_character_ratio,
        formula_line_ratio=formula_line_ratio,
        math_structure_per_line=math_per_line,
    )
    feat: dict[str, Any] = {
        "image_id": image_id,
        "output_character_count": len(text),
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
        "non_empty_line_ratio": len(non_empty) / n_lines,
        "mean_line_length": float(np.mean([len(ln) for ln in non_empty])) if non_empty else 0.0,
        "formula_line_ratio": float(formula_line_ratio),
        **ratios,
        **structs,
        "math_structure_per_line": float(math_per_line),
        "formula_span_count": int(formula_span_count),
        "formula_character_ratio": float(formula_character_ratio),
        "content_morphology": morphology,
    }
    return feat


def run_parse_recognition(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    raw_dir = out_dir / "recognition_raw"
    ensure_dir(raw_dir)
    ocr_index = out_dir / "ocr_generations.parquet"
    if not ocr_index.exists():
        raise FileNotFoundError(f"Missing {ocr_index}")
    ocr_df = pd.read_parquet(ocr_index)
    rows = []
    for _, r in ocr_df.iterrows():
        if str(r.get("status")) != "ok":
            continue
        image_id = str(r["image_id"])
        text_path = raw_dir / f"{image_id}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(r.get("text") or "")
        rows.append(extract_recognition_features(image_id, text))
    df = pd.DataFrame(rows)
    for c in RECOG_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[RECOG_COLUMNS]
    atomic_replace_parquet(df, out_dir / "recognition_features.parquet")
    atomic_write_json(out_dir / "recognition_morphology_rules.json", MORPHOLOGY_RULES)
    reports = Path(cfg["paths"]["reports_dir"])
    ensure_dir(reports)
    atomic_write_json(reports / "recognition_morphology_rules.json", MORPHOLOGY_RULES)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    df = run_parse_recognition(cfg)
    print(f"[parse_recognition] n={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
