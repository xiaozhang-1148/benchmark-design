"""Parse recognition / OCR text into explicit statistical features."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .feature_store import atomic_replace_parquet
from .utils import ensure_dir

RECOG_COLUMNS = [
    "image_id",
    "output_token_count",
    "output_character_count",
    "line_count",
    "markdown_heading_count",
    "formula_count",
    "formula_character_ratio",
    "digit_ratio",
    "latin_ratio",
    "chinese_ratio",
    "math_symbol_ratio",
    "whitespace_ratio",
    "repetition_ratio",
    "mean_generated_token_logprob",
    "logprob_available",
]

MATH_SYMBOLS = set("∑∏∫√∞≈≠≤≥±×÷∂∇∈∉⊂⊃∪∩∧∨⇒⇔∀∃°′″^_{}[]\\|<>±µπθαβγΔΩ")


def _char_class_ratios(text: str) -> dict[str, float]:
    if not text:
        return {
            "digit_ratio": 0.0,
            "latin_ratio": 0.0,
            "chinese_ratio": 0.0,
            "math_symbol_ratio": 0.0,
            "whitespace_ratio": 0.0,
        }
    n = len(text)
    digit = sum(ch.isdigit() for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    math = sum((ch in MATH_SYMBOLS) or (ch in "+-*/=<>^_") for ch in text)
    ws = sum(ch.isspace() for ch in text)
    return {
        "digit_ratio": digit / n,
        "latin_ratio": latin / n,
        "chinese_ratio": chinese / n,
        "math_symbol_ratio": math / n,
        "whitespace_ratio": ws / n,
    }


def _repetition_ratio(text: str, ngram: int = 8) -> float:
    """Fraction of characters covered by the most frequent repeated n-gram window."""
    if len(text) < ngram * 2:
        return 0.0
    grams = [text[i : i + ngram] for i in range(0, len(text) - ngram + 1, ngram)]
    if not grams:
        return 0.0
    c = Counter(grams)
    top, cnt = c.most_common(1)[0]
    if cnt <= 1:
        return 0.0
    return min(1.0, (cnt * len(top)) / max(len(text), 1))


def _formula_stats(text: str) -> tuple[int, float]:
    patterns = [
        r"\$\$.*?\$\$",
        r"\$[^$]+\$",
        r"\\\(.*?\\\)",
        r"\\\[.*?\\\]",
        r"<\|ref\|>formula<\|/ref\|>",
    ]
    spans = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.DOTALL):
            spans.append((m.start(), m.end()))
    # merge overlaps
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    formula_chars = sum(e - s for s, e in merged)
    return len(merged), (formula_chars / max(len(text), 1))


def extract_recognition_features(
    image_id: str,
    text: str,
    output_token_count: int | None = None,
    mean_logprob: float | None = None,
) -> dict[str, Any]:
    text = text or ""
    lines = text.splitlines()
    formula_count, formula_ratio = _formula_stats(text)
    ratios = _char_class_ratios(text)
    feat = {
        "image_id": image_id,
        "output_token_count": int(output_token_count) if output_token_count is not None else len(text.split()),
        "output_character_count": len(text),
        "line_count": len(lines),
        "markdown_heading_count": sum(1 for ln in lines if ln.lstrip().startswith("#")),
        "formula_count": formula_count,
        "formula_character_ratio": formula_ratio,
        **ratios,
        "repetition_ratio": _repetition_ratio(text),
        "mean_generated_token_logprob": mean_logprob,
        "logprob_available": mean_logprob is not None and not (isinstance(mean_logprob, float) and math.isnan(mean_logprob)),
    }
    # log1p versions for count-like fields
    for key in ["output_token_count", "output_character_count", "line_count", "markdown_heading_count", "formula_count"]:
        feat[f"{key}_log1p"] = float(np.log1p(feat[key]))
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
        mean_lp = r.get("mean_generated_token_logprob")
        if mean_lp is not None and (isinstance(mean_lp, float) and (math.isnan(mean_lp))):
            mean_lp = None
        if pd.isna(mean_lp):
            mean_lp = None
        tok_count = r.get("output_token_count")
        if tok_count is not None and pd.isna(tok_count):
            tok_count = None
        rows.append(
            extract_recognition_features(
                image_id,
                text,
                output_token_count=int(tok_count) if tok_count is not None else None,
                mean_logprob=float(mean_lp) if mean_lp is not None else None,
            )
        )
    df = pd.DataFrame(rows)
    atomic_replace_parquet(df, out_dir / "recognition_features.parquet")
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
