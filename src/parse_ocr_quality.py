"""OCR quality channel — filter/diagnostics only; never enters recognition PCA."""

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
from .utils import atomic_write_json, ensure_dir

QUALITY_COLUMNS = [
    "image_id",
    "ocr_quality_status",
    "hit_max_tokens",
    "output_token_count",
    "repetition_ratio",
    "max_ngram_repeat_count",
    "empty",
    "invalid_char_rate",
    "extreme_line_count",
    "extreme_formula_count",
    "parse_failed",
    "mean_generated_token_logprob",
    "logprob_available",
    "suspicious_reasons",
]


def _repetition_ratio(text: str, ngram: int = 8) -> float:
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


def _max_ngram_repeat(text: str, ngram: int = 12) -> int:
    if len(text) < ngram * 3:
        return 0
    grams = [text[i : i + ngram] for i in range(0, len(text) - ngram + 1, max(1, ngram // 2))]
    if not grams:
        return 0
    return int(Counter(grams).most_common(1)[0][1])


def _invalid_char_rate(text: str) -> float:
    if not text:
        return 0.0
    # replacement char / null / most C0 controls (except tab/newline)
    bad = 0
    for ch in text:
        o = ord(ch)
        if ch == "\ufffd" or o == 0 or (o < 32 and ch not in "\t\n\r"):
            bad += 1
    return bad / len(text)


def _formula_span_count(text: str) -> int:
    pats = [r"\$\$.*?\$\$", r"\$[^$]+\$", r"\\\(.*?\\\)", r"\\\[.*?\\\]"]
    n = 0
    for pat in pats:
        n += len(re.findall(pat, text, flags=re.DOTALL))
    return n


def classify_status(
    *,
    empty: bool,
    hit_max_tokens: bool,
    repetition_ratio: float,
    max_ngram_repeat: int,
    parse_failed: bool,
    invalid_char_rate: float,
    extreme_line_count: bool,
    extreme_formula_count: bool,
    qcfg: dict[str, Any],
) -> tuple[str, list[str]]:
    """Priority: empty > truncated > repetitive > parse_failed > suspicious > valid."""
    rep_thr = float(qcfg.get("repetition_ratio_threshold", 0.35))
    ngram_thr = int(qcfg.get("catastrophic_ngram_repeat", 20))
    inv_thr = float(qcfg.get("invalid_char_rate_threshold", 0.05))
    reasons: list[str] = []

    if empty:
        return "empty", ["empty"]
    if hit_max_tokens:
        return "truncated", ["hit_max_tokens"]
    if repetition_ratio >= rep_thr or max_ngram_repeat >= ngram_thr:
        return "repetitive", ["high_repetition"]
    if parse_failed:
        return "parse_failed", ["layout_parse_failed"]

    if invalid_char_rate >= inv_thr:
        reasons.append("invalid_chars")
    if extreme_line_count:
        reasons.append("extreme_line_count")
    if extreme_formula_count:
        reasons.append("extreme_formula_count")
    if reasons:
        return "suspicious", reasons
    return "valid", []


def extract_quality_row(
    image_id: str,
    text: str,
    *,
    output_token_count: int | None,
    mean_logprob: float | None,
    max_tokens: int,
    parse_failed: bool,
    qcfg: dict[str, Any],
) -> dict[str, Any]:
    text = text or ""
    lines = text.splitlines()
    n_lines = len(lines)
    formula_n = _formula_span_count(text)
    tok = int(output_token_count) if output_token_count is not None else 0
    empty = len(text.strip()) == 0
    hit_max = (not empty) and tok >= int(max_tokens)
    rep = _repetition_ratio(text)
    ngram_rep = _max_ngram_repeat(text)
    inv = _invalid_char_rate(text)
    extreme_lines = n_lines >= int(qcfg.get("extreme_line_count", 400))
    extreme_formulas = formula_n >= int(qcfg.get("extreme_formula_count", 80))

    status, reasons = classify_status(
        empty=empty,
        hit_max_tokens=hit_max,
        repetition_ratio=rep,
        max_ngram_repeat=ngram_rep,
        parse_failed=parse_failed,
        invalid_char_rate=inv,
        extreme_line_count=extreme_lines,
        extreme_formula_count=extreme_formulas,
        qcfg=qcfg,
    )
    lp_ok = mean_logprob is not None and not (
        isinstance(mean_logprob, float) and (math.isnan(mean_logprob) or math.isinf(mean_logprob))
    )
    return {
        "image_id": image_id,
        "ocr_quality_status": status,
        "hit_max_tokens": bool(hit_max),
        "output_token_count": tok,
        "repetition_ratio": float(rep),
        "max_ngram_repeat_count": int(ngram_rep),
        "empty": bool(empty),
        "invalid_char_rate": float(inv),
        "extreme_line_count": bool(extreme_lines),
        "extreme_formula_count": bool(extreme_formulas),
        "parse_failed": bool(parse_failed),
        "mean_generated_token_logprob": float(mean_logprob) if lp_ok else None,
        "logprob_available": bool(lp_ok),
        "suspicious_reasons": "|".join(reasons),
    }


def run_parse_ocr_quality(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    raw_dir = out_dir / "recognition_raw"
    ocr_index = out_dir / "ocr_generations.parquet"
    if not ocr_index.exists():
        raise FileNotFoundError(f"Missing {ocr_index}")

    qcfg = dict(cfg.get("ocr_quality") or {})
    max_tokens = int(cfg.get("vllm", {}).get("max_tokens", 4096))

    # parse_failed: grounding markers present but layout unavailable (not mere markdown OCR)
    parse_failed_ids: set[str] = set()
    lay_path = out_dir / "layout_features.parquet"
    layout_avail: dict[str, bool] = {}
    if lay_path.exists():
        lay = pd.read_parquet(lay_path)
        if "layout_available" in lay.columns:
            layout_avail = {
                str(i): bool(a) for i, a in zip(lay["image_id"].astype(str), lay["layout_available"])
            }

    ocr_df = pd.read_parquet(ocr_index)
    rows: list[dict[str, Any]] = []
    for _, r in ocr_df.iterrows():
        image_id = str(r["image_id"])
        status_ocr = str(r.get("status") or "")
        text_path = raw_dir / f"{image_id}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(r.get("text") or "")

        has_grounding = ("<|ref|>" in text) or ("<|det|>" in text)
        lay_ok = layout_avail.get(image_id)
        parse_failed = bool(has_grounding and lay_ok is False)

        if status_ocr != "ok":
            rows.append(
                {
                    "image_id": image_id,
                    "ocr_quality_status": "empty" if not text.strip() else "suspicious",
                    "hit_max_tokens": False,
                    "output_token_count": int(r["output_token_count"])
                    if r.get("output_token_count") is not None and not pd.isna(r.get("output_token_count"))
                    else 0,
                    "repetition_ratio": 0.0,
                    "max_ngram_repeat_count": 0,
                    "empty": not bool(text.strip()),
                    "invalid_char_rate": 0.0,
                    "extreme_line_count": False,
                    "extreme_formula_count": False,
                    "parse_failed": parse_failed,
                    "mean_generated_token_logprob": None,
                    "logprob_available": False,
                    "suspicious_reasons": f"ocr_status={status_ocr}",
                }
            )
            continue

        mean_lp = r.get("mean_generated_token_logprob")
        if mean_lp is not None and pd.isna(mean_lp):
            mean_lp = None
        tok = r.get("output_token_count")
        if tok is not None and pd.isna(tok):
            tok = None
        rows.append(
            extract_quality_row(
                image_id,
                text,
                output_token_count=int(tok) if tok is not None else None,
                mean_logprob=float(mean_lp) if mean_lp is not None else None,
                max_tokens=max_tokens,
                parse_failed=parse_failed,
                qcfg=qcfg,
            )
        )

    df = pd.DataFrame(rows)
    # Ensure column order
    for c in QUALITY_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[QUALITY_COLUMNS]
    atomic_replace_parquet(df, out_dir / "ocr_quality.parquet")

    counts = df["ocr_quality_status"].value_counts().to_dict()
    meta = {
        "max_tokens": max_tokens,
        "thresholds": qcfg,
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "n": int(len(df)),
        "priority": ["empty", "truncated", "repetitive", "parse_failed", "suspicious", "valid"],
    }
    atomic_write_json(out_dir / "ocr_quality_meta.json", meta)
    reports = Path(cfg["paths"]["reports_dir"])
    ensure_dir(reports)
    atomic_write_json(reports / "ocr_quality_meta.json", meta)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    df = run_parse_ocr_quality(cfg)
    print(f"[parse_ocr_quality] n={len(df)} counts={df['ocr_quality_status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
