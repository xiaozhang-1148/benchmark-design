"""Multi-label OCR quality status (non-exclusive flags)."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_config
from ..feature_store import atomic_replace_parquet
from ..utils import atomic_write_json, ensure_dir
from .paths import analysis_v2_dir, reports_v2_dir

REF_DET = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)


def _repetition_ratio(text: str, ngram: int = 8) -> float:
    if len(text) < ngram * 2:
        return 0.0
    grams = [text[i : i + ngram] for i in range(0, len(text) - ngram + 1, ngram)]
    if not grams:
        return 0.0
    top, cnt = Counter(grams).most_common(1)[0]
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


def _primary_status(flags: dict[str, bool]) -> str:
    """Non-exclusive labels; primary is only for plotting priority."""
    order = [
        ("is_empty", "empty"),
        ("hit_max_tokens", "truncated"),
        ("has_catastrophic_ngram_repeat", "catastrophic_repeat"),
        ("has_high_repetition_ratio", "high_repetition"),
        ("has_grounding_parse_failure", "grounding_parse_failure"),
        ("has_extreme_line_count", "extreme_line_count"),
        ("has_extreme_formula_count", "extreme_formula_count"),
        ("has_invalid_characters", "invalid_characters"),
        ("has_invalid_bounding_boxes", "invalid_bounding_boxes"),
    ]
    for key, name in order:
        if flags.get(key):
            return name
    return "ok"


def extract_quality_v2(
    image_id: str,
    text: str,
    *,
    output_token_count: int | None,
    max_tokens: int,
    qcfg: dict[str, Any],
    has_grounding_parse_failure: bool,
    has_invalid_bounding_boxes: bool,
) -> dict[str, Any]:
    text = text or ""
    lines = text.splitlines()
    n_lines = len(lines)
    formula_n = _formula_span_count(text)
    tok = int(output_token_count) if output_token_count is not None else 0

    is_empty = len(text.strip()) == 0
    hit_max_tokens = (not is_empty) and tok >= int(max_tokens)
    rep = _repetition_ratio(text)
    ngram_rep = _max_ngram_repeat(text)
    inv = _invalid_char_rate(text)

    rep_thr = float(qcfg.get("repetition_ratio_threshold", 0.35))
    ngram_thr = int(qcfg.get("catastrophic_ngram_repeat", 20))
    inv_thr = float(qcfg.get("invalid_char_rate_threshold", 0.05))
    line_thr = int(qcfg.get("extreme_line_count", 400))
    formula_thr = int(qcfg.get("extreme_formula_count", 80))

    flags = {
        "is_empty": bool(is_empty),
        "hit_max_tokens": bool(hit_max_tokens),
        "has_high_repetition_ratio": bool(rep >= rep_thr),
        "has_catastrophic_ngram_repeat": bool(ngram_rep >= ngram_thr),
        "has_extreme_line_count": bool(n_lines >= line_thr),
        "has_extreme_formula_count": bool(formula_n >= formula_thr),
        "has_invalid_characters": bool(inv >= inv_thr),
        "has_grounding_parse_failure": bool(has_grounding_parse_failure),
        "has_invalid_bounding_boxes": bool(has_invalid_bounding_boxes),
    }

    strict_blockers = [
        "is_empty",
        "hit_max_tokens",
        "has_high_repetition_ratio",
        "has_catastrophic_ngram_repeat",
        "has_extreme_line_count",
        "has_extreme_formula_count",
        "has_grounding_parse_failure",
    ]
    lenient_blockers = [
        "is_empty",
        "hit_max_tokens",
        "has_catastrophic_ngram_repeat",
        "has_grounding_parse_failure",
    ]
    ocr_usable_strict = not any(flags[k] for k in strict_blockers)
    ocr_usable_lenient = not any(flags[k] for k in lenient_blockers)

    return {
        "image_id": image_id,
        **flags,
        "output_token_count": tok,
        "repetition_ratio": float(rep),
        "max_ngram_repeat_count": int(ngram_rep),
        "invalid_char_rate": float(inv),
        "line_count": int(n_lines),
        "formula_count": int(formula_n),
        "ocr_usable_strict": bool(ocr_usable_strict),
        "ocr_usable_lenient": bool(ocr_usable_lenient),
        "ocr_primary_status": _primary_status(flags),
    }


def run_ocr_quality_v2(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    raw_dir = out_dir / "recognition_raw"
    ocr_index = out_dir / "ocr_generations.parquet"
    v2 = analysis_v2_dir(cfg)
    reports = reports_v2_dir(cfg)

    if not ocr_index.exists():
        raise FileNotFoundError(f"Missing {ocr_index}")

    qcfg = dict(cfg.get("ocr_quality") or {})
    max_tokens = int(cfg.get("vllm", {}).get("max_tokens", 4096))

    # layout availability / invalid boxes from v2 if present
    lay_path = v2 / "layout_features_v2.parquet"
    inv_path = v2 / "layout_invalid_boxes.parquet"
    layout_avail: dict[str, bool] = {}
    invalid_box_ids: set[str] = set()
    if lay_path.exists():
        lay = pd.read_parquet(lay_path)
        layout_avail = {
            str(i): bool(a) for i, a in zip(lay["image_id"].astype(str), lay["layout_available"])
        }
    if inv_path.exists():
        inv = pd.read_parquet(inv_path)
        if len(inv) and "image_id" in inv.columns:
            invalid_box_ids = set(inv["image_id"].astype(str))

    ocr_df = pd.read_parquet(ocr_index)
    rows: list[dict[str, Any]] = []
    for _, r in ocr_df.iterrows():
        image_id = str(r["image_id"])
        text_path = raw_dir / f"{image_id}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(r.get("text") or "")

        has_grounding = ("<|ref|>" in text) or ("<|det|>" in text)
        lay_ok = layout_avail.get(image_id)
        # grounding present but no usable layout
        parse_fail = bool(has_grounding and lay_ok is False)
        # also: grounding tags present but regex yields nothing parseable → covered by lay_ok False after layout_v2

        tok = r.get("output_token_count")
        if tok is not None and pd.isna(tok):
            tok = None

        rows.append(
            extract_quality_v2(
                image_id,
                text,
                output_token_count=int(tok) if tok is not None else None,
                max_tokens=max_tokens,
                qcfg=qcfg,
                has_grounding_parse_failure=parse_fail,
                has_invalid_bounding_boxes=image_id in invalid_box_ids,
            )
        )

    df = pd.DataFrame(rows)
    atomic_replace_parquet(df, v2 / "ocr_quality_v2.parquet")

    flag_cols = [
        "is_empty",
        "hit_max_tokens",
        "has_high_repetition_ratio",
        "has_catastrophic_ngram_repeat",
        "has_extreme_line_count",
        "has_extreme_formula_count",
        "has_invalid_characters",
        "has_grounding_parse_failure",
        "has_invalid_bounding_boxes",
    ]
    summary = {
        "n": int(len(df)),
        "max_tokens": max_tokens,
        "thresholds": qcfg,
        "flag_counts": {c: int(df[c].sum()) for c in flag_cols if c in df.columns},
        "ocr_usable_strict": int(df["ocr_usable_strict"].sum()),
        "ocr_usable_lenient": int(df["ocr_usable_lenient"].sum()),
        "ocr_primary_status_counts": {str(k): int(v) for k, v in df["ocr_primary_status"].value_counts().items()},
        "note": "Flags are non-exclusive; primary_status is plot-only priority.",
    }
    atomic_write_json(reports / "ocr_quality_summary.json", summary)
    atomic_write_json(v2 / "ocr_quality_summary.json", summary)
    print(
        f"[ocr_quality_v2] n={len(df)} strict={summary['ocr_usable_strict']} "
        f"lenient={summary['ocr_usable_lenient']} flags={summary['flag_counts']}"
    )
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run_ocr_quality_v2(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
