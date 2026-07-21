"""Hard quality checks before / for the analysis report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_write_json, ensure_dir


def run_quality_gates(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    analysis = out_dir / "analysis"
    ensure_dir(reports)

    gates: dict[str, Any] = {"checks": [], "summary": {}, "pass": True}

    def add(name: str, ok: bool, detail: Any = None) -> None:
        gates["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            gates["pass"] = False

    man_n = 0
    if (out_dir / "manifest.parquet").exists():
        man = pd.read_parquet(out_dir / "manifest.parquet")
        man_n = len(man)
        readable = int((man["status"] != "corrupt").sum()) if "status" in man else man_n
    else:
        readable = 0
        add("manifest_exists", False, "missing manifest.parquet")

    vis_n = len(pd.read_parquet(out_dir / "visual_index.parquet")) if (out_dir / "visual_index.parquet").exists() else 0
    lay_n = 0
    lay_avail = 0
    if (out_dir / "layout_features.parquet").exists():
        lay = pd.read_parquet(out_dir / "layout_features.parquet")
        lay_n = len(lay)
        lay_avail = int(lay["layout_available"].sum()) if "layout_available" in lay else 0
        if "content_area_ratio" in lay.columns:
            over = float((pd.to_numeric(lay["content_area_ratio"], errors="coerce") > 1.0 + 1e-6).mean())
            add("layout_content_ratio_le_1", over == 0.0, {"fraction_gt_1": over})
        # box bounds already clipped in parser; check occupancy finite
        occ = [c for c in lay.columns if c.startswith("occupancy_")]
        if occ:
            bad = int((~np.isfinite(lay[occ].to_numpy(dtype=float))).sum())
            add("layout_occupancy_finite", bad == 0, {"nonfinite_cells": bad})

    rec_n = len(pd.read_parquet(out_dir / "recognition_features.parquet")) if (out_dir / "recognition_features.parquet").exists() else 0

    q_counts: dict[str, int] = {}
    valid_n = 0
    trunc_n = rep_n = empty_n = parse_fail_n = 0
    if (out_dir / "ocr_quality.parquet").exists():
        q = pd.read_parquet(out_dir / "ocr_quality.parquet")
        q_counts = {str(k): int(v) for k, v in q["ocr_quality_status"].value_counts().items()}
        valid_n = int(q_counts.get("valid", 0))
        trunc_n = int(q_counts.get("truncated", 0))
        rep_n = int(q_counts.get("repetitive", 0))
        empty_n = int(q_counts.get("empty", 0))
        parse_fail_n = int(q_counts.get("parse_failed", 0))
        nq = max(len(q), 1)
        trunc_rate = trunc_n / nq
        rep_rate = rep_n / nq
        parse_rate = parse_fail_n / nq
        add("ocr_truncation_rate", trunc_rate < 0.01, {"rate": trunc_rate, "threshold": 0.01})
        add("ocr_repetitive_rate", rep_rate < 0.005, {"rate": rep_rate, "threshold": 0.005})
        add("layout_parse_fail_rate", parse_rate < 0.01, {"rate": parse_rate, "threshold": 0.01})
    else:
        add("ocr_quality_exists", False, "missing ocr_quality.parquet")

    # Embedding NaN/Inf / norms
    if (analysis / "feature_metrics.json").exists():
        metrics = json.loads((analysis / "feature_metrics.json").read_text())
    elif (reports / "feature_metrics.json").exists():
        metrics = json.loads((reports / "feature_metrics.json").read_text())
    else:
        metrics = {}

    for ch in ("visual", "layout", "recognition"):
        qrep = metrics.get("channels", {}).get(ch, {}).get("quality", {})
        if not qrep:
            continue
        add(f"{ch}_no_nan_inf", int(qrep.get("nan_inf_cells", 0)) == 0, qrep.get("nan_inf_cells"))
        add(f"{ch}_constant_dims_ok", int(qrep.get("constant_dims", 0)) == 0, qrep.get("constant_dims"))
        n_used = metrics.get("channels", {}).get(ch, {}).get("n_used") or qrep.get("n_samples")
        add(f"{ch}_pca_n_recorded", n_used is not None and int(n_used) > 0, n_used)

    # ID alignment among filtered channels
    try:
        ids_v = set(pd.read_parquet(analysis / "visual_index_aligned.parquet")["image_id"].astype(str))
        ids_l = set(pd.read_parquet(analysis / "layout_index_aligned.parquet")["image_id"].astype(str))
        ids_r = set(pd.read_parquet(analysis / "recognition_index_aligned.parquet")["image_id"].astype(str))
        common = ids_v & ids_l & ids_r
        add(
            "channel_ids_alignable",
            len(common) >= 20 or (len(ids_v) + len(ids_l) + len(ids_r) < 60),
            {"n_visual": len(ids_v), "n_layout": len(ids_l), "n_recognition": len(ids_r), "n_common": len(common)},
        )
    except Exception as e:  # noqa: BLE001
        add("channel_ids_alignable", False, str(e))

    # Recognition must not include quality cols in PCA feature list
    rec_cols = metrics.get("channels", {}).get("recognition", {}).get("feature_cols") or []
    banned = {"repetition_ratio", "mean_generated_token_logprob", "hit_max_tokens", "logprob_available"}
    leak = sorted(banned.intersection(set(rec_cols)))
    add("recognition_pca_excludes_quality", len(leak) == 0, {"leaked": leak})

    # Dropped constants reported
    for ch in ("layout", "recognition"):
        dropped = metrics.get("channels", {}).get(ch, {}).get("dropped_constant_cols") or []
        # Informational — constants already dropped from PCA; fail only if still in feature_cols
        feat_cols = set(metrics.get("channels", {}).get(ch, {}).get("feature_cols") or [])
        still = [c for c in dropped if c in feat_cols]
        add(f"{ch}_no_constant_in_pca", len(still) == 0, {"still_present": still, "dropped": dropped})

    gates["summary"] = {
        "total_images": man_n,
        "readable_images": readable,
        "visual_valid": vis_n,
        "layout_total": lay_n,
        "layout_valid": lay_avail,
        "recognition_total": rec_n,
        "recognition_valid": valid_n,
        "truncated": trunc_n,
        "repetitive": rep_n,
        "empty": empty_n,
        "parse_failed": parse_fail_n,
        "ocr_quality_status_counts": q_counts,
        "excluded_from_recognition_pca": {
            "reason": "ocr_quality_status != valid",
            "n": int(rec_n - valid_n) if rec_n else None,
        },
    }

    atomic_write_json(reports / "quality_gates.json", gates)
    ensure_dir(out_dir / "analysis")
    atomic_write_json(out_dir / "analysis" / "quality_gates.json", gates)
    return gates
