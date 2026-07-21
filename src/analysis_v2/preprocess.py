"""Shared feature matrix preprocessing: log1p, quantile clip, drop const/dup, RobustScaler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def drop_constant_and_duplicate(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], dict[str, Any]]:
    use = [c for c in cols if c in df.columns]
    dropped_const: list[str] = []
    kept: list[str] = []
    for c in use:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            dropped_const.append(c)
        else:
            kept.append(c)

    dup_pairs: list[tuple[str, str]] = []
    final: list[str] = []
    seen: list[tuple[str, pd.Series]] = []
    for c in kept:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        is_dup = False
        for prev_c, prev_s in seen:
            if np.allclose(s.to_numpy(dtype=np.float64), prev_s.to_numpy(dtype=np.float64), equal_nan=True):
                dup_pairs.append((c, prev_c))
                is_dup = True
                break
        if not is_dup:
            final.append(c)
            seen.append((c, s))
    return final, {"dropped_constant": dropped_const, "dropped_duplicate_of": dup_pairs}


def apply_log1p(df: pd.DataFrame, cols: list[str], suffix: str = "_transformed") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[f"{c}{suffix}"] = np.log1p(pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0))
    return out


def fit_clip_robust_scale(
    df: pd.DataFrame,
    cols: list[str],
    *,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    out_joblib: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Clip to [0.5%, 99.5%] then RobustScaler. Persist bounds + scaler for reuse."""
    use_cols, drop_meta = drop_constant_and_duplicate(df, cols)
    X = df[use_cols].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)

    lower = X.quantile(lower_q)
    upper = X.quantile(upper_q)
    Xc = X.clip(lower=lower, upper=upper, axis=1)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(Xc.values).astype(np.float32)

    meta = {
        "columns": use_cols,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "clip_lower": {c: float(lower[c]) for c in use_cols},
        "clip_upper": {c: float(upper[c]) for c in use_cols},
        "drop_meta": drop_meta,
        "max_abs_scaled": float(np.nanmax(np.abs(Xs))) if Xs.size else 0.0,
        "nan_inf_count": int((~np.isfinite(Xs)).sum()),
    }
    joblib.dump({"scaler": scaler, "meta": meta, "columns": use_cols}, out_joblib)
    return Xs, use_cols, meta


def transform_with_saved(
    df: pd.DataFrame,
    joblib_path: Path,
) -> tuple[np.ndarray, list[str]]:
    blob = joblib.load(joblib_path)
    cols: list[str] = list(blob["columns"])
    meta = blob["meta"]
    scaler: RobustScaler = blob["scaler"]
    X = df[cols].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lower = pd.Series(meta["clip_lower"])
    upper = pd.Series(meta["clip_upper"])
    Xc = X.clip(lower=lower, upper=upper, axis=1)
    return scaler.transform(Xc.values).astype(np.float32), cols
