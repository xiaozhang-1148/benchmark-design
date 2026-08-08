"""Result auditing for DeepSeek-OCR2 feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def audit_outputs(output_dir: str | Path, *, n_expected: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    g = np.load(output_dir / "deepseek_ocr2_global_mean_fp32.npy")
    l = np.load(output_dir / "deepseek_ocr2_local_mean_fp32.npy")
    c = np.load(output_dir / "deepseek_ocr2_global_local_concat_fp32.npy")

    index_rows: list[dict[str, Any]] = []
    with (output_dir / "sample_index.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                index_rows.append(json.loads(line))

    n = len(index_rows)
    report: dict[str, Any] = {
        "n_index": n,
        "global_shape": list(g.shape),
        "local_shape": list(l.shape),
        "concat_shape": list(c.shape),
        "global_dtype": str(g.dtype),
        "local_dtype": str(l.dtype),
        "concat_dtype": str(c.dtype),
        "checks": {},
    }

    checks = report["checks"]
    checks["shapes_match"] = g.shape == (n, 896) and l.shape == (n, 896) and c.shape == (n, 1792)
    checks["dtypes_float32"] = g.dtype == np.float32 and l.dtype == np.float32 and c.dtype == np.float32
    checks["all_finite"] = bool(
        np.isfinite(g).all() and np.isfinite(l).all() and np.isfinite(c).all()
    )
    checks["row_index_contiguous"] = [r["row_index"] for r in index_rows] == list(range(n))
    ids = [r["sample_id"] for r in index_rows]
    paths = [r["image_path"] for r in index_rows]
    checks["sample_id_unique"] = len(ids) == len(set(ids))
    checks["image_path_unique"] = len(paths) == len(set(paths))
    checks["concat_matches_global"] = bool(np.allclose(c[:, :896], g, rtol=0, atol=0))
    checks["concat_matches_local"] = bool(np.allclose(c[:, 896:], l, rtol=0, atol=0))
    checks["not_all_zero_global"] = bool(np.any(g != 0))
    checks["not_all_zero_local"] = bool(np.any(l != 0))
    checks["not_all_zero_concat"] = bool(np.any(c != 0))
    # Different images should not be identical
    if n >= 2:
        same = np.all(c[0] == c[1])
        checks["first_two_rows_identical"] = bool(same)
        checks["features_vary_across_rows"] = not same
    else:
        checks["features_vary_across_rows"] = None
    # Variance not collapsed
    var = c.var(axis=0)
    checks["frac_dims_near_zero_var"] = float((var < 1e-12).mean())
    checks["median_dim_var"] = float(np.median(var))
    success = sum(1 for r in index_rows if r.get("status") == "success")
    checks["all_success"] = success == n
    if n_expected is not None:
        checks["n_equals_expected"] = n == int(n_expected)

    report["n_success"] = success
    report["passed"] = all(
        v is True
        for k, v in checks.items()
        if k
        not in {
            "first_two_rows_identical",
            "frac_dims_near_zero_var",
            "median_dim_var",
            "features_vary_across_rows",
        }
    ) and checks.get("features_vary_across_rows") in (True, None)

    # Soft check: variance collapse
    if checks["frac_dims_near_zero_var"] > 0.5:
        report["passed"] = False
        report["failure_reason"] = "majority of dims have near-zero variance"

    return report
