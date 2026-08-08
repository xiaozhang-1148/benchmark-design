"""Audit text embedding outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def audit_outputs(output_dir: str | Path, *, n_expected: int | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    emb = np.load(output_dir / "qwen3_embedding_0.6b_last_token_raw_fp32.npy")
    index = [json.loads(l) for l in (output_dir / "text_sample_index.jsonl").open(encoding="utf-8") if l.strip()]
    assembled = [
        json.loads(l) for l in (output_dir / "assembled_page_texts.jsonl").open(encoding="utf-8") if l.strip()
    ]
    n = len(index)
    checks: dict[str, Any] = {
        "shape_ok": emb.shape == (n, 1024),
        "dtype_f32": emb.dtype == np.float32,
        "all_finite": bool(np.isfinite(emb).all()),
        "row_contiguous": [r["row_index"] for r in index] == list(range(n)),
        "sample_id_unique": len({r["sample_id"] for r in index}) == n,
        "json_path_unique": len({r["json_path"] for r in index}) == n,
        "assembled_aligned": len(assembled) == n
        and [a["row_index"] for a in assembled] == list(range(n)),
        "all_success": all(r.get("status") == "success" for r in index),
        "not_all_zero": bool(np.any(emb != 0)),
        "features_vary": bool(n < 2 or not np.all(emb[0] == emb[1])),
        "not_l2_unit": float(np.mean(np.abs(np.linalg.norm(emb, axis=1) - 1.0))) > 0.05,
    }
    if n_expected is not None:
        checks["n_equals_expected"] = n == int(n_expected)
    norms = np.linalg.norm(emb, axis=1)
    report = {
        "n": n,
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "checks": checks,
        "norm_stats": {
            "min": float(norms.min()) if n else None,
            "mean": float(norms.mean()) if n else None,
            "max": float(norms.max()) if n else None,
        },
        "token_stats": _token_stats([r.get("token_count") for r in index]),
        "passed": all(bool(v) for v in checks.values()),
    }
    return report


def _token_stats(vals: list[Any]) -> dict[str, Any]:
    arr = np.array([int(v) for v in vals if v is not None], dtype=np.int64)
    if arr.size == 0:
        return {}
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "gt_1600": int((arr > 1600).sum()),
        "gt_2048": int((arr > 2048).sum()),
    }
