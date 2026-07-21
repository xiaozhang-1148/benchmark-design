"""Image-level embedding extraction quality (no OCR text / layout)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .feature_store import EmbeddingStore, atomic_replace_parquet
from .utils import atomic_write_json, ensure_dir

QUALITY_COLUMNS = [
    "image_id",
    "image_readable",
    "embedding_finite",
    "embedding_norm_valid",
    "embedding_dimension_valid",
    "embedding_usable",
    "embedding_norm",
    "embedding_dim",
    "token_count",
    "fail_reasons",
]


def run_embedding_quality(cfg: dict[str, Any]) -> pd.DataFrame:
    """Join manifest + visual index/embeddings into per-image quality flags."""
    out_dir = Path(cfg["paths"]["outputs_dir"])
    reports = Path(cfg["paths"]["reports_dir"])
    ensure_dir(reports)

    man_path = out_dir / "manifest.parquet"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}")
    man = pd.read_parquet(man_path)

    dim = 896
    meta_p = out_dir / "visual_embeddings.f32.meta.json"
    if meta_p.exists():
        dim = int(json.loads(meta_p.read_text()).get("dim", 896))
    intro = reports / "model_introspection.json"
    if intro.exists():
        shape = json.loads(intro.read_text()).get("selected_layer_output_shape")
        if shape and len(shape) >= 3:
            dim = int(shape[-1])

    store = EmbeddingStore(
        mmap_path=out_dir / "visual_embeddings.f32.mmap",
        index_path=out_dir / "visual_index.parquet",
        dim=dim,
    )
    X, vis_idx = store.load_matrix()
    id_to_row = {str(i): int(r) for i, r in zip(vis_idx["image_id"].astype(str), range(len(vis_idx)))}
    token_by_id = {}
    if "token_count" in vis_idx.columns:
        token_by_id = dict(zip(vis_idx["image_id"].astype(str), vis_idx["token_count"]))

    fail_ids: set[str] = set()
    fail_path = out_dir / "visual_failures.parquet"
    if fail_path.exists():
        fails = pd.read_parquet(fail_path)
        if "image_id" in fails.columns:
            fail_ids = set(fails["image_id"].astype(str))

    # Norm range: reject near-zero or absurd pre-L2 norms if recorded; else check L2 unit vector
    norm_lo = float((cfg.get("embedding_quality") or {}).get("norm_min", 1e-6))
    norm_hi = float((cfg.get("embedding_quality") or {}).get("norm_max", 1e6))

    rows: list[dict[str, Any]] = []
    for _, r in man.iterrows():
        image_id = str(r["image_id"])
        reasons: list[str] = []
        status = str(r.get("status") or "")
        image_readable = status != "corrupt" and image_id not in fail_ids
        if status == "corrupt":
            reasons.append("manifest_corrupt")
        if image_id in fail_ids:
            reasons.append("visual_extract_failed")
            image_readable = False

        emb_finite = False
        emb_norm_ok = False
        emb_dim_ok = False
        emb_norm = None
        emb_dim = None
        tok = token_by_id.get(image_id)

        if image_id in id_to_row:
            vec = np.asarray(X[id_to_row[image_id]], dtype=np.float64)
            emb_dim = int(vec.shape[0])
            emb_dim_ok = emb_dim == int(dim)
            if not emb_dim_ok:
                reasons.append("bad_dimension")
            emb_finite = bool(np.isfinite(vec).all())
            if not emb_finite:
                reasons.append("nonfinite")
            nrm = float(np.linalg.norm(vec))
            emb_norm = nrm
            emb_norm_ok = bool(norm_lo <= nrm <= norm_hi)
            if not emb_norm_ok:
                reasons.append("bad_norm")
        else:
            reasons.append("missing_embedding")
            if image_readable:
                # readable image but no embedding yet / failed silently
                image_readable = image_readable and status != "corrupt"

        usable = bool(image_readable and emb_finite and emb_norm_ok and emb_dim_ok)
        rows.append(
            {
                "image_id": image_id,
                "image_readable": bool(image_readable and status != "corrupt"),
                "embedding_finite": bool(emb_finite),
                "embedding_norm_valid": bool(emb_norm_ok),
                "embedding_dimension_valid": bool(emb_dim_ok),
                "embedding_usable": usable,
                "embedding_norm": emb_norm,
                "embedding_dim": emb_dim,
                "token_count": int(tok) if tok is not None and not (isinstance(tok, float) and np.isnan(tok)) else None,
                "fail_reasons": "|".join(reasons),
            }
        )

    df = pd.DataFrame(rows)
    for c in QUALITY_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[QUALITY_COLUMNS]
    atomic_replace_parquet(df, out_dir / "embedding_quality.parquet")

    summary = {
        "n": int(len(df)),
        "expected_dim": int(dim),
        "counts": {
            "image_readable": int(df["image_readable"].sum()),
            "embedding_finite": int(df["embedding_finite"].sum()),
            "embedding_norm_valid": int(df["embedding_norm_valid"].sum()),
            "embedding_dimension_valid": int(df["embedding_dimension_valid"].sum()),
            "embedding_usable": int(df["embedding_usable"].sum()),
        },
        "fail_reason_top": (
            df.loc[df["fail_reasons"].astype(str).str.len() > 0, "fail_reasons"]
            .value_counts()
            .head(20)
            .to_dict()
        ),
    }
    atomic_write_json(out_dir / "embedding_quality_summary.json", summary)
    atomic_write_json(reports / "embedding_quality_summary.json", summary)
    print(f"[embedding_quality] n={len(df)} usable={summary['counts']['embedding_usable']} dim={dim}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run_embedding_quality(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
