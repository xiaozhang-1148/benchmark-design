"""Interface verification & determinism checks (stage 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..utils import atomic_write_json, load_image_rgb
from .extractor import ProjectedTokenExtractor


def run_verify(cfg: dict[str, Any]) -> dict[str, Any]:
    meta_dir = Path(cfg["paths"]["metadata_dir"])
    diag = Path(cfg["paths"]["diagnostics_dir"])
    man = pd.read_parquet(meta_dir / "manifest.parquet")
    man = man[man["status"] != "corrupt"].head(int(cfg["stages"]["verify_n"]))
    rows = man.to_dict("records")
    if not rows:
        raise RuntimeError("no images for verify")

    extractor = ProjectedTokenExtractor(cfg)
    # First image: print shapes
    img0 = load_image_rgb(rows[0]["image_path"])
    out0 = extractor.embed_image(img0, debug=True)

    # Determinism: repeat subset
    n_det = min(int(cfg["stages"].get("determinism_n", 30)), len(rows))
    cos_sims = []
    for r in rows[:n_det]:
        img = load_image_rgb(r["image_path"])
        a = extractor.embed_image(img)["embedding"]
        b = extractor.embed_image(img)["embedding"]
        cos_sims.append(float(np.dot(a, b)))

    # Batch vs single consistency on a few (sequential encode already)
    report = {
        "run_id": cfg.get("run_id"),
        "method": extractor.method_name,
        "use_local_patches": bool(cfg.get("use_local_patches", False)),
        "first_image": {
            "shapes": out0["shapes"],
            "token_count": out0["token_count"],
            "embedding_dim": out0["embedding_dim"],
            "token_dtype": out0["token_dtype"],
            "token_min": out0["token_min"],
            "token_max": out0["token_max"],
            "norm_before_l2": out0["norm_before"],
            "n_local_patches": out0["n_local_patches"],
        },
        "determinism_n": n_det,
        "repeat_cosine_min": float(np.min(cos_sims)),
        "repeat_cosine_mean": float(np.mean(cos_sims)),
        "repeat_cosine_max": float(np.max(cos_sims)),
        "cuda": torch.cuda.is_available(),
        "pass_determinism": bool(np.min(cos_sims) > 0.999),
        "note": "Uses SAM→Qwen2→Projector projected tokens (not mid-layer hooks; no generate()).",
    }
    atomic_write_json(diag / "verify_report.json", report)
    print(f"[verify] dim={out0['embedding_dim']} repeat_cos_min={report['repeat_cosine_min']:.6f}")
    extractor.close()
    return report
