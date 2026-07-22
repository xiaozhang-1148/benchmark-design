"""Load experiment run_config and resolve per-run directories."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..utils import ensure_dir
from .io_util import make_run_id

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "run_config.yaml"


def load_run_config(
    path: str | Path | None = None,
    *,
    run_id: str | None = None,
    reuse_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CFG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(cfg_path.resolve())

    root = Path(cfg["paths"]["experiment_root"])
    ensure_dir(root)
    cfg["paths"]["experiment_root"] = str(root)

    if reuse_run_dir is not None:
        run_dir = Path(reuse_run_dir)
        cfg["run_id"] = run_dir.name
    else:
        cfg["run_id"] = run_id or make_run_id()
        run_dir = root / "runs" / cfg["run_id"]

    ensure_dir(run_dir)
    cfg["paths"]["run_dir"] = str(run_dir)
    cfg["paths"]["config_dir"] = str(ensure_dir(run_dir / "config"))
    cfg["paths"]["metadata_dir"] = str(ensure_dir(run_dir / "metadata"))
    cfg["paths"]["embeddings_dir"] = str(ensure_dir(run_dir / "embeddings"))
    cfg["paths"]["diagnostics_dir"] = str(ensure_dir(run_dir / "diagnostics"))
    cfg["paths"]["projections_dir"] = str(ensure_dir(run_dir / "projections"))
    cfg["paths"]["clustering_dir"] = str(ensure_dir(run_dir / "clustering"))
    cfg["paths"]["galleries_dir"] = str(ensure_dir(run_dir / "galleries"))
    cfg["paths"]["report_dir"] = str(ensure_dir(run_dir / "report"))
    for sub in ("nearest_neighbors", "cluster_centers", "cluster_boundaries", "outliers"):
        ensure_dir(Path(cfg["paths"]["galleries_dir"]) / sub)

    # defaults for P1
    cfg.setdefault("use_local_patches", False)
    cfg.setdefault("pooling", "global_mean")
    return cfg


def model_path(cfg: dict[str, Any]) -> str:
    local = (cfg.get("model") or {}).get("local_path")
    if local and Path(local).exists():
        return local
    return cfg["model"]["name_or_path"]


def dump_frozen_run_config(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["paths"]["config_dir"]) / "run_config.yaml"
    slim = {
        "run_id": cfg.get("run_id"),
        "method_name": cfg.get("method_name"),
        "model_name": cfg["model"]["name_or_path"],
        "model_revision": cfg["model"].get("revision"),
        "dtype": cfg["model"].get("dtype", "bfloat16"),
        "base_size": cfg["preprocess"]["base_size"],
        "image_size": cfg["preprocess"]["image_size"],
        "crop_mode": cfg["preprocess"]["crop_mode"],
        "use_local_patches": cfg.get("use_local_patches", False),
        "pooling": cfg.get("pooling", "global_mean"),
        "normalization": cfg.get("normalization", "l2"),
        "exclude_view_separator": cfg.get("exclude_view_separator", True),
        "random_seed": cfg.get("random_seed", 42),
        "attn_implementation": cfg["model"].get("attn_implementation"),
        "embedding_sha256": cfg.get("embedding_sha256"),
    }
    from ..utils import atomic_write_text

    atomic_write_text(out, yaml.safe_dump(slim, allow_unicode=True, sort_keys=False))
    return out
