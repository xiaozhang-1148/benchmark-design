"""Configuration loading and path helpers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(cfg_path.resolve())
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any]) -> None:
    root = Path(cfg["paths"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    reports = cfg["paths"].get("reports_dir")
    if not reports:
        reports = root / "reports"
    else:
        reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["reports_dir"] = str(reports)
    cfg["paths"]["output_root"] = str(root)
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["outputs_dir"] = str(outputs)


def model_resolved_path(cfg: dict[str, Any]) -> str:
    local = cfg["model"].get("local_path")
    if local and Path(local).exists():
        return local
    return cfg["model"]["name_or_path"]


def fingerprint_config(cfg: dict[str, Any], keys: list[str] | None = None) -> str:
    """Stable hash of inference-critical config fields for cache invalidation."""
    subset = {
        "model_name": cfg["model"]["name_or_path"],
        "revision": cfg["model"].get("revision"),
        "prompt": cfg.get("prompt"),
        "selected_layer": cfg["model"].get("selected_layer"),
        "visual_encoder": cfg["model"].get("visual_encoder"),
        "dtype": cfg["model"].get("dtype"),
        "base_size": cfg["visual"].get("base_size"),
        "image_size": cfg["visual"].get("image_size"),
        "crop_mode": cfg["visual"].get("crop_mode"),
    }
    if keys:
        subset = {k: subset[k] for k in keys if k in subset}
    blob = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def get_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("seed", 42))
