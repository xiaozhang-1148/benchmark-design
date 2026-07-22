"""Load joint-cluster experiment config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..utils import ensure_dir

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "joint_cluster" / "experiment_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    cfg["_config_path"] = str(cfg_path.resolve())

    root = Path(cfg["paths"]["output_root"])
    ensure_dir(root)
    cfg["paths"]["output_root"] = str(root)
    for name in (
        "config",
        "data",
        "models",
        "metrics",
        "figures",
        "representatives",
        "cluster_cards",
        "reports",
    ):
        cfg["paths"][f"{name}_dir"] = str(ensure_dir(root / name))
    return cfg
