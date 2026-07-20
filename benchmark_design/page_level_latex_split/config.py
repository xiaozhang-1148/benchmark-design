"""Frozen split configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class BinSpec:
    label: str
    min: float
    max: float | None


@dataclass(frozen=True, slots=True)
class SplitConfig:
    dataset_version: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    candidate_seeds: tuple[int, ...]
    expression_count_bins: tuple[BinSpec, ...]
    max_expression_token_bins: tuple[BinSpec, ...]
    page_token_bins: tuple[BinSpec, ...]
    foreground_density_bins: tuple[BinSpec, ...]
    line_aspect_ratio_bins: tuple[BinSpec, ...]
    ast_depth_labels: dict[str, tuple[int, ...]]
    min_support_pages: int
    min_expected_per_split: int
    min_split_ratio_for_support: float
    require_train_vocab_coverage: bool
    tie_break: str
    global_swap_max_iterations: int
    repair_top_k_train: int
    max_relative_deviation_tolerance: float
    family_tolerances: dict[str, float]
    algorithm_version: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ratios(self) -> dict[str, float]:
        return {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio}


def _parse_bins(items: list[dict[str, Any]]) -> tuple[BinSpec, ...]:
    bins: list[BinSpec] = []
    for item in items:
        bins.append(
            BinSpec(
                label=str(item["label"]),
                min=float(item["min"]),
                max=None if item.get("max") is None else float(item["max"]),
            )
        )
    return tuple(bins)


def load_split_config(path: Path) -> SplitConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split config not found: {path.resolve()}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"split config must be a mapping: {path}")
    missing = [key for key in ("train_ratio", "val_ratio", "test_ratio") if key not in payload]
    if missing:
        raise KeyError(
            f"split config {path.resolve()} missing keys {missing}; "
            f"found={sorted(payload)}"
        )
    train = float(payload["train_ratio"])
    val = float(payload["val_ratio"])
    test = float(payload["test_ratio"])
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {train + val + test}")
    ast_raw = dict(payload.get("ast_depth_labels") or {})
    ast_depth_labels = {str(k): tuple(int(x) for x in v) for k, v in ast_raw.items()}
    seeds = payload.get("candidate_seeds") or [payload.get("random_seed", 42)]
    return SplitConfig(
        dataset_version=str(payload.get("dataset_version", "")),
        train_ratio=train,
        val_ratio=val,
        test_ratio=test,
        random_seed=int(payload.get("random_seed", 42)),
        candidate_seeds=tuple(int(s) for s in seeds),
        expression_count_bins=_parse_bins(list(payload["expression_count_bins"])),
        max_expression_token_bins=_parse_bins(list(payload["max_expression_token_bins"])),
        page_token_bins=_parse_bins(list(payload["page_token_bins"])),
        foreground_density_bins=_parse_bins(
            list(
                payload.get("foreground_density_bins")
                or [
                    {"label": "density_lt_2pct", "min": 0.0, "max": 0.02},
                    {"label": "density_2_4pct", "min": 0.02, "max": 0.04},
                    {"label": "density_4_6pct", "min": 0.04, "max": 0.06},
                    {"label": "density_6_8pct", "min": 0.06, "max": 0.08},
                    {"label": "density_8_10pct", "min": 0.08, "max": 0.10},
                    {"label": "density_ge_10pct", "min": 0.10, "max": None},
                ]
            )
        ),
        line_aspect_ratio_bins=_parse_bins(
            list(
                payload.get("line_aspect_ratio_bins")
                or [
                    {"label": "aspect_lt_3", "min": 0.0, "max": 3.0},
                    {"label": "aspect_3_5", "min": 3.0, "max": 5.0},
                    {"label": "aspect_5_8", "min": 5.0, "max": 8.0},
                    {"label": "aspect_8_12", "min": 8.0, "max": 12.0},
                    {"label": "aspect_gt_12", "min": 12.0, "max": None},
                ]
            )
        ),
        ast_depth_labels=ast_depth_labels,
        min_support_pages=int(payload.get("min_support_pages", 30)),
        min_expected_per_split=int(payload.get("min_expected_per_split", 1)),
        min_split_ratio_for_support=float(payload.get("min_split_ratio_for_support", 0.05)),
        require_train_vocab_coverage=bool(payload.get("require_train_vocab_coverage", True)),
        tie_break=str(payload.get("tie_break", "seeded_hash")),
        global_swap_max_iterations=int(payload.get("global_swap_max_iterations", 500)),
        repair_top_k_train=int(payload.get("repair_top_k_train", 50)),
        max_relative_deviation_tolerance=float(payload.get("max_relative_deviation_tolerance", 0.15)),
        family_tolerances={str(k): float(v) for k, v in dict(payload.get("family_tolerances") or {}).items()},
        algorithm_version=str(payload.get("algorithm_version", "page_level_latex_split_v1")),
        raw=payload,
    )
