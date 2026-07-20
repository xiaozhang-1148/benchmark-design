"""End-to-end pipeline orchestration."""

from __future__ import annotations

import argparse
import traceback
from typing import Any, Callable

from .analyze_features import analyze_all
from .build_manifest import build_manifest
from .build_report import build_report
from .config import load_config
from .inspect_model import run_introspection
from .parse_layout import run_parse_layout
from .parse_recognition import run_parse_recognition
from .plot_features import run_plots
from .vllm_ocr_runner import run_benchmark, run_vllm_ocr
from .visual_feature_runner import run_visual_features


def run_pipeline(cfg: dict[str, Any]) -> None:
    stages = cfg.get("pipeline", {}).get("stages") or [
        "inspect_model",
        "build_manifest",
        "benchmark",
        "vllm_ocr",
        "parse_layout",
        "parse_recognition",
        "visual_features",
        "analyze",
        "plot",
        "report",
    ]
    mapping: dict[str, Callable[[], Any]] = {
        "inspect_model": lambda: run_introspection(cfg),
        "build_manifest": lambda: build_manifest(cfg),
        "benchmark": lambda: run_benchmark(cfg),
        "vllm_ocr": lambda: run_vllm_ocr(cfg),
        "parse_layout": lambda: run_parse_layout(cfg),
        "parse_recognition": lambda: run_parse_recognition(cfg),
        "visual_features": lambda: run_visual_features(cfg),
        "analyze": lambda: analyze_all(cfg),
        "plot": lambda: run_plots(cfg),
        "report": lambda: build_report(cfg),
    }
    for stage in stages:
        print(f"===== STAGE: {stage} =====")
        try:
            mapping[stage]()
        except Exception:
            print(f"[pipeline] stage failed: {stage}")
            traceback.print_exc()
            raise
        print(f"===== DONE: {stage} =====")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 feature pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit images for smoke runs")
    parser.add_argument("--stages", nargs="*", default=None, help="Optional subset of stages")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("pipeline", {})["limit"] = args.limit
    if args.stages:
        cfg.setdefault("pipeline", {})["stages"] = args.stages
    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
