"""Orchestrate analysis_v2: visual embedding space only."""

from __future__ import annotations

import argparse
from typing import Any

from ..config import load_config
from .analyze import run_analyze_v2
from .quality_gates import run_quality_gates_v2
from .report import build_report_v2


def run_analysis_v2(cfg: dict[str, Any]) -> None:
    print("===== analysis_v2: analyze/plot (visual) =====")
    run_analyze_v2(cfg)
    print("===== analysis_v2: quality_gates =====")
    run_quality_gates_v2(cfg)
    print("===== analysis_v2: report =====")
    build_report_v2(cfg)
    print("===== analysis_v2: DONE =====")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek analysis_v2 (visual only)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Subset: analyze gates report",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    mapping = {
        "analyze": run_analyze_v2,
        "gates": run_quality_gates_v2,
        "report": build_report_v2,
    }
    stages = args.stages or list(mapping.keys())
    for s in stages:
        if s in {"layout", "ocr_quality", "recognition"}:
            print(f"[analysis_v2] skip removed stage: {s}")
            continue
        print(f"===== STAGE: {s} =====")
        mapping[s](cfg)
        print(f"===== DONE: {s} =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
