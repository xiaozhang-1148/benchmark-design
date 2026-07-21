"""Four-stage visual experiment pipeline."""

from __future__ import annotations

import argparse
from typing import Any

from .clustering import run_clustering
from .config import dump_frozen_run_config, load_run_config
from .diagnostics import run_diagnostics
from .extract_run import run_extract
from .galleries import run_galleries
from .manifest import build_manifest
from .projections import run_projections
from .report import build_html_report
from .verify import run_verify


def run_visual_experiment(cfg: dict[str, Any], stages: list[str] | None = None) -> None:
    dump_frozen_run_config(cfg)
    mapping = {
        "manifest": lambda: build_manifest(cfg),
        "verify": lambda: _verify(cfg),
        "smoke": lambda: _smoke(cfg),
        "extract": lambda: run_extract(cfg, limit=None),
        "diagnostics": lambda: run_diagnostics(cfg),
        "projections": lambda: run_projections(cfg),
        "clustering": lambda: run_clustering(cfg),
        "galleries": lambda: run_galleries(cfg),
        "report": lambda: build_html_report(cfg),
        "analyze": lambda: _analyze(cfg),
    }
    stages = stages or ["manifest", "verify", "extract", "analyze"]
    for s in stages:
        print(f"===== STAGE: {s} =====")
        mapping[s]()
        print(f"===== DONE: {s} =====")


def _verify(cfg: dict[str, Any]) -> None:
    n = int(cfg["stages"]["verify_n"])
    build_manifest(cfg, limit=n)
    run_extract(cfg, limit=n, resume=False)
    run_verify(cfg)


def _smoke(cfg: dict[str, Any]) -> None:
    n = int(cfg["stages"]["smoke_n"])
    build_manifest(cfg, limit=n)
    run_extract(cfg, limit=n, resume=True)
    run_diagnostics(cfg)
    run_projections(cfg)


def _analyze(cfg: dict[str, Any]) -> None:
    run_diagnostics(cfg)
    run_projections(cfg)
    run_clustering(cfg)
    run_galleries(cfg)
    build_html_report(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR2 mean-pooled projected-token embedding experiment"
    )
    parser.add_argument("--config", default="configs/experiment/run_config.yaml")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="manifest verify smoke extract diagnostics projections clustering galleries report analyze",
    )
    args = parser.parse_args(argv)
    cfg = load_run_config(args.config)
    run_visual_experiment(cfg, stages=args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
