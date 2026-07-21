"""End-to-end pipeline: pure visual projected-token embeddings (no OCR)."""

from __future__ import annotations

import argparse
import traceback
from typing import Any

from .visual_exp.config import load_run_config
from .visual_exp.pipeline import run_visual_experiment


def run_pipeline(cfg: dict[str, Any] | None = None, *, stages: list[str] | None = None) -> None:
    if cfg is None:
        cfg = load_run_config()
    # Allow legacy yaml that still has paths.output_root — prefer experiment config
    run_visual_experiment(cfg, stages=stages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR2 mean-pooled projected-token embedding (visual only)"
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
    try:
        run_pipeline(cfg, stages=args.stages)
    except Exception:
        traceback.print_exc()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
