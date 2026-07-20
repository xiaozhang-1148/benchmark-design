"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from heatmap_analysis.config import load_config
from heatmap_analysis.pipeline import extract_all, run_all
from heatmap_analysis.aggregation import run_aggregation
from heatmap_analysis.comparison import compare_groups
from heatmap_analysis.cluster_study import run_cluster_study
from heatmap_analysis.reporting import generate_report
from heatmap_analysis.visualization import generate_all_visualizations
from heatmap_analysis.io import run_preprocessing_checks
from heatmap_analysis.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heatmap_analysis",
        description="Handwriting answer-sheet heatmap statistics and clustering",
    )
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", "-c", type=Path, required=True, help="YAML config path")
    parent.add_argument("--limit", type=int, default=None, help="Max images to process (debug)")
    parent.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", parents=[parent], help="Run data validation checks")
    sub.add_parser("extract", parents=[parent], help="Extract per-image heatmaps and metrics")
    sub.add_parser("aggregate", parents=[parent], help="Aggregate dataset-level heatmaps")

    cluster_p = sub.add_parser("cluster", parents=[parent], help="Run dual-track clustering study")
    cluster_p.add_argument("--skip-extract", action="store_true", help="Skip extract if cache exists")
    cluster_p.add_argument("--methods", type=str, default=None, help="Subset: K, G, M, KG, KGM, etc.")

    compare_p = sub.add_parser("compare", parents=[parent], help="Group comparison")
    compare_p.add_argument("--group-by", action="append", default=None, help="Metadata column(s)")

    sub.add_parser("report", parents=[parent], help="Generate HTML report and visualizations")
    sub.add_parser("run-all", parents=[parent], help="Run full pipeline")

    viz_p = sub.add_parser("visualize", parents=[parent], help="Generate aggregate/cluster PNG visualizations")
    viz_p.add_argument("--group-by", action="append", default=None)

    # backward-compatible alias
    exp_p = sub.add_parser("experiments", parents=[parent], help="Alias for cluster study")
    exp_p.add_argument("--skip-extract", action="store_true")
    exp_p.add_argument("--only", type=str, default=None, help="Deprecated; use --methods on cluster")
    exp_p.add_argument("--methods", type=str, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if getattr(args, "no_gpu", False):
        cfg.gpu.enabled = False
    log_file = cfg.output.output_dir / "heatmap_analysis.log"
    setup_logging(log_file)

    cfg.output.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "preprocess":
        run_preprocessing_checks(cfg)
    elif args.command == "extract":
        extract_all(cfg, limit=args.limit, cfg_path=args.config)
    elif args.command == "aggregate":
        run_aggregation(cfg)
    elif args.command == "compare":
        compare_groups(cfg, args.group_by)
    elif args.command == "cluster":
        run_cluster_study(
            cfg,
            skip_extract=args.skip_extract,
            cfg_path=args.config,
            limit=args.limit,
            methods=args.methods,
        )
    elif args.command == "experiments":
        methods = getattr(args, "methods", None) or getattr(args, "only", None)
        run_cluster_study(
            cfg,
            skip_extract=args.skip_extract,
            cfg_path=args.config,
            limit=args.limit,
            methods=methods,
        )
    elif args.command == "report":
        generate_all_visualizations(cfg)
        path = generate_report(cfg)
        print(f"Report written to {path}")
    elif args.command == "visualize":
        generate_all_visualizations(cfg)
    elif args.command == "run-all":
        run_all(cfg, limit=args.limit, group_by=getattr(args, "group_by", None), cfg_path=args.config)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
