"""Deprecated: use heatmap_analysis.cluster_study instead."""

from heatmap_analysis.cluster_study import run_cluster_study

__all__ = ["run_cluster_study"]

# Backward-compatible alias
run_experiments = run_cluster_study
