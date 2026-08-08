"""Canonical paths under candidate_select_cluster/."""

from __future__ import annotations

from pathlib import Path

CLUSTER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLUSTER_ROOT.parent

DOCS_DIR = CLUSTER_ROOT / "docs"
DATA_DIR = CLUSTER_ROOT / "data"
FIGURES_DIR = CLUSTER_ROOT / "figures"
SCRIPTS_DIR = CLUSTER_ROOT / "scripts"
SRC_DIR = CLUSTER_ROOT / "src"

FLOW_DOC = DOCS_DIR / "候选样本筛选与聚类修正流程.md"
TAXONOMY_DOC = DOCS_DIR / "高中数学三级标签框架_一级与二级优化版_v0.2.md"
LABELED_CSV = DATA_DIR / "第二批数据题目_seed21pro_latex.csv"
KNOWLEDGE_COVERAGE_PNG = FIGURES_DIR / "知识点结构与题目占比.png"
KNOWLEDGE_COVERAGE_SVG = FIGURES_DIR / "知识点结构与题目占比.svg"
