"""Tables, figures, and quality reports from the final split_manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from benchmark_design.export_layout import cross_domain_inputs_available
from benchmark_design.page_level_latex_split.io_validate import STRUCTURE_HAS_COLUMNS
from benchmark_design.page_level_latex_split.cross_domain_tables import write_cross_domain_tables
from benchmark_design.page_level_latex_split.figures_ch7 import export_ch7_figures, remove_legacy_figures
from benchmark_design.page_level_latex_split.labels import BINARY_LABEL_COLUMNS, PageLabels
from benchmark_design.page_level_latex_split.refine import VocabAudit
from benchmark_design.page_level_latex_split.stratify import (
    SPLITS,
    LabelQuota,
    observed_label_counts,
    relative_deviation,
)

SIMILAR_COLS = (
    "has_digit_letter_pair",
    "has_circle_like_pair",
    "has_latin_greek_pair",
    "has_greek_variant_pair",
    "has_operator_variable_pair",
    "has_relation_stroke_pair",
)


def _merge_features(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    return manifest.merge(features, on="page_id", how="inner", validate="one_to_one")


def write_table1_scale(merged: pd.DataFrame, path: Path) -> None:
    rows = []
    n = len(merged)
    for split in ("overall", *SPLITS):
        sub = merged if split == "overall" else merged[merged["split"] == split]
        rows.append(
            {
                "split": split,
                "page_count": len(sub),
                "page_ratio": len(sub) / n if n else 0.0,
                "ast_tree_count": int(sub["ast_tree_count"].sum()),
                "total_ast_node_count": int(sub["total_ast_node_count"].sum()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _bin_distribution(merged: pd.DataFrame, col: str, path: Path) -> None:
    labels = sorted(merged[col].unique())
    rows = []
    n = len(merged)
    for label in labels:
        for split in ("overall", *SPLITS):
            sub = merged if split == "overall" else merged[merged["split"] == split]
            count = int((sub[col] == label).sum())
            rows.append(
                {
                    "bin": label,
                    "split": split,
                    "count": count,
                    "ratio": count / (len(sub) if len(sub) else 1),
                    "overall_ratio": count / n if split == "overall" and n else (
                        int((merged[col] == label).sum()) / n if n else 0.0
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table2_bins(merged: pd.DataFrame, labels_frame: pd.DataFrame, out_dir: Path) -> None:
    frame = merged.merge(
        labels_frame.loc[:, ["page_id", "expr_bin", "page_token_bin", "maxlen_bin"]],
        on="page_id",
    )
    _bin_distribution(frame, "expr_bin", out_dir / "table2_expression_count_bins.csv")
    _bin_distribution(frame, "page_token_bin", out_dir / "table2_page_token_bins.csv")
    _bin_distribution(frame, "maxlen_bin", out_dir / "table2_max_expression_length_bins.csv")


def write_table3_continuous(merged: pd.DataFrame, path: Path) -> None:
    metrics = ("ast_tree_count", "total_ast_node_count", "max_ast_depth")
    rows = []
    for metric in metrics:
        for split in ("overall", *SPLITS):
            sub = merged if split == "overall" else merged[merged["split"] == split]
            values = sub[metric].astype(float)
            rows.append(
                {
                    "metric": metric,
                    "split": split,
                    "mean": float(values.mean()) if len(values) else 0.0,
                    "median": float(values.median()) if len(values) else 0.0,
                    "p25": float(values.quantile(0.25)) if len(values) else 0.0,
                    "p75": float(values.quantile(0.75)) if len(values) else 0.0,
                    "p90": float(values.quantile(0.90)) if len(values) else 0.0,
                    "p95": float(values.quantile(0.95)) if len(values) else 0.0,
                    "maximum": float(values.max()) if len(values) else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table4_ast(merged: pd.DataFrame, path: Path) -> None:
    rows = []
    n = len(merged)
    depth_cols = [f"has_expr_depth_{depth}" for depth in range(6)]
    has_depth_cols = all(col in merged.columns for col in depth_cols)
    for depth in range(0, 6):
        for split in ("overall", *SPLITS):
            sub = merged if split == "overall" else merged[merged["split"] == split]
            if has_depth_cols:
                cover = int(sub[f"has_expr_depth_{depth}"].sum())
                coverage_note = "page_has_expression_at_depth"
            else:
                cover = int((sub["max_ast_depth"] >= depth).sum())
                coverage_note = "max_ast_depth_ge_fallback"
            max_count = int((sub["max_ast_depth"] == depth).sum())
            rows.append(
                {
                    "ast_depth": depth,
                    "split": split,
                    "pages_with_max_depth": max_count,
                    "max_depth_ratio": max_count / (len(sub) if len(sub) else 1),
                    "pages_with_expression_at_depth": cover,
                    "coverage_ratio": cover / (len(sub) if len(sub) else 1),
                    "coverage_definition": coverage_note,
                    "overall_page_count": n,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table5_structure(merged: pd.DataFrame, path: Path) -> None:
    rows = []
    overall_n = len(merged)
    for col in STRUCTURE_HAS_COLUMNS:
        overall_count = int(merged[col].sum())
        overall_ratio = overall_count / overall_n if overall_n else 0.0
        for split in ("overall", *SPLITS):
            sub = merged if split == "overall" else merged[merged["split"] == split]
            count = int(sub[col].sum())
            ratio = count / (len(sub) if len(sub) else 1)
            rows.append(
                {
                    "structure": col,
                    "split": split,
                    "page_count": count,
                    "page_ratio": ratio,
                    "relative_deviation": 0.0
                    if split == "overall"
                    else (ratio - overall_ratio) / max(1e-9, overall_ratio),
                }
            )
    # StructureCount distribution
    for sc in sorted(merged["structure_type_count"].unique()):
        for split in ("overall", *SPLITS):
            sub = merged if split == "overall" else merged[merged["split"] == split]
            count = int((sub["structure_type_count"] == sc).sum())
            rows.append(
                {
                    "structure": f"structure_count_{sc}",
                    "split": split,
                    "page_count": count,
                    "page_ratio": count / (len(sub) if len(sub) else 1),
                    "relative_deviation": "",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table6_rare_similar(merged: pd.DataFrame, path: Path) -> None:
    rows = []
    for split in ("overall", *SPLITS):
        sub = merged if split == "overall" else merged[merged["split"] == split]
        rows.append(
            {
                "attribute": "has_rare8",
                "split": split,
                "page_count": int(sub["has_rare8"].sum()),
                "page_ratio": float(sub["has_rare8"].mean()) if len(sub) else 0.0,
                "rare8_token_instances": int(sub["rare8_token_count"].sum()),
            }
        )
        for col in SIMILAR_COLS:
            rows.append(
                {
                    "attribute": col,
                    "split": split,
                    "page_count": int(sub[col].sum()),
                    "page_ratio": float(sub[col].mean()) if len(sub) else 0.0,
                    "rare8_token_instances": "",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table7_joint(merged: pd.DataFrame, path: Path) -> None:
    rows = []
    for split in ("overall", *SPLITS):
        sub = merged if split == "overall" else merged[merged["split"] == split]
        for sc in range(0, 7):
            for depth in range(0, 6):
                count = int(
                    ((sub["structure_type_count"] == sc) & (sub["max_ast_depth"].clip(upper=5) == depth)).sum()
                )
                rows.append(
                    {
                        "split": split,
                        "structure_type_count": sc,
                        "max_ast_depth": depth,
                        "page_count": count,
                        "page_ratio": count / (len(sub) if len(sub) else 1),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table8_quality(
    quotas: dict[str, LabelQuota],
    observed: dict[str, Counter],
    vocab: VocabAudit,
    config: SplitConfig,
    path: Path,
) -> None:
    families = {
        "scale": [l for l in quotas if l.startswith(("expr_", "page_token_", "maxlen_"))],
        "structure": [
            l
            for l in quotas
            if l.startswith("depth_")
            or l.startswith("struc_cnt_")
            or l.startswith("joint_sc")
            or l.startswith("has_expr_depth_")
            or l in set(STRUCTURE_HAS_COLUMNS)
        ],
        "rare8": [l for l in quotas if l == "has_rare8"],
        "similar_token": [l for l in quotas if l in SIMILAR_COLS],
        "vocabulary": [],
    }
    rows = []
    for family, labels in families.items():
        if family == "vocabulary":
            ok = (len(vocab.val_unseen) == 0 and len(vocab.test_unseen) == 0) or not config.require_train_vocab_coverage
            rows.append(
                {
                    "family": family,
                    "max_relative_deviation": 0.0 if ok else 1.0,
                    "mean_relative_deviation": 0.0 if ok else 1.0,
                    "passed": "yes" if ok else "no",
                }
            )
            continue
        devs = []
        for label in labels:
            q = quotas[label]
            if not q.formal:
                continue
            for split in SPLITS:
                devs.append(relative_deviation(observed[label][split], q.targets[split]))
        max_dev = max(devs) if devs else 0.0
        mean_dev = sum(devs) / len(devs) if devs else 0.0
        tol = config.family_tolerances.get(family, config.max_relative_deviation_tolerance)
        rows.append(
            {
                "family": family,
                "max_relative_deviation": max_dev,
                "mean_relative_deviation": mean_dev,
                "passed": "yes" if max_dev <= tol else "no",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_quota_audit(
    quotas: dict[str, LabelQuota],
    observed: dict[str, Counter],
    path: Path,
) -> None:
    rows = []
    for label, q in sorted(quotas.items()):
        max_dev = max(relative_deviation(observed[label][s], q.targets[s]) for s in SPLITS)
        rows.append(
            {
                "label": label,
                "formal": int(q.formal),
                "overall": q.support,
                "train_target": q.targets["train"],
                "train_actual": observed[label]["train"],
                "val_target": q.targets["val"],
                "val_actual": observed[label]["val"],
                "test_target": q.targets["test"],
                "test_actual": observed[label]["test"],
                "max_relative_deviation": max_dev,
                "low_support_reason": q.reason if not q.formal else "",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_low_support(quotas: dict[str, LabelQuota], observed: dict[str, Counter], path: Path) -> None:
    rows = []
    for label, q in sorted(quotas.items()):
        if q.formal:
            continue
        rows.append(
            {
                "label": label,
                "overall": q.support,
                "train_expected": q.expected_raw["train"],
                "val_expected": q.expected_raw["val"],
                "test_expected": q.expected_raw["test"],
                "reason": q.reason,
                "train_actual": observed[label]["train"],
                "val_actual": observed[label]["val"],
                "test_actual": observed[label]["test"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_vocabulary_audit_csv(vocab: VocabAudit, path: Path) -> None:
    rows = [
        {
            "metric": "vocab_size",
            "overall": vocab.overall_size,
            "train": vocab.train_size,
            "val": vocab.val_size,
            "test": vocab.test_size,
        },
        {
            "metric": "unseen_token_count",
            "overall": "",
            "train": 0,
            "val": len(vocab.val_unseen),
            "test": len(vocab.test_unseen),
        },
        {
            "metric": "rare8_pages",
            "overall": sum(vocab.rare8_by_split.values()),
            "train": vocab.rare8_by_split["train"],
            "val": vocab.rare8_by_split["val"],
            "test": vocab.rare8_by_split["test"],
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    detail = path.with_name("vocabulary_unseen_tokens.csv")
    pd.DataFrame(list(vocab.anomalies)).to_csv(detail, index=False)


def export_reports(
    *,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    labels_frame: pd.DataFrame,
    page_labels: list[PageLabels],
    assignment: dict[str, str],
    quotas: dict[str, LabelQuota],
    vocab: VocabAudit,
    config: SplitConfig,
    output_root: Path,
    skip_figures: bool = False,
) -> dict[str, str]:
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    merged = _merge_features(manifest, features)
    observed = observed_label_counts(page_labels, assignment)

    write_table1_scale(merged, tables / "table1_split_scale.csv")
    write_table2_bins(merged, labels_frame, tables)
    write_table3_continuous(merged, tables / "table3_continuous_stats.csv")
    write_table4_ast(merged, tables / "table4_ast_depth.csv")
    write_table5_structure(merged, tables / "table5_structure.csv")
    write_table6_rare_similar(merged, tables / "table6_rare8_similar.csv")
    write_table7_joint(merged, tables / "table7_structure_depth_joint.csv")
    write_table8_quality(quotas, observed, vocab, config, tables / "table8_quality_summary.csv")
    write_quota_audit(quotas, observed, output_root / "split_quota_audit.csv")
    write_low_support(quotas, observed, output_root / "low_support_features.csv")
    write_vocabulary_audit_csv(vocab, output_root / "vocabulary_audit.csv")

    benchmark_export_root = output_root.parent
    cross_domain_manifest: dict[str, str] = {}
    if cross_domain_inputs_available(benchmark_export_root):
        cross_domain_manifest = write_cross_domain_tables(
            manifest,
            benchmark_export_root,
            config,
            tables,
        )

    if not skip_figures:
        remove_legacy_figures(figures)
        figure_manifest = export_ch7_figures(tables, figures)
    else:
        figure_manifest = {}

    return {
        "tables": "tables/",
        "figures": "figures/",
        "split_quota_audit": "split_quota_audit.csv",
        "low_support_features": "low_support_features.csv",
        "vocabulary_audit": "vocabulary_audit.csv",
        **cross_domain_manifest,
        **figure_manifest,
    }
