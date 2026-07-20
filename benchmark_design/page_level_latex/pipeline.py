"""End-to-end page-level LaTeX metrics pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.page_level_latex.consistency import run_consistency_checks
from benchmark_design.page_level_latex.expression_latex_metrics import (
    apply_rare10_pass2,
    build_expression_metrics_pass1,
    write_expression_latex_metrics,
)

from benchmark_design.page_level_latex.page_latex_metrics import (
    aggregate_page_latex_metrics,
    page_metrics_to_frame,
)
from benchmark_design.page_level_latex.plot_data import build_fig6_5_joint_grouped
from benchmark_design.page_level_latex.plotting import export_page_latex_figures
from benchmark_design.page_level_latex.similar_tokens import (
    compute_similar_token_stats,
    load_similar_token_groups,
    write_similar_token_samples_stub,
)
from benchmark_design.page_level_latex.tables import (
    write_ast_depth_coverage,
    write_distinct_token_distribution,
    write_distinct_token_summary,
    write_length_coverage,
    write_max_length_distribution,
    write_protocol_audit,
    write_rare8_occurrence_distribution,
    write_rare10_occurrence_distribution,
    write_rare10_summary,
    write_rare10_token_detail,
    write_scale_summary,
    write_structure_combinations,
    write_structure_coverage,
    write_structure_depth_joint_distribution,
    write_structure_type_count,
    write_token_category_coverage,
)


@dataclass(frozen=True, slots=True)
class PageLatexExportResult:
    output_root: Path
    passed_consistency: bool
    manifest: dict[str, str]


def run_page_level_latex_export(
    input_dir: Path,
    output_root: Path,
    *,
    workers: int | None = None,
    show_progress: bool = True,
    skip_figures: bool = False,
    strict_consistency: bool = True,
) -> PageLatexExportResult:
    metrics_dir = output_root / "metrics"
    summary_dir = output_root / "summary"
    figures_dir = output_root / "figures"
    plot_data_dir = output_root / "plot_data"
    for path in (metrics_dir, summary_dir, figures_dir, plot_data_dir):
        path.mkdir(parents=True, exist_ok=True)

    pass1 = build_expression_metrics_pass1(
        input_dir,
        show_progress=show_progress,
        workers=workers,
    )
    expression_rows = apply_rare10_pass2(pass1)
    all_image_ids = sorted({row.image_id for row in pass1.rows})
    page_rows = aggregate_page_latex_metrics(expression_rows, all_image_ids=all_image_ids)

    expression_path = write_expression_latex_metrics(
        expression_rows,
        metrics_dir / "expression_latex_metrics.csv",
    )
    page_frame = page_metrics_to_frame(page_rows)
    page_path = metrics_dir / "page_latex_metrics.csv"
    page_frame.to_csv(page_path, index=False)

    audit = dict(pass1.audit)
    write_protocol_audit(audit, summary_dir / "page_latex_protocol_audit.csv")

    consistency = run_consistency_checks(
        expression_rows,
        page_rows,
        vocab_size=int(audit.get("vocab_size", 0)),
    )
    consistency.chapter5_frame.to_csv(summary_dir / "page_latex_chapter5_consistency.csv", index=False)
    (summary_dir / "page_latex_invariant_errors.json").write_text(
        json.dumps({"errors": list(consistency.invariant_errors)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_scale_summary(page_rows, summary_dir / "page_latex_scale_summary.csv")
    write_length_coverage(expression_rows, page_rows, summary_dir / "page_latex_length_coverage.csv")
    write_max_length_distribution(page_rows, summary_dir / "page_latex_max_length_distribution.csv")
    write_ast_depth_coverage(expression_rows, page_rows, summary_dir / "page_latex_ast_depth_coverage.csv")
    write_structure_coverage(expression_rows, page_rows, summary_dir / "page_latex_structure_coverage.csv")
    write_structure_type_count(page_rows, summary_dir / "page_latex_structure_type_count.csv")
    write_structure_combinations(page_rows, summary_dir / "page_latex_structure_combinations.csv")
    write_structure_depth_joint_distribution(
        page_rows,
        summary_dir / "page_latex_structure_depth_joint_distribution.csv",
    )
    build_fig6_5_joint_grouped(page_rows).to_csv(
        plot_data_dir / "fig6_5_structure_depth_joint_distribution.csv",
        index=False,
    )
    write_structure_depth_joint_distribution(
        page_rows,
        plot_data_dir / "page_structure_depth_joint_distribution_exact.csv",
    )
    write_token_category_coverage(
        expression_rows,
        page_rows,
        summary_dir / "page_latex_token_category_coverage.csv",
    )
    write_distinct_token_summary(page_rows, summary_dir / "page_token_distinct_count_summary.csv")
    write_distinct_token_distribution(
        page_rows,
        summary_dir / "page_token_distinct_count_distribution.csv",
    )
    write_token_category_coverage(
        expression_rows,
        page_rows,
        summary_dir / "page_token_category_coverage.csv",
    )

    write_rare10_summary(
        expression_rows,
        page_rows,
        pass1.token_counter,
        summary_dir / "page_latex_rare10_summary.csv",
    )
    write_rare10_occurrence_distribution(
        page_rows,
        summary_dir / "page_latex_rare10_occurrence_distribution.csv",
    )
    from benchmark_design.page_level_latex.rare8 import compute_rare8_token_set

    rare8_tokens = compute_rare8_token_set(pass1.token_counter)
    write_rare8_occurrence_distribution(
        expression_rows,
        page_rows,
        rare8_tokens,
        summary_dir / "page_latex_rare8_occurrence_distribution.csv",
    )
    write_rare10_token_detail(
        expression_rows,
        pass1.token_counter,
        summary_dir / "page_latex_rare10_token_detail.csv",
        total_pages=len(page_rows),
    )

    # Similar-token Chapter 6.6.3
    config_rows = load_similar_token_groups()
    similar = compute_similar_token_stats(
        expression_rows,
        page_rows,
        pass1.token_counter,
        config_rows=config_rows,
    )
    similar["validation"].to_csv(summary_dir / "page_latex_similar_token_validation.csv", index=False)
    similar["detail"].to_csv(summary_dir / "page_latex_similar_token_detail.csv", index=False)
    similar["group_summary"].to_csv(summary_dir / "page_latex_similar_token_group_summary.csv", index=False)
    similar["pair_cooccurrence"].to_csv(
        summary_dir / "page_latex_similar_token_pair_cooccurrence.csv",
        index=False,
    )
    write_similar_token_samples_stub(
        expression_rows,
        config_rows=config_rows,
        token_counter=pass1.token_counter,
        output_path=summary_dir / "page_latex_similar_token_samples.csv",
    )

    figure_manifest: dict[str, str] = {}
    if not skip_figures and page_rows:
        figure_paths = export_page_latex_figures(
            expression_rows,
            page_rows,
            figures_dir,
            rare_tokens=rare8_tokens,
            similar_group_summary=similar["group_summary"],
        )
        figure_manifest = {
            key: path.relative_to(output_root).as_posix() for key, path in figure_paths.items()
        }

    summary_payload = {
        "input_dir": str(input_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "raw_page_count": audit.get("raw_page_count", 0),
        "raw_expression_count": audit.get("raw_expression_count", 0),
        "valid_expression_count": audit.get("valid_expression_count", 0),
        "page_count": len(page_rows),
        "total_token_count": audit.get("total_token_count", 0),
        "vocab_size": audit.get("vocab_size", 0),
        "consistency_passed": consistency.passed,
        "invariant_errors": list(consistency.invariant_errors),
        "chapter5_mismatches": consistency.chapter5_frame.loc[
            ~consistency.chapter5_frame["match"], "metric"
        ].tolist(),
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if strict_consistency and not consistency.passed:
        mismatch = ", ".join(summary_payload["chapter5_mismatches"]) or "none"
        invariants = "; ".join(consistency.invariant_errors) or "none"
        raise RuntimeError(
            "page_level_latex consistency failed; refusing to treat paper tables as authoritative. "
            f"chapter5 mismatches=[{mismatch}]; invariants=[{invariants}]"
        )

    manifest = {
        "expression_latex_metrics": expression_path.relative_to(output_root).as_posix(),
        "page_latex_metrics": page_path.relative_to(output_root).as_posix(),
        "dataset_summary": "dataset_summary.json",
        "chapter5_consistency": "summary/page_latex_chapter5_consistency.csv",
        "protocol_audit": "summary/page_latex_protocol_audit.csv",
        "consistency_passed": str(consistency.passed).lower(),
    }
    manifest.update(figure_manifest)
    return PageLatexExportResult(
        output_root=output_root,
        passed_consistency=consistency.passed,
        manifest=manifest,
    )
