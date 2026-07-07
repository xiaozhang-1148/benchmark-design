"""Generate the cross-benchmark comparison Markdown report."""

from __future__ import annotations

import math
from pathlib import Path

from benchmark_design.ocr.cross_benchmark import (
    CrossBenchmarkProfile,
    LengthBinStat,
    AST_DEPTH_COUNT_LABELS,
    STRUCTURE_ANY_COLUMN,
    STRUCTURE_TYPE_COUNT_LABELS,
)
from benchmark_design.ocr.structure_distribution import (
    MATRIX_STRUCTURE_REPORT_COLUMN,
    MATRIX_STRUCTURE_TRIGGER_TOKENS,
)


def _fmt_int(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{int(value):,}"


def _fmt_pct(share: float, *, decimals: int = 2) -> str:
    return f"{share * 100:.{decimals}f}%"


def _fmt_rate_pct(rate: float, *, decimals: int = 2) -> str:
    return _fmt_pct(rate, decimals=decimals)


def _fmt_length(value: float | int) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "TBD"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def _fmt_gini(value: float) -> str:
    return f"{value:.3f}"


def _fmt_bin_cell(item: LengthBinStat) -> str:
    return f"{_fmt_int(item.count)} ({_fmt_pct(item.share)})"


def _fmt_count_share(count: int, share: float) -> str:
    return f"{_fmt_int(count)} ({_fmt_pct(share)})"


def _structure_within_structured_share(count: int, structured_count: int) -> float:
    return count / structured_count if structured_count else 0.0


def _fmt_optional_pct(value: float) -> str:
    pct = value * 100.0
    if pct == 0.0:
        return "0.00%"
    if pct < 0.01:
        return f"{pct:.4f}%"
    return f"{pct:.2f}%"


def build_cross_benchmark_comparison_markdown(profiles: list[CrossBenchmarkProfile]) -> str:
    lines = [
        "# Cross-Benchmark Comparison",
        "",
        "This report compares **Ours** with CROHME, HME100K, MathWriting, and MNE under a unified "
        "LaTeX tokenization and structure-analysis protocol.",
        "",
        "## 1. Dataset Scale and Effective Diversity",
        "",
        "| Dataset | Expr. Count | Unique Expr. | Duplicate Rate | Vocab Size | Parse OK |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in profiles:
        lines.append(
            f"| {profile.display_name} | {_fmt_int(profile.expression_count)} | "
            f"{_fmt_int(profile.unique_expression_count)} | "
            f"{_fmt_rate_pct(profile.duplicate_rate)} | "
            f"{_fmt_int(profile.vocabulary_size)} | "
            f"{_fmt_rate_pct(profile.parse_success_rate)} |"
        )

    lines.extend(
        [
            "",
            "**Takeaway:** Ours provides a large-scale benchmark with high effective expression diversity "
            "and substantially richer vocabulary coverage.",
            "",
            "## 2. Expression Length Complexity",
            "",
            "| Dataset | Mean Len | P50 | P90 | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile in profiles:
        lines.append(
            f"| {profile.display_name} | {_fmt_length(profile.mean_length)} | "
            f"{_fmt_length(profile.p50_length)} | {_fmt_length(profile.p90_length)} | "
            f"{profile.max_length} |"
        )

    bin_headers = [spec.label for spec in profiles[0].length_bins] if profiles else []
    lines.extend(
        [
            "",
            "## 3. Length Bin Distribution",
            "",
            "| Dataset | " + " | ".join(bin_headers) + " |",
            "| --- | " + " | ".join("---:" for _ in bin_headers) + " |",
        ]
    )
    for profile in profiles:
        cells = [_fmt_bin_cell(item) for item in profile.length_bins]
        lines.append(f"| {profile.display_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "**Takeaway:** Ours has a much larger absolute number of long expressions, especially in the "
            "`41-80` and `>80` token ranges.",
            "",
            "## 4. Token Long-Tail and Vocabulary Diversity",
            "",
            "| Dataset | Vocab | Top-10 Cov. | Top-50 Cov. | Top-100 Cov. | Gini | "
            "Rare-1 Vocab | Rare-5 Vocab | Rare-10 Expr. |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile in profiles:
        lines.append(
            f"| {profile.display_name} | {profile.vocabulary_size} | "
            f"{_fmt_pct(profile.top_10_coverage)} | {_fmt_pct(profile.top_50_coverage)} | "
            f"{_fmt_pct(profile.top_100_coverage)} | {_fmt_gini(profile.gini)} | "
            f"{_fmt_pct(profile.rare_1_vocab_ratio)} | {_fmt_pct(profile.rare_5_vocab_ratio)} | "
            f"{_fmt_pct(profile.rare_10_expression_ratio)} |"
        )

    lines.extend(
        [
            "",
            "## 5. Structure Combination",
            "",
            "Expression counts and shares by exact Table-6 structure-type count per expression. "
            f"**{STRUCTURE_ANY_COLUMN}** uses corpus-wide share (`count / all expressions`). "
            "The four type-count columns and **Matrix env** use **within-structure share** "
            "(`count / expressions with ≥1 structure type`). The four type-count columns are "
            "**mutually exclusive** (`1` / `2` / `3` / `≥4` distinct types; expressions with zero "
            "structure types appear only in the gap between **Any Structure ≥1** and 100%). "
            "Each cell shows `count (share%)`. "
            f"**{MATRIX_STRUCTURE_REPORT_COLUMN}** is a **non-exclusive subset tag** (Matrix is one of "
            "the eight structure types). Do **not** sum **Matrix env** with the other columns. "
            f"Triggers: {MATRIX_STRUCTURE_TRIGGER_TOKENS}.",
            "",
            f"| Dataset | {STRUCTURE_ANY_COLUMN} | "
            + " | ".join(STRUCTURE_TYPE_COUNT_LABELS)
            + f" | {MATRIX_STRUCTURE_REPORT_COLUMN} |",
            "| --- | ---: | " + " | ".join("---:" for _ in STRUCTURE_TYPE_COUNT_LABELS) + " | ---: |",
        ]
    )
    for profile in profiles:
        expression_count = profile.expression_count
        structured_count = profile.count_any_structure
        any_structure_cell = _fmt_count_share(
            structured_count,
            structured_count / expression_count if expression_count else 0.0,
        )
        structure_cells = " | ".join(
            _fmt_count_share(count, _structure_within_structured_share(count, structured_count))
            for count in profile.structure_type_count_bins
        )
        matrix_cell = _fmt_count_share(
            profile.count_matrix_structure,
            _structure_within_structured_share(profile.count_matrix_structure, structured_count),
        )
        lines.append(
            f"| {profile.display_name} | {any_structure_cell} | {structure_cells} | {matrix_cell} |"
        )

    ast_headers = " | ".join(f"depth {label}" for label in AST_DEPTH_COUNT_LABELS)
    lines.extend(
        [
            "",
            "## 6. AST Depth Distribution",
            "",
            "Expression counts and shares by PosFormer max nested level (`ast_depth`). "
            "Each depth cell shows `count (share%)`.",
            "",
            f"| Dataset | {ast_headers} | Max |",
            "| --- | " + " | ".join("---:" for _ in AST_DEPTH_COUNT_LABELS) + " | ---: |",
        ]
    )
    for profile in profiles:
        expression_count = profile.expression_count
        depth_cells = " | ".join(
            _fmt_count_share(count, count / expression_count if expression_count else 0.0)
            for count in profile.ast_depth_counts
        )
        lines.append(
            f"| {profile.display_name} | {depth_cells} | {profile.max_ast_depth} |"
        )

    lines.extend(
        [
            "",
            "## 7. Joint Difficulty Profile",
            "",
            "| Dataset | >40 Tokens | >80 Tokens | AST >=3 | >40 & AST >=2 | >80 & AST >=3 | "
            "Multi-Struct >=3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for profile in profiles:
        lines.append(
            f"| {profile.display_name} | {_fmt_int(profile.count_gt_40_tokens)} | "
            f"{_fmt_int(profile.count_gt_80_tokens)} | {_fmt_int(profile.count_ast_ge_3)} | "
            f"{_fmt_int(profile.count_gt_40_and_ast_ge_2)} | "
            f"{_fmt_int(profile.count_gt_80_and_ast_ge_3)} | "
            f"{_fmt_int(profile.count_multi_struct_ge_3)} |"
        )

    lines.extend(
        [
            "",
            "**Takeaway:** This table captures compound difficulty: expressions that are long, "
            "structurally deep, and compositionally complex.",
            "",
            "## 8. Token Taxonomy Composition",
            "",
            "Token occurrence counts and shares under the unified `LATEX_DICT` taxonomy "
            "(same categories as Table 4). Each cell shows `count (share%)`. "
            "Punctuation and layout / alignment tokens are excluded from category columns.",
            "",
            "| Dataset | Total tokens | Latin variable | Digit | Special symbol | Operator | Grouping | Structural | "
            "CJK | Other / unknown | Notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    taxonomy_shares = (
        lambda p: (
            p.latin_variable_token_ratio,
            p.digit_token_ratio,
            p.special_symbol_token_ratio,
            p.operator_token_ratio,
            p.grouping_token_ratio,
            p.structural_token_ratio,
            p.cjk_token_ratio,
            p.other_unknown_token_ratio,
        )
    )
    for profile in profiles:
        shares = taxonomy_shares(profile)
        category_cells = " | ".join(
            _fmt_count_share(count, share)
            for count, share in zip(profile.taxonomy_token_counts, shares, strict=True)
        )
        lines.append(
            f"| {profile.display_name} | {_fmt_int(profile.total_token_count)} | "
            f"{category_cells} | "
            f"{profile.notes} |"
        )

    ours = next(profile for profile in profiles if profile.display_name == "Ours")
    lines.extend(
        [
            "",
            "## 9. Summary of Advantages",
            "",
            "| Dimension | Evidence | Interpretation |",
            "| --- | --- | --- |",
            f"| Scale | {_fmt_int(ours.expression_count)} expressions | Large benchmark size |",
            f"| Effective diversity | {_fmt_int(ours.unique_expression_count)} unique normalized expressions | "
            "Not dominated by duplicates |",
            f"| Vocabulary richness | {_fmt_int(ours.vocabulary_size)} tokens | Broader token coverage |",
            f"| Long-expression coverage | {_fmt_int(ours.count_gt_80_tokens)} expressions >80 tokens | "
            "Strong long-tail difficulty |",
            f"| Mixed text-math OCR | CJK {_fmt_pct(ours.cjk_token_ratio)} + structural {_fmt_pct(ours.structural_token_ratio)} | "
            "More realistic OCR setting |",
            "| Structure complexity | fraction, superscript, subscript, radical, matrix, limit, sum | "
            "Covers diverse mathematical structures |",
            "",
        ]
    )
    return "\n".join(lines)


def write_cross_benchmark_comparison_markdown(
    profiles: list[CrossBenchmarkProfile],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_cross_benchmark_comparison_markdown(profiles), encoding="utf-8")
