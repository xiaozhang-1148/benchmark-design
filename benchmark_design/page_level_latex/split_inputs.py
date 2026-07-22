"""Export frozen split-input tables from Chapter-6 page LaTeX metrics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark_design.config.block_level import BLOCK_LEVEL_IMAGE_EXTENSIONS
from benchmark_design.page_level_latex.expression_latex_metrics import (
    ExpressionLatexMetricsRow,
    apply_rare10_pass2,
    build_expression_metrics_pass1,
    read_expression_latex_metrics_csv,
)
from benchmark_design.page_level_latex.latex_protocol import STRUCTURE_TYPE_ORDER
from benchmark_design.page_level_latex.rare8 import (
    compute_rare8_page_stats,
    compute_rare8_token_set,
    summarize_rare8,
)
from benchmark_design.page_level_latex.loader import iter_benchmark_json_paths
from benchmark_design.page_level_latex.page_latex_metrics import (
    PageLatexMetricsRow,
    aggregate_page_latex_metrics,
    read_page_latex_metrics_csv,
)
from benchmark_design.page_level_latex.similar_tokens import (
    _page_token_counters,
    _subgroup_cooccurrence_count,
    load_similar_token_groups,
    validate_similar_tokens,
)

# group_name -> page_hmer_features column
SIMILAR_GROUP_FLAG_COLUMNS: dict[str, str] = {
    "digit-letter": "has_digit_letter_pair",
    "circle-like": "has_circle_like_pair",
    "latin-greek": "has_latin_greek_pair",
    "digit-greek": "has_greek_variant_pair",
    "operator-variable": "has_operator_variable_pair",
    "relation-stroke": "has_relation_stroke_pair",
}


@dataclass(frozen=True, slots=True)
class SplitInputsResult:
    output_dir: Path
    page_count: int
    manifest: dict[str, str]


def _resolve_image_path(image_name: str, input_dir: Path) -> Path | None:
    stem = Path(image_name).stem
    for ext in BLOCK_LEVEL_IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    candidate = input_dir / image_name
    if candidate.is_file():
        return candidate
    return None


def _build_dataset_manifest(
    input_dir: Path,
    page_ids: Sequence[str],
    *,
    annotation_by_page: dict[str, Path],
) -> pd.DataFrame:
    records = []
    seen: set[str] = set()
    for page_id in page_ids:
        if page_id in seen:
            raise ValueError(f"duplicate page_id in dataset: {page_id}")
        seen.add(page_id)
        annotation_path = annotation_by_page.get(page_id)
        if annotation_path is None or not annotation_path.is_file():
            raise FileNotFoundError(f"annotation missing for page_id={page_id}")
        with annotation_path.open(encoding="utf-8") as handle:
            page = json.load(handle)
        image_name = str(page.get("image_name", annotation_path.name.removesuffix(".json")))
        image_path = _resolve_image_path(image_name, input_dir)
        if image_path is None:
            raise FileNotFoundError(f"image missing for page_id={page_id} image_name={image_name}")
        records.append(
            {
                "page_id": page_id,
                "image_path": str(image_path.resolve()),
                "annotation_path": str(annotation_path.resolve()),
            }
        )
    if len(records) != len(page_ids):
        raise ValueError(f"page total mismatch: expected {len(page_ids)}, got {len(records)}")
    return pd.DataFrame.from_records(records)


def _page_structure_flags(page: PageLatexMetricsRow) -> dict[str, int]:
    present = {
        name: int(getattr(page, f"{name}_expression_count") > 0) for name in STRUCTURE_TYPE_ORDER
    }
    structure_type_count = sum(present.values())
    if structure_type_count != page.distinct_structure_type_count:
        # Prefer presence sum as the official split feature; still hard-check consistency.
        raise ValueError(
            f"{page.image_id}: structure_type_count mismatch "
            f"presence_sum={structure_type_count} distinct={page.distinct_structure_type_count}"
        )
    return {
        **{f"has_{name}": present[name] for name in STRUCTURE_TYPE_ORDER},
        "structure_type_count": structure_type_count,
    }


def _similar_group_page_flags(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_ids: Sequence[str],
    token_counter: Counter[str],
) -> dict[str, dict[str, int]]:
    config_rows = load_similar_token_groups()
    validation = validate_similar_tokens(config_rows, token_counter)
    valid_matches = {
        (str(r.group_name), str(r.configured_token)): str(r.matched_token)
        for r in validation.itertuples(index=False)
        if bool(r.exists_in_vocab) and str(r.matched_token)
    }
    subgroups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in config_rows:
        matched = valid_matches.get((row.group_name, row.token))
        if not matched:
            continue
        key = (row.group_name, row.subgroup_id)
        if matched not in subgroups[key]:
            subgroups[key].append(matched)

    page_counters = _page_token_counters(expression_rows)
    flags_by_page: dict[str, dict[str, int]] = {
        page_id: {col: 0 for col in SIMILAR_GROUP_FLAG_COLUMNS.values()} for page_id in page_ids
    }
    for group_name, column in SIMILAR_GROUP_FLAG_COLUMNS.items():
        group_subgroups = [(sid, toks) for (g, sid), toks in subgroups.items() if g == group_name]
        for page_id in page_ids:
            counter = page_counters.get(page_id, Counter())
            hit = any(_subgroup_cooccurrence_count(counter, toks) > 0 for _, toks in group_subgroups)
            flags_by_page[page_id][column] = int(hit)
    return flags_by_page


def _page_expr_depth_presence(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
) -> dict[str, dict[int, int]]:
    """page_id -> depth -> 1 if page has at least one valid expression at that AST depth."""
    presence: dict[str, dict[int, int]] = defaultdict(dict)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        depth = int(row.ast_depth)
        presence[row.image_id][depth] = 1
    return presence


def _rare8_page_stats(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    rare8_tokens: set[str],
) -> dict[str, tuple[int, int]]:
    return compute_rare8_page_stats(expression_rows, rare8_tokens)


def build_page_hmer_features(
    page_rows: Sequence[PageLatexMetricsRow],
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    token_counter: Counter[str],
) -> pd.DataFrame:
    rare8_tokens = compute_rare8_token_set(token_counter)
    rare8_by_page = _rare8_page_stats(expression_rows, rare8_tokens)
    depth_presence = _page_expr_depth_presence(expression_rows)
    page_ids = [page.image_id for page in page_rows]
    similar_flags = _similar_group_page_flags(expression_rows, page_ids, token_counter)

    records = []
    for page in page_rows:
        structure = _page_structure_flags(page)
        has_rare8, rare8_count = rare8_by_page.get(page.image_id, (0, 0))
        depths = depth_presence.get(page.image_id, {})
        record = {
            "page_id": page.image_id,
            "ast_tree_count": page.ast_tree_count,
            "total_ast_node_count": page.total_ast_node_count,
            "max_ast_depth": page.max_ast_depth,
            **structure,
            "has_rare8": has_rare8,
            "rare8_token_count": rare8_count,
            **{f"has_expr_depth_{depth}": int(depths.get(depth, 0)) for depth in range(6)},
            **similar_flags[page.image_id],
        }
        # Hard check structure identity.
        expected = sum(record[f"has_{name}"] for name in STRUCTURE_TYPE_ORDER)
        if record["structure_type_count"] != expected:
            raise ValueError(
                f"{page.image_id}: structure_type_count={record['structure_type_count']} "
                f"!= sum(has_*)={expected}"
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_page_token_counts(expression_rows: Sequence[ExpressionLatexMetricsRow]) -> pd.DataFrame:
    page_counters = _page_token_counters(expression_rows)
    records = []
    for page_id in sorted(page_counters):
        for token, count in sorted(page_counters[page_id].items()):
            records.append({"page_id": page_id, "token": token, "count": int(count)})
    return pd.DataFrame.from_records(records, columns=["page_id", "token", "count"])


def _load_cached_latex_metrics(
    page_level_latex_output: Path,
) -> tuple[list[PageLatexMetricsRow], list[ExpressionLatexMetricsRow], Counter[str]]:
    metrics_dir = page_level_latex_output / "metrics"
    expression_path = metrics_dir / "expression_latex_metrics.csv"
    page_path = metrics_dir / "page_latex_metrics.csv"
    if not expression_path.is_file() or not page_path.is_file():
        raise FileNotFoundError(
            "page_level_HMER metrics missing; expected "
            f"{expression_path} and {page_path}"
        )
    expression_rows = list(read_expression_latex_metrics_csv(expression_path))
    page_rows = read_page_latex_metrics_csv(page_path)
    token_counter: Counter[str] = Counter()
    for row in expression_rows:
        if row.valid_for_latex:
            token_counter.update(row.tokens)
    return page_rows, expression_rows, token_counter


def prepare_split_inputs(
    input_dir: Path,
    output_dir: Path,
    *,
    dataset_version: str = "",
    workers: int | None = None,
    show_progress: bool = True,
    page_level_latex_output: Path | None = None,
) -> SplitInputsResult:
    """Build dataset_manifest / page_hmer_features / page_token_counts.

    When ``page_level_latex_output`` points at a completed ``page_level_HMER/``
    export, reuse its metrics CSVs instead of re-parsing benchmark JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = iter_benchmark_json_paths(input_dir)
    if not json_paths:
        raise FileNotFoundError(f"no benchmark JSON pages under {input_dir}")

    annotation_by_page: dict[str, Path] = {}
    for path in json_paths:
        with path.open(encoding="utf-8") as handle:
            page = json.load(handle)
        image_name = str(page.get("image_name", path.name.removesuffix(".json")))
        page_id = Path(image_name).stem
        if page_id in annotation_by_page:
            raise ValueError(f"duplicate page assignment for page_id={page_id}")
        annotation_by_page[page_id] = path

    if page_level_latex_output is not None:
        page_rows, expression_rows, token_counter = _load_cached_latex_metrics(page_level_latex_output)
        page_ids = sorted(page.image_id for page in page_rows)
        missing_annotations = sorted(set(page_ids) - set(annotation_by_page))
        if missing_annotations:
            raise ValueError(
                f"cached page_level_HMER pages missing benchmark annotations: {missing_annotations[:10]}"
            )
    else:
        pass1 = build_expression_metrics_pass1(
            input_dir,
            show_progress=show_progress,
            workers=workers,
        )
        expression_rows = list(apply_rare10_pass2(pass1))
        page_ids = sorted(annotation_by_page)
        page_rows = aggregate_page_latex_metrics(expression_rows, all_image_ids=page_ids)
        token_counter = pass1.token_counter

    if len(page_rows) != len(page_ids):
        raise ValueError(f"page count mismatch: rows={len(page_rows)} ids={len(page_ids)}")

    manifest_df = _build_dataset_manifest(input_dir, page_ids, annotation_by_page=annotation_by_page)
    features_df = build_page_hmer_features(page_rows, expression_rows, token_counter)
    tokens_df = build_page_token_counts(expression_rows)
    rare8_tokens = compute_rare8_token_set(token_counter)
    rare8_summary = summarize_rare8(expression_rows, rare8_tokens, total_pages=len(page_rows))

    # Merge consistency: every page has features; no orphans.
    if set(manifest_df["page_id"]) != set(features_df["page_id"]):
        raise ValueError("manifest and features page_id sets differ")
    token_pages = set(tokens_df["page_id"]) if not tokens_df.empty else set()
    empty_pages = set(manifest_df["page_id"]) - token_pages
    # Empty OCR pages are allowed only if ast_tree_count == 0.
    zero_expr = set(features_df.loc[features_df["ast_tree_count"] == 0, "page_id"])
    unexpected_empty = empty_pages - zero_expr
    if unexpected_empty:
        raise ValueError(f"pages with expressions but no token rows: {sorted(unexpected_empty)[:10]}")

    manifest_path = output_dir / "dataset_manifest.csv"
    features_path = output_dir / "page_hmer_features.csv"
    tokens_path = output_dir / "page_token_counts.csv"
    meta_path = output_dir / "dataset_meta.json"

    manifest_df.to_csv(manifest_path, index=False)
    features_df.to_csv(features_path, index=False)
    tokens_df.to_csv(tokens_path, index=False)

    version = dataset_version or input_dir.name
    meta = {
        "dataset_version": version,
        "input_dir": str(input_dir.resolve()),
        "page_count": len(manifest_df),
        "ast_tree_count": int(features_df["ast_tree_count"].sum()),
        "token_instance_count": int(tokens_df["count"].sum()) if not tokens_df.empty else 0,
        "metrics_source": (
            str(page_level_latex_output.resolve())
            if page_level_latex_output is not None
            else "recomputed_from_benchmark_json"
        ),
        "rare8_summary": {
            "rare_vocab_count": rare8_summary.rare_vocab_count,
            "token_instance_count": rare8_summary.token_instance_count,
            "expression_count": rare8_summary.expression_count,
            "page_count": rare8_summary.page_count,
            "page_ratio": rare8_summary.page_ratio,
        },
        "files": {
            "dataset_manifest": manifest_path.name,
            "page_hmer_features": features_path.name,
            "page_token_counts": tokens_path.name,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return SplitInputsResult(
        output_dir=output_dir,
        page_count=len(manifest_df),
        manifest={
            "dataset_manifest": manifest_path.name,
            "page_hmer_features": features_path.name,
            "page_token_counts": tokens_path.name,
            "dataset_meta": meta_path.name,
        },
    )
