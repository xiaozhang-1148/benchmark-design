"""Similar-token configuration, validation, and page-level statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.page_latex_metrics import PageLatexMetricsRow
from benchmark_design.page_level_latex.plot_style import page_ratio

DEFAULT_SIMILAR_TOKEN_GROUPS_CSV = Path(__file__).resolve().parent / "data" / "similar_token_groups.csv"

# Display labels for Figure 6-8 (match HMER Table 9 style).
GROUP_DISPLAY_LABELS: dict[str, str] = {
    "digit-letter": "1/l, 2/z, 5/s, 6/b, 9/g/q",
    "circle-like": r"0/o/\theta",
    "latin-greek": r"p/\rho, u/\mu",
    "digit-greek": r"4/\varphi",
    "operator-variable": r"x/\times",
    "relation-stroke": r"</\leq, >/\geq",
}

TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    r"\leq": (r"\leq", r"\le"),
    r"\le": (r"\le", r"\leq"),
    r"\geq": (r"\geq", r"\ge"),
    r"\ge": (r"\ge", r"\geq"),
}


@dataclass(frozen=True, slots=True)
class SimilarTokenConfigRow:
    group_id: str
    group_name: str
    subgroup_id: str
    token: str
    token_display: str
    source: str
    note: str


def load_similar_token_groups(config_path: Path | None = None) -> list[SimilarTokenConfigRow]:
    path = config_path or DEFAULT_SIMILAR_TOKEN_GROUPS_CSV
    frame = pd.read_csv(path)
    required = {"group_id", "group_name", "token", "token_display", "source", "note"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"similar_token_groups.csv missing columns: {sorted(missing)}")
    if "subgroup_id" not in frame.columns:
        frame = frame.copy()
        frame["subgroup_id"] = frame["group_name"]
    rows: list[SimilarTokenConfigRow] = []
    for rec in frame.itertuples(index=False):
        rows.append(
            SimilarTokenConfigRow(
                group_id=str(rec.group_id),
                group_name=str(rec.group_name),
                subgroup_id=str(rec.subgroup_id),
                token=str(rec.token),
                token_display=str(rec.token_display),
                source=str(rec.source),
                note="" if pd.isna(rec.note) else str(rec.note),
            )
        )
    return rows


def _match_token(configured: str, vocab: set[str]) -> tuple[str | None, str]:
    if configured in vocab:
        return configured, "exact match"
    for alias in TOKEN_ALIASES.get(configured, ()):
        if alias in vocab:
            return alias, f"alias match via {alias}"
    return None, "not found in vocabulary"


def validate_similar_tokens(
    config_rows: Sequence[SimilarTokenConfigRow],
    token_counter: Counter[str],
) -> pd.DataFrame:
    vocab = set(token_counter)
    records = []
    for row in config_rows:
        matched, note = _match_token(row.token, vocab)
        records.append(
            {
                "group_name": row.group_name,
                "configured_token": row.token,
                "matched_token": matched or "",
                "exists_in_vocab": matched is not None,
                "corpus_frequency": int(token_counter.get(matched or row.token, 0)),
                "validation_note": note,
            }
        )
    return pd.DataFrame(records)


def _page_token_counters(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
) -> dict[str, Counter[str]]:
    page_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        page_counters[row.image_id].update(row.tokens)
    return page_counters


def _page_expression_token_sets(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
) -> dict[str, list[set[str]]]:
    page_exprs: dict[str, list[set[str]]] = defaultdict(list)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        page_exprs[row.image_id].append(set(row.tokens))
    return page_exprs


def _subgroup_cooccurrence_count(page_counter: Counter[str], members: Sequence[str]) -> int:
    """Count same-page co-occurrence intensity for one lookalike subgroup.

    Only counts when at least two distinct subgroup members appear on the page.
    The contribution equals the sum of frequencies of present members
    (each appearance of those co-occurring characters is counted).
    """
    present = [tok for tok in members if page_counter.get(tok, 0) > 0]
    if len(present) < 2:
        return 0
    return int(sum(page_counter[tok] for tok in present))


def compute_similar_token_stats(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    token_counter: Counter[str],
    *,
    config_rows: Sequence[SimilarTokenConfigRow] | None = None,
) -> dict[str, pd.DataFrame]:
    config_rows = list(config_rows or load_similar_token_groups())
    validation = validate_similar_tokens(config_rows, token_counter)
    valid_matches = {
        (str(r.group_name), str(r.configured_token)): str(r.matched_token)
        for r in validation.itertuples(index=False)
        if bool(r.exists_in_vocab) and str(r.matched_token)
    }

    page_counters = _page_token_counters(expression_rows)
    page_exprs = _page_expression_token_sets(expression_rows)
    total_pages = len(page_rows)
    all_image_ids = [page.image_id for page in page_rows]

    # subgroup_id -> matched tokens
    subgroups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in config_rows:
        matched = valid_matches.get((row.group_name, row.token))
        if not matched:
            continue
        key = (row.group_name, row.subgroup_id)
        if matched not in subgroups[key]:
            subgroups[key].append(matched)

    # Detail per configured token.
    detail_records = []
    for row in config_rows:
        matched = valid_matches.get((row.group_name, row.token))
        if not matched:
            continue
        expr_count = 0
        pages: set[str] = set()
        for image_id, expr_sets in page_exprs.items():
            hit = False
            for tokens in expr_sets:
                if matched in tokens:
                    expr_count += 1
                    hit = True
            if hit:
                pages.add(image_id)
        detail_records.append(
            {
                "group_name": row.group_name,
                "subgroup_id": row.subgroup_id,
                "token": matched,
                "token_occurrence_count": int(token_counter[matched]),
                "expression_count": expr_count,
                "page_count": len(pages),
                "page_ratio": page_ratio(len(pages), total_pages),
            }
        )
    detail = pd.DataFrame(detail_records)

    # Group summary: only same-page subgroup co-occurrence events.
    group_order = list(dict.fromkeys(row.group_name for row in config_rows))
    group_records = []
    pair_records = []
    for group_name in group_order:
        group_subgroups = [(sid, toks) for (g, sid), toks in subgroups.items() if g == group_name]
        members = []
        for _, toks in group_subgroups:
            for tok in toks:
                if tok not in members:
                    members.append(tok)

        co_event_total = 0
        co_pages: set[str] = set()
        page_event_counts: list[int] = []
        for image_id in all_image_ids:
            counter = page_counters.get(image_id, Counter())
            page_events = 0
            for _, toks in group_subgroups:
                page_events += _subgroup_cooccurrence_count(counter, toks)
            if page_events > 0:
                co_pages.add(image_id)
                co_event_total += page_events
                page_event_counts.append(page_events)

        group_records.append(
            {
                "group_name": group_name,
                "group_display": GROUP_DISPLAY_LABELS.get(group_name, group_name),
                "member_token_count": len(members),
                "token_occurrence_count": sum(token_counter[tok] for tok in members),
                "cooccurrence_event_count": co_event_total,
                "cooccurrence_page_count": len(co_pages),
                "cooccurrence_page_ratio": page_ratio(len(co_pages), total_pages),
                "cooccurrence_event_ratio": page_ratio(co_event_total, total_pages),
            }
        )

        # Pairwise co-occurrence within each subgroup (for appendix).
        for _, toks in group_subgroups:
            for i, tok_a in enumerate(toks):
                for tok_b in toks[i + 1 :]:
                    shared_pages = 0
                    shared_events = 0
                    for image_id in all_image_ids:
                        counter = page_counters.get(image_id, Counter())
                        if counter.get(tok_a, 0) > 0 and counter.get(tok_b, 0) > 0:
                            shared_pages += 1
                            shared_events += counter[tok_a] + counter[tok_b]
                    pair_records.append(
                        {
                            "group_name": group_name,
                            "token_a": tok_a,
                            "token_b": tok_b,
                            "cooccurrence_page_count": shared_pages,
                            "cooccurrence_event_count": shared_events,
                            "all_page_ratio": page_ratio(shared_pages, total_pages),
                        }
                    )

    return {
        "validation": validation,
        "detail": detail,
        "group_summary": pd.DataFrame(group_records),
        "pair_cooccurrence": pd.DataFrame(pair_records),
    }


def write_similar_token_samples_stub(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    *,
    config_rows: Sequence[SimilarTokenConfigRow],
    token_counter: Counter[str],
    output_path: Path,
    samples_per_token: int = 3,
    random_seed: int = 42,
) -> Path:
    """Index expression-level sample candidates (no token bbox available)."""
    import random

    validation = validate_similar_tokens(config_rows, token_counter)
    matched = {
        (str(r.group_name), str(r.configured_token)): str(r.matched_token)
        for r in validation.itertuples(index=False)
        if bool(r.exists_in_vocab) and str(r.matched_token)
    }
    rng = random.Random(random_seed)
    by_token: dict[tuple[str, str], list[ExpressionLatexMetricsRow]] = defaultdict(list)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        token_set = set(row.tokens)
        for (group_name, configured), matched_token in matched.items():
            if matched_token in token_set:
                by_token[(group_name, matched_token)].append(row)

    records = []
    for (group_name, token), candidates in sorted(by_token.items()):
        by_page: dict[str, list[ExpressionLatexMetricsRow]] = defaultdict(list)
        for row in candidates:
            by_page[row.image_id].append(row)
        pages = list(by_page.keys())
        rng.shuffle(pages)
        selected: list[ExpressionLatexMetricsRow] = []
        for page in pages:
            selected.append(rng.choice(by_page[page]))
            if len(selected) >= samples_per_token:
                break
        for row in selected:
            records.append(
                {
                    "group_name": group_name,
                    "token": token,
                    "image_id": row.image_id,
                    "expression_id": f"{row.image_id}:{row.line_id}",
                    "sample_path": "",
                    "sample_level": "expression",
                    "random_seed": random_seed,
                }
            )
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
