"""Export tokenizer vocabulary, taxonomy maps, and documentation."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from benchmark_design import commands as cmd
from benchmark_design.ocr.expression_features import ExpressionFeatures, resolve_token_counter
from benchmark_design.ocr.token_taxonomy import (
    LAYOUT_ALIGNMENT_TOKENS,
    PUNCTUATION_TOKENS,
    TOKEN_CATEGORY_ORDER,
    TokenCategory,
    classify_token,
)
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

CURATED_EXAMPLES: tuple[tuple[str, str], ...] = (
    (r"x", "single variable"),
    (r"x^{2}", "superscript"),
    (r"x_{1}", "subscript"),
    (r"\frac{1}{n}", "fraction"),
    (r"\sqrt{S}", "radical"),
    (r"4^{x-\frac{1}{4}}", "nested superscript fraction"),
    (r"\lim \limits _ { x \rightarrow 0 } f(x)", "limit with subscript"),
    (r"\begin{cases} a \\ b \end{cases}", "matrix environment"),
    (r"解 : ( 1 ) 依 题", "CJK mixed"),
    (r"Ay^{3}_{1}+\frac{y^{\beta_{1}}_{2}B}{C}", "multi-structure"),
)

_VOCAB_GROUP_LABELS: dict[str, str] = {
    **{token: "greek_lower" for token in cmd._GREEK_LOWER},
    **{token: "greek_upper" for token in cmd._GREEK_UPPER},
    **{token: "fraction_binomial" for token in cmd._FRACTION_BINOMIAL},
    **{token: "environment" for token in cmd._ENVIRONMENTS},
    **{token: "operator" for token in cmd._OPERATORS},
    **{token: "delimiter" for token in cmd._DELIMITERS},
    **{token: "custom_symbol" for token in cmd._CUSTOM_SYMBOLS},
}


def _vocab_group(token: str) -> str:
    return _VOCAB_GROUP_LABELS.get(token, "latex_dict")


def write_tokenizer_rules(output_path: Path) -> None:
    lines = [
        "# LaTeX Tokenizer Rules",
        "",
        "This project tokenizes normalized LaTeX OCR strings with a greedy longest-match",
        "dictionary (`LATEX_DICT` from `benchmark_design/commands.py`).",
        "",
        "## Algorithm",
        "",
        "1. Skip whitespace characters.",
        "2. At each position, try the longest substring (up to 32 chars) that exists in `LATEX_DICT`.",
        "3. If no dictionary entry matches, emit the current single character as its own token.",
        "4. Repeat until the string is consumed.",
        "",
        "## Implications",
        "",
        "- Multi-character LaTeX commands (e.g. `\\frac`, `\\begin`) are kept atomic when listed in the dictionary.",
        "- Digits, Latin letters, and CJK characters fall back to single-character tokens when unmatched.",
        "- Plain punctuation (`,`, `:`, `.`, `;`, `!`, `?`, `'`) and layout tokens (`\\\\`, `&`) are classified",
        "  explicitly; see `taxonomy_rules.md`.",
        "- Spaces in the source string are not emitted as tokens.",
        "",
        "## Taxonomy classification",
        "",
        "After tokenization, each token is assigned exactly one category using mutually exclusive sets",
        "derived from `commands.py` groups plus explicit punctuation / layout / CJK rules.",
        "Priority follows `TOKEN_CATEGORY_ORDER` in `token_taxonomy.py` (documented in `taxonomy_rules.md`).",
        "",
        "Only tokens that are truly unclassifiable or suspected OCR noise are labeled",
        "`other / unknown tokens`.",
        "",
        "## Documentation",
        "",
        "- [`latex_dictionary.md`](latex_dictionary.md) — dictionary source and vocab groups",
        "- [`taxonomy_rules.md`](taxonomy_rules.md) — category priority and examples",
        "- [`metric_definitions.md`](metric_definitions.md) — benchmark metric denominators",
        "- [`data_schema.md`](data_schema.md) — `expression_level_statistics.csv` column reference",
        "- [`known_limitations.md`](known_limitations.md) — OCR artifact handling notes",
        "",
        "## Reproducibility files",
        "",
        "- `latex_vocab.csv`: all dictionary entries",
        "- `token_taxonomy_map.csv`: token → category for dictionary + corpus tokens (under `resources/`)",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_vocab_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "group", "category"])
        for token in sorted(build_latex_vocab()):
            writer.writerow([token, _vocab_group(token), classify_token(token).value])


def write_token_taxonomy_map_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "category", "in_latex_dict", "corpus_count"])
        for token in sorted(build_latex_vocab()):
            writer.writerow([token, classify_token(token).value, "true", counter.get(token, 0)])
        for token, count in sorted(counter.items()):
            if token in build_latex_vocab():
                continue
            writer.writerow([token, classify_token(token).value, "false", count])


def write_taxonomy_unknown_tokens_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    counter = corpus_token_counter(features)
    total = sum(counter.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "count", "share", "category"])
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if classify_token(token) is not TokenCategory.OTHER:
                continue
            writer.writerow([token, count, f"{count / total:.6f}" if total else "0", TokenCategory.OTHER.value])


def write_unknown_token_ratio_by_token_csv(features: list[ExpressionFeatures], output_path: Path) -> None:
    counter = corpus_token_counter(features)
    total = sum(counter.values())
    expr_hits = Counter[str]()
    for feature in features:
        for token in set(feature.token_sequence):
            if classify_token(token) is TokenCategory.OTHER:
                expr_hits[token] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "count", "token_share", "expression_hit_count", "expression_hit_ratio"])
        expression_count = len(features)
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if classify_token(token) is not TokenCategory.OTHER:
                continue
            writer.writerow(
                [
                    token,
                    count,
                    f"{count / total:.6f}" if total else "0",
                    expr_hits[token],
                    f"{expr_hits[token] / expression_count:.6f}" if expression_count else "0",
                ]
            )


def write_tokenization_examples_md(features: list[ExpressionFeatures], output_path: Path) -> None:
    vocab = build_latex_vocab()
    lines = [
        "# Tokenization Examples",
        "",
        "## Curated examples",
        "",
        "| Note | LaTeX | Tokens |",
        "| --- | --- | --- |",
    ]
    for latex, note in CURATED_EXAMPLES:
        tokens = " ".join(tokenize_greedy(latex, vocab))
        lines.append(f"| {note} | `{latex}` | `{tokens}` |")

    lines.extend(["", "## Corpus samples", ""])
    step = max(1, len(features) // 5)
    for feature in features[::step][:5]:
        tokens = " ".join(feature.token_sequence)
        lines.append(f"- `{feature.normalized_latex}` → `{tokens}`")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_taxonomy_examples_md(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> None:
    counter = resolve_token_counter(features, token_counter)
    lines = ["# Token Taxonomy Examples", ""]
    for category in TOKEN_CATEGORY_ORDER:
        tokens = [token for token in counter if classify_token(token) is category]
        top_tokens = sorted(tokens, key=lambda token: counter[token], reverse=True)[:20]
        lines.append(f"## {category.value}")
        lines.append("")
        lines.append("Top tokens: " + ", ".join(f"`{token}`" for token in top_tokens))
        example_feature = next(
            (feature for feature in features if any(classify_token(t) is category for t in feature.token_sequence)),
            None,
        )
        if example_feature:
            lines.append(f"Example expression: `{example_feature.normalized_latex}`")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _vocab_group_counts() -> dict[str, int]:
    counts: dict[str, int] = Counter()
    for token in build_latex_vocab():
        counts[_vocab_group(token)] += 1
    return dict(counts)


def write_latex_dictionary_md(output_path: Path) -> None:
    vocab = build_latex_vocab()
    group_counts = _vocab_group_counts()
    lines = [
        "# LaTeX Dictionary",
        "",
        "## Source",
        "",
        "The greedy tokenizer vocabulary is `LATEX_DICT` in `benchmark_design/commands.py`.",
        "It is built by flattening and deduplicating all command groups listed in that module",
        "(Greek letters, operators, delimiters, environments, spacing commands, OCR-specific",
        "custom symbols such as `\\delete` / `\\insertion`, and related LaTeX fragments).",
        "Entries are sorted **longest first** so greedy matching prefers multi-character commands.",
        "",
        f"**Total dictionary size:** {len(vocab):,} unique tokens",
        "",
        "## Dictionary groups (`latex_vocab.csv` column `group`)",
        "",
        "The CSV at [`../../resources/latex_vocab.csv`](../../resources/latex_vocab.csv) lists every dictionary",
        "entry with its source group and post-tokenization taxonomy category.",
        "",
        "| Group | Count | Meaning |",
        "| --- | ---: | --- |",
        ("| `greek_lower` | "
         f"{group_counts.get('greek_lower', 0):,} | Lower-case Greek letter commands (`\\alpha`, …) |"),
        ("| `greek_upper` | "
         f"{group_counts.get('greek_upper', 0):,} | Upper-case Greek letter commands (`\\Gamma`, …) |"),
        ("| `fraction_binomial` | "
         f"{group_counts.get('fraction_binomial', 0):,} | Fraction / binomial structure triggers |"),
        ("| `environment` | "
         f"{group_counts.get('environment', 0):,} | Env. / cases / align-like environment names |"),
        ("| `operator` | "
         f"{group_counts.get('operator', 0):,} | Binary operators, relations, large operators |"),
        ("| `delimiter` | "
         f"{group_counts.get('delimiter', 0):,} | Delimiters and bracket-like commands |"),
        ("| `custom_symbol` | "
         f"{group_counts.get('custom_symbol', 0):,} | Non-standard OCR pipeline symbols |"),
        ("| `latex_dict` | "
         f"{group_counts.get('latex_dict', 0):,} | Other grouped LaTeX commands (arrows, accents, spacing, …) |"),
        "",
        "## Taxonomy categories (`latex_vocab.csv` column `category`)",
        "",
        "Dictionary entries are also mapped to mutually exclusive **taxonomy categories** used in",
        "Table 4 and expression-level `token_type_counts`. See [`taxonomy_rules.md`](taxonomy_rules.md)",
        "for priority rules. Categories include latin/digit/CJK singles, operators, grouping,",
        "structural, special symbols, punctuation, layout/alignment, and other/unknown.",
        "",
        "## Related files",
        "",
        "- [`../../resources/latex_vocab.csv`](../../resources/latex_vocab.csv) — full dictionary export",
        "- [`../../resources/token_taxonomy_map.csv`](../../resources/token_taxonomy_map.csv) — dictionary + corpus singles",
        "- [`../tables/unclassified_token_summary.csv`](../tables/unclassified_token_summary.csv) — unclassified corpus tokens",
        "- [`tokenizer_rules.md`](tokenizer_rules.md) — greedy tokenization algorithm",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_taxonomy_rules_md(output_path: Path) -> None:
    punct = ", ".join(f"`{token}`" for token in sorted(PUNCTUATION_TOKENS))
    layout = ", ".join(f"`{token}`" for token in sorted(LAYOUT_ALIGNMENT_TOKENS))
    priority_lines = [f"{index}. `{category.value}`" for index, category in enumerate(TOKEN_CATEGORY_ORDER, start=1)]
    lines = [
        "# Token Taxonomy Rules",
        "",
        "Each token receives **exactly one** category. Classification is deterministic and",
        "mutually exclusive. Priority is fixed; the first matching rule wins.",
        "",
        "## Priority order",
        "",
        *priority_lines,
        "",
        "## Category definitions",
        "",
        "### Core mathematical tokens",
        "",
        "- **latin variable tokens** — single ASCII letters (`a`–`z`, `A`–`Z`) not matched as dictionary commands.",
        "- **digit tokens** — single ASCII digits `0`–`9`.",
        "- **special symbol tokens** — Greek commands, arrows, misc symbols, spacing commands such as `\\therefore`,",
        "  and other dictionary entries classified as symbols (excluding structural / operator / grouping sets).",
        "- **operator tokens** — `+`, `-`, `=`, `<`, `>`, `*`, `/`, and relation / large-operator commands.",
        "- **grouping tokens** — parentheses, braces, brackets, and `\\left` / `\\right` delimiters.",
        "- **structural tokens** — `^`, `_`, fractions, roots, sums, integrals, limits, matrices / environments.",
        "- **CJK tokens** — single-character Unified CJK ideographs detected by Unicode range.",
        "",
        "### Punctuation tokens",
        "",
        f"Fixed set: {punct}.",
        "",
        "These are **plain-text punctuation** emitted as single-character fallback tokens when they",
        "are not absorbed into a longer dictionary match. They include sentence / clause punctuation",
        "common in mixed Chinese solution text (`解 : …`) and numeric lists.",
        "",
        "They are **not** unknown noise: they are expected OCR punctuation and counted separately from",
        "`other / unknown tokens`.",
        "",
        "### Layout / alignment tokens",
        "",
        f"Fixed set: {layout}.",
        "",
        "- `\\\\` — row break / line break in matrices, cases, and aligned environments.",
        "- `&` — column alignment separator in tabular / matrix layouts.",
        "",
        "These tokens encode **layout structure**, not mathematical operators. They are classified",
        "before falling through to unknown.",
        "",
        "### Other / unknown tokens",
        "",
        "Any token that matches **none** of the above rules. This category is reserved for:",
        "",
        "- characters or fragments that cannot be explained by the dictionary or taxonomy sets,",
        "- suspected OCR noise or corruption,",
        "- rare literal symbols not yet mapped (e.g. unexpected Unicode punctuation).",
        "",
        "**Not unknown:** dictionary commands, CJK characters, standard punctuation above, or `\\\\` / `&`.",
        "",
        "## Worked examples",
        "",
        "| Token | Category | Reason |",
        "| --- | --- | --- |",
        "| `:` | punctuation tokens | explicit punctuation set |",
        "| `\\\\` | layout / alignment tokens | matrix row break |",
        "| `&` | layout / alignment tokens | alignment column separator |",
        "| `\\frac` | structural tokens | fraction trigger in dictionary |",
        "| `解` | CJK tokens | CJK Unicode range |",
        "| `\\delete` | special symbol tokens | OCR custom symbol in dictionary |",
        "",
        "## Related tables",
        "",
        "- [`../../resources/token_taxonomy_map.csv`](../../resources/token_taxonomy_map.csv)",
        "- [`../tables/unclassified_token_summary.csv`](../tables/unclassified_token_summary.csv)",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_definitions_md(output_path: Path) -> None:
    lines = [
        "# Metric Definitions",
        "",
        "All metrics below use the **same tokenization pass** (`LATEX_DICT` greedy tokenizer) unless",
        "noted. Expression counts refer to one OCR line / one benchmark expression record.",
        "",
        "## Duplicate rate",
        "",
        "**Definition:** two benchmark expression records are duplicates iff their **full**",
        "``normalized_latex`` strings are exactly equal (character-for-character). Only",
        "whole-expression equality counts; partial or token-level matching is never used.",
        "",
        "**Formula:** `(expression_count - unique_normalized_latex_count) / expression_count`",
        "",
        "- **Numerator:** redundant expressions — records whose ``normalized_latex`` exactly matches at least one other record.",
        "- **Denominator:** total expression count.",
        "- **Normalization:** ``normalized_latex = raw OCR LaTeX stripped of leading/trailing whitespace`` (see ``duplicates.py``).",
        "",
        "## Unknown token ratio (`other / unknown token ratio`)",
        "",
        "**Formula:** `count(tokens classified as other / unknown) / total_token_count`",
        "",
        "- **Numerator:** token occurrences labeled `other / unknown tokens` in taxonomy (see `taxonomy_rules.md`).",
        "- **Denominator:** all tokens in the corpus after tokenization.",
        "- Punctuation and layout tokens are **excluded** from this ratio.",
        "",
        "## Vocab coverage (cross-benchmark)",
        "",
        "**Formula:** `count(non-unknown tokens) / total_token_count`",
        "",
        "Equivalently: `1 - unknown_token_ratio` when unknown is defined strictly as `other / unknown tokens`.",
        "Measures how much of the corpus token mass falls into known taxonomy categories.",
        "",
        "## AST depth (PosFormer max nested level)",
        "",
        "Computed by PosFormer position-forest encoding (`position_forest.py`) on each expression's token list.",
        "",
        "- **AST depth** — maximum nested structural level returned by the encoder for that expression.",
        "- **Corpus summaries** — mean / max / histogram over expressions; see `tables/ast_statistics_*`.",
        "- **Parse status** — separate brace / substructure validation (`ok`, `unbalanced_braces`,",
        "  `incomplete_substructure`); encoding still runs but low parse confidence is flagged.",
        "",
        "## Structure co-occurrence",
        "",
        "Defined over the eight Table-6 structure types (fraction, superscript, subscript, radical, sum,",
        "integral, limit, matrix). For each expression, collect the set of structure types present.",
        "",
        "- **Co-occurrence count** — for types A and B, number of expressions where **both** appear.",
        "- **Co-occurrence ratio** in `tables/appendix/structure_cooccurrence_matrix.csv`:",
        "  `cooccurrence_count / expression_count`.",
        "- Diagonal entries count expressions containing that type at least once.",
        "",
        "## Structure type distribution (Table 6)",
        "",
        "- **Expr. ratio** — expressions containing the structure / all expressions.",
        "- **Occ. ratio** — structure trigger token occurrences / all structural token occurrences.",
        "",
        "## Token long-tail (Table 5)",
        "",
        "- **Gini** — inequality over token frequency counts.",
        "- **Top-k coverage** — sum of frequencies of k most common tokens / total tokens.",
        "- **Rare vocab ratio** — fraction of vocabulary items with corpus frequency ≤ threshold (1 / 5 / 10).",
        "",
        "## Length distribution",
        "",
        "Per-expression **token length** = number of tokens after greedy tokenization.",
        "Percentiles (P50, P90, P95, P99) use linear interpolation on sorted lengths.",
        "",
        "## Related outputs",
        "",
        "- `tables/` — per-metric core CSV summaries (see `tables/README.md`)",
        "- `cross_benchmark/` — cross-dataset comparison CSVs",
        "- `details/expression_level_statistics.csv` — per-expression evidence (see `data_schema.md`)",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_schema_md(output_path: Path) -> None:
    lines = [
        "# Data Schema: `expression_level_statistics.csv`",
        "",
        "Primary audit table under `details/`. One row per expression after tokenization and feature extraction.",
        "",
        "| Column | Type | Description |",
        "| --- | --- | --- |",
        "| `expression_id` | string | Stable ID `{dataset}:{source_key}` (e.g. image line or caption id). |",
        "| `dataset` | string | Dataset name (`ours`, `CROHME2014`, …). |",
        "| `source_file` | string | Relative path to source JSON / caption / shard file. |",
        "| `line_id` | string | Line or block index within the source file. |",
        "| `normalized_latex` | string | Canonical duplicate key: OCR LaTeX with leading/trailing whitespace stripped. |",
        "| `token_sequence` | string | Space-joined greedy token sequence (CSV-quoted when needed). |",
        "| `token_length` | int | Number of tokens in `token_sequence`. |",
        "| `length_bin` | string | Fixed bin label (1–10, 11–20, 21–40, 41–80, >80 tokens). |",
        "| `is_duplicate` | 0/1 | 1 if another record shares the exact same full `normalized_latex`. |",
        "| `duplicate_group_id` | int | Internal group id for exact full-expression duplicates. |",
        "| `duplicate_count` | int | Number of records sharing this full `normalized_latex` (1 = unique). |",
        "| `token_type_counts` | JSON | Per-taxonomy-category token counts for this expression. |",
        "| `has_rare_1` | 0/1 | Expression contains a token with corpus frequency ≤ 1. |",
        "| `has_rare_5` | 0/1 | Expression contains a token with corpus frequency ≤ 5. |",
        "| `has_rare_10` | 0/1 | Expression contains a token with corpus frequency ≤ 10. |",
        "| `structure_types` | string | Pipe-separated Table-6 structure types present (`分式|下标|…`). |",
        "| `structure_type_count` | int | Number of distinct structure types in this expression. |",
        "| `structure_max_depths` | JSON | Per-structure-type max nesting depth within this expression. |",
        "| `ast_depth` | int | PosFormer position-forest max nested level. |",
        "| `parse_status` | string | `ok`, `unknown_token`, `unbalanced_braces`, or `incomplete_substructure`. |",
        "",
        "## Notes",
        "",
        "- `token_type_counts` keys match taxonomy category names (including `punctuation tokens`,",
        "  `layout / alignment tokens`, and `other / unknown tokens`).",
        "- Rare flags are computed from **corpus-level** token frequencies computed in the same export pass.",
        "- Duplicate fields use exact full-string match on `normalized_latex` (see `metric_definitions.md`).",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_known_limitations_md(output_path: Path) -> None:
    lines = [
        "# Known Limitations and Handling Strategies",
        "",
        "## OCR pipeline artifacts (`\\delete`, `\\insertion`, …)",
        "",
        "Non-standard commands such as `\\delete`, `\\insertion`, and `\\insert` appear in the OCR",
        "post-processing vocabulary as **custom symbols**. They are tokenized as dictionary entries",
        "and classified as `special symbol tokens`, not as unknown noise.",
        "",
        "## Chinese solution text",
        "",
        "Mixed Chinese + math expressions (e.g. `解 : ( 1 ) 依 题`) are tokenized character-wise for",
        "CJK ideographs. Chinese characters become `CJK tokens`; ASCII punctuation such as `:` becomes",
        "`punctuation tokens`. Spaces are skipped and not emitted.",
        "",
        "## Punctuation",
        "",
        "Plain punctuation (`,`, `:`, `.`, `;`, `!`, `?`, `'`) is **explicitly categorized** and excluded",
        "from `other / unknown tokens`. LaTeX spacing commands (`\\,`, `\\quad`, …) remain dictionary",
        "entries and map to `special symbol tokens`.",
        "",
        "## Env. alignment and line breaks",
        "",
        "- `\\\\` (double backslash) — row breaks in Env. / cases layouts → `layout / alignment tokens`.",
        "- `&` — column separators in aligned layouts → `layout / alignment tokens`.",
        "- Env. **nesting depth** in Table 6 counts nested valid "
        "``\\begin{env} ... \\\\ ... \\end{env}`` blocks only.",
        "",
        "## Duplicate detection",
        "",
        "Duplicates are detected by **exact string equality** on OCR LaTeX. Whitespace or notation",
        "variants that humans would consider equivalent are **not** merged.",
        "",
        "## Parse validation vs AST encoding",
        "",
        "Parse OK requires **Dictionary OK** (every token in the frozen HMER vocabulary / taxonomy)",
        "and **Structural AST OK** (balanced ``{``/``}``; structure operators must have atom or",
        "``{group}`` arguments). PosFormer encoding always runs; low parse success rates are",
        "data-quality signals.",
        "",
        "## Unknown tokens",
        "",
        "Residue that is neither dictionary-covered nor in punctuation / layout / CJK / standard math",
        "classes is reported as `unclassified`. Review `tables/unclassified_token_summary.csv`",
        "and `examples/unknown_token_examples.csv` for audit samples.",
        "",
        "## Cross-benchmark comparability",
        "",
        "All datasets share the same tokenizer and taxonomy. Loader-specific source formats differ,",
        "but metrics are computed on the unified token sequence representation.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tokenizer_docs(
    features: list[ExpressionFeatures],
    docs_dir: Path,
    resources_dir: Path,
    *,
    token_counter: Counter[str] | None = None,
) -> dict[str, Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)
    resolved_counter = resolve_token_counter(features, token_counter)
    paths = {
        "tokenizer_rules": docs_dir / "tokenizer_rules.md",
        "latex_dictionary": docs_dir / "latex_dictionary.md",
        "taxonomy_rules": docs_dir / "taxonomy_rules.md",
        "metric_definitions": docs_dir / "metric_definitions.md",
        "data_schema": docs_dir / "data_schema.md",
        "known_limitations": docs_dir / "known_limitations.md",
        "tokenization_examples": docs_dir / "tokenization_examples.md",
        "taxonomy_examples": docs_dir / "taxonomy_examples.md",
        "latex_vocab": resources_dir / "latex_vocab.csv",
        "token_taxonomy_map": resources_dir / "token_taxonomy_map.csv",
    }
    write_tokenizer_rules(paths["tokenizer_rules"])
    write_latex_dictionary_md(paths["latex_dictionary"])
    write_taxonomy_rules_md(paths["taxonomy_rules"])
    write_metric_definitions_md(paths["metric_definitions"])
    write_data_schema_md(paths["data_schema"])
    write_known_limitations_md(paths["known_limitations"])
    write_tokenization_examples_md(features, paths["tokenization_examples"])
    write_taxonomy_examples_md(features, paths["taxonomy_examples"], token_counter=resolved_counter)
    write_latex_vocab_csv(paths["latex_vocab"])
    write_token_taxonomy_map_csv(features, paths["token_taxonomy_map"], token_counter=resolved_counter)
    return paths
