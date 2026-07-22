"""Load and validate split input tables."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark_design.page_level_latex_split.config import SplitConfig

STRUCTURE_HAS_COLUMNS = (
    "has_frac",
    "has_sup",
    "has_sub",
    "has_sqrt",
    "has_env",
    "has_bigop",
    "has_accent",
    "has_stackrel",
    "has_textcircled",
)

REQUIRED_FEATURE_COLUMNS = (
    "page_id",
    "ast_tree_count",
    "total_ast_node_count",
    "max_ast_depth",
    *STRUCTURE_HAS_COLUMNS,
    "structure_type_count",
    "has_rare8",
    "rare8_token_count",
    "has_digit_letter_pair",
    "has_circle_like_pair",
    "has_latin_greek_pair",
    "has_greek_variant_pair",
    "has_operator_variable_pair",
    "has_relation_stroke_pair",
)

OPTIONAL_EXPR_DEPTH_COLUMNS = tuple(f"has_expr_depth_{depth}" for depth in range(6))


@dataclass(frozen=True, slots=True)
class SplitInputBundle:
    manifest: pd.DataFrame
    features: pd.DataFrame
    token_counts: pd.DataFrame
    dataset_version: str
    input_hashes: dict[str, str]
    inputs_dir: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_and_validate_inputs(inputs_dir: Path, config: SplitConfig) -> SplitInputBundle:
    manifest_path = inputs_dir / "dataset_manifest.csv"
    features_path = inputs_dir / "page_hmer_features.csv"
    tokens_path = inputs_dir / "page_token_counts.csv"
    meta_path = inputs_dir / "dataset_meta.json"

    for path in (manifest_path, features_path, tokens_path):
        if not path.is_file():
            raise FileNotFoundError(f"required split input missing: {path}")

    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(features_path)
    token_counts = pd.read_csv(tokens_path)

    _require_columns(manifest, ("page_id", "image_path", "annotation_path"), "dataset_manifest")
    _require_columns(features, REQUIRED_FEATURE_COLUMNS, "page_hmer_features")
    _require_columns(token_counts, ("page_id", "token", "count"), "page_token_counts")

    if manifest["page_id"].duplicated().any():
        dupes = manifest.loc[manifest["page_id"].duplicated(), "page_id"].tolist()
        raise ValueError(f"duplicate page_id in manifest: {dupes[:10]}")
    if features["page_id"].duplicated().any():
        dupes = features.loc[features["page_id"].duplicated(), "page_id"].tolist()
        raise ValueError(f"duplicate page_id in features: {dupes[:10]}")

    manifest_ids = set(manifest["page_id"].astype(str))
    feature_ids = set(features["page_id"].astype(str))
    if manifest_ids != feature_ids:
        missing = sorted(manifest_ids - feature_ids)
        extra = sorted(feature_ids - manifest_ids)
        raise ValueError(
            "manifest/features page mismatch; "
            f"missing_features={missing[:10]} extra_features={extra[:10]}"
        )

    # File existence
    for row in manifest.itertuples(index=False):
        if not Path(str(row.image_path)).is_file():
            raise FileNotFoundError(f"image missing: {row.image_path}")
        if not Path(str(row.annotation_path)).is_file():
            raise FileNotFoundError(f"annotation missing: {row.annotation_path}")

    # Numeric ranges and structure identity
    for col in (
        "ast_tree_count",
        "total_ast_node_count",
        "max_ast_depth",
        "structure_type_count",
        "rare8_token_count",
    ):
        if features[col].isna().any():
            raise ValueError(f"missing values in {col}")
        if (features[col] < 0).any():
            raise ValueError(f"negative values in {col}")

    for col in STRUCTURE_HAS_COLUMNS + (
        "has_rare8",
        "has_digit_letter_pair",
        "has_circle_like_pair",
        "has_latin_greek_pair",
        "has_greek_variant_pair",
        "has_operator_variable_pair",
        "has_relation_stroke_pair",
    ):
        if features[col].isna().any():
            raise ValueError(f"missing values in {col}")
        bad = ~features[col].isin([0, 1, True, False])
        if bad.any():
            raise ValueError(f"non-binary values in {col}")

    features = features.copy()
    for depth, col in enumerate(OPTIONAL_EXPR_DEPTH_COLUMNS):
        if col not in features.columns:
            features[col] = (features["max_ast_depth"].astype(int) == depth).astype(int)
    for col in OPTIONAL_EXPR_DEPTH_COLUMNS:
        if features[col].isna().any():
            raise ValueError(f"missing values in {col}")
        bad = ~features[col].isin([0, 1, True, False])
        if bad.any():
            raise ValueError(f"non-binary values in {col}")
        features[col] = features[col].astype(int)

    for col in STRUCTURE_HAS_COLUMNS + (
        "has_rare8",
        "has_digit_letter_pair",
        "has_circle_like_pair",
        "has_latin_greek_pair",
        "has_greek_variant_pair",
        "has_operator_variable_pair",
        "has_relation_stroke_pair",
    ):
        features[col] = features[col].astype(int)

    summed = features[list(STRUCTURE_HAS_COLUMNS)].sum(axis=1)
    mismatch = features["structure_type_count"].astype(int) != summed
    if mismatch.any():
        bad_ids = features.loc[mismatch, "page_id"].astype(str).tolist()
        raise ValueError(
            "structure_type_count != sum(has_frac+...+has_env) for pages: "
            f"{bad_ids[:10]}"
        )

    # Token table pages must cover pages with tokens
    token_pages = set(token_counts["page_id"].astype(str)) if not token_counts.empty else set()
    zero_expr = set(features.loc[features["ast_tree_count"] == 0, "page_id"].astype(str))
    unexpected = (manifest_ids - token_pages) - zero_expr
    if unexpected:
        raise ValueError(f"pages with expressions but no token rows: {sorted(unexpected)[:10]}")
    orphan_tokens = token_pages - manifest_ids
    if orphan_tokens:
        raise ValueError(f"token rows for unknown pages: {sorted(orphan_tokens)[:10]}")

    dataset_version = config.dataset_version
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dataset_version = str(meta.get("dataset_version") or dataset_version)

    hashes = {
        "dataset_manifest.csv": file_sha256(manifest_path),
        "page_hmer_features.csv": file_sha256(features_path),
        "page_token_counts.csv": file_sha256(tokens_path),
    }

    # Sort deterministically
    manifest = manifest.sort_values("page_id").reset_index(drop=True)
    features = features.sort_values("page_id").reset_index(drop=True)
    token_counts = token_counts.sort_values(["page_id", "token"]).reset_index(drop=True)

    return SplitInputBundle(
        manifest=manifest,
        features=features,
        token_counts=token_counts,
        dataset_version=dataset_version,
        input_hashes=hashes,
        inputs_dir=inputs_dir,
    )
