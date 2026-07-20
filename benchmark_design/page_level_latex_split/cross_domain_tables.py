"""Split-aware tables for Chapter 7 cross-domain figures (7-3 to 7-6)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from benchmark_design.block_level.flow_structure.block_roles import (
    is_deleted_text_block,
    is_txt_block,
    normalize_block_type,
)
from benchmark_design.ocr.lbd_coordinates import (
    STRUCTURAL_DIFFICULTY_TIERS,
    assign_b_bin,
    assign_d_bin,
    assign_l_bin,
    classify_lbd,
)
from benchmark_design.page_level_latex_split.config import BinSpec, SplitConfig
from benchmark_design.page_level_latex_split.stratify import SPLITS
from benchmark_design.export_layout import (
    BenchmarkExportLayout,
    expression_level_statistics_csv,
    image_features_csv,
    line_metrics_csv,
)

LAYOUT_DOMAIN_CATEGORIES = ("structure_layout", "hybrid_layout")
BLOCK_TYPE_CATEGORIES = ("Txtblock", "deleted_text_block", "figure", "chart")
INTERFERENCE_CATEGORIES = (
    "interference_zero",
    "interference_low",
    "interference_mid",
    "interference_high",
)


def normalize_page_id(page_id: str) -> str:
    return str(page_id).removesuffix(".jpg")


def assign_numeric_bin(value: float, bins: tuple[BinSpec, ...]) -> str:
    for spec in bins:
        if value < spec.min:
            continue
        if spec.max is None or value < spec.max:
            return spec.label
    if bins:
        return bins[-1].label
    raise ValueError("empty bin specification")


def write_split_category_table(
    path: Path,
    categories: Iterable[str],
    split_counts: dict[str, Counter[str]],
) -> None:
    cats = list(categories)
    rows: list[dict[str, object]] = []
    overall_counter: Counter[str] = Counter()
    for split in SPLITS:
        overall_counter.update(split_counts.get(split, Counter()))
    overall_total = sum(overall_counter.values())

    for category in cats:
        for split in ("overall", *SPLITS):
            counter = overall_counter if split == "overall" else split_counts.get(split, Counter())
            total = overall_total if split == "overall" else sum(counter.values())
            count = int(counter.get(category, 0))
            rows.append(
                {
                    "category": category,
                    "split": split,
                    "count": count,
                    "ratio": count / total if total else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _split_entity_category_counts(
    manifest: pd.DataFrame,
    entity_frame: pd.DataFrame,
    *,
    page_id_col: str = "page_id",
    category_col: str = "category",
) -> dict[str, Counter[str]]:
    entity_cols = entity_frame.loc[:, [page_id_col, category_col]].copy()
    if page_id_col != "page_id":
        entity_cols = entity_cols.rename(columns={page_id_col: "page_id"})
    merged = entity_cols.merge(manifest[["page_id", "split"]], on="page_id", how="inner")
    split_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for split, group in merged.groupby("split"):
        split_counts[str(split)] = Counter(group[category_col].tolist())
    return split_counts


def _block_type_category(raw_type: str) -> str:
    if is_txt_block(raw_type):
        return "Txtblock"
    if is_deleted_text_block(raw_type):
        return "deleted_text_block"
    normalized = normalize_block_type(raw_type)
    if normalized == "figure":
        return "figure"
    if normalized == "chart":
        return "chart"
    return "other"


def _load_block_instances(manifest: pd.DataFrame, export_root: Path | None = None) -> pd.DataFrame:
    if export_root is not None:
        layout_frames: list[pd.DataFrame] = []
        layout = BenchmarkExportLayout(export_root)
        for layout_dir in (layout.structure_layout, layout.hybrid_layout):
            density_path = layout_dir / "tables" / "block_foreground_density.csv"
            if density_path.is_file():
                layout_frames.append(
                    pd.read_csv(density_path, usecols=["page_id", "block_type"])
                )
        if layout_frames:
            blocks = pd.concat(layout_frames, ignore_index=True)
            blocks["page_id"] = blocks["page_id"].map(normalize_page_id)
            blocks["category"] = blocks["block_type"].map(_block_type_category)
            return blocks.loc[:, ["page_id", "category"]]

    rows: list[dict[str, str]] = []
    for record in manifest.itertuples(index=False):
        annotation_path = Path(str(record.annotation_path))
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for block in payload.get("blocks") or []:
            rows.append(
                {
                    "page_id": str(record.page_id),
                    "category": _block_type_category(str(block.get("type", ""))),
                }
            )
    return pd.DataFrame(rows)


def _assign_interference_category(status: str, ratio: float) -> str:
    if str(status) == "zero_no_foreground":
        return "interference_zero"
    if ratio < 0.03:
        return "interference_low"
    if ratio < 0.06:
        return "interference_mid"
    return "interference_high"


def write_table9_foreground_density(
    manifest: pd.DataFrame,
    export_root: Path,
    config: SplitConfig,
    path: Path,
) -> None:
    image_features = pd.read_csv(image_features_csv(export_root))
    merged = manifest.merge(
        image_features[["image_id", "foreground_density"]],
        left_on="page_id",
        right_on="image_id",
        how="inner",
        validate="one_to_one",
    )
    categories = [spec.label for spec in config.foreground_density_bins]
    page_categories = merged["foreground_density"].map(
        lambda value: assign_numeric_bin(float(value), config.foreground_density_bins)
    )
    page_frame = pd.DataFrame({"page_id": merged["page_id"], "category": page_categories})
    write_split_category_table(
        path,
        categories,
        _split_entity_category_counts(manifest, page_frame),
    )


def write_table10_flow_structure(
    manifest: pd.DataFrame,
    export_root: Path,
    path: Path,
) -> None:
    layout = BenchmarkExportLayout(export_root)
    page_rows: list[dict[str, str]] = []
    for layout_domain, layout_dir in (
        ("structure_layout", layout.structure_layout),
        ("hybrid_layout", layout.hybrid_layout),
    ):
        metrics_path = layout_dir / "tables" / "flow_structure_page_metrics.csv"
        if not metrics_path.is_file():
            continue
        pages = pd.read_csv(metrics_path, usecols=["page_id"])
        for page_id in pages["page_id"].map(normalize_page_id):
            page_rows.append({"page_id": str(page_id), "category": layout_domain})
    page_frame = pd.DataFrame(page_rows)
    write_split_category_table(
        path,
        LAYOUT_DOMAIN_CATEGORIES,
        _split_entity_category_counts(manifest, page_frame),
    )


def write_table10_block_types(manifest: pd.DataFrame, path: Path, *, export_root: Path | None = None) -> None:
    blocks = _load_block_instances(manifest, export_root)
    categories = list(BLOCK_TYPE_CATEGORIES)
    if (blocks["category"] == "other").any():
        categories = [*categories, "other"]
    write_split_category_table(
        path,
        categories,
        _split_entity_category_counts(manifest, blocks),
    )


def _load_line_metrics_frame(export_root: Path) -> pd.DataFrame:
    line_path = line_metrics_csv(export_root)
    header = pd.read_csv(line_path, nrows=0).columns.tolist()
    usecols = [
        "image_id",
        "aspect_ratio",
        "is_valid",
        "interference_status",
        "interference_ratio",
        "bbox_outside_pixel_count",
        "bbox_outside_ink_count",
        "bbox_outside_ink_ratio",
    ]
    selected = [col for col in usecols if col in header]
    return pd.read_csv(line_path, usecols=selected)


def write_table11_line_aspect_ratio(
    manifest: pd.DataFrame,
    export_root: Path,
    config: SplitConfig,
    path: Path,
    *,
    lines: pd.DataFrame | None = None,
) -> None:
    if lines is None:
        lines = pd.read_csv(
            line_metrics_csv(export_root),
            usecols=["image_id", "aspect_ratio", "is_valid"],
        )
    valid = lines[lines["is_valid"] == True].copy()  # noqa: E712
    valid["category"] = valid["aspect_ratio"].map(
        lambda value: assign_numeric_bin(float(value), config.line_aspect_ratio_bins)
    )
    categories = [spec.label for spec in config.line_aspect_ratio_bins]
    write_split_category_table(
        path,
        categories,
        _split_entity_category_counts(manifest, valid, page_id_col="image_id"),
    )


def _line_interference_category(
    *,
    status: str | None = None,
    ratio: float | None = None,
    bbox_outside_pixel_count: int | None = None,
    bbox_outside_ink_count: int | None = None,
    bbox_outside_ink_ratio: float | None = None,
) -> str:
    if status is not None:
        return _assign_interference_category(status, float(ratio or 0.0))

    outside_px = int(bbox_outside_pixel_count or 0)
    ink_count = int(bbox_outside_ink_count or 0)
    ink_ratio = float(bbox_outside_ink_ratio or 0.0)
    if outside_px <= 0 or ink_count <= 0 or ink_ratio <= 0.0:
        return "interference_zero"
    return _assign_interference_category("computed", ink_ratio)


def write_table11_line_interference(
    manifest: pd.DataFrame,
    export_root: Path,
    path: Path,
    *,
    lines: pd.DataFrame | None = None,
) -> None:
    line_path = line_metrics_csv(export_root)
    if lines is None:
        header = pd.read_csv(line_path, nrows=0).columns.tolist()
    else:
        header = lines.columns.tolist()
    if {"interference_status", "interference_ratio"}.issubset(header):
        if lines is None:
            usecols = ["image_id", "interference_status", "interference_ratio", "is_valid"]
            lines = pd.read_csv(line_path, usecols=usecols)
        else:
            lines = lines.loc[
                :,
                ["image_id", "interference_status", "interference_ratio", "is_valid"],
            ]
        valid = lines[lines["is_valid"] == True].copy()  # noqa: E712
        valid["category"] = [
            _line_interference_category(status=str(status), ratio=float(ratio))
            for status, ratio in zip(
                valid["interference_status"],
                valid["interference_ratio"],
                strict=True,
            )
        ]
    else:
        usecols = [
            "image_id",
            "bbox_outside_pixel_count",
            "bbox_outside_ink_count",
            "bbox_outside_ink_ratio",
            "is_valid",
        ]
        missing = [col for col in usecols if col not in header]
        if missing:
            raise ValueError(
                f"{line_path} missing columns required for interference table: {missing}"
            )
        if lines is None:
            lines = pd.read_csv(line_path, usecols=usecols)
        else:
            lines = lines.loc[:, usecols]
        valid = lines[lines["is_valid"] == True].copy()  # noqa: E712
        valid["category"] = [
            _line_interference_category(
                bbox_outside_pixel_count=int(outside_px),
                bbox_outside_ink_count=int(ink_count),
                bbox_outside_ink_ratio=float(ink_ratio),
            )
            for outside_px, ink_count, ink_ratio in zip(
                valid["bbox_outside_pixel_count"],
                valid["bbox_outside_ink_count"],
                valid["bbox_outside_ink_ratio"],
                strict=True,
            )
        ]
    write_split_category_table(
        path,
        INTERFERENCE_CATEGORIES,
        _split_entity_category_counts(manifest, valid, page_id_col="image_id"),
    )


def write_table12_expression_difficulty(
    manifest: pd.DataFrame,
    export_root: Path,
    path: Path,
) -> None:
    expressions = pd.read_csv(
        expression_level_statistics_csv(export_root),
        usecols=["expression_id", "token_length", "structure_type_count", "ast_depth"],
    )
    expressions["page_id"] = expressions["expression_id"].str.split(":", n=2).str[1].map(normalize_page_id)
    expressions["category"] = [
        classify_lbd(
            assign_l_bin(int(token_length)),
            assign_b_bin(int(structure_type_count)),
            assign_d_bin(int(ast_depth)),
        )
        for token_length, structure_type_count, ast_depth in zip(
            expressions["token_length"],
            expressions["structure_type_count"],
            expressions["ast_depth"],
            strict=True,
        )
    ]
    write_split_category_table(
        path,
        STRUCTURAL_DIFFICULTY_TIERS,
        _split_entity_category_counts(manifest, expressions),
    )


def write_cross_domain_tables(
    manifest: pd.DataFrame,
    export_root: Path,
    config: SplitConfig,
    tables_dir: Path,
) -> dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    line_metrics = _load_line_metrics_frame(export_root)
    writers: tuple[tuple[str, Callable[[], None]], ...] = (
        (
            "table9_foreground_density_bins.csv",
            lambda: write_table9_foreground_density(
                manifest,
                export_root,
                config,
                tables_dir / "table9_foreground_density_bins.csv",
            ),
        ),
        (
            "table10_flow_structure.csv",
            lambda: write_table10_flow_structure(
                manifest,
                export_root,
                tables_dir / "table10_flow_structure.csv",
            ),
        ),
        (
            "table10_block_type_composition.csv",
            lambda: write_table10_block_types(
                manifest,
                tables_dir / "table10_block_type_composition.csv",
                export_root=export_root,
            ),
        ),
        (
            "table11_line_aspect_ratio_bins.csv",
            lambda: write_table11_line_aspect_ratio(
                manifest,
                export_root,
                config,
                tables_dir / "table11_line_aspect_ratio_bins.csv",
                lines=line_metrics,
            ),
        ),
        (
            "table11_line_interference_bins.csv",
            lambda: write_table11_line_interference(
                manifest,
                export_root,
                tables_dir / "table11_line_interference_bins.csv",
                lines=line_metrics,
            ),
        ),
        (
            "table12_expression_difficulty.csv",
            lambda: write_table12_expression_difficulty(
                manifest,
                export_root,
                tables_dir / "table12_expression_difficulty.csv",
            ),
        ),
    )
    outputs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(writers), 6)) as pool:
        futures = [pool.submit(writer) for _, writer in writers]
        for future in futures:
            future.result()
    for name, _ in writers:
        outputs[name] = name
    return outputs
