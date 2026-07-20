"""Image I/O, dataset discovery, and data validation."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.utils import file_hash, save_json

logger = logging.getLogger("heatmap_analysis.io")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageRecord:
    image_id: str
    image_path: Path
    rel_path: str
    template_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationIssue:
    image_id: str
    image_path: str
    issue_type: str
    message: str


def read_image(path: Path) -> np.ndarray:
    """Read image as BGR uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def read_grayscale(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read grayscale image: {path}")
    return gray


def discover_images(dataset_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    paths: list[Path] = []
    for p in sorted(dataset_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in ext_set:
            paths.append(p)
    return paths


def build_metadata_index(cfg: AnalysisConfig) -> pd.DataFrame:
    """Build or load metadata table for all images."""
    dataset_dir = cfg.input.dataset_dir
    if cfg.input.metadata_file and cfg.input.metadata_file.exists():
        df = pd.read_csv(cfg.input.metadata_file)
        if cfg.input.image_column not in df.columns:
            # Try relative path from dataset dir
            if "filename" in df.columns:
                df[cfg.input.image_column] = df["filename"].apply(
                    lambda x: str(dataset_dir / x)
                )
            else:
                raise ValueError(f"missing column {cfg.input.image_column} in metadata")
        return df

    records: list[dict] = []
    for img_path in discover_images(dataset_dir, cfg.input.image_extensions):
        rel = str(img_path.relative_to(dataset_dir))
        image_id = img_path.stem
        meta: dict = {"image_id": image_id, "image_path": str(img_path), "rel_path": rel}
        sidecar = img_path.with_suffix(img_path.suffix + ".json")
        if not sidecar.exists():
            sidecar = img_path.with_suffix(".json")
        if sidecar.exists():
            import json

            with sidecar.open("r", encoding="utf-8") as f:
                side = json.load(f)
            for k, v in side.items():
                if k not in meta and not isinstance(v, (list, dict)):
                    meta[k] = v
        records.append(meta)
    return pd.DataFrame(records)


def records_from_metadata(cfg: AnalysisConfig, df: pd.DataFrame) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for _, row in df.iterrows():
        img_path = Path(str(row[cfg.input.image_column]))
        if not img_path.is_absolute():
            img_path = (cfg.input.dataset_dir / img_path).resolve()
        image_id = str(row.get(cfg.input.id_column, img_path.stem))
        template_id = None
        if cfg.input.template_column in row and pd.notna(row[cfg.input.template_column]):
            template_id = str(row[cfg.input.template_column])
        meta = {k: row[k] for k in df.columns if k not in (cfg.input.image_column, cfg.input.id_column)}
        rel = str(img_path.relative_to(cfg.input.dataset_dir)) if img_path.is_relative_to(cfg.input.dataset_dir) else str(img_path)
        records.append(
            ImageRecord(
                image_id=image_id,
                image_path=img_path,
                rel_path=rel,
                template_id=template_id,
                metadata={k: (None if pd.isna(v) else v) for k, v in meta.items()},
            )
        )
    return records


def validate_image(
    record: ImageRecord,
    cfg: AnalysisConfig,
    ink_mask: np.ndarray | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    path = record.image_path
    try:
        gray = read_grayscale(path)
    except ValueError as e:
        issues.append(ValidationIssue(record.image_id, str(path), "read_error", str(e)))
        return issues

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        issues.append(ValidationIssue(record.image_id, str(path), "empty_image", "zero dimension"))
        return issues

    if h * w < cfg.preprocessing.min_page_pixels:
        issues.append(
            ValidationIssue(record.image_id, str(path), "small_page", f"page pixels {h*w} < min")
        )

    aspect = max(h, w) / max(min(h, w), 1)
    if aspect > cfg.preprocessing.max_aspect_ratio:
        issues.append(
            ValidationIssue(record.image_id, str(path), "aspect_ratio", f"aspect ratio {aspect:.2f}")
        )

    # Orientation heuristic: if width >> height for document-like image, flag
    if w > h * 1.8:
        issues.append(
            ValidationIssue(record.image_id, str(path), "orientation", "width much larger than height")
        )

    if ink_mask is not None:
        ink_total = float(np.sum(ink_mask > 0))
        if ink_total / ink_mask.size < cfg.preprocessing.blank_ink_threshold:
            issues.append(
                ValidationIssue(record.image_id, str(path), "blank_page", "no significant ink detected")
            )

    return issues


def find_duplicate_files(records: list[ImageRecord]) -> list[dict]:
    """Detect duplicate files by content hash."""
    hash_map: dict[str, list[str]] = {}
    for rec in records:
        try:
            h = file_hash(rec.image_path)
        except OSError:
            continue
        hash_map.setdefault(h, []).append(rec.image_id)
    dups = [{"hash": h, "image_ids": ids} for h, ids in hash_map.items() if len(ids) > 1]
    return dups


def run_preprocessing_checks(cfg: AnalysisConfig) -> dict:
    """Validate dataset and write check report."""
    df = build_metadata_index(cfg)
    records = records_from_metadata(cfg, df)
    all_issues: list[dict] = []
    readable = 0
    for rec in records:
        issues = validate_image(rec, cfg)
        if not any(i.issue_type == "read_error" for i in issues):
            readable += 1
        for issue in issues:
            all_issues.append(
                {
                    "image_id": issue.image_id,
                    "image_path": issue.image_path,
                    "issue_type": issue.issue_type,
                    "message": issue.message,
                }
            )

    duplicates = find_duplicate_files(records)

    report = {
        "total_images": len(records),
        "readable_images": readable,
        "issue_count": len(all_issues),
        "duplicate_groups": len(duplicates),
        "issues": all_issues,
        "duplicates": duplicates,
    }

    out_dir = cfg.output.output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_df = df.copy()
    if cfg.input.image_column in export_df.columns:
        export_df[cfg.input.image_column] = export_df[cfg.input.image_column].apply(
            lambda x: str((cfg.input.dataset_dir / x).resolve()) if not Path(str(x)).is_absolute() else str(x)
        )
    export_df.to_csv(out_dir / "metadata_index.csv", index=False)
    if all_issues:
        pd.DataFrame(all_issues).to_csv(out_dir / "validation_issues.csv", index=False)
    save_json(cfg.output.output_dir / "report" / "data_checks.json", report)
    logger.info("Preprocessing checks: %d images, %d issues", len(records), len(all_issues))
    return report


def load_template(template_dir: Path | None, template_id: str | None) -> np.ndarray | None:
    if template_dir is None or template_id is None:
        return None
    for ext in IMAGE_SUFFIXES:
        p = template_dir / f"{template_id}{ext}"
        if p.exists():
            return read_grayscale(p)
    return None


def cache_path_for_image(cfg: AnalysisConfig, image_id: str, suffix: str) -> Path:
    return cfg.cache_dir / "per_image" / f"{image_id}.{suffix}"
