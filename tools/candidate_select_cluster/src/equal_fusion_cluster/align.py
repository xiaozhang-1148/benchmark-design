"""Align Batch02 samples to vision/text feature rows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, ensure_dir
from .discover import PairedSample, discover_all_pairs
from .feature_index import FeatureStore, load_text_store, load_vision_store
from .fusion import equal_weight_fuse


@dataclass
class AlignedSample:
    group_id: str
    sample_id: str
    image_path: str
    json_path: str
    image_basename: str
    vision_feature_source: str
    vision_feature_row: int
    text_feature_source: str
    text_feature_row: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def align_all(
    data_root: str | Path,
    vision_root: str | Path,
    text_root: str | Path,
) -> tuple[list[AlignedSample], dict[str, Any], list[dict[str, Any]]]:
    pairs = discover_all_pairs(data_root)
    vis = load_vision_store(vision_root)
    txt = load_text_store(text_root)

    aligned: list[AlignedSample] = []
    errors: list[dict[str, Any]] = []

    for p in pairs:
        try:
            if p.image_basename not in vis.basename_to_row:
                raise KeyError(f"missing vision feature for {p.image_basename}")
            if p.image_basename not in txt.basename_to_row:
                raise KeyError(f"missing text feature for {p.image_basename}")
            vr = vis.basename_to_row[p.image_basename]
            tr = txt.basename_to_row[p.image_basename]
            # dimension sanity via store dims
            aligned.append(
                AlignedSample(
                    group_id=p.group_id,
                    sample_id=p.sample_id,
                    image_path=p.image_path,
                    json_path=p.json_path,
                    image_basename=p.image_basename,
                    vision_feature_source=vis.matrix_path,
                    vision_feature_row=vr,
                    text_feature_source=txt.matrix_path,
                    text_feature_row=tr,
                )
            )
        except Exception as e:  # noqa: BLE001
            errors.append(
                {
                    "group_id": p.group_id,
                    "sample_id": p.sample_id,
                    "image_path": p.image_path,
                    "json_path": p.json_path,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # duplicate feature row checks
    vis_rows = [a.vision_feature_row for a in aligned]
    txt_rows = [a.text_feature_row for a in aligned]
    report = {
        "n_pairs_discovered": len(pairs),
        "n_aligned": len(aligned),
        "n_errors": len(errors),
        "n_groups": len({a.group_id for a in aligned}),
        "vision": {
            "matrix": vis.matrix_path,
            "index": vis.index_path,
            "dim": vis.dim,
            "n": vis.n,
            "selection_reason": vis.selection_reason,
        },
        "text": {
            "matrix": txt.matrix_path,
            "index": txt.index_path,
            "dim": txt.dim,
            "n": txt.n,
            "selection_reason": txt.selection_reason,
        },
        "mapping_basis": (
            "Batch02 image basename → vision sample_index.sample_id / image_path name; "
            "text text_sample_index.image_name. Features extracted from flat ALL_Benchmark "
            "with unique basenames."
        ),
        "vision_row_unique": len(vis_rows) == len(set(vis_rows)),
        "text_row_unique": len(txt_rows) == len(set(txt_rows)),
        "group_depth_note": (
            "Batch02 question groups are relative depth-2 dirs (exam_id/question_id); "
            "task brief depth-3 A/B/C does not exist in this tree."
        ),
    }
    if not report["vision_row_unique"] or not report["text_row_unique"]:
        errors.append({"error": "duplicate feature row mapping detected"})
    aligned.sort(key=lambda a: (a.group_id, a.sample_id))
    return aligned, report, errors


def write_alignment(
    aligned: list[AlignedSample],
    report: dict[str, Any],
    errors: list[dict[str, Any]],
    output_root: Path,
) -> None:
    adir = output_root / "alignment"
    ensure_dir(adir)
    tmp = adir / "all_aligned_samples.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        for a in aligned:
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(adir / "all_aligned_samples.jsonl")
    atomic_write_json(adir / "alignment_report.json", report)
    with (adir / "alignment_errors.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def load_group_raw_matrices(
    aligned_group: list[AlignedSample],
    vis: FeatureStore,
    txt: FeatureStore,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return raw vision [N,1792], text [N,1024], sample_ids (no fusion)."""
    n = len(aligned_group)
    V = np.empty((n, 1792), dtype=np.float32)
    T = np.empty((n, 1024), dtype=np.float32)
    ids: list[str] = []
    for i, a in enumerate(aligned_group):
        V[i] = vis.get_by_basename(a.image_basename)
        T[i] = txt.get_by_basename(a.image_basename)
        ids.append(a.sample_id)
        if not np.isfinite(V[i]).all() or not np.isfinite(T[i]).all():
            raise RuntimeError(f"non-finite features for {a.sample_id}")
        if float(np.linalg.norm(V[i])) < 1e-12 or float(np.linalg.norm(T[i])) < 1e-12:
            raise RuntimeError(f"near-zero feature norm for {a.sample_id}")
    return V, T, ids


def load_group_matrices(
    aligned_group: list[AlignedSample],
    vis: FeatureStore,
    txt: FeatureStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Legacy helper: equal-weight fuse without PCA.

    Prefer ``prepare_group_pca_fusion`` for the production pipeline.
    """
    V, T, ids = load_group_raw_matrices(aligned_group, vis, txt)
    x_hat, t_hat, z = equal_weight_fuse(V, T)
    return x_hat, t_hat, z, ids
