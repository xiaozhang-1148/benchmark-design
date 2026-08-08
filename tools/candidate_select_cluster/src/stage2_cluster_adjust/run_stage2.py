"""Stage 2: adjust 8 selected candidates using frozen remaining-sample clusters."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.equal_fusion_cluster.feature_index import load_text_store, load_vision_store
from src.equal_fusion_cluster.fusion import l2_normalize
from src.equal_fusion_cluster.pca_fuse import load_pca_pickle, transform_with_saved_pca
from src.equal_fusion_cluster.run import safe_group_dirname
from src.utils import atomic_write_json, ensure_dir

DEFAULT_CLUSTER = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/raw_dataset_cluster"
DEFAULT_RAW = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/raw_dataset"
DEFAULT_SELECT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/select_dataset"
DEFAULT_AUDIT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/selected_samples_by_type.json"
DEFAULT_VIS = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/vision_deatures"
DEFAULT_TXT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/text_feature"
DEFAULT_OUT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/stage2_cluster_adjusted_1"
REASON_UNDER_COVERAGE = "pre_unique_cluster_count_lt_ceil_K_half"
REASON_RANDOM_DEDUPE = "random_multi_per_cluster_keep_nearest"
# backward-compatible alias used in older audits
REPLACEMENT_REASON = REASON_UNDER_COVERAGE

SOURCE_MAP_CN = {"随机": "random", "难度": "hard", "多样性": "diverse"}
SOURCE_MAP_EN = {"random": "random", "hard": "hard", "diversity": "diverse", "diverse": "diverse"}


@dataclass
class SampleRef:
    group_id: str
    exam_id: str
    question_id: str
    sample_id: str  # exam/question/basename.jpg
    basename: str
    image_path: Path
    json_path: Path
    selection_source: str | None  # random/hard/diverse/None for remaining
    status: str  # selected | remaining


@dataclass
class AssignedSample:
    ref: SampleRef
    cluster_id: int
    cluster_size: int
    cosine_to_centroid: float
    z: np.ndarray | None = None


@dataclass
class QuestionResult:
    group_id: str
    exam_id: str
    question_id: str
    K: int
    reference_sample_count: int
    pre_selected: list[AssignedSample]
    pre_unique_cluster_count: int
    pre_cluster_histogram: dict[str, int]
    triggered_replacement: bool
    min_unique_clusters_required: int = 0
    removed_random: list[AssignedSample] = field(default_factory=list)
    added_reps: list[AssignedSample] = field(default_factory=list)
    post_selected: list[AssignedSample] = field(default_factory=list)
    post_unique_cluster_count: int = 0
    post_cluster_histogram: dict[str, int] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def split_group_id(group_id: str) -> tuple[str, str]:
    parts = Path(group_id).parts
    if len(parts) != 2:
        raise RuntimeError(f"expected exam/question group_id, got {group_id!r}")
    return parts[0], parts[1]


def list_paired_jpgs(qdir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(qdir.glob("*.jpg")):
        j = Path(str(img) + ".json")
        if not j.is_file():
            raise RuntimeError(f"missing json for {img}")
        pairs.append((img, j))
    return pairs


def load_selection_audit(path: Path) -> dict[str, dict[str, list[str]]]:
    """group_id -> {random/hard/diverse: [sample_id,...]}"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[str]]] = {}
    questions = raw.get("questions")
    if not isinstance(questions, list):
        raise RuntimeError(f"audit missing questions list: {path}")
    for q in questions:
        gid = q["group_id"]
        if "samples_en" in q and isinstance(q["samples_en"], dict):
            src = q["samples_en"]
            mapped = {
                "random": list(src.get("random") or []),
                "hard": list(src.get("hard") or []),
                "diverse": list(src.get("diversity") or src.get("diverse") or []),
            }
        elif "samples" in q:
            src = q["samples"]
            mapped = {
                "random": list(src.get("随机") or src.get("random") or []),
                "hard": list(src.get("难度") or src.get("hard") or []),
                "diverse": list(src.get("多样性") or src.get("diverse") or src.get("diversity") or []),
            }
        else:
            raise RuntimeError(f"no samples fields for {gid}")
        n = sum(len(v) for v in mapped.values())
        if n != 8:
            raise RuntimeError(f"{gid}: expected 8 selected samples in audit, got {n}")
        if len(mapped["random"]) != 4 or len(mapped["hard"]) != 2 or len(mapped["diverse"]) != 2:
            raise RuntimeError(f"{gid}: expected random=4 hard=2 diverse=2, got { {k:len(v) for k,v in mapped.items()} }")
        out[gid] = mapped
    return out


def discover_successful_cluster_groups(cluster_root: Path) -> dict[str, Path]:
    groups_dir = cluster_root / "groups"
    if not groups_dir.is_dir():
        raise FileNotFoundError(groups_dir)
    mapping: dict[str, Path] = {}
    for gd in sorted(groups_dir.iterdir()):
        if not gd.is_dir():
            continue
        man_path = gd / "group_manifest.json"
        if not man_path.exists():
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if man.get("status") != "success":
            continue
        gid = man["group_id"]
        required = [
            "centroids.npy",
            "labels.npy",
            "fused_features.npy",
            "clusters.jsonl",
            "image_pca.pkl",
            "text_pca.pkl",
        ]
        miss = [n for n in required if not (gd / n).exists()]
        if miss:
            raise RuntimeError(f"cluster group {gid} missing {miss}")
        mapping[gid] = gd
    return mapping


@dataclass
class FrozenClusterRef:
    group_id: str
    gdir: Path
    K: int
    n_ref: int
    centroids: np.ndarray  # [K,D] unit
    labels: np.ndarray
    fused: np.ndarray
    cluster_rows: list[dict[str, Any]]
    cluster_sizes: list[int]
    image_pca: dict[str, Any]
    text_pca: dict[str, Any]
    basename_to_idx: dict[str, int]
    basename_to_row: dict[str, dict[str, Any]]


def load_frozen_cluster(group_id: str, gdir: Path) -> FrozenClusterRef:
    man = json.loads((gdir / "group_manifest.json").read_text(encoding="utf-8"))
    cents = np.load(gdir / "centroids.npy").astype(np.float32)
    labels = np.load(gdir / "labels.npy").astype(np.int64)
    fused = np.load(gdir / "fused_features.npy").astype(np.float32)
    rows = [json.loads(l) for l in (gdir / "clusters.jsonl").open(encoding="utf-8") if l.strip()]
    if not (len(rows) == fused.shape[0] == labels.shape[0]):
        raise RuntimeError(
            f"{group_id}: alignment mismatch clusters={len(rows)} fused={fused.shape[0]} labels={labels.shape[0]}"
        )
    K = int(man.get("k") if man.get("k") is not None else cents.shape[0])
    if cents.shape[0] != K:
        raise RuntimeError(f"{group_id}: centroids rows {cents.shape[0]} != K={K}")
    cnorm_err = float(np.max(np.abs(np.linalg.norm(cents, axis=1) - 1.0)))
    if cnorm_err > 1e-4:
        raise RuntimeError(f"{group_id}: centroids not unit (max_err={cnorm_err})")
    sizes = [int(np.sum(labels == c)) for c in range(K)]
    bn_to_idx: dict[str, int] = {}
    bn_to_row: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        bn = Path(r["image_path"]).name
        if bn in bn_to_idx:
            raise RuntimeError(f"{group_id}: duplicate basename in clusters.jsonl: {bn}")
        if int(r["cluster_id"]) != int(labels[i]):
            raise RuntimeError(f"{group_id}: cluster_id mismatch at row {i}")
        bn_to_idx[bn] = i
        bn_to_row[bn] = r
    img_pca = load_pca_pickle(gdir / "image_pca.pkl")
    txt_pca = load_pca_pickle(gdir / "text_pca.pkl")
    if int(img_pca["output_dim"]) != int(txt_pca["output_dim"]):
        raise RuntimeError(f"{group_id}: PCA output dims differ")
    if bool(img_pca.get("whiten", True)) or bool(txt_pca.get("whiten", True)):
        raise RuntimeError(f"{group_id}: PCA whiten must be False")
    return FrozenClusterRef(
        group_id=group_id,
        gdir=gdir,
        K=K,
        n_ref=int(fused.shape[0]),
        centroids=cents,
        labels=labels,
        fused=fused,
        cluster_rows=rows,
        cluster_sizes=sizes,
        image_pca=img_pca,
        text_pca=txt_pca,
        basename_to_idx=bn_to_idx,
        basename_to_row=bn_to_row,
    )


def project_sample_to_fused(
    vision: np.ndarray,
    text: np.ndarray,
    image_pca: dict[str, Any],
    text_pca: dict[str, Any],
) -> np.ndarray:
    """Frozen pipeline: L2→PCA→L2→concat→L2 (no refit)."""
    v = np.asarray(vision, dtype=np.float32).reshape(1, -1)
    t = np.asarray(text, dtype=np.float32).reshape(1, -1)
    if v.shape[1] != 1792 or t.shape[1] != 1024:
        raise RuntimeError(f"bad feature dims {v.shape}/{t.shape}")
    if not np.isfinite(v).all() or not np.isfinite(t).all():
        raise RuntimeError("non-finite raw features")
    x_hat = l2_normalize(v)
    t_hat = l2_normalize(t)
    x_pca = transform_with_saved_pca(image_pca, x_hat)
    t_pca = transform_with_saved_pca(text_pca, t_hat)
    x_p = l2_normalize(x_pca)
    t_p = l2_normalize(t_pca)
    z = l2_normalize(np.concatenate([x_p, t_p], axis=-1))
    return z[0].astype(np.float32, copy=False)


def assign_to_centroid(z: np.ndarray, centroids: np.ndarray) -> tuple[int, float]:
    sims = centroids @ z  # [K]
    cid = int(np.argmax(sims))
    return cid, float(sims[cid])


def build_question_samples(
    group_id: str,
    *,
    select_root: Path,
    raw_root: Path,
    sources: dict[str, list[str]],
) -> tuple[list[SampleRef], list[SampleRef]]:
    exam_id, question_id = split_group_id(group_id)
    sel_dir = select_root / group_id
    raw_dir = raw_root / group_id
    if not sel_dir.is_dir():
        raise FileNotFoundError(sel_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    # map basename -> source from audit sample_ids
    bn_source: dict[str, str] = {}
    for src, ids in sources.items():
        for sid in ids:
            bn = Path(sid).name
            if bn in bn_source:
                raise RuntimeError(f"{group_id}: duplicate basename in audit {bn}")
            bn_source[bn] = src

    selected: list[SampleRef] = []
    for img, jpath in list_paired_jpgs(sel_dir):
        bn = img.name
        if bn not in bn_source:
            raise RuntimeError(f"{group_id}: select file {bn} missing selection_source in audit")
        selected.append(
            SampleRef(
                group_id=group_id,
                exam_id=exam_id,
                question_id=question_id,
                sample_id=f"{group_id}/{bn}",
                basename=bn,
                image_path=img,
                json_path=jpath,
                selection_source=bn_source[bn],
                status="selected",
            )
        )
    if len(selected) != 8:
        raise RuntimeError(f"{group_id}: select has {len(selected)} jpgs, expected 8")
    missing_src = set(bn_source) - {s.basename for s in selected}
    if missing_src:
        raise RuntimeError(f"{group_id}: audit basenames not on disk: {sorted(missing_src)}")

    remaining: list[SampleRef] = []
    for img, jpath in list_paired_jpgs(raw_dir):
        bn = img.name
        if bn in bn_source:
            raise RuntimeError(f"{group_id}: sample {bn} present in both select and raw")
        remaining.append(
            SampleRef(
                group_id=group_id,
                exam_id=exam_id,
                question_id=question_id,
                sample_id=f"{group_id}/{bn}",
                basename=bn,
                image_path=img,
                json_path=jpath,
                selection_source=None,
                status="remaining",
            )
        )
    return selected, remaining


def assign_selected_via_frozen(
    selected: list[SampleRef],
    cref: FrozenClusterRef,
    vis,
    txt,
) -> list[AssignedSample]:
    out: list[AssignedSample] = []
    for s in selected:
        if s.basename in cref.basename_to_idx:
            raise RuntimeError(
                f"{cref.group_id}: selected sample {s.basename} unexpectedly in remaining cluster matrix"
            )
        v = vis.get_by_basename(s.basename)
        t = txt.get_by_basename(s.basename)
        z = project_sample_to_fused(v, t, cref.image_pca, cref.text_pca)
        if z.shape[0] != cref.centroids.shape[1]:
            raise RuntimeError(f"fused dim mismatch {z.shape} vs centroids {cref.centroids.shape}")
        cid, cos = assign_to_centroid(z, cref.centroids)
        out.append(
            AssignedSample(
                ref=s,
                cluster_id=cid,
                cluster_size=cref.cluster_sizes[cid],
                cosine_to_centroid=cos,
                z=z,
            )
        )
    return out


def assign_remaining_from_cluster(
    remaining: list[SampleRef],
    cref: FrozenClusterRef,
) -> dict[str, AssignedSample]:
    """Map remaining basenames to frozen cluster assignments."""
    out: dict[str, AssignedSample] = {}
    for s in remaining:
        if s.basename not in cref.basename_to_idx:
            raise RuntimeError(f"{cref.group_id}: remaining {s.basename} missing from cluster matrix")
        idx = cref.basename_to_idx[s.basename]
        cid = int(cref.labels[idx])
        # prefer stored cosine; recompute for consistency
        cos = float(np.dot(cref.fused[idx], cref.centroids[cid]))
        out[s.basename] = AssignedSample(
            ref=s,
            cluster_id=cid,
            cluster_size=cref.cluster_sizes[cid],
            cosine_to_centroid=cos,
            z=cref.fused[idx],
        )
    if len(out) != cref.n_ref:
        raise RuntimeError(
            f"{cref.group_id}: remaining disk={len(out)} != cluster N={cref.n_ref}"
        )
    return out


def cluster_label(cid: int | str) -> str:
    """Format cluster category ids as cluster_x (never bare digits)."""
    s = str(cid)
    if s.startswith("cluster_"):
        return s
    return f"cluster_{int(cid)}"


def histogram(assigned: list[AssignedSample]) -> dict[str, int]:
    c = Counter(int(a.cluster_id) for a in assigned)
    return {cluster_label(k): int(v) for k, v in sorted(c.items())}


def min_unique_clusters_required(K: int) -> int:
    """Coverage threshold τ = ceil(K/2). E.g. K=7/8 → 4, K=9/10 → 5."""
    if K < 1:
        raise ValueError(f"invalid K={K}")
    return int(math.ceil(K / 2.0))


def unique_cluster_count(assigned: list[AssignedSample]) -> int:
    return len({int(a.cluster_id) for a in assigned})


def pick_centroid_representative(
    cref: FrozenClusterRef,
    cluster_id: int,
    *,
    remaining_by_bn: dict[str, AssignedSample],
    forbidden: set[str],
) -> AssignedSample:
    """Argmax cosine to μ_c among remaining members of cluster c, not in forbidden."""
    members = []
    for bn, a in remaining_by_bn.items():
        if bn in forbidden:
            continue
        if int(a.cluster_id) != int(cluster_id):
            continue
        members.append(a)
    if not members:
        raise RuntimeError(f"{cref.group_id}: no available members in cluster {cluster_id}")
    # sort by (-cos, sample_id) for stable tie-break
    members.sort(key=lambda a: (-a.cosine_to_centroid, a.ref.sample_id))
    return members[0]


def dedupe_randoms_per_cluster(randoms: list[AssignedSample]) -> tuple[list[AssignedSample], list[AssignedSample]]:
    """Within each cluster, keep only the random nearest to the centroid; drop the rest.

    hard/diverse are ignored by the caller. Returns (kept_randoms, removed_randoms).
    """
    by_cluster: dict[int, list[AssignedSample]] = defaultdict(list)
    for a in randoms:
        by_cluster[int(a.cluster_id)].append(a)
    kept: list[AssignedSample] = []
    removed: list[AssignedSample] = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        members.sort(key=lambda a: (-a.cosine_to_centroid, a.ref.sample_id))
        kept.append(members[0])
        removed.extend(members[1:])
    kept.sort(key=lambda a: a.ref.sample_id)
    removed.sort(key=lambda a: a.ref.sample_id)
    return kept, removed


def top_cluster_ids_by_size(cref: FrozenClusterRef, n: int = 4) -> list[int]:
    """Reference clusters ordered by sample count desc, then id asc; take up to n."""
    order = sorted(range(cref.K), key=lambda c: (-cref.cluster_sizes[c], c))
    return order[: max(0, min(n, cref.K))]


def process_question(
    group_id: str,
    sources: dict[str, list[str]],
    cref: FrozenClusterRef,
    *,
    select_root: Path,
    raw_root: Path,
    vis,
    txt,
) -> QuestionResult:
    exam_id, question_id = split_group_id(group_id)
    selected_refs, remaining_refs = build_question_samples(
        group_id, select_root=select_root, raw_root=raw_root, sources=sources
    )
    pre_sel = assign_selected_via_frozen(selected_refs, cref, vis, txt)
    rem_map = assign_remaining_from_cluster(remaining_refs, cref)

    pre_unique = unique_cluster_count(pre_sel)
    pre_hist = histogram(pre_sel)
    keep_threshold = min_unique_clusters_required(cref.K)
    result = QuestionResult(
        group_id=group_id,
        exam_id=exam_id,
        question_id=question_id,
        K=cref.K,
        reference_sample_count=cref.n_ref,
        pre_selected=pre_sel,
        pre_unique_cluster_count=pre_unique,
        pre_cluster_histogram=pre_hist,
        triggered_replacement=False,
        min_unique_clusters_required=keep_threshold,
    )

    def base_transition(a: AssignedSample, **kwargs: Any) -> dict[str, Any]:
        row = {
            "exam_id": exam_id,
            "question_id": question_id,
            "sample_id": a.ref.sample_id,
            "original_selection_source": a.ref.selection_source,
            "pre_status": "selected",
            "pre_cluster_id": cluster_label(a.cluster_id),
            "pre_cluster_size": a.cluster_size,
            "pre_cosine_to_centroid": a.cosine_to_centroid,
            "action": "keep",
            "replacement_reason": "",
            "replaces_sample_id": "",
            "post_status": "selected",
            "post_selection_source": a.ref.selection_source,
            "post_cluster_id": cluster_label(a.cluster_id),
            "post_cluster_size": a.cluster_size,
            "post_cosine_to_centroid": a.cosine_to_centroid,
        }
        row.update(kwargs)
        return row

    def append_remaining_keeps(
        *,
        moved_bn: set[str],
    ) -> None:
        for bn, a in sorted(rem_map.items()):
            if bn in moved_bn:
                continue
            result.transitions.append(
                {
                    "exam_id": exam_id,
                    "question_id": question_id,
                    "sample_id": a.ref.sample_id,
                    "original_selection_source": "",
                    "pre_status": "remaining",
                    "pre_cluster_id": cluster_label(a.cluster_id),
                    "pre_cluster_size": a.cluster_size,
                    "pre_cosine_to_centroid": a.cosine_to_centroid,
                    "action": "keep",
                    "replacement_reason": "",
                    "replaces_sample_id": "",
                    "post_status": "remaining",
                    "post_selection_source": "",
                    "post_cluster_id": cluster_label(a.cluster_id),
                    "post_cluster_size": a.cluster_size,
                    "post_cosine_to_centroid": a.cosine_to_centroid,
                }
            )

    protected = [a for a in pre_sel if a.ref.selection_source in ("hard", "diverse")]
    randoms = [a for a in pre_sel if a.ref.selection_source == "random"]
    if len(protected) != 4 or len(randoms) != 4:
        raise RuntimeError(
            f"{group_id}: expected 4 protected + 4 random, got {len(protected)}+{len(randoms)}"
        )

    # Case C >= τ → among randoms only, keep nearest-to-centroid per cluster
    if pre_unique >= keep_threshold:
        kept_randoms, removed = dedupe_randoms_per_cluster(randoms)
        post = protected + kept_randoms
        result.removed_random = removed
        result.added_reps = []
        result.post_selected = post
        result.post_unique_cluster_count = unique_cluster_count(post)
        result.post_cluster_histogram = histogram(post)
        if removed:
            result.triggered_replacement = True
            result.notes.append(REASON_RANDOM_DEDUPE)
        else:
            result.notes.append("pre_unique_ge_ceil_K_half_no_random_dup_clusters")

        for a in protected + kept_randoms:
            result.transitions.append(base_transition(a, action="keep"))
        for a in removed:
            result.transitions.append(
                base_transition(
                    a,
                    action="remove_random",
                    replacement_reason=REASON_RANDOM_DEDUPE,
                    post_status="remaining",
                    post_selection_source="",
                )
            )
        append_remaining_keeps(moved_bn=set())
        return result

    # Case UNDER: C < τ → drop all randoms; add centroid reps from 4 largest UNCOVERED clusters
    result.triggered_replacement = True
    result.notes.append(REASON_UNDER_COVERAGE)
    protected_clusters = {int(a.cluster_id) for a in protected}
    uncovered = sorted(
        [c for c in range(cref.K) if c not in protected_clusters],
        key=lambda c: (-cref.cluster_sizes[c], c),
    )
    target_clusters = uncovered[:4]
    # Prefer uncovered largest clusters; fall back to covered largest clusters if needed.
    size_order = uncovered + sorted(
        [c for c in range(cref.K) if c in protected_clusters],
        key=lambda c: (-cref.cluster_sizes[c], c),
    )
    forbidden = {a.ref.basename for a in protected} | {a.ref.basename for a in randoms}
    added: list[AssignedSample] = []
    used_clusters: list[int] = []
    for c in size_order:
        if len(added) >= 4:
            break
        try:
            rep = pick_centroid_representative(
                cref, c, remaining_by_bn=rem_map, forbidden=forbidden
            )
        except RuntimeError:
            continue
        rep.ref.selection_source = "replacement"
        added.append(rep)
        used_clusters.append(c)
        forbidden.add(rep.ref.basename)

    if len(added) != 4:
        raise RuntimeError(
            f"{group_id}: need 4 centroid replacements from largest uncovered clusters, got {len(added)} "
            f"(uncovered={uncovered}, size_order={size_order})"
        )

    post = protected + added
    if len(post) != 8:
        raise RuntimeError(f"{group_id}: post selected size {len(post)} != 8")

    result.removed_random = randoms
    result.added_reps = added
    result.post_selected = post
    result.post_unique_cluster_count = unique_cluster_count(post)
    result.post_cluster_histogram = histogram(post)
    result.notes.append(f"replacement_clusters={used_clusters}")

    for a in protected:
        result.transitions.append(base_transition(a, action="keep"))
    for i, a in enumerate(randoms):
        result.transitions.append(
            base_transition(
                a,
                action="remove_random",
                replacement_reason=REASON_UNDER_COVERAGE,
                post_status="remaining",
                post_selection_source="",
            )
        )
    for i, a in enumerate(added):
        replaces = randoms[i].ref.sample_id if i < len(randoms) else ""
        result.transitions.append(
            {
                "exam_id": exam_id,
                "question_id": question_id,
                "sample_id": a.ref.sample_id,
                "original_selection_source": "",
                "pre_status": "remaining",
                "pre_cluster_id": cluster_label(a.cluster_id),
                "pre_cluster_size": a.cluster_size,
                "pre_cosine_to_centroid": a.cosine_to_centroid,
                "action": "add_representative",
                "replacement_reason": REASON_UNDER_COVERAGE,
                "replaces_sample_id": replaces,
                "post_status": "selected",
                "post_selection_source": "replacement",
                "post_cluster_id": cluster_label(a.cluster_id),
                "post_cluster_size": a.cluster_size,
                "post_cosine_to_centroid": a.cosine_to_centroid,
            }
        )
    append_remaining_keeps(moved_bn={a.ref.basename for a in added})
    return result



def validate_results(
    results: list[QuestionResult],
    *,
    select_root: Path,
    raw_root: Path,
    cluster_map: dict[str, Path] | None = None,
) -> list[str]:
    errors: list[str] = []
    for r in results:
        keep_threshold = r.min_unique_clusters_required or min_unique_clusters_required(r.K)
        prot_pre = {
            a.ref.basename for a in r.pre_selected if a.ref.selection_source in ("hard", "diverse")
        }
        prot_post = {
            a.ref.basename
            for a in r.post_selected
            if a.ref.selection_source in ("hard", "diverse")
        }
        if prot_pre != prot_post:
            errors.append(f"{r.group_id}: hard/diverse not preserved")

        rand_pre = {
            a.ref.basename for a in r.pre_selected if a.ref.selection_source == "random"
        }
        removed = {a.ref.basename for a in r.removed_random}
        if not removed.issubset(rand_pre):
            errors.append(f"{r.group_id}: removed non-random samples")

        pre_bn = {a.ref.basename for a in r.pre_selected}
        post_bn = {a.ref.basename for a in r.post_selected}

        if r.pre_unique_cluster_count >= keep_threshold:
            if r.added_reps:
                errors.append(f"{r.group_id}: C>=τ should not add representatives")
            post_randoms = [
                a for a in r.post_selected if a.ref.selection_source == "random"
            ]
            by_c: dict[int, list[AssignedSample]] = defaultdict(list)
            for a in post_randoms:
                by_c[int(a.cluster_id)].append(a)
            for cid, members in by_c.items():
                if len(members) > 1:
                    errors.append(
                        f"{r.group_id}: cluster {cid} still has {len(members)} randoms after dedupe"
                    )
            pre_randoms = [a for a in r.pre_selected if a.ref.selection_source == "random"]
            kept_exp, removed_exp = dedupe_randoms_per_cluster(pre_randoms)
            if {a.ref.basename for a in removed_exp} != removed:
                errors.append(f"{r.group_id}: removed random set != expected per-cluster dedupe")
            kept_bn = {a.ref.basename for a in kept_exp}
            post_rand_bn = {a.ref.basename for a in post_randoms}
            if post_rand_bn != kept_bn:
                errors.append(f"{r.group_id}: kept randoms != expected nearest-per-cluster")
            if removed and not r.triggered_replacement:
                errors.append(f"{r.group_id}: dedupe removed samples but triggered_replacement=False")
            if not removed and r.triggered_replacement:
                errors.append(f"{r.group_id}: no dedupe removals but triggered_replacement=True")
            if len(r.post_selected) != 4 + len(kept_exp):
                errors.append(
                    f"{r.group_id}: C>=τ post size {len(r.post_selected)} != 4+{len(kept_exp)}"
                )

        else:  # under coverage
            if not r.triggered_replacement:
                errors.append(f"{r.group_id}: under-coverage should trigger replacement")
            if removed != rand_pre:
                errors.append(f"{r.group_id}: under-coverage must remove all randoms")
            if len(r.added_reps) != 4:
                errors.append(f"{r.group_id}: under-coverage added={len(r.added_reps)} != 4")
            if len(r.post_selected) != 8:
                errors.append(f"{r.group_id}: under-coverage post_selected={len(r.post_selected)}")
            rem_bns = {p.name for p in (raw_root / r.group_id).glob("*.jpg")}
            for a in r.added_reps:
                if a.ref.basename not in rem_bns:
                    errors.append(f"{r.group_id}: added {a.ref.basename} not from stage1 raw")
                if not str(a.ref.image_path).startswith(str(raw_root)):
                    errors.append(f"{r.group_id}: added {a.ref.basename} path not under raw_root")

            if cluster_map is not None and r.group_id in cluster_map:
                cref = load_frozen_cluster(r.group_id, cluster_map[r.group_id])
                protected_cids = {
                    int(a.cluster_id)
                    for a in r.pre_selected
                    if a.ref.selection_source in ("hard", "diverse")
                }
                uncovered = sorted(
                    [c for c in range(cref.K) if c not in protected_cids],
                    key=lambda c: (-cref.cluster_sizes[c], c),
                )
                forbidden = set(prot_pre) | set(removed)
                expected_bns: list[str] = []
                # first try from uncovered in size order
                for c in uncovered:
                    if len(expected_bns) >= 4:
                        break
                    members: list[tuple[str, float]] = []
                    for bn, idx in cref.basename_to_idx.items():
                        if bn in forbidden:
                            continue
                        if int(cref.labels[idx]) != int(c):
                            continue
                        cos = float(np.dot(cref.fused[idx], cref.centroids[c]))
                        members.append((bn, cos))
                    if not members:
                        continue
                    members.sort(key=lambda t: (-t[1], t[0]))
                    best_bn, _ = members[0]
                    expected_bns.append(best_bn)
                    forbidden.add(best_bn)
                # fallback: if uncovered exhausted, use remaining size order
                if len(expected_bns) < 4:
                    size_order = sorted(range(cref.K), key=lambda c: (-cref.cluster_sizes[c], c))
                    for c in size_order:
                        if len(expected_bns) >= 4:
                            break
                        if c in protected_cids and c in {int(cref.labels[cref.basename_to_idx[bn]])
                                                          for bn in forbidden}:
                            continue
                        members: list[tuple[str, float]] = []
                        for bn, idx in cref.basename_to_idx.items():
                            if bn in forbidden:
                                continue
                            if int(cref.labels[idx]) != int(c):
                                continue
                            cos = float(np.dot(cref.fused[idx], cref.centroids[c]))
                            members.append((bn, cos))
                        if not members:
                            continue
                        members.sort(key=lambda t: (-t[1], t[0]))
                        best_bn, _ = members[0]
                        expected_bns.append(best_bn)
                        forbidden.add(best_bn)
                got = [a.ref.basename for a in r.added_reps]
                if got != expected_bns:
                    errors.append(
                        f"{r.group_id}: added reps {got} != expected from uncovered largest clusters {expected_bns}"
                    )

        # set algebra vs stage1 disk
        stage1_sel = {p.name for p in (select_root / r.group_id).glob("*.jpg")}
        stage1_raw = {p.name for p in (raw_root / r.group_id).glob("*.jpg")}
        universe = stage1_sel | stage1_raw
        post_sel = {a.ref.basename for a in r.post_selected}
        if post_sel - universe:
            errors.append(f"{r.group_id}: post select outside universe")
        if post_sel & (universe - post_sel):
            errors.append(f"{r.group_id}: select/raw intersect logically")
        post_raw = universe - post_sel
        if post_sel & post_raw:
            errors.append(f"{r.group_id}: intersect nonempty")
        if post_sel | post_raw != universe:
            errors.append(f"{r.group_id}: union mismatch")
        if len(universe) != len(stage1_sel) + len(stage1_raw):
            errors.append(f"{r.group_id}: stage1 select/raw already intersect")

        for root in (select_root / r.group_id, raw_root / r.group_id):
            for img in root.glob("*.jpg"):
                if not Path(str(img) + ".json").is_file():
                    errors.append(f"missing json {img}")
    return errors



def link_or_copy(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize(results: list[QuestionResult], out_root: Path, select_root: Path, raw_root: Path) -> None:
    sel_out = out_root / "select_dataset"
    raw_out = out_root / "raw_dataset"
    for r in results:
        stage1_sel = {p.name: p for p in (select_root / r.group_id).glob("*.jpg")}
        stage1_raw = {p.name: p for p in (raw_root / r.group_id).glob("*.jpg")}
        universe_img = {**stage1_raw, **stage1_sel}
        post_sel_bn = {a.ref.basename for a in r.post_selected}
        # select
        for bn in sorted(post_sel_bn):
            img = universe_img[bn]
            j = Path(str(img) + ".json")
            dest = sel_out / r.group_id
            link_or_copy(img, dest / bn)
            link_or_copy(j, dest / j.name)
        # raw
        for bn, img in sorted(universe_img.items()):
            if bn in post_sel_bn:
                continue
            j = Path(str(img) + ".json")
            dest = raw_out / r.group_id
            link_or_copy(img, dest / bn)
            link_or_copy(j, dest / j.name)


def build_modified_question_record(r: QuestionResult) -> dict[str, Any]:
    """Rich audit row for a question that triggered random→representative replacement."""
    pre_by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in r.pre_selected:
        pre_by_cluster[int(a.cluster_id)].append(
            {
                "filename": a.ref.basename,
                "selection_source": a.ref.selection_source,
                "sample_id": a.ref.sample_id,
            }
        )

    removed_samples = [
        {
            "filename": a.ref.basename,
            "sample_id": a.ref.sample_id,
            "cluster_id": cluster_label(a.cluster_id),
            "selection_source": a.ref.selection_source,
        }
        for a in r.removed_random
    ]
    removed_by_cluster: dict[int, list[str]] = defaultdict(list)
    for a in r.removed_random:
        removed_by_cluster[int(a.cluster_id)].append(a.ref.basename)

    added_samples = [
        {
            "filename": a.ref.basename,
            "sample_id": a.ref.sample_id,
            "cluster_id": cluster_label(a.cluster_id),
            "selection_source": a.ref.selection_source or "replacement",
        }
        for a in r.added_reps
    ]
    added_by_cluster: dict[int, list[str]] = defaultdict(list)
    for a in r.added_reps:
        added_by_cluster[int(a.cluster_id)].append(a.ref.basename)

    kept = [
        {
            "filename": a.ref.basename,
            "sample_id": a.ref.sample_id,
            "cluster_id": cluster_label(a.cluster_id),
            "selection_source": a.ref.selection_source,
        }
        for a in r.pre_selected
        if a.ref.selection_source in ("hard", "diverse")
    ]

    post_by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in r.post_selected:
        post_by_cluster[int(a.cluster_id)].append(
            {
                "filename": a.ref.basename,
                "selection_source": a.ref.selection_source,
            }
        )

    return {
        "folder": r.group_id,
        "K": r.K,
        "min_unique_clusters_required": r.min_unique_clusters_required,
        "original_unique_cluster_count": r.pre_unique_cluster_count,
        "original_clusters": {
            cluster_label(cid): entries for cid, entries in sorted(pre_by_cluster.items())
        },
        "kept": kept,
        "removed": {
            "count": len(removed_samples),
            "by_cluster": {
                cluster_label(cid): fns for cid, fns in sorted(removed_by_cluster.items())
            },
            "samples": removed_samples,
        },
        "added": {
            "count": len(added_samples),
            "by_cluster": {
                cluster_label(cid): fns for cid, fns in sorted(added_by_cluster.items())
            },
            "samples": added_samples,
        },
        "post_unique_cluster_count": r.post_unique_cluster_count,
        "post_clusters": {
            cluster_label(cid): entries for cid, entries in sorted(post_by_cluster.items())
        },
    }


def write_manifests(results: list[QuestionResult], out_root: Path, *, dry_run: bool) -> dict[str, Any]:
    man = out_root / "manifests"
    aud = out_root / "audit"
    ensure_dir(man)
    ensure_dir(aud)

    # final_selected / final_remaining / sample_transition / question_cluster_summary
    with (man / "final_selected.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "exam_id",
                "question_id",
                "sample_id",
                "post_selection_source",
                "post_cluster_id",
                "post_cluster_size",
                "post_cosine_to_centroid",
                "triggered_replacement",
            ],
        )
        w.writeheader()
        for r in results:
            for a in r.post_selected:
                src = a.ref.selection_source or ""
                w.writerow(
                    {
                        "exam_id": r.exam_id,
                        "question_id": r.question_id,
                        "sample_id": a.ref.sample_id,
                        "post_selection_source": src,
                        "post_cluster_id": cluster_label(a.cluster_id),
                        "post_cluster_size": a.cluster_size,
                        "post_cosine_to_centroid": a.cosine_to_centroid,
                        "triggered_replacement": r.triggered_replacement,
                    }
                )

    with (man / "sample_transition.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "exam_id",
            "question_id",
            "sample_id",
            "original_selection_source",
            "pre_status",
            "pre_cluster_id",
            "pre_cluster_size",
            "pre_cosine_to_centroid",
            "action",
            "replacement_reason",
            "replaces_sample_id",
            "post_status",
            "post_selection_source",
            "post_cluster_id",
            "post_cluster_size",
            "post_cosine_to_centroid",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for t in r.transitions:
                w.writerow({k: t.get(k, "") for k in fields})

    with (man / "final_remaining.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "exam_id",
                "question_id",
                "sample_id",
                "post_cluster_id",
                "post_cluster_size",
                "post_cosine_to_centroid",
            ],
        )
        w.writeheader()
        for r in results:
            for t in r.transitions:
                if t.get("post_status") == "remaining":
                    w.writerow(
                        {
                            "exam_id": t["exam_id"],
                            "question_id": t["question_id"],
                            "sample_id": t["sample_id"],
                            "post_cluster_id": t["post_cluster_id"],
                            "post_cluster_size": t["post_cluster_size"],
                            "post_cosine_to_centroid": t["post_cosine_to_centroid"],
                        }
                    )

    with (man / "question_cluster_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "exam_id",
            "question_id",
            "K",
            "min_unique_clusters_required",
            "reference_sample_count",
            "pre_selected_count",
            "pre_unique_cluster_count",
            "pre_cluster_histogram",
            "triggered_replacement",
            "removed_random_count",
            "removed_random_sample_ids",
            "added_representative_count",
            "added_representative_sample_ids",
            "added_source_cluster_ids",
            "post_selected_count",
            "post_unique_cluster_count",
            "post_cluster_histogram",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "exam_id": r.exam_id,
                    "question_id": r.question_id,
                    "K": r.K,
                    "min_unique_clusters_required": r.min_unique_clusters_required,
                    "reference_sample_count": r.reference_sample_count,
                    "pre_selected_count": len(r.pre_selected),
                    "pre_unique_cluster_count": r.pre_unique_cluster_count,
                    "pre_cluster_histogram": json.dumps(r.pre_cluster_histogram, ensure_ascii=False),
                    "triggered_replacement": r.triggered_replacement,
                    "removed_random_count": len(r.removed_random),
                    "removed_random_sample_ids": json.dumps(
                        [a.ref.sample_id for a in r.removed_random], ensure_ascii=False
                    ),
                    "added_representative_count": len(r.added_reps),
                    "added_representative_sample_ids": json.dumps(
                        [a.ref.sample_id for a in r.added_reps], ensure_ascii=False
                    ),
                    "added_source_cluster_ids": json.dumps(
                        [cluster_label(a.cluster_id) for a in r.added_reps], ensure_ascii=False
                    ),
                    "post_selected_count": len(r.post_selected),
                    "post_unique_cluster_count": r.post_unique_cluster_count,
                    "post_cluster_histogram": json.dumps(r.post_cluster_histogram, ensure_ascii=False),
                }
            )

    question_audit_rows = []
    for r in results:
        question_audit_rows.append(
            {
                "group_id": r.group_id,
                "K": r.K,
                "min_unique_clusters_required": r.min_unique_clusters_required,
                "pre_unique_cluster_count": r.pre_unique_cluster_count,
                "post_unique_cluster_count": r.post_unique_cluster_count,
                "triggered_replacement": r.triggered_replacement,
                "pre_cluster_histogram": r.pre_cluster_histogram,
                "post_cluster_histogram": r.post_cluster_histogram,
                "pre_selected": [
                    {
                        "sample_id": a.ref.sample_id,
                        "selection_source": a.ref.selection_source,
                        "cluster_id": cluster_label(a.cluster_id),
                        "cosine": a.cosine_to_centroid,
                    }
                    for a in r.pre_selected
                ],
                "post_selected": [
                    {
                        "sample_id": a.ref.sample_id,
                        "selection_source": a.ref.selection_source,
                        "cluster_id": cluster_label(a.cluster_id),
                        "cosine": a.cosine_to_centroid,
                    }
                    for a in r.post_selected
                ],
            }
        )
    with (aud / "question_audit.json").open("w", encoding="utf-8") as f:
        json.dump(question_audit_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    unchanged_rows = [
        {
            "folder": r.group_id,
            "K": r.K,
            "min_unique_clusters_required": r.min_unique_clusters_required,
            "original_unique_cluster_count": r.pre_unique_cluster_count,
        }
        for r in results
        if not r.triggered_replacement
    ]
    with (aud / "unchanged_questions.json").open("w", encoding="utf-8") as f:
        json.dump(unchanged_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    modified_rows = [
        build_modified_question_record(r) for r in results if r.triggered_replacement
    ]
    with (aud / "modified_questions.json").open("w", encoding="utf-8") as f:
        json.dump(modified_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    n_mod = sum(1 for r in results if r.triggered_replacement)
    n_keep = len(results) - n_mod
    pre_dist = Counter(r.pre_unique_cluster_count for r in results)
    post_dist = Counter(r.post_unique_cluster_count for r in results)
    n_removed = sum(len(r.removed_random) for r in results)
    n_added = sum(len(r.added_reps) for r in results)
    n_final_selected = len(results) * 8
    n_final_remaining = sum(
        1 for r in results for t in r.transitions if t.get("post_status") == "remaining"
    )
    summary = {
        "dry_run": dry_run,
        "n_questions": len(results),
        "n_unchanged": n_keep,
        "n_modified": n_mod,
        "pre_unique_cluster_count_distribution": {str(k): v for k, v in sorted(pre_dist.items())},
        "post_unique_cluster_count_distribution": {str(k): v for k, v in sorted(post_dist.items())},
        "n_removed_random_total": n_removed,
        "n_added_representative_total": n_added,
        "n_final_selected_pairs": n_final_selected,
        "n_final_remaining_pairs": n_final_remaining,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(man / "stage2_summary.json", summary)
    return summary


def inspect_and_print(args: argparse.Namespace) -> None:
    print("=== Stage2 input inspection ===", flush=True)
    for name, p in [
        ("CLUSTER", args.cluster_root),
        ("RAW", args.raw_root),
        ("SELECT", args.select_root),
        ("AUDIT", args.audit_json),
    ]:
        print(f"{name}: exists={Path(p).exists()} path={p}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage2: adjust selected candidates via frozen clusters")
    p.add_argument("--cluster-root", default=DEFAULT_CLUSTER)
    p.add_argument("--raw-root", default=DEFAULT_RAW)
    p.add_argument("--select-root", default=DEFAULT_SELECT)
    p.add_argument("--audit-json", default=DEFAULT_AUDIT)
    p.add_argument("--vision-feature-root", default=DEFAULT_VIS)
    p.add_argument("--text-feature-root", default=DEFAULT_TXT)
    p.add_argument("--output-root", default=DEFAULT_OUT)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    inspect_and_print(args)
    t0 = time.time()

    select_root = Path(args.select_root)
    raw_root = Path(args.raw_root)
    cluster_root = Path(args.cluster_root)
    audit_path = Path(args.audit_json)
    out_root = Path(args.output_root)

    if not audit_path.is_file():
        print(f"FATAL: missing selection audit {audit_path}", flush=True)
        return 2
    sources_by_q = load_selection_audit(audit_path)
    print(f"[audit] questions={len(sources_by_q)}", flush=True)

    cluster_map = discover_successful_cluster_groups(cluster_root)
    print(f"[cluster] successful_groups={len(cluster_map)}", flush=True)

    missing = sorted(set(sources_by_q) - set(cluster_map))
    extra = sorted(set(cluster_map) - set(sources_by_q))
    if missing:
        print(
            f"FATAL: {len(missing)} select questions lack successful frozen cluster results. "
            f"Examples: {missing[:10]}",
            flush=True,
        )
        print("Wait for raw_dataset_cluster full run to finish (need PCA/centroids).", flush=True)
        return 3
    if extra:
        print(f"[warn] {len(extra)} cluster groups not in select audit (ignored)", flush=True)

    vis = load_vision_store(args.vision_feature_root)
    txt = load_text_store(args.text_feature_root)

    group_ids = sorted(sources_by_q.keys())
    if args.limit > 0:
        group_ids = group_ids[: args.limit]

    results: list[QuestionResult] = []
    for i, gid in enumerate(group_ids, 1):
        print(f"[{i}/{len(group_ids)}] {gid}", flush=True)
        cref = load_frozen_cluster(gid, cluster_map[gid])
        r = process_question(
            gid,
            sources_by_q[gid],
            cref,
            select_root=select_root,
            raw_root=raw_root,
            vis=vis,
            txt=txt,
        )
        results.append(r)
        print(
            f"  K={r.K} τ={r.min_unique_clusters_required} "
            f"pre_unique={r.pre_unique_cluster_count} trigger={r.triggered_replacement} "
            f"removed={len(r.removed_random)} added={len(r.added_reps)} "
            f"post_n={len(r.post_selected)} post_unique={r.post_unique_cluster_count}",
            flush=True,
        )

    errors = validate_results(
        results, select_root=select_root, raw_root=raw_root, cluster_map=cluster_map
    )
    if errors:
        print("VALIDATION FAILED:", flush=True)
        for e in errors[:50]:
            print(" -", e, flush=True)
        return 4

    # write to temp then move
    ensure_dir(out_root.parent)
    tmp = Path(tempfile.mkdtemp(prefix="stage2_", dir=str(out_root.parent)))
    try:
        summary = write_manifests(results, tmp, dry_run=bool(args.dry_run))
        cfg = {
            "cluster_root": args.cluster_root,
            "raw_root": args.raw_root,
            "select_root": args.select_root,
            "audit_json": args.audit_json,
            "vision_feature_root": args.vision_feature_root,
            "text_feature_root": args.text_feature_root,
            "output_root": args.output_root,
            "mode": "dry-run" if args.dry_run else "materialize",
            "rule": (
                "τ=ceil(K/2); "
                "if C>=τ: per-cluster keep nearest random only (drop other randoms); "
                "if C<τ: drop all randoms and add centroid reps from 4 largest clusters"
            ),
            "frozen_cluster": True,
            "elapsed_sec_so_far": time.time() - t0,
        }
        with (tmp / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        if args.materialize:
            materialize(results, tmp, select_root, raw_root)
            # post materialize path checks
            for r in results:
                for a in r.post_selected:
                    ip = tmp / "select_dataset" / r.group_id / a.ref.basename
                    jp = Path(str(ip) + ".json")
                    if not ip.is_file() or not jp.is_file():
                        raise RuntimeError(f"missing materialized {ip}")

        # replace output root atomically
        if out_root.exists():
            bak = out_root.with_name(out_root.name + f".bak_{int(time.time())}")
            out_root.rename(bak)
        tmp.rename(out_root)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # print dry-run summary
    print("\n========== Stage2 summary ==========", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("validation: PASSED", flush=True)
    print(f"output: {out_root}", flush=True)
    print(f"elapsed_sec: {time.time() - t0:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
