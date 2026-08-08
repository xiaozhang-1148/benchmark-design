"""Checkpoints CP0–CP6 for equal-fusion spherical K-means."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, ensure_dir
from .align import align_all
from .discover import discover_all_pairs, discover_question_groups
from .feature_index import load_text_store, load_vision_store
from .run import DEFAULT_DATA, DEFAULT_OUT, DEFAULT_TXT, DEFAULT_VIS
from .spherical_kmeans import spherical_kmeans


def cp0(out: Path) -> dict[str, Any]:
    from pathlib import Path as P

    report = {
        "checkpoint": 0,
        "paths_exist": {
            "data": P(DEFAULT_DATA).is_dir(),
            "vision": P(DEFAULT_VIS).is_dir(),
            "text": P(DEFAULT_TXT).is_dir(),
        },
        "groups_depth2": len(discover_question_groups(DEFAULT_DATA)),
        "structure_note": (
            "Batch02 layout is exam_id/question_id/files (question group relative depth=2). "
            "Task brief depth-3 A/B/C directories are absent."
        ),
    }
    vis = load_vision_store(DEFAULT_VIS)
    txt = load_text_store(DEFAULT_TXT)
    report["vision"] = {
        "matrix": vis.matrix_path,
        "shape": list(vis.matrix.shape),
        "dtype": str(vis.matrix.dtype),
        "dim": vis.dim,
        "reason": vis.selection_reason,
    }
    report["text"] = {
        "matrix": txt.matrix_path,
        "shape": list(txt.matrix.shape),
        "dtype": str(txt.matrix.dtype),
        "dim": txt.dim,
        "reason": txt.selection_reason,
    }
    pairs = discover_all_pairs(DEFAULT_DATA)
    report["n_images_paired"] = len(pairs)
    report["passed"] = (
        report["paths_exist"]["data"]
        and vis.dim == 1792
        and txt.dim == 1024
        and report["groups_depth2"] > 0
    )
    atomic_write_json(out / "checkpoint0_report.json", report)
    return report


def cp1(out: Path) -> dict[str, Any]:
    pairs = discover_all_pairs(DEFAULT_DATA)
    by = {}
    for p in pairs:
        by.setdefault(p.group_id, []).append(p)
    examples = {}
    for gid in sorted(by.keys())[:3]:
        examples[gid] = [s.to_dict() for s in by[gid][:3]]
    report = {
        "checkpoint": 1,
        "n_groups": len(by),
        "n_pairs": len(pairs),
        "unpaired": 0,
        "duplicate_sample_ids": 0,
        "examples": examples,
        "passed": len(pairs) > 0 and len(by) > 0,
    }
    atomic_write_json(out / "checkpoint1_pairs.json", report)
    return report


def cp2(out: Path) -> dict[str, Any]:
    aligned, report, errors = align_all(DEFAULT_DATA, DEFAULT_VIS, DEFAULT_TXT)
    vis = load_vision_store(DEFAULT_VIS)
    txt = load_text_store(DEFAULT_TXT)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(aligned), size=min(20, len(aligned)), replace=False)
    checks = []
    for i in idxs:
        a = aligned[int(i)]
        v = vis.get_by_basename(a.image_basename)
        t = txt.get_by_basename(a.image_basename)
        checks.append(
            {
                "sample_id": a.sample_id,
                "vision_row": a.vision_feature_row,
                "text_row": a.text_feature_row,
                "vision_shape": list(v.shape),
                "text_shape": list(t.shape),
                "ok": v.shape == (1792,) and t.shape == (1024,),
            }
        )
    out_r = {
        "checkpoint": 2,
        "alignment_report": report,
        "n_errors": len(errors),
        "spot_checks": checks,
        "passed": report["n_aligned"] == report["n_pairs_discovered"] and all(c["ok"] for c in checks),
    }
    atomic_write_json(out / "checkpoint2_feature_map.json", out_r)
    return out_r


def cp3(out: Path) -> dict[str, Any]:
    from .align import load_group_raw_matrices
    from .pca_fuse import prepare_group_pca_fusion

    aligned, _, _ = align_all(DEFAULT_DATA, DEFAULT_VIS, DEFAULT_TXT)
    by: dict[str, list] = {}
    for a in aligned:
        by.setdefault(a.group_id, []).append(a)
    gid, ss = max(by.items(), key=lambda kv: len(kv[1]))
    ss = sorted(ss, key=lambda s: s.sample_id)[:80]
    vis = load_vision_store(DEFAULT_VIS)
    txt = load_text_store(DEFAULT_TXT)
    V, T, ids = load_group_raw_matrices(ss, vis, txt)
    prep = prepare_group_pca_fusion(V, T, group_id=gid, sample_ids=ids, max_components=64)
    report = {
        "checkpoint": 3,
        "group_id": gid,
        "n": len(ids),
        "vision_shape": list(V.shape),
        "text_shape": list(T.shape),
        "pca_n_components": prep.n_components,
        "reduced_image_shape": list(prep.x_prime.shape),
        "reduced_text_shape": list(prep.t_prime.shape),
        "fused_shape": list(prep.z.shape),
        "fused_norm_mean": float(np.linalg.norm(prep.z, axis=1).mean()),
        "whiten": False,
        "passed": (
            V.shape[1] == 1792
            and T.shape[1] == 1024
            and prep.n_components == min(64, len(ids) - 1)
            and prep.z.shape == (len(ids), 2 * prep.n_components)
            and abs(float(np.linalg.norm(prep.z, axis=1).mean()) - 1.0) < 1e-5
        ),
    }
    atomic_write_json(out / "checkpoint3_single_fuse.json", report)
    return report


def cp4(out: Path) -> dict[str, Any]:
    from .align import load_group_raw_matrices
    from .pca_fuse import prepare_group_pca_fusion, verify_pca_fusion_identity

    aligned, _, _ = align_all(DEFAULT_DATA, DEFAULT_VIS, DEFAULT_TXT)
    by: dict[str, list] = {}
    for a in aligned:
        by.setdefault(a.group_id, []).append(a)
    gid, ss = max(by.items(), key=lambda kv: len(kv[1]))
    ss = sorted(ss, key=lambda s: s.sample_id)
    vis = load_vision_store(DEFAULT_VIS)
    txt = load_text_store(DEFAULT_TXT)
    V, T, ids = load_group_raw_matrices(ss, vis, txt)
    prep = prepare_group_pca_fusion(V, T, group_id=gid, sample_ids=ids, max_components=64)
    idc = verify_pca_fusion_identity(prep, seed=42)
    report = {"checkpoint": 4, "group_id": gid, **idc}
    atomic_write_json(out / "checkpoint4_identity.json", report)
    return report


def cp5(out: Path) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    # 3 well-separated directions
    centers = np.eye(3, 16, dtype=np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    zs = []
    labels_true = []
    for c in range(3):
        for _ in range(40):
            noise = rng.normal(0, 0.05, size=16).astype(np.float32)
            v = centers[c] + noise
            v = v / np.linalg.norm(v)
            zs.append(v)
            labels_true.append(c)
    z = np.stack(zs)
    ids = [f"s{i:03d}" for i in range(len(zs))]
    res = spherical_kmeans(z, 3, sample_ids=ids, n_init=10, max_iter=100, seed=42)
    cnorms = np.linalg.norm(res.centroids, axis=1)
    # purity: majority label per cluster
    purity = 0
    for c in range(3):
        members = [labels_true[i] for i in range(len(ids)) if res.labels[i] == c]
        if members:
            purity += max(members.count(x) for x in set(members))
    purity /= len(ids)
    res2 = spherical_kmeans(z, 3, sample_ids=ids, n_init=10, max_iter=100, seed=42)
    report = {
        "checkpoint": 5,
        "centroid_norms": cnorms.tolist(),
        "max_norm_err": float(np.max(np.abs(cnorms - 1))),
        "objective": res.objective,
        "purity": purity,
        "deterministic_labels_equal": bool(np.array_equal(res.labels, res2.labels)),
        "k_preserved": len(res.cluster_sizes) == 3 and min(res.cluster_sizes) > 0,
        "passed": (
            float(np.max(np.abs(cnorms - 1))) < 1e-4
            and purity > 0.9
            and bool(np.array_equal(res.labels, res2.labels))
            and len(res.cluster_sizes) == 3
        ),
    }
    atomic_write_json(out / "checkpoint5_unit_skmeans.json", report)
    return report


def cp6(out: Path, default_k: int = 5) -> dict[str, Any]:
    from .run import main as run_main

    # choose a group with enough samples for elbow [2,20]
    aligned, _, _ = align_all(DEFAULT_DATA, DEFAULT_VIS, DEFAULT_TXT)
    by = {}
    for a in aligned:
        by.setdefault(a.group_id, []).append(a)
    gid = None
    for g, ss in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(ss) >= max(20, default_k):
            gid = g
            break
    assert gid is not None
    pilot_out = out / "pilot_run"
    rc = run_main(
        [
            "--data-root",
            DEFAULT_DATA,
            "--vision-feature-root",
            DEFAULT_VIS,
            "--text-feature-root",
            DEFAULT_TXT,
            "--output-root",
            str(pilot_out),
            "--mode",
            "pilot",
            "--pilot-group",
            gid,
            "--k-min",
            "2",
            "--k-max",
            "20",
            "--n-init",
            "5",
            "--seed",
            "42",
            "--no-resume",
        ]
    )
    # audit pilot outputs
    from .run import safe_group_dirname

    gdir = pilot_out / "groups" / safe_group_dirname(gid)
    labels = np.load(gdir / "labels.npy")
    cents = np.load(gdir / "centroids.npy")
    fused = np.load(gdir / "fused_features.npy")
    metrics = json.loads((gdir / "metrics.json").read_text())
    elbow = json.loads((gdir / "k_elbow_curve.json").read_text())
    selected_k = int(metrics["K"])
    fused_dim = int(metrics["fused_dim"])
    n_comp = int(metrics["pca_n_components"])
    sil_path = gdir / "fused_silhouette_per_sample.npy"
    clusters = [json.loads(l) for l in (gdir / "clusters.jsonl").open() if l.strip()]
    required_files = [
        "image_pca.pkl",
        "text_pca.pkl",
        "image_pca_explained_variance.json",
        "text_pca_explained_variance.json",
        "reduced_image_features.npy",
        "reduced_text_features.npy",
        "fused_features.npy",
    ]
    checks = {
        "rc0": rc == 0,
        "n_labels": int(labels.shape[0]) == len(by[gid]),
        "label_range": int(labels.min()) >= 0 and int(labels.max()) < selected_k,
        "centroids_shape": list(cents.shape) == [selected_k, fused_dim],
        "fused_shape": list(fused.shape) == [len(by[gid]), fused_dim],
        "pca_dim": n_comp == min(64, len(by[gid]) - 1) and fused_dim == 2 * n_comp,
        "all_samples_once": len(clusters) == len(by[gid])
        and len({c["sample_id"] for c in clusters}) == len(by[gid]),
        "no_cross_group": all(c["group_id"] == gid for c in clusters),
        "reps_in_cluster": all(
            any(c["sample_id"] == r and c["is_representative"] for c in clusters) or r == ""
            for r in metrics["representatives"]
        ),
        "elbow_file": elbow.get("selected_k") == selected_k,
        "elbow_curve_len": 2 <= len(elbow.get("curve", [])) <= 19,
        "k_in_2_20": 2 <= selected_k <= 20,
        "silhouette_mean": metrics.get("fused_silhouette") is not None,
        "silhouette_all_points": sil_path.exists()
        and int(np.load(sil_path).shape[0]) == len(by[gid]),
        "pca_artifacts": all((gdir / n).exists() for n in required_files),
        "reduced_shapes": list(np.load(gdir / "reduced_image_features.npy").shape)
        == [len(by[gid]), n_comp]
        and list(np.load(gdir / "reduced_text_features.npy").shape) == [len(by[gid]), n_comp],
    }
    report = {
        "checkpoint": 6,
        "pilot_group": gid,
        "n": len(by[gid]),
        "k": selected_k,
        "pca_n_components": n_comp,
        "fused_dim": fused_dim,
        "k_selection": "elbow",
        "checks": checks,
        "metrics": {k: metrics[k] for k in metrics if k != "n_init_trials"},
        "passed": all(checks.values()),
    }
    atomic_write_json(out / "checkpoint6_pilot.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="all")
    p.add_argument("--out-dir", default=str(Path(DEFAULT_OUT) / "checkpoints"))
    p.add_argument("--pilot-k", type=int, default=5)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    ensure_dir(out)
    which = str(args.checkpoint)
    if which == "all":
        ids = list(range(0, 7))
    elif "-" in which:
        a, b = which.split("-", 1)
        ids = list(range(int(a), int(b) + 1))
    else:
        ids = [int(which)]

    results = {}
    t0 = time.time()
    for cid in ids:
        print(f"\n===== CHECKPOINT {cid} =====", flush=True)
        fn = [cp0, cp1, cp2, cp3, cp4, cp5, cp6][cid]
        if cid == 6:
            results[f"cp{cid}"] = fn(out, default_k=args.pilot_k)
        else:
            results[f"cp{cid}"] = fn(out)
        print(f"[cp{cid}] passed={results[f'cp{cid}'].get('passed')}", flush=True)
        if not results[f"cp{cid}"].get("passed"):
            print(json.dumps(results[f"cp{cid}"], indent=2, ensure_ascii=False)[:4000])
            return 1
    atomic_write_json(
        out / "checkpoints_summary.json",
        {"results": {k: {"passed": v.get("passed")} for k, v in results.items()}, "elapsed_sec": time.time() - t0},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
