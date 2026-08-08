"""CLI: equal-weight fusion + per-question spherical K-means."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, ensure_dir
from .align import align_all, load_group_raw_matrices, write_alignment
from .feature_index import load_text_store, load_vision_store
from .pca_fuse import (
    DEFAULT_PCA_MAX_COMPONENTS,
    prepare_group_pca_fusion,
    save_group_pca_artifacts,
    verify_pca_fusion_identity,
)
from .plots import plot_k_selection_curves
from .spherical_kmeans import (
    check_unit_norms,
    compute_assigned_cosine_and_distortions,
    cosine_silhouette,
    spherical_kmeans,
    sweep_spherical_kmeans_elbow,
)

DEFAULT_DATA = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/raw_dataset"
DEFAULT_VIS = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/vision_deatures"
DEFAULT_TXT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/text_feature"
DEFAULT_OUT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/raw_dataset_cluster"


def safe_group_dirname(group_id: str) -> str:
    # Keep readable path separators as __
    return group_id.replace("/", "__").replace("\\", "__")


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp"
    np.save(str(tmp), arr)
    saved = Path(str(tmp) + ".npy")
    os.replace(saved, path)


def load_k_map(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in raw.items()}


def resolve_fixed_k(group_id: str, k_map: dict[str, int], default_k: int | None) -> int | None:
    """Explicit K from k-map or --default-k. None means use elbow sweep."""
    if group_id in k_map:
        return int(k_map[group_id])
    if default_k is not None:
        return int(default_k)
    return None


def config_fingerprint(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Equal-weight fusion + spherical K-means by question")
    p.add_argument("--data-root", default=DEFAULT_DATA)
    p.add_argument("--vision-feature-root", default=DEFAULT_VIS)
    p.add_argument("--text-feature-root", default=DEFAULT_TXT)
    p.add_argument("--output-root", default=DEFAULT_OUT)
    p.add_argument(
        "--default-k",
        type=int,
        default=None,
        help="If set, fix K for all groups (overrides elbow). Prefer k-map when both set per group.",
    )
    p.add_argument("--k-map-json", type=str, default=None)
    p.add_argument("--k-min", type=int, default=2, help="Elbow sweep lower bound (inclusive)")
    p.add_argument("--k-max", type=int, default=30, help="Elbow sweep upper bound (inclusive)")
    p.add_argument(
        "--pca-max-components",
        type=int,
        default=DEFAULT_PCA_MAX_COMPONENTS,
        help="Per-group PCA n_components = min(this, N-1, dim)",
    )
    p.add_argument(
        "--no-elbow",
        action="store_true",
        help="Disable elbow; require --default-k or --k-map-json for every group",
    )
    p.add_argument("--n-init", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["preflight", "pilot", "full"], default="full")
    p.add_argument("--pilot-group", type=str, default=None, help="group_id for pilot mode")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--materialize-symlinks", action="store_true")
    p.add_argument(
        "--skip-silhouette",
        action="store_true",
        help="Skip cosine silhouette (default: compute on all points for each K and final)",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip writing k_selection_curves.png / k_vs_cosine_silhouette.png",
    )
    return p


def _attach_silhouette_metrics(
    metrics: dict[str, Any],
    *,
    z: np.ndarray,
    x_hat: np.ndarray,
    t_hat: np.ndarray,
    labels: np.ndarray,
    gdir: Path,
    skip: bool,
) -> None:
    if skip:
        return
    fused = cosine_silhouette(z, labels, return_per_sample=True)
    img = cosine_silhouette(x_hat, labels, return_per_sample=True)
    txt = cosine_silhouette(t_hat, labels, return_per_sample=True)
    if fused is not None:
        mean_f, per_f = fused
        metrics["fused_silhouette"] = mean_f
        metrics["fused_silhouette_n"] = int(per_f.shape[0])
        atomic_save_npy(gdir / "fused_silhouette_per_sample.npy", per_f)
    else:
        metrics["fused_silhouette"] = None
    if img is not None:
        mean_i, per_i = img
        metrics["image_space_silhouette"] = mean_i
        atomic_save_npy(gdir / "image_silhouette_per_sample.npy", per_i)
    else:
        metrics["image_space_silhouette"] = None
    if txt is not None:
        mean_t, per_t = txt
        metrics["text_space_silhouette"] = mean_t
        atomic_save_npy(gdir / "text_silhouette_per_sample.npy", per_t)
    else:
        metrics["text_space_silhouette"] = None


def run_group(
    group_id: str,
    samples: list[Any],
    *,
    vis,
    txt,
    k: int | None,
    out_dir: Path,
    args: argparse.Namespace,
    cfg_fp: str,
) -> dict[str, Any]:
    """
    Cluster one question group.

    If ``k`` is None, sweep K in [k_min, k_max] (clamped to N) and pick elbow K
    on distortion = N - spherical_objective. Cosine silhouette is computed on
    all points for the selected clustering (unless --skip-silhouette).
    """
    gdir = out_dir / "groups" / safe_group_dirname(group_id)
    ensure_dir(gdir)
    manifest_path = gdir / "group_manifest.json"
    use_elbow = k is None

    group_cfg: dict[str, Any] = {
        "group_id": group_id,
        "k": k,
        "k_selection": "elbow" if use_elbow else "fixed",
        "k_min": args.k_min,
        "k_max": args.k_max,
        "pca_max_components": args.pca_max_components,
        "pca_whiten": False,
        "n": len(samples),
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "config_fingerprint": cfg_fp,
        "vision_source": vis.matrix_path,
        "text_source": txt.matrix_path,
        "pipeline": "rowL2->PCA(min(64,N-1))->rowL2->concat->rowL2->spherical_kmeans(Z)",
    }

    resume = args.resume and not args.no_resume
    if resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            old.get("status") == "success"
            and old.get("config_fingerprint") == cfg_fp
            and old.get("n") == len(samples)
            and (
                (use_elbow and old.get("k_selection") == "elbow")
                or (not use_elbow and old.get("k") == k)
            )
            and (gdir / "image_pca.pkl").exists()
            and (gdir / "text_pca.pkl").exists()
            and (gdir / "fused_features.npy").exists()
        ):
            return {
                "group_id": group_id,
                "status": "skipped_resume",
                "k": old.get("k"),
                "n": len(samples),
            }

    try:
        V, T, ids = load_group_raw_matrices(samples, vis, txt)
        prep = prepare_group_pca_fusion(
            V,
            T,
            group_id=group_id,
            sample_ids=ids,
            max_components=args.pca_max_components,
        )
        save_group_pca_artifacts(prep, gdir)
        x_prime, t_prime, z = prep.x_prime, prep.t_prime, prep.z
        n = len(ids)
        fused_dim = prep.fused_dim
        group_cfg["pca_n_components"] = prep.n_components
        group_cfg["fused_dim"] = fused_dim
        elbow_info: dict[str, Any] | None = None

        if use_elbow:
            if n < 3:
                raise RuntimeError(f"elbow clustering requires N>=3, got N={n}")
            sweep = sweep_spherical_kmeans_elbow(
                z,
                ids,
                k_min=args.k_min,
                k_max=args.k_max,
                n_init=args.n_init,
                max_iter=args.max_iter,
                tol=args.tol,
                seed=args.seed,
                compute_silhouette=not args.skip_silhouette,
            )
            result = sweep["result"]
            k = int(sweep["selected_k"])
            elbow_info = {
                "k_min": sweep["k_min"],
                "k_max": sweep["k_max"],
                "n_init": sweep["n_init"],
                "selection_method": sweep["selection_method"],
                "objective_definition": sweep["objective_definition"],
                "distortion_definition": sweep["distortion_definition"],
                "distortion_name": sweep["distortion_name"],
                "selected_k": k,
                "broad_elbow": sweep.get("broad_elbow"),
                "elbow_95pct_interval": sweep.get("elbow_95pct_interval"),
                "elbow_95pct_ks": sweep["elbow"].get("elbow_95pct_ks"),
                "top_elbow_candidates": sweep.get("top_elbow_candidates"),
                "method": sweep["elbow"]["method"],
                "tie": sweep["elbow"].get("tie"),
                "tied_ks": sweep["elbow"].get("tied_ks"),
                "curve": sweep["curve"],
                "perpendicular_distances": sweep["elbow"].get("perpendicular_distances"),
                "normalized_k": sweep["elbow"].get("normalized_k"),
                "normalized_distortion": sweep["elbow"].get("normalized_distortion"),
                "monotonicity_warnings": sweep.get("monotonicity_warnings") or [],
                "silhouette_role": "diagnostic_only",
            }
            # diagnostic max-silhouette K (does NOT select K_q)
            sils = [c.get("fused_silhouette") for c in sweep["curve"]]
            max_sil_k = None
            if all(s is not None for s in sils):
                best_sil_i = int(np.argmax(np.asarray(sils, dtype=np.float64)))
                max_sil_k = int(sweep["curve"][best_sil_i]["k"])
                elbow_info["max_silhouette_k"] = max_sil_k
                elbow_info["max_silhouette"] = float(sils[best_sil_i])  # type: ignore[arg-type]
            if sweep.get("monotonicity_warnings"):
                elbow_info["elbow_curve_monotonicity_warning"] = True
            atomic_write_json(gdir / "k_elbow_curve.json", elbow_info)
            group_cfg["k"] = k
            if not args.skip_plots:
                plot_k_selection_curves(
                    sweep["curve"],
                    selected_k=k,
                    out_path=gdir / "k_selection_curves.png",
                    title="elbow method: max perpendicular distance to normalized endpoint chord",
                    group_id=group_id,
                    max_silhouette_k=max_sil_k,
                )
        else:
            assert k is not None
            if k > n:
                raise RuntimeError(f"K={k} > N={n}")
            if k < 1:
                raise RuntimeError(f"K={k} invalid")
            result = spherical_kmeans(
                z,
                k,
                sample_ids=ids,
                n_init=args.n_init,
                max_iter=args.max_iter,
                tol=args.tol,
                seed=args.seed,
            )

        # metrics
        check_unit_norms(z, result.centroids, atol=1e-5)
        dist_info = compute_assigned_cosine_and_distortions(z, result.labels, result.centroids)
        cos_assigned = dist_info["assigned_cosine"]
        metrics: dict[str, Any] = {
            "N_group": n,
            "K": k,
            "k_selection": "elbow" if use_elbow else "fixed",
            "k_search": {
                "k_min": args.k_min,
                "k_max": args.k_max,
                "n_init": args.n_init,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "seed": args.seed,
                "seed_formula": "base_seed + 1000 * K + init_index",
                "selection_method": "max_perpendicular_distance_to_chord",
                "silhouette_role": "diagnostic_only",
            },
            "pca_n_components": prep.n_components,
            "fused_dim": fused_dim,
            "best_objective": float(result.objective),
            "spherical_kmeans_total_cosine_distortion": dist_info[
                "spherical_kmeans_total_cosine_distortion"
            ],
            "spherical_kmeans_mean_cosine_distortion": dist_info[
                "spherical_kmeans_mean_cosine_distortion"
            ],
            "distortion_crosscheck_abs_err": dist_info["distortion_crosscheck_abs_err"],
            "distortion": dist_info["spherical_kmeans_total_cosine_distortion"],
            "mean_cosine_to_assigned_centroid": float(cos_assigned.mean()),
            "min_cosine_to_assigned_centroid": float(cos_assigned.min()),
            "iterations": result.iterations,
            "converged": result.converged,
            "cluster_sizes": result.cluster_sizes,
            "empty_cluster_reinitializations": result.empty_reinitializations,
            "best_init_index": result.best_init_index,
            "n_init_trials": result.n_init_trials,
            "representatives": result.representatives,
            "pca_fusion_identity": verify_pca_fusion_identity(prep, seed=args.seed),
        }
        if elbow_info is not None:
            metrics["elbow"] = {
                "method": elbow_info["method"],
                "selection_method": elbow_info["selection_method"],
                "k_min": elbow_info["k_min"],
                "k_max": elbow_info["k_max"],
                "n_init": elbow_info["n_init"],
                "selected_k": elbow_info["selected_k"],
                "broad_elbow": elbow_info.get("broad_elbow"),
                "elbow_95pct_interval": elbow_info.get("elbow_95pct_interval"),
                "elbow_95pct_ks": elbow_info.get("elbow_95pct_ks"),
                "top_elbow_candidates": elbow_info.get("top_elbow_candidates"),
                "max_silhouette_k": elbow_info.get("max_silhouette_k"),
                "silhouette_role": "diagnostic_only",
                "monotonicity_warnings": elbow_info.get("monotonicity_warnings"),
                "curve": elbow_info["curve"],
            }
        _attach_silhouette_metrics(
            metrics,
            z=z,
            x_hat=x_prime,
            t_hat=t_prime,
            labels=result.labels,
            gdir=gdir,
            skip=args.skip_silhouette,
        )
        if metrics.get("fused_silhouette") is not None:
            metrics["silhouette_diagnostic_only"] = True

        # clustering outputs (PCA/fusion already saved)
        atomic_save_npy(gdir / "labels.npy", result.labels)
        atomic_save_npy(gdir / "centroids.npy", result.centroids)

        with (gdir / "aligned_samples.jsonl").open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        with (gdir / "clusters.jsonl").open("w", encoding="utf-8") as f:
            for i, s in enumerate(samples):
                cid = int(result.labels[i])
                row = {
                    "group_id": group_id,
                    "sample_id": s.sample_id,
                    "image_path": s.image_path,
                    "json_path": s.json_path,
                    "cluster_id": cid,
                    "cosine_to_centroid": float(cos_assigned[i]),
                    "is_representative": s.sample_id == result.representatives[cid],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        atomic_write_json(gdir / "metrics.json", metrics)
        (gdir / "errors.jsonl").write_text("", encoding="utf-8")

        if args.materialize_symlinks:
            for cid in range(k):
                cdir = gdir / "cluster_views" / f"c{cid:03d}"
                ensure_dir(cdir)
                for i, s in enumerate(samples):
                    if int(result.labels[i]) != cid:
                        continue
                    for src, tag in [(s.image_path, "img"), (s.json_path, "json")]:
                        if tag == "json":
                            dst = cdir / Path(s.json_path).name
                        else:
                            dst = cdir / Path(s.image_path).name
                        if dst.exists() or dst.is_symlink():
                            continue
                        try:
                            os.symlink(src, dst)
                        except OSError:
                            pass

        # validate membership
        assert len(result.labels) == len(samples)
        assert set(result.labels.tolist()) <= set(range(k))
        assert result.centroids.shape == (k, fused_dim)
        cnorms = np.linalg.norm(result.centroids, axis=1)
        if float(np.max(np.abs(cnorms - 1.0))) > 1e-5:
            raise RuntimeError(f"centroid norms not unit: {cnorms}")
        # labels must equal argmax reassignment
        re_sims = z @ result.centroids.T
        re_lab = np.argmax(re_sims, axis=1).astype(np.int64)
        if not np.array_equal(re_lab, result.labels):
            raise RuntimeError("saved labels != argmax assignment to centroids")
        required = [
            "image_pca.pkl",
            "text_pca.pkl",
            "image_pca_explained_variance.json",
            "text_pca_explained_variance.json",
            "reduced_image_features.npy",
            "reduced_text_features.npy",
            "fused_features.npy",
        ]
        for name in required:
            if not (gdir / name).exists():
                raise RuntimeError(f"missing required artifact: {name}")

        man = {
            **group_cfg,
            "status": "success",
            "objective": result.objective,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(manifest_path, man)
        return {
            "group_id": group_id,
            "status": "success",
            "k": k,
            "n": len(samples),
            "metrics": metrics,
        }

    except Exception as e:  # noqa: BLE001
        err = {"group_id": group_id, "error": f"{type(e).__name__}: {e}"}
        with (gdir / "errors.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
        atomic_write_json(
            manifest_path,
            {**group_cfg, "status": "failed", "error": err["error"]},
        )
        return {
            "group_id": group_id,
            "status": "failed",
            "k": k,
            "n": len(samples),
            "error": err["error"],
        }

def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out = Path(args.output_root)
    ensure_dir(out)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    print("[preflight] aligning…", flush=True)
    aligned, report, errors = align_all(args.data_root, args.vision_feature_root, args.text_feature_root)
    write_alignment(aligned, report, errors, out)
    print(
        f"[preflight] pairs={report['n_pairs_discovered']} aligned={report['n_aligned']} "
        f"errors={report['n_errors']} groups={report['n_groups']}",
        flush=True,
    )
    print(f"[preflight] note: {report['group_depth_note']}", flush=True)

    if report["n_errors"] > 0 or report["n_aligned"] == 0:
        print("[fatal] alignment incomplete; refusing to cluster", flush=True)
        atomic_write_json(
            out / "run_manifest.json",
            {"status": "failed_alignment", "report": report, "started_utc": started},
        )
        return 2

    # Per-group PCA+fusion identity on one large-enough question group
    vis = load_vision_store(args.vision_feature_root)
    txt = load_text_store(args.text_feature_root)
    by_tmp: dict[str, list] = {}
    for a in aligned:
        by_tmp.setdefault(a.group_id, []).append(a)
    probe_gid, probe_ss = max(by_tmp.items(), key=lambda kv: len(kv[1]))
    probe_ss = sorted(probe_ss, key=lambda s: s.sample_id)
    V, T, ids = load_group_raw_matrices(probe_ss, vis, txt)
    prep = prepare_group_pca_fusion(
        V,
        T,
        group_id=probe_gid,
        sample_ids=ids,
        max_components=args.pca_max_components,
    )
    id_check = verify_pca_fusion_identity(prep, seed=args.seed)
    id_check["probe_group_id"] = probe_gid
    id_check["probe_n"] = len(ids)
    atomic_write_json(out / "alignment" / "fusion_identity_check.json", id_check)
    print(f"[preflight] PCA+fusion identity: {id_check}", flush=True)
    if not id_check["passed"]:
        print("[fatal] PCA equal-weight identity failed", flush=True)
        return 3

    if args.mode == "preflight":
        atomic_write_json(
            out / "run_manifest.json",
            {
                "status": "preflight_ok",
                "alignment": report,
                "fusion_identity": id_check,
                "started_utc": started,
                "elapsed_sec": time.time() - t0,
            },
        )
        return 0

    k_map = load_k_map(args.k_map_json)
    by_group: dict[str, list] = {}
    for a in aligned:
        by_group.setdefault(a.group_id, []).append(a)
    for gid in by_group:
        by_group[gid].sort(key=lambda s: s.sample_id)

    if args.mode == "pilot":
        if args.pilot_group:
            gid = args.pilot_group
        else:
            # Prefer a group with N large enough for a meaningful elbow sweep
            candidates = sorted(by_group.items(), key=lambda kv: -len(kv[1]))
            gid = None
            for g, ss in candidates:
                if len(ss) >= max(args.k_min + 1, 10):
                    gid = g
                    break
            if gid is None:
                print("[fatal] no suitable pilot group", flush=True)
                return 4
        groups_to_run = [gid]
    else:
        groups_to_run = sorted(by_group.keys())

    use_elbow_default = not args.no_elbow
    base_cfg = {
        "default_k": args.default_k,
        "k_map": k_map,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "elbow_default": use_elbow_default,
        "pca_max_components": args.pca_max_components,
        "pca_whiten": False,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "vision": report["vision"],
        "text": report["text"],
        "fusion": (
            "per-group: rowL2(X/T)->PCA(min(pca_max,N-1),whiten=False)->rowL2->"
            "concat->rowL2; spherical K-means on Z only"
        ),
        "clustering": "spherical_kmeans",
        "k_selection": "elbow_max_perp_distance_to_normalized_chord" if use_elbow_default else "fixed",
        "silhouette": "cosine_all_points_diagnostic_only",
    }

    results: list[dict[str, Any]] = []
    missing_k: list[str] = []
    for gid in groups_to_run:
        samples = by_group[gid]
        fixed_k = resolve_fixed_k(gid, k_map, args.default_k)
        if fixed_k is None and not use_elbow_default:
            missing_k.append(gid)
            gdir = out / "groups" / safe_group_dirname(gid)
            ensure_dir(gdir)
            V, T, ids = load_group_raw_matrices(samples, vis, txt)
            prep = prepare_group_pca_fusion(
                V,
                T,
                group_id=gid,
                sample_ids=ids,
                max_components=args.pca_max_components,
            )
            save_group_pca_artifacts(prep, gdir)
            with (gdir / "aligned_samples.jsonl").open("w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            atomic_write_json(
                gdir / "group_manifest.json",
                {
                    "group_id": gid,
                    "status": "missing_k",
                    "n": len(samples),
                    "pca_n_components": prep.n_components,
                    "fused_dim": prep.fused_dim,
                },
            )
            results.append({"group_id": gid, "status": "missing_k", "n": len(samples)})
            continue

        k_for_run = fixed_k  # None → elbow inside run_group
        cfg_fp = config_fingerprint(
            {
                **base_cfg,
                "group_id": gid,
                "k": k_for_run,
                "n": len(samples),
                "k_selection_mode": "fixed" if k_for_run is not None else "elbow",
            }
        )
        mode_s = f"K={k_for_run}" if k_for_run is not None else f"elbow[{args.k_min},{args.k_max}]"
        print(f"[cluster] {gid} N={len(samples)} {mode_s}", flush=True)
        results.append(
            run_group(
                gid,
                samples,
                vis=vis,
                txt=txt,
                k=k_for_run,
                out_dir=out,
                args=args,
                cfg_fp=cfg_fp,
            )
        )

    # summary
    sdir = out / "summary"
    ensure_dir(sdir)
    summary_rows = []
    failed = []
    for r in results:
        summary_rows.append(
            {
                "group_id": r["group_id"],
                "status": r["status"],
                "n": r.get("n"),
                "k": r.get("k"),
                "best_objective": (r.get("metrics") or {}).get("best_objective"),
                "fused_silhouette": (r.get("metrics") or {}).get("fused_silhouette"),
                "mean_cosine": (r.get("metrics") or {}).get("mean_cosine_to_assigned_centroid"),
                "cluster_sizes": (r.get("metrics") or {}).get("cluster_sizes"),
                "error": r.get("error"),
            }
        )
        if r["status"] in ("failed", "missing_k"):
            failed.append(r)

    atomic_write_json(sdir / "group_summary.json", summary_rows)
    with (sdir / "group_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "status",
                "n",
                "k",
                "best_objective",
                "fused_silhouette",
                "mean_cosine",
                "cluster_sizes",
                "error",
            ],
        )
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    with (sdir / "failed_groups.jsonl").open("w", encoding="utf-8") as f:
        for r in failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_success = sum(1 for r in results if r["status"] in ("success", "skipped_resume"))
    n_fail = sum(1 for r in results if r["status"] == "failed")
    n_missing = sum(1 for r in results if r["status"] == "missing_k")
    overall = "success" if n_fail == 0 and n_missing == 0 else "partial_or_failed"
    run_man = {
        "status": overall,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
        "mode": args.mode,
        "n_groups_attempted": len(results),
        "n_success": n_success,
        "n_failed": n_fail,
        "n_missing_k": n_missing,
        "missing_k_groups": missing_k if args.mode != "pilot" else [g for g in missing_k if g in groups_to_run],
        "alignment": report,
        "fusion_identity": id_check,
        "config": base_cfg,
        "paths": {
            "data_root": args.data_root,
            "vision": args.vision_feature_root,
            "text": args.text_feature_root,
            "output": str(out),
        },
    }
    atomic_write_json(out / "run_manifest.json", run_man)
    print(json.dumps({k: run_man[k] for k in ("status", "n_success", "n_failed", "n_missing_k", "elapsed_sec")}, indent=2))
    return 0 if overall == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
