"""Backfill per-K cosine silhouette + plots for existing group outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json
from .plots import plot_k_selection_curves
from .spherical_kmeans import sweep_spherical_kmeans_elbow

DEFAULT_GROUPS = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/"
    "equal_fusion_spherical_kmeans/groups"
)


def _load_sample_ids(gdir: Path) -> list[str]:
    path = gdir / "aligned_samples.jsonl"
    ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["sample_id"])
    return ids


def backfill_group(
    gdir: Path,
    *,
    force: bool = False,
    n_init_override: int | None = None,
) -> dict[str, Any]:
    man_path = gdir / "group_manifest.json"
    elbow_path = gdir / "k_elbow_curve.json"
    fused_path = gdir / "fused_features.npy"
    plot_path = gdir / "k_selection_curves.png"
    sil_plot = gdir / "k_vs_cosine_silhouette.png"

    if not man_path.exists() or not fused_path.exists():
        return {"group": gdir.name, "status": "skip_missing_inputs"}

    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("status") != "success":
        return {"group": gdir.name, "status": "skip_not_success", "manifest_status": man.get("status")}

    elbow: dict[str, Any] = {}
    if elbow_path.exists():
        elbow = json.loads(elbow_path.read_text(encoding="utf-8"))
    curve = elbow.get("curve") or []
    has_sil = bool(curve) and all(c.get("fused_silhouette") is not None for c in curve)
    has_plots = plot_path.exists() and sil_plot.exists()
    if has_sil and has_plots and not force:
        return {
            "group": gdir.name,
            "status": "skipped_resume",
            "selected_k": elbow.get("selected_k"),
        }

    z = np.load(fused_path)
    ids = _load_sample_ids(gdir)
    if len(ids) != z.shape[0]:
        raise RuntimeError(f"{gdir.name}: sample_ids={len(ids)} vs fused N={z.shape[0]}")

    k_min = int(man.get("k_min", elbow.get("k_min", 2)))
    k_max = int(man.get("k_max", elbow.get("k_max", 20)))
    n_init = int(n_init_override if n_init_override is not None else man.get("n_init", 10))
    max_iter = int(man.get("max_iter", 300))
    tol = float(man.get("tol", 1e-6))
    seed = int(man.get("seed", 42))

    sweep = sweep_spherical_kmeans_elbow(
        z,
        ids,
        k_min=k_min,
        k_max=k_max,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        compute_silhouette=True,
    )
    selected_k = int(sweep["selected_k"])
    # Prefer previously recorded elbow K if present (should match given same seed/n_init)
    prev_k = elbow.get("selected_k")
    if prev_k is not None and int(prev_k) != selected_k:
        # Keep previous elbow choice for consistency with already-written labels;
        # still refresh silhouette curve from the new sweep.
        selected_k = int(prev_k)

    sils = [c.get("fused_silhouette") for c in sweep["curve"]]
    elbow_out = {
        "method": sweep["elbow"]["method"],
        "selected_k": selected_k,
        "k_min": sweep["k_min"],
        "k_max": sweep["k_max"],
        "curve": sweep["curve"],
        "normalized_distances": sweep["elbow"].get("normalized_distances"),
    }
    if all(s is not None for s in sils):
        best_sil_i = int(np.argmax(np.asarray(sils, dtype=np.float64)))
        elbow_out["max_silhouette_k"] = int(sweep["curve"][best_sil_i]["k"])
        elbow_out["max_silhouette"] = float(sils[best_sil_i])  # type: ignore[arg-type]

    atomic_write_json(elbow_path, elbow_out)
    plot_k_selection_curves(
        sweep["curve"],
        selected_k=selected_k,
        out_path=plot_path,
        title="Elbow (distortion) + cosine silhouette vs K",
        group_id=str(man.get("group_id", gdir.name)),
    )

    # patch metrics with silhouette curve summary if metrics exist
    met_path = gdir / "metrics.json"
    if met_path.exists():
        met = json.loads(met_path.read_text(encoding="utf-8"))
        met["elbow"] = {
            "method": elbow_out["method"],
            "k_min": elbow_out["k_min"],
            "k_max": elbow_out["k_max"],
            "curve": elbow_out["curve"],
            "max_silhouette_k": elbow_out.get("max_silhouette_k"),
            "max_silhouette": elbow_out.get("max_silhouette"),
        }
        atomic_write_json(met_path, met)

    return {
        "group": gdir.name,
        "status": "success",
        "selected_k": selected_k,
        "max_silhouette_k": elbow_out.get("max_silhouette_k"),
        "max_silhouette": elbow_out.get("max_silhouette"),
        "n": int(z.shape[0]),
        "curve_len": len(sweep["curve"]),
    }


def audit_groups(groups_root: Path) -> dict[str, Any]:
    required = [
        "image_pca.pkl",
        "text_pca.pkl",
        "image_pca_explained_variance.json",
        "text_pca_explained_variance.json",
        "reduced_image_features.npy",
        "reduced_text_features.npy",
        "fused_features.npy",
        "labels.npy",
        "centroids.npy",
        "metrics.json",
        "group_manifest.json",
        "k_elbow_curve.json",
        "clusters.jsonl",
        "k_selection_curves.png",
        "k_vs_cosine_silhouette.png",
    ]
    gs = sorted([p for p in groups_root.iterdir() if p.is_dir()])
    ok = 0
    bad: list[dict[str, Any]] = []
    for g in gs:
        names = {p.name for p in g.iterdir()}
        miss = [r for r in required if r not in names]
        if miss:
            bad.append({"group": g.name, "missing": miss})
            continue
        man = json.loads((g / "group_manifest.json").read_text(encoding="utf-8"))
        met = json.loads((g / "metrics.json").read_text(encoding="utf-8"))
        elbow = json.loads((g / "k_elbow_curve.json").read_text(encoding="utf-8"))
        n = int(met["N_group"])
        k = int(met["K"])
        nc = int(met["pca_n_components"])
        fd = int(met["fused_dim"])
        fused = np.load(g / "fused_features.npy")
        labels = np.load(g / "labels.npy")
        cents = np.load(g / "centroids.npy")
        curve = elbow.get("curve") or []
        checks = {
            "status": man.get("status") == "success",
            "elbow_mode": man.get("k_selection") == "elbow",
            "pca": nc == min(64, n - 1),
            "fused": fused.shape == (n, fd) and fd == 2 * nc,
            "labels": labels.shape == (n,) and int(labels.min()) >= 0 and int(labels.max()) < k,
            "centroids": list(cents.shape) == [k, fd],
            "elbow_k": elbow.get("selected_k") == k,
            "curve_len": len(curve) == min(20, n) - 1,
            "sil_in_curve": all(c.get("fused_silhouette") is not None for c in curve),
            "whiten_false": man.get("pca_whiten") is False,
        }
        bad_keys = [kk for kk, vv in checks.items() if not vv]
        if bad_keys:
            bad.append({"group": g.name, "bad": bad_keys})
        else:
            ok += 1
    return {
        "n_dirs": len(gs),
        "ok": ok,
        "bad": len(bad),
        "issues_head": bad[:20],
        "expected_total_groups_note": "Batch02 has 327 question groups; partial dirs mean run still in progress or incomplete.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill K–silhouette curves/plots for group outputs")
    p.add_argument("--groups-root", default=DEFAULT_GROUPS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n-init", type=int, default=None, help="Override n_init for backfill sweep")
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.groups_root)
    if args.audit_only:
        report = audit_groups(root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["bad"] == 0 else 1

    gs = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.limit > 0:
        gs = gs[: args.limit]
    results = []
    for i, g in enumerate(gs, 1):
        print(f"[{i}/{len(gs)}] {g.name}", flush=True)
        try:
            r = backfill_group(g, force=args.force, n_init_override=args.n_init)
        except Exception as e:  # noqa: BLE001
            r = {"group": g.name, "status": "failed", "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(f"  -> {r.get('status')} k={r.get('selected_k')} maxsil_k={r.get('max_silhouette_k')}", flush=True)

    audit = audit_groups(root)
    out = {"results": results, "audit": audit}
    atomic_write_json(root.parent / "backfill_k_plots_report.json", out)
    print(json.dumps({"n": len(results), "audit": audit}, indent=2, ensure_ascii=False))
    return 0 if audit["bad"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
