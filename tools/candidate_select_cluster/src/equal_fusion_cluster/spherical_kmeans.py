"""True spherical K-means (cosine / inner-product on the unit sphere)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SphericalKMeansResult:
    labels: np.ndarray  # [N] int64 in 0..K-1 after renumber
    centroids: np.ndarray  # [K, D] unit vectors
    objective: float
    iterations: int
    converged: bool
    cluster_sizes: list[int]
    empty_reinitializations: int
    n_init_trials: list[dict[str, Any]]
    representatives: list[str]  # sample_id per cluster after renumber
    best_init_index: int = 0


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (x / n).astype(np.float32, copy=False)


def _cosine_kmeans_pp(z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ on unit sphere using cosine distance = 1 - dot."""
    n, d = z.shape
    centers = np.empty((k, d), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = z[first]
    # closest cosine distance to chosen centers
    closest = 1.0 - z @ centers[0]
    closest = np.clip(closest, 0.0, 2.0)
    for c in range(1, k):
        # sample proportional to distance^2
        weights = closest ** 2
        s = float(weights.sum())
        if s <= 0 or not np.isfinite(s):
            idx = int(rng.integers(0, n))
        else:
            probs = weights / s
            idx = int(rng.choice(n, p=probs))
        centers[c] = z[idx]
        dist = 1.0 - z @ centers[c]
        dist = np.clip(dist, 0.0, 2.0)
        closest = np.minimum(closest, dist)
    return _l2_normalize_rows(centers)


def _assign(z: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sims = z @ centers.T  # [N, K]
    labels = np.argmax(sims, axis=1).astype(np.int64)
    return labels, sims


def _update_centers(
    z: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
    sample_ids: list[str] | None,
) -> tuple[np.ndarray, int]:
    """Mean of assigned unit vectors, then L2-normalize each center."""
    n, d = z.shape
    centers = np.zeros((k, d), dtype=np.float32)
    empty_reinit = 0
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            continue
        # mean then normalize (equivalent direction to sum-then-normalize)
        centers[c] = z[mask].mean(axis=0)

    for c in range(k):
        nrm = float(np.linalg.norm(centers[c]))
        if nrm < 1e-12:
            empty_reinit += 1
            sims = z @ centers.T
            for cc in range(k):
                if float(np.linalg.norm(centers[cc])) < 1e-12:
                    sims[:, cc] = -np.inf
            max_sim = sims.max(axis=1)
            finite = np.isfinite(max_sim)
            if not finite.any():
                idx = int(rng.integers(0, n))
            else:
                vals = np.where(finite, max_sim, np.inf)
                idx = int(np.argmin(vals))
            centers[c] = z[idx]
        else:
            centers[c] = centers[c] / nrm
    return centers.astype(np.float32, copy=False), empty_reinit


def _run_once(
    z: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    sample_ids: list[str] | None,
) -> dict[str, Any]:
    centers = _cosine_kmeans_pp(z, k, rng)
    labels = np.full(z.shape[0], -1, dtype=np.int64)
    objective = -np.inf
    empty_total = 0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        new_labels, sims = _assign(z, centers)
        new_obj = float(sims[np.arange(z.shape[0]), new_labels].sum())
        centers, empty_re = _update_centers(z, new_labels, k, rng, sample_ids)
        empty_total += empty_re
        # re-assign after center update for objective reporting consistency optional
        label_same = np.array_equal(new_labels, labels)
        obj_delta = abs(new_obj - objective) if np.isfinite(objective) else np.inf
        labels = new_labels
        # Spherical K-means maximizes objective; allow tiny float noise
        if new_obj + 1e-9 < objective:
            # should not decrease meaningfully; still continue but record
            pass
        objective = new_obj
        if label_same or obj_delta < tol:
            converged = True
            break
    # Final Voronoi assignment to the last centers (centers may have moved after labels)
    labels, sims = _assign(z, centers)
    objective = float(sims[np.arange(z.shape[0]), labels].sum())
    return {
        "labels": labels,
        "centroids": centers,
        "objective": float(objective),
        "iterations": int(it),
        "converged": bool(converged),
        "empty_reinitializations": int(empty_total),
        "cluster_sizes": [int(np.sum(labels == c)) for c in range(k)],
    }


def _renumber_by_representative(
    labels: np.ndarray,
    centroids: np.ndarray,
    z: np.ndarray,
    sample_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    k = centroids.shape[0]
    reps: list[tuple[str, int]] = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            # should not happen after empty handling; keep placeholder
            reps.append(("", c))
            continue
        sims = z[idxs] @ centroids[c]
        local = int(idxs[int(np.argmax(sims))])
        reps.append((sample_ids[local], c))
    # sort by sample_id dictionary order; empty last
    order = sorted(range(k), key=lambda i: (reps[i][0] == "", reps[i][0]))
    old_to_new = {old: new for new, old in enumerate(order)}
    new_labels = np.array([old_to_new[int(x)] for x in labels], dtype=np.int64)
    new_centroids = centroids[order].copy()
    new_reps = [reps[old][0] for old in order]
    return new_labels, new_centroids, new_reps


def spherical_kmeans(
    z: np.ndarray,
    k: int,
    *,
    sample_ids: list[str],
    n_init: int = 30,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
) -> SphericalKMeansResult:
    """
    Spherical K-means with ``n_init`` independent runs.

    Each init uses a distinct reproducible seed:
      ``seed_init = base_seed + 1000 * K + init_index``
    and the trial with maximum objective ``sum_i max_k <z_i, c_k>`` is kept.
    """
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2:
        raise RuntimeError(f"Z must be 2D, got {z.shape}")
    n, d = z.shape
    if len(sample_ids) != n:
        raise RuntimeError("sample_ids length mismatch")
    if k < 1 or k > n:
        raise RuntimeError(f"K={k} invalid for N={n}")
    norms = np.linalg.norm(z, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-6):
        raise RuntimeError("input features must be finite unit (approx) vectors")
    z = _l2_normalize_rows(z)

    if n == 1 and k == 1:
        return SphericalKMeansResult(
            labels=np.array([0], dtype=np.int64),
            centroids=z.copy(),
            objective=float(np.dot(z[0], z[0])),
            iterations=0,
            converged=True,
            cluster_sizes=[1],
            empty_reinitializations=0,
            n_init_trials=[],
            representatives=[sample_ids[0]],
            best_init_index=0,
        )

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_init_index = 0
    for t in range(n_init):
        trial_seed = int(seed) + 1000 * int(k) + int(t)
        rng = np.random.default_rng(trial_seed)
        trial = _run_once(z, k, rng=rng, max_iter=max_iter, tol=tol, sample_ids=sample_ids)
        trial_meta = {
            "init_index": t,
            "seed": trial_seed,
            "iterations": trial["iterations"],
            "converged": trial["converged"],
            "objective": trial["objective"],
            "cluster_sizes": trial["cluster_sizes"],
            "empty_reinitializations": trial["empty_reinitializations"],
        }
        trials.append(trial_meta)
        if best is None or trial["objective"] > best["objective"]:
            best = trial
            best_init_index = t

    assert best is not None
    labels, centroids, reps = _renumber_by_representative(
        best["labels"], best["centroids"], z, sample_ids
    )
    # Final assignment must match argmax dot with saved centroids
    reassigned, sims = _assign(z, centroids)
    if not np.array_equal(reassigned, labels):
        mismatch = reassigned != labels
        # Large-N float / exact cosine ties: allow if stored label is within atol of max
        assigned_cos = sims[np.arange(n), labels]
        best_cos = sims[np.arange(n), reassigned]
        if np.any(assigned_cos[mismatch] < best_cos[mismatch] - 1e-5):
            raise RuntimeError("labels inconsistent with argmax assignment to final centroids")
        labels = reassigned
        # refresh representatives under the resolved Voronoi assignment
        reps = []
        for c in range(k):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                reps.append("")
                continue
            local = int(idxs[int(np.argmax(z[idxs] @ centroids[c]))])
            reps.append(sample_ids[local])
    # Canonical objective after renumber (float64 sum of assigned cosines)
    objective = float(sims.astype(np.float64)[np.arange(n), labels].sum())
    sizes = [int(np.sum(labels == c)) for c in range(k)]
    return SphericalKMeansResult(
        labels=labels,
        centroids=centroids,
        objective=objective,
        iterations=int(best["iterations"]),
        converged=bool(best["converged"]),
        cluster_sizes=sizes,
        empty_reinitializations=int(best["empty_reinitializations"]),
        n_init_trials=trials,
        representatives=reps,
        best_init_index=int(best_init_index),
    )


def cosine_silhouette(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    return_per_sample: bool = False,
    sample_size: int | None = None,
    seed: int = 42,
) -> float | tuple[float, np.ndarray] | None:
    """
    Cosine-distance silhouette.

    distance(i,j) = 1 - cos(i,j) on L2-normalized rows.

    For large N, pass ``sample_size`` to compute a stratified subsample diagnostic
    (full pairwise is O(N^2) memory/time and unsuitable for N≈10^4+).
    """
    z = _l2_normalize_rows(np.asarray(z, dtype=np.float32))
    labels = np.asarray(labels, dtype=np.int64)
    n = z.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2 or n < 3:
        return None

    if sample_size is not None and n > int(sample_size):
        rng = np.random.default_rng(int(seed))
        # Stratify by cluster so every non-empty cluster is represented when possible.
        chosen: list[int] = []
        remaining = int(sample_size)
        sizes = {int(c): int(np.sum(labels == c)) for c in uniq}
        # proportional allocation with at least 2 per cluster when possible
        alloc = {}
        for c, sz in sizes.items():
            alloc[c] = min(sz, max(2, int(round(sample_size * sz / n))))
        # adjust to exactly sample_size
        total_alloc = sum(alloc.values())
        keys = sorted(alloc.keys())
        i = 0
        while total_alloc > sample_size and keys:
            c = keys[i % len(keys)]
            if alloc[c] > 2 and alloc[c] < sizes[c] + 1:
                # shrink preferentially clusters with room above 2
                if alloc[c] > 2:
                    alloc[c] -= 1
                    total_alloc -= 1
            i += 1
            if i > sample_size * 3:
                break
        while total_alloc < sample_size and keys:
            c = keys[i % len(keys)]
            if alloc[c] < sizes[c]:
                alloc[c] += 1
                total_alloc += 1
            i += 1
            if i > sample_size * 3:
                break
        for c, take in alloc.items():
            idxs = np.where(labels == c)[0]
            if idxs.size <= take:
                chosen.extend(idxs.tolist())
            else:
                chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())
        chosen_arr = np.asarray(sorted(set(chosen)), dtype=np.int64)
        return cosine_silhouette(
            z[chosen_arr],
            labels[chosen_arr],
            return_per_sample=return_per_sample,
            sample_size=None,
            seed=seed,
        )

    # Full pairwise cosine distance for all points (groups are small; or subsample).
    sims = z @ z.T
    dist = (1.0 - sims).astype(np.float64, copy=False)
    np.fill_diagonal(dist, 0.0)
    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        li = int(labels[i])
        same = labels == li
        same[i] = False
        if not np.any(same):
            sil[i] = 0.0
            continue
        a = float(dist[i, same].mean())
        b = np.inf
        for c in uniq:
            if int(c) == li:
                continue
            mask = labels == c
            if np.any(mask):
                b = min(b, float(dist[i, mask].mean()))
        if not np.isfinite(b):
            sil[i] = 0.0
        else:
            sil[i] = (b - a) / max(a, b, 1e-12)
    mean_sil = float(sil.mean())
    if return_per_sample:
        return mean_sil, sil.astype(np.float32)
    return mean_sil


def compute_assigned_cosine_and_distortions(
    z: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, Any]:
    """Cross-check D = N - J and D = sum(1 - cos_assigned)."""
    z = np.asarray(z, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float32)
    n = z.shape[0]
    sims = z @ centroids.T
    cos_assigned = sims[np.arange(n), labels].astype(np.float64)
    if not np.isfinite(cos_assigned).all():
        raise RuntimeError("non-finite assigned cosine similarities")
    objective = float(cos_assigned.sum())
    d1 = float(n - objective)
    d2 = float(np.sum(1.0 - cos_assigned))
    if abs(d1 - d2) > 1e-5:
        raise RuntimeError(
            f"distortion cross-check failed: |D1-D2|={abs(d1 - d2)} "
            f"(D1=N-J={d1}, D2=sum(1-cos)={d2})"
        )
    return {
        "objective": objective,
        "spherical_kmeans_total_cosine_distortion": d1,
        "spherical_kmeans_mean_cosine_distortion": float(d1 / max(n, 1)),
        "distortion_crosscheck_abs_err": float(abs(d1 - d2)),
        "assigned_cosine": cos_assigned,
    }


def check_unit_norms(z: np.ndarray, centroids: np.ndarray, *, atol: float = 1e-5) -> dict[str, float]:
    zn = np.linalg.norm(np.asarray(z, dtype=np.float64), axis=1)
    cn = np.linalg.norm(np.asarray(centroids, dtype=np.float64), axis=1)
    max_z = float(np.max(np.abs(zn - 1.0)))
    max_c = float(np.max(np.abs(cn - 1.0)))
    if max_z > atol:
        raise RuntimeError(f"sample norm error {max_z} > {atol}")
    if max_c > atol:
        raise RuntimeError(f"centroid norm error {max_c} > {atol}")
    return {"max_sample_norm_error": max_z, "max_centroid_norm_error": max_c}


def elbow_select_k(
    ks: list[int],
    distortions: list[float],
    *,
    tie_atol: float = 1e-12,
) -> tuple[int, dict[str, Any]]:
    """
    Elbow via max perpendicular distance to the **normalized** endpoint chord.

    Normalize:
      x_K = (K - K_min) / (K_max - K_min)
      y_K = (D(K) - D(K_max)) / (D(K_min) - D(K_max))
    so endpoints are (0,1) and (1,0); chord is x + y - 1 = 0.
    Distance: |x + y - 1| / sqrt(2).

    Only interior K in {K_min+1, ..., K_max-1} may be selected.
    On float ties, prefer the smaller K.
    """
    if len(ks) != len(distortions) or len(ks) == 0:
        raise ValueError("ks/distortions length mismatch")
    if len(ks) == 1:
        return int(ks[0]), {
            "method": "max_perpendicular_distance_to_chord",
            "normalized_k": [0.0],
            "normalized_distortion": [1.0],
            "perpendicular_distances": [0.0],
            "chosen_index": 0,
            "tie": False,
            "interior_only": False,
            "warning": "single_k",
        }

    ks_arr = np.asarray(ks, dtype=np.float64)
    d_arr = np.asarray(distortions, dtype=np.float64)
    k_min = float(ks_arr[0])
    k_max = float(ks_arr[-1])
    d_lo = float(d_arr[0])  # D(K_min)
    d_hi = float(d_arr[-1])  # D(K_max)
    denom_k = k_max - k_min
    denom_d = d_lo - d_hi
    if abs(denom_k) < 1e-15:
        raise RuntimeError("K_min == K_max; cannot normalize elbow abscissa")
    if abs(denom_d) < 1e-15:
        # flat distortion: all distances ~0; pick smallest interior (or k_min+1)
        x_n = (ks_arr - k_min) / denom_k
        y_n = np.zeros_like(ks_arr)
        dists = np.abs(x_n + y_n - 1.0) / np.sqrt(2.0)
        warning = "flat_distortion_curve"
    else:
        x_n = (ks_arr - k_min) / denom_k
        y_n = (d_arr - d_hi) / denom_d
        dists = np.abs(x_n + y_n - 1.0) / np.sqrt(2.0)
        warning = None

    interior = [i for i in range(len(ks)) if ks[i] > ks[0] and ks[i] < ks[-1]]
    if not interior:
        # e.g. only K=2,3 — no interior; fall back with warning
        candidates = list(range(len(ks)))
        warning = (warning + ";" if warning else "") + "elbow_no_interior_points"
    else:
        candidates = interior

    best_d = max(float(dists[i]) for i in candidates)
    tied = [i for i in candidates if abs(float(dists[i]) - best_d) <= tie_atol]
    # prefer smaller K
    best_i = min(tied, key=lambda i: int(ks[i]))
    tie = len(tied) > 1

    # top-5 by distance among interior (or all if no interior), then smaller K
    ranked = sorted(
        (candidates if interior else list(range(len(ks)))),
        key=lambda i: (-float(dists[i]), int(ks[i])),
    )
    top5 = [{"k": int(ks[i]), "distance": float(dists[i])} for i in ranked[:5]]

    max_d = best_d
    e95_ks = sorted(
        int(ks[i])
        for i in (interior if interior else range(len(ks)))
        if float(dists[i]) >= 0.95 * max_d - 1e-15
    )
    # broad if E95 contains a contiguous run of >=3 K values
    broad = False
    if e95_ks:
        run = 1
        for a, b in zip(e95_ks, e95_ks[1:]):
            if b == a + 1:
                run += 1
                if run >= 3:
                    broad = True
                    break
            else:
                run = 1

    meta: dict[str, Any] = {
        "method": "max_perpendicular_distance_to_chord",
        "normalized_k": [float(x) for x in x_n],
        "normalized_distortion": [float(y) for y in y_n],
        "perpendicular_distances": [float(d) for d in dists],
        "chosen_index": int(best_i),
        "tie": tie,
        "tied_ks": [int(ks[i]) for i in tied],
        "interior_only": bool(interior),
        "top_elbow_candidates": top5,
        "elbow_95pct_interval": [e95_ks[0], e95_ks[-1]] if e95_ks else None,
        "elbow_95pct_ks": e95_ks,
        "broad_elbow": broad,
    }
    if warning:
        meta["warning"] = warning
    return int(ks[best_i]), meta


def sweep_spherical_kmeans_elbow(
    z: np.ndarray,
    sample_ids: list[str],
    *,
    k_min: int = 2,
    k_max: int = 30,
    n_init: int = 30,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
    compute_silhouette: bool = True,
    silhouette_sample_size: int | None = None,
    monotonic_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Sweep K in [k_min, min(k_max, N-1)], keep best of ``n_init`` spherical K-means
    runs per K, select elbow on spherical_kmeans_total_cosine_distortion = N - J.

    Silhouette (if enabled) is diagnostic only and does **not** select K.
    For large N, set ``silhouette_sample_size`` (e.g. 3000) to avoid O(N^2).
    """
    z = np.asarray(z, dtype=np.float32)
    n = z.shape[0]
    if n < 3:
        raise RuntimeError(f"need N>=3 for K sweep / elbow clustering, got N={n}")
    k_lo = max(2, int(k_min))
    k_hi = min(int(k_max), n - 1)
    if k_lo > k_hi:
        raise RuntimeError(f"empty K range after clamp: [{k_lo},{k_hi}] for N={n}")

    # Auto sample silhouette for large corpora unless explicitly disabled (None + compute).
    sil_sample = silhouette_sample_size
    if compute_silhouette and sil_sample is None and n > 2500:
        sil_sample = 3000

    # sample unit-norm check on input Z
    zn = np.linalg.norm(z.astype(np.float64), axis=1)
    max_z_err = float(np.max(np.abs(zn - 1.0)))
    if max_z_err > 1e-5:
        raise RuntimeError(f"input Z rows are not unit within 1e-5 (max_err={max_z_err})")

    curve: list[dict[str, Any]] = []
    results_by_k: dict[int, SphericalKMeansResult] = {}
    monotonic_warnings: list[dict[str, Any]] = []

    for k in range(k_lo, k_hi + 1):
        res = spherical_kmeans(
            z,
            k,
            sample_ids=sample_ids,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        check_unit_norms(z, res.centroids, atol=1e-5)
        dist_info = compute_assigned_cosine_and_distortions(z, res.labels, res.centroids)
        # objective from best trial must match recomputed (scale tolerance with N)
        obj_atol = max(1e-4, 1e-8 * float(n))
        if abs(dist_info["objective"] - res.objective) > obj_atol:
            raise RuntimeError(
                f"K={k}: recomputed objective {dist_info['objective']} "
                f"!= stored {res.objective} (atol={obj_atol})"
            )
        if not np.isfinite(res.objective) or not np.isfinite(dist_info["spherical_kmeans_total_cosine_distortion"]):
            raise RuntimeError(f"K={k}: non-finite objective/distortion")

        objs = [float(t["objective"]) for t in res.n_init_trials]
        if len(objs) != n_init:
            raise RuntimeError(f"K={k}: expected {n_init} inits, got {len(objs)}")
        # distinct seeds
        seeds = [t["seed"] for t in res.n_init_trials]
        if len(set(seeds)) != n_init:
            raise RuntimeError(f"K={k}: init seeds not all distinct: {seeds}")

        row: dict[str, Any] = {
            "k": k,
            "n_init": n_init,
            "objective_best": float(res.objective),
            "objective_mean": float(np.mean(objs)),
            "objective_std": float(np.std(objs)),
            "objective_min": float(np.min(objs)),
            "distortion_total": dist_info["spherical_kmeans_total_cosine_distortion"],
            "distortion_mean": dist_info["spherical_kmeans_mean_cosine_distortion"],
            "spherical_kmeans_total_cosine_distortion": dist_info[
                "spherical_kmeans_total_cosine_distortion"
            ],
            "spherical_kmeans_mean_cosine_distortion": dist_info[
                "spherical_kmeans_mean_cosine_distortion"
            ],
            "distortion_crosscheck_abs_err": dist_info["distortion_crosscheck_abs_err"],
            "best_init_index": res.best_init_index,
            "converged": bool(res.converged),
            "n_iter": int(res.iterations),
            "cluster_sizes": res.cluster_sizes,
            # legacy aliases used by older plot code
            "objective": float(res.objective),
            "distortion": dist_info["spherical_kmeans_total_cosine_distortion"],
            "iterations": int(res.iterations),
        }
        if compute_silhouette:
            sil = cosine_silhouette(
                z,
                res.labels,
                sample_size=sil_sample,
                seed=seed + 17 * k,
            )
            row["fused_silhouette"] = sil
            row["silhouette_diagnostic_only"] = True
            if sil_sample is not None:
                row["silhouette_sample_size"] = int(min(sil_sample, n))
        results_by_k[k] = res
        curve.append(row)
        print(f"  [sweep] K={k}/{k_hi} done", flush=True)

    # monotonicity: D[K+1] <= D[K] + tol
    for i in range(len(curve) - 1):
        d_cur = float(curve[i]["distortion_total"])
        d_nxt = float(curve[i + 1]["distortion_total"])
        if d_nxt > d_cur + monotonic_tolerance:
            monotonic_warnings.append(
                {
                    "flag": "elbow_curve_monotonicity_warning",
                    "k": int(curve[i]["k"]),
                    "k_next": int(curve[i + 1]["k"]),
                    "D_k": d_cur,
                    "D_k_next": d_nxt,
                    "delta": d_nxt - d_cur,
                }
            )

    ks = [c["k"] for c in curve]
    distortions = [c["distortion_total"] for c in curve]
    best_k, elbow_meta = elbow_select_k(ks, distortions)

    # attach normalized coords / distances onto curve rows
    for i, row in enumerate(curve):
        row["normalized_k"] = elbow_meta["normalized_k"][i]
        row["normalized_distortion"] = elbow_meta["normalized_distortion"][i]
        row["perpendicular_distance"] = elbow_meta["perpendicular_distances"][i]

    best = results_by_k[best_k]
    return {
        "selected_k": best_k,
        "elbow": elbow_meta,
        "k_min": k_lo,
        "k_max": k_hi,
        "n_init": n_init,
        "curve": curve,
        "result": best,
        "results_by_k": results_by_k,
        "selection_method": "max_perpendicular_distance_to_chord",
        "objective_definition": "sum_i max_k dot(z_i, c_k)",
        "distortion_definition": "N - objective",
        "distortion_name": "spherical_kmeans_total_cosine_distortion",
        "broad_elbow": elbow_meta.get("broad_elbow"),
        "elbow_95pct_interval": elbow_meta.get("elbow_95pct_interval"),
        "top_elbow_candidates": elbow_meta.get("top_elbow_candidates"),
        "monotonicity_warnings": monotonic_warnings,
        "silhouette_role": "diagnostic_only",
    }
