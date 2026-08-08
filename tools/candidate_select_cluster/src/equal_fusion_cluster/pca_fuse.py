"""Per-question centered PCA → row-L2 → equal-weight concat fusion."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ..utils import atomic_write_json, ensure_dir
from .fusion import EPS, l2_normalize, verify_equal_weight_identity

DEFAULT_PCA_MAX_COMPONENTS = 64


@dataclass
class FittedPCABundle:
    modality: str
    pca: PCA
    sample_ids: list[str]
    group_id: str
    n_samples: int
    input_dim: int
    output_dim: int
    whiten: bool
    n_components_requested: int
    config: dict[str, Any]

    def to_pickle_dict(self) -> dict[str, Any]:
        p = self.pca
        return {
            "modality": self.modality,
            "group_id": self.group_id,
            "sample_ids": list(self.sample_ids),
            "n_samples": self.n_samples,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "whiten": bool(self.whiten),
            "n_components_requested": self.n_components_requested,
            "config": self.config,
            # sklearn fields required for reproducibility / transform of new samples
            "mean_": np.asarray(p.mean_, dtype=np.float64),
            "components_": np.asarray(p.components_, dtype=np.float64),
            "explained_variance_": np.asarray(p.explained_variance_, dtype=np.float64),
            "explained_variance_ratio_": np.asarray(p.explained_variance_ratio_, dtype=np.float64),
            "n_features_in_": int(p.n_features_in_),
            "n_components_": int(p.n_components_),
            "sklearn_pca": p,
        }

    def explained_variance_report(self) -> dict[str, Any]:
        ratio = np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64)
        var = np.asarray(self.pca.explained_variance_, dtype=np.float64)
        return {
            "modality": self.modality,
            "group_id": self.group_id,
            "n_samples": self.n_samples,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "whiten": self.whiten,
            "n_components": int(self.pca.n_components_),
            "explained_variance": var.tolist(),
            "explained_variance_ratio": ratio.tolist(),
            "cumulative_explained_variance_ratio": np.cumsum(ratio).tolist(),
            "total_explained_variance_ratio": float(ratio.sum()),
            "sample_ids": list(self.sample_ids),
            "config": self.config,
        }


@dataclass
class GroupPCAFusionResult:
    sample_ids: list[str]
    x_l2: np.ndarray  # row-L2 of raw vision
    t_l2: np.ndarray  # row-L2 of raw text
    x_pca: np.ndarray  # PCA scores (not yet L2)
    t_pca: np.ndarray
    x_prime: np.ndarray  # row-L2 after PCA
    t_prime: np.ndarray
    z: np.ndarray  # fused unit vectors
    image_pca: FittedPCABundle
    text_pca: FittedPCABundle
    n_components: int
    fused_dim: int


def pca_n_components(n_samples: int, input_dim: int, max_components: int = DEFAULT_PCA_MAX_COMPONENTS) -> int:
    if n_samples < 2:
        raise RuntimeError(f"PCA requires N>=2, got N={n_samples}")
    n = min(int(max_components), int(n_samples) - 1, int(input_dim))
    if n < 1:
        raise RuntimeError(f"invalid PCA n_components={n} for N={n_samples} dim={input_dim}")
    return n


def fit_centered_pca(
    x: np.ndarray,
    *,
    modality: str,
    group_id: str,
    sample_ids: list[str],
    max_components: int = DEFAULT_PCA_MAX_COMPONENTS,
    whiten: bool = False,
) -> tuple[FittedPCABundle, np.ndarray]:
    """Fit standard centered PCA (sklearn default) with whiten=False; return bundle + scores."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if len(sample_ids) != n:
        raise RuntimeError("sample_ids length must match N")
    n_comp = pca_n_components(n, d, max_components=max_components)
    if whiten:
        raise RuntimeError("whiten must be False for this pipeline")
    pca = PCA(n_components=n_comp, whiten=False, svd_solver="full")
    scores = pca.fit_transform(x).astype(np.float32, copy=False)
    bundle = FittedPCABundle(
        modality=modality,
        pca=pca,
        sample_ids=list(sample_ids),
        group_id=group_id,
        n_samples=n,
        input_dim=d,
        output_dim=int(pca.n_components_),
        whiten=False,
        n_components_requested=n_comp,
        config={
            "n_components": n_comp,
            "whiten": False,
            "svd_solver": "full",
            "centered": True,
            "max_components_cap": int(max_components),
        },
    )
    return bundle, scores


def prepare_group_pca_fusion(
    vision: np.ndarray,
    text: np.ndarray,
    *,
    group_id: str,
    sample_ids: list[str],
    max_components: int = DEFAULT_PCA_MAX_COMPONENTS,
    eps: float = EPS,
) -> GroupPCAFusionResult:
    """
    Per-question pipeline:
      X̂=rowL2(X), T̂=rowL2(T)
      X_pca=PCA_image(X̂), T_pca=PCA_text(T̂)   # centered, whiten=False
      X'=rowL2(X_pca), T'=rowL2(T_pca)
      Z=rowL2([X', T'])
    """
    vision = np.asarray(vision, dtype=np.float32)
    text = np.asarray(text, dtype=np.float32)
    if vision.ndim != 2 or text.ndim != 2:
        raise RuntimeError("vision/text must be 2D matrices")
    if vision.shape[0] != text.shape[0]:
        raise RuntimeError(f"N mismatch vision={vision.shape[0]} text={text.shape[0]}")
    if vision.shape[1] != 1792 or text.shape[1] != 1024:
        raise RuntimeError(f"expected dims 1792/1024, got {vision.shape}/{text.shape}")
    if len(sample_ids) != vision.shape[0]:
        raise RuntimeError("sample_ids length mismatch")
    if not np.isfinite(vision).all() or not np.isfinite(text).all():
        raise RuntimeError("non-finite features")

    x_l2 = l2_normalize(vision, eps=eps)
    t_l2 = l2_normalize(text, eps=eps)

    image_pca, x_pca = fit_centered_pca(
        x_l2,
        modality="image",
        group_id=group_id,
        sample_ids=sample_ids,
        max_components=max_components,
        whiten=False,
    )
    text_pca, t_pca = fit_centered_pca(
        t_l2,
        modality="text",
        group_id=group_id,
        sample_ids=sample_ids,
        max_components=max_components,
        whiten=False,
    )
    # Both use min(64, N-1, dim); dims differ so n_components should match via N-1/64.
    if image_pca.output_dim != text_pca.output_dim:
        raise RuntimeError(
            f"PCA output dims differ image={image_pca.output_dim} text={text_pca.output_dim}"
        )

    x_prime = l2_normalize(x_pca, eps=eps)
    t_prime = l2_normalize(t_pca, eps=eps)
    z = l2_normalize(np.concatenate([x_prime, t_prime], axis=-1), eps=eps)

    return GroupPCAFusionResult(
        sample_ids=list(sample_ids),
        x_l2=x_l2,
        t_l2=t_l2,
        x_pca=x_pca.astype(np.float32, copy=False),
        t_pca=t_pca.astype(np.float32, copy=False),
        x_prime=x_prime,
        t_prime=t_prime,
        z=z,
        image_pca=image_pca,
        text_pca=text_pca,
        n_components=image_pca.output_dim,
        fused_dim=int(z.shape[1]),
    )


def save_group_pca_artifacts(result: GroupPCAFusionResult, gdir: Path) -> None:
    """Persist required PCA/fusion artifacts for one question group."""
    ensure_dir(gdir)

    def _atomic_pickle(path: Path, obj: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    _atomic_pickle(gdir / "image_pca.pkl", result.image_pca.to_pickle_dict())
    _atomic_pickle(gdir / "text_pca.pkl", result.text_pca.to_pickle_dict())
    atomic_write_json(gdir / "image_pca_explained_variance.json", result.image_pca.explained_variance_report())
    atomic_write_json(gdir / "text_pca_explained_variance.json", result.text_pca.explained_variance_report())

    # reduced_* = post-PCA row-L2 features used in fusion (X', T')
    np.save(str(gdir / "reduced_image_features.npy"), result.x_prime)
    np.save(str(gdir / "reduced_text_features.npy"), result.t_prime)
    np.save(str(gdir / "fused_features.npy"), result.z)

    # also keep raw PCA scores for audit/debug
    np.save(str(gdir / "image_pca_scores.npy"), result.x_pca)
    np.save(str(gdir / "text_pca_scores.npy"), result.t_pca)


def load_pca_pickle(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def transform_with_saved_pca(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Apply a saved PCA transform to new samples (same modality space)."""
    pca: PCA | None = bundle.get("sklearn_pca")
    x = np.asarray(x, dtype=np.float64)
    if pca is not None:
        return pca.transform(x).astype(np.float32, copy=False)
    # Fallback if only arrays were kept
    mean = np.asarray(bundle["mean_"], dtype=np.float64)
    components = np.asarray(bundle["components_"], dtype=np.float64)
    return ((x - mean) @ components.T).astype(np.float32, copy=False)


def verify_pca_fusion_identity(result: GroupPCAFusionResult, *, seed: int = 42) -> dict[str, Any]:
    idc = verify_equal_weight_identity(
        result.x_prime,
        result.t_prime,
        result.z,
        n_pairs=min(100, max(1, result.z.shape[0] * (result.z.shape[0] - 1) // 2)),
        seed=seed,
    )
    return {
        **idc,
        "n_components": result.n_components,
        "fused_dim": result.fused_dim,
        "pipeline": "rowL2->PCA->rowL2->concat->rowL2",
    }
