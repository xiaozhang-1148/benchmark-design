"""Equal-weight L2-normalize → concat → L2-normalize fusion (2816-d)."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2_normalize(x: np.ndarray, *, axis: int = -1, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    if np.any(n < eps):
        raise RuntimeError("vector norm below eps during L2 normalize")
    return (x / n).astype(np.float32, copy=False)


def equal_weight_fuse(
    vision: np.ndarray,
    text: np.ndarray,
    *,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_hat [..., 1792], t_hat [..., 1024], z [..., 2816]  all unit L2 (approx).
    """
    vision = np.asarray(vision, dtype=np.float32)
    text = np.asarray(text, dtype=np.float32)
    if vision.shape[-1] != 1792:
        raise RuntimeError(f"vision dim must be 1792, got {vision.shape}")
    if text.shape[-1] != 1024:
        raise RuntimeError(f"text dim must be 1024, got {text.shape}")
    if not np.isfinite(vision).all():
        raise RuntimeError("non-finite vision features")
    if not np.isfinite(text).all():
        raise RuntimeError("non-finite text features")

    x_hat = l2_normalize(vision, eps=eps)
    t_hat = l2_normalize(text, eps=eps)
    z_raw = np.concatenate([x_hat, t_hat], axis=-1)
    z = l2_normalize(z_raw, eps=eps)
    return x_hat, t_hat, z


def verify_equal_weight_identity(
    x_hat: np.ndarray,
    t_hat: np.ndarray,
    z: np.ndarray,
    *,
    n_pairs: int = 100,
    seed: int = 42,
    atol: float = 1e-5,
) -> dict[str, float | bool | int]:
    """Check dot(z_i,z_j) ≈ 0.5*dot(x_i,x_j)+0.5*dot(t_i,t_j) and ||z||≈1."""
    n = z.shape[0]
    if n < 2:
        raise RuntimeError("need at least 2 samples for identity check")
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(z, axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0)))
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    errs = []
    # sample pairs with replacement for simplicity when N large
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        while j == i:
            j = int(rng.integers(0, n))
        lhs = float(np.dot(z[i], z[j]))
        rhs = 0.5 * float(np.dot(x_hat[i], x_hat[j])) + 0.5 * float(np.dot(t_hat[i], t_hat[j]))
        errs.append(abs(lhs - rhs))
    max_err = float(max(errs))
    mean_err = float(np.mean(errs))
    return {
        "n_pairs": n_pairs,
        "max_identity_error": max_err,
        "mean_identity_error": mean_err,
        "max_norm_error": max_norm_err,
        "atol": atol,
        "passed": max_err < atol and max_norm_err < atol,
    }
