"""GPU-accelerated image operations via CuPy (with CPU fallback)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from heatmap_analysis.gpu import get_xp, to_numpy


def otsu_threshold_from_histogram(hist: np.ndarray) -> float:
    """Global Otsu threshold from 256-bin histogram."""
    hist = np.asarray(hist, dtype=np.float64).ravel()
    if hist.size != 256:
        raise ValueError("histogram must have 256 bins")
    total = hist.sum()
    if total <= 0:
        return 128.0
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.divide(
        (mu_t * omega - mu) ** 2,
        denom,
        out=np.zeros_like(omega),
        where=denom > 1e-12,
    )
    return float(np.argmax(sigma_b))


def gaussian_blur(gray: np.ndarray, ksize: int, *, xp: Any) -> Any:
    sigma = max(ksize / 6.0, 0.5)
    if xp is np:
        return cv2.GaussianBlur(gray, (ksize | 1, ksize | 1), 0)
    from cupyx.scipy.ndimage import gaussian_filter

    g = xp.asarray(gray, dtype=xp.float32)
    return gaussian_filter(g, sigma=sigma)


def otsu_binary_inv(gray: np.ndarray, *, xp: Any) -> Any:
    """Return uint8 mask (0/255) of dark foreground."""
    blur = gaussian_blur(gray, 3, xp=xp)
    if xp is np:
        blur_u8 = np.clip(blur, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(blur_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask
    blur_u8 = xp.clip(blur, 0, 255).astype(xp.uint8)
    hist = xp.bincount(blur_u8.ravel(), minlength=256)
    thr = otsu_threshold_from_histogram(to_numpy(hist))
    return ((blur_u8 <= thr).astype(xp.uint8)) * 255


def adaptive_binary_inv(gray: np.ndarray, block_size: int, c: int, *, xp: Any) -> Any:
    bs = block_size | 1
    blur = gaussian_blur(gray, 3, xp=xp)
    if xp is np:
        blur_u8 = np.clip(blur, 0, 255).astype(np.uint8)
        return cv2.adaptiveThreshold(
            blur_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, c
        )
    from cupyx.scipy.ndimage import uniform_filter

    blur_f = xp.asarray(blur, dtype=xp.float32)
    local_mean = uniform_filter(blur_f, size=bs, mode="nearest")
    return ((blur_f < local_mean - c).astype(xp.uint8)) * 255


def morph_close(mask: Any, ksize: int, iterations: int, *, xp: Any) -> Any:
    if xp is np:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    from cupyx.scipy.ndimage import binary_closing

    struct = xp.ones((ksize, ksize), dtype=xp.bool_)
    out = mask > 0
    for _ in range(iterations):
        out = binary_closing(out, structure=struct)
    return (out.astype(xp.uint8)) * 255


def morph_open(mask: Any, kw: int, kh: int, *, xp: Any) -> Any:
    if xp is np:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    from cupyx.scipy.ndimage import binary_opening

    struct = xp.ones((kh, kw), dtype=xp.bool_)
    out = binary_opening(mask > 0, structure=struct)
    return (out.astype(xp.uint8)) * 255


def mask_edges(mask: Any, edge_ratio: float, *, xp: Any) -> Any:
    h, w = mask.shape[:2]
    mx = max(int(w * edge_ratio), 1)
    my = max(int(h * edge_ratio), 1)
    out = xp.array(mask, copy=True) if xp is not np else mask.copy()
    out[:my, :] = 0
    out[-my:, :] = 0
    out[:, :mx] = 0
    out[:, -mx:] = 0
    return out


def remove_large_components(mask: Any, max_area_ratio: float, *, xp: Any) -> Any:
    h, w = mask.shape[:2]
    max_area = h * w * max_area_ratio
    if xp is np:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(mask, dtype=np.uint8), connectivity=8
        )
        out = np.asarray(mask, dtype=np.uint8).copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                out[labels == i] = 0
        return out

    from cupyx.scipy.ndimage import generate_binary_structure, label

    structure = generate_binary_structure(rank=2, connectivity=2)
    labeled, num = label(mask > 0, structure=structure)
    if num == 0:
        return mask
    counts = xp.bincount(labeled.ravel())
    large_ids = xp.flatnonzero(counts > max_area)
    large_ids = large_ids[large_ids > 0]
    if large_ids.size == 0:
        return mask
    remove = xp.isin(labeled, large_ids)
    out = xp.where(remove, 0, mask)
    return out.astype(xp.uint8)


def remove_hv_lines(mask: Any, *, xp: Any) -> Any:
    h, w = mask.shape[:2]
    kw = max(w // 20, 15)
    kh = max(h // 20, 15)
    horiz = morph_open(mask, kw, 1, xp=xp)
    vert = morph_open(mask, 1, kh, xp=xp)
    if xp is np:
        lines = cv2.bitwise_or(horiz, vert)
        return cv2.bitwise_and(mask, cv2.bitwise_not(lines))
    lines = ((horiz > 0) | (vert > 0)).astype(xp.uint8) * 255
    return xp.where(lines > 0, 0, mask).astype(xp.uint8)


def bounding_box_from_binary(binary: Any, *, xp: Any) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) from foreground pixels."""
    if xp is np:
        ys, xs = np.nonzero(binary)
    else:
        ys, xs = xp.nonzero(binary > 0)
        if ys.size:
            ys = to_numpy(ys)
            xs = to_numpy(xs)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def template_subtract(
    answer: np.ndarray,
    template: np.ndarray,
    diff_threshold: int,
    *,
    xp: Any,
) -> Any:
    if xp is np:
        if answer.shape != template.shape:
            template = cv2.resize(template, (answer.shape[1], answer.shape[0]))
        diff = cv2.absdiff(answer, template)
        _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        dark = answer < 200
        return cv2.bitwise_and(mask, dark.astype(np.uint8) * 255)

    ans = xp.asarray(answer, dtype=xp.uint8)
    tmpl = xp.asarray(template, dtype=xp.uint8)
    if ans.shape != tmpl.shape:
        tmpl = xp.asarray(
            cv2.resize(to_numpy(tmpl), (ans.shape[1], ans.shape[0])),
            dtype=xp.uint8,
        )
    diff = xp.abs(ans.astype(xp.int16) - tmpl.astype(xp.int16))
    mask = (diff > diff_threshold).astype(xp.uint8) * 255
    dark = ans < 200
    return xp.where(dark & (mask > 0), 255, 0).astype(xp.uint8)


def ink_weights_from_mask(mask: Any, *, xp: Any) -> Any:
    """Binary ink mask -> float32 weights (0/1), stays on xp."""
    if xp is np:
        return (np.asarray(mask) > 0).astype(np.float32)
    return (xp.asarray(mask) > 0).astype(xp.float32)
