"""Page region detection and template alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from heatmap_analysis.gpu import get_xp, to_numpy
from heatmap_analysis import image_ops as iops


@dataclass
class PageRegion:
    """Valid page bounding box in pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    scale_to_norm: float = 1.0

    @property
    def width(self) -> int:
        return max(self.x1 - self.x0, 1)

    @property
    def height(self) -> int:
        return max(self.y1 - self.y0, 1)


def detect_page_region(
    gray: np.ndarray,
    edge_mask_ratio: float = 0.02,
    *,
    use_gpu: bool = False,
    xp: Any | None = None,
) -> PageRegion:
    """Locate content bounding box using Otsu threshold and foreground bbox."""
    h, w = gray.shape[:2]
    xp = xp if xp is not None else get_xp(use_gpu)

    binary = iops.otsu_binary_inv(gray, xp=xp)
    binary = iops.morph_close(binary, 5, 2, xp=xp)

    bbox = iops.bounding_box_from_binary(binary, xp=xp)
    if bbox is None:
        margin_x = int(w * edge_mask_ratio)
        margin_y = int(h * edge_mask_ratio)
        return PageRegion(margin_x, margin_y, w - margin_x, h - margin_y)

    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    if bw * bh < 0.05 * h * w:
        margin_x = int(w * edge_mask_ratio)
        margin_y = int(h * edge_mask_ratio)
        return PageRegion(margin_x, margin_y, w - margin_x, h - margin_y)
    return PageRegion(x0, y0, x1, y1)


def crop_region(gray: np.ndarray, region: PageRegion) -> np.ndarray:
    return gray[region.y0 : region.y1, region.x0 : region.x1].copy()


def align_to_template(
    answer: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align answer sheet to template using ORB feature matching + homography."""
    answer_g = answer if answer.ndim == 2 else cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
    template_g = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template_g, None)
    kp2, des2 = orb.detectAndCompute(answer_g, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return answer_g, template_g

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return answer_g, template_g
    matches = sorted(matches, key=lambda m: m.distance)[:50]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return answer_g, template_g
    aligned = cv2.warpPerspective(answer_g, H, (template_g.shape[1], template_g.shape[0]))
    return aligned, template_g


def normalize_to_canvas(
    ink_mask: np.ndarray,
    preserve_aspect_ratio: bool = True,
    target_size: int = 512,
) -> tuple[np.ndarray, dict]:
    """
    Map ink mask to a square canvas while preserving aspect ratio (letterbox padding).
    Returns normalized mask and transform metadata.
    """
    h, w = ink_mask.shape[:2]
    if preserve_aspect_ratio:
        scale = target_size / max(h, w)
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        resized = cv2.resize(ink_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target_size, target_size), dtype=ink_mask.dtype)
        y_off = (target_size - new_h) // 2
        x_off = (target_size - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        meta = {
            "scale": scale,
            "x_offset": x_off,
            "y_offset": y_off,
            "original_size": (h, w),
            "content_size": (new_h, new_w),
        }
        return canvas, meta
    stretched = cv2.resize(ink_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return stretched, {"scale": (target_size / h, target_size / w), "original_size": (h, w)}


def pixel_to_normalized(
    ys: np.ndarray,
    xs: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel coordinates to [0,1] normalized coordinates."""
    nx = xs.astype(np.float64) / max(width - 1, 1)
    ny = ys.astype(np.float64) / max(height - 1, 1)
    return np.clip(nx, 0.0, 1.0), np.clip(ny, 0.0, 1.0)
