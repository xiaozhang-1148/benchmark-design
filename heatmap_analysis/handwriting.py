"""Handwriting ink extraction from answer sheets (CPU + GPU)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from heatmap_analysis.alignment import PageRegion
from heatmap_analysis.config import PreprocessingConfig
from heatmap_analysis.gpu import get_xp, to_numpy
from heatmap_analysis import image_ops as iops


def otsu_ink_mask(gray: np.ndarray) -> np.ndarray:
    return to_numpy(iops.otsu_binary_inv(gray, xp=np))


def adaptive_ink_mask(gray: np.ndarray, block_size: int = 35, c: int = 10) -> np.ndarray:
    return to_numpy(iops.adaptive_binary_inv(gray, block_size, c, xp=np))


def mask_page_edges(mask: np.ndarray, edge_ratio: float) -> np.ndarray:
    return to_numpy(iops.mask_edges(mask, edge_ratio, xp=np))


def remove_large_structures(mask: np.ndarray, max_area_ratio: float = 0.15) -> np.ndarray:
    return to_numpy(iops.remove_large_components(mask, max_area_ratio, xp=np))


def remove_horizontal_vertical_lines(mask: np.ndarray) -> np.ndarray:
    return to_numpy(iops.remove_hv_lines(mask, xp=np))


def template_subtract_ink(
    answer_gray: np.ndarray,
    template_gray: np.ndarray,
    diff_threshold: int = 25,
) -> np.ndarray:
    return to_numpy(iops.template_subtract(answer_gray, template_gray, diff_threshold, xp=np))


def _extract_threshold_path(page: np.ndarray, config: PreprocessingConfig, *, xp: Any) -> Any:
    if config.threshold_method == "adaptive":
        ink = iops.adaptive_binary_inv(page, 35, 10, xp=xp)
    else:
        ink = iops.otsu_binary_inv(page, xp=xp)
    ink = iops.mask_edges(ink, config.edge_mask_ratio, xp=xp)
    ink = iops.remove_large_components(ink, 0.15, xp=xp)
    ink = iops.remove_hv_lines(ink, xp=xp)
    return ink


def extract_ink_mask(
    gray: np.ndarray,
    config: PreprocessingConfig,
    template: np.ndarray | None = None,
    aligned: bool = False,
    *,
    use_gpu: bool = False,
    xp: Any | None = None,
) -> tuple[Any, PageRegion, dict]:
    """
    Extract handwriting ink mask on the full image (no page crop).
    Returns (ink_weights float32 on xp, full-image region, info dict).
    """
    xp = xp if xp is not None else get_xp(use_gpu)
    on_gpu = xp is not np

    h, w = gray.shape[:2]
    region = PageRegion(0, 0, w, h)
    page = gray
    info: dict = {"mode": "no_template", "aligned": aligned, "backend": "gpu" if on_gpu else "cpu", "page_crop": False}

    if config.use_template_subtraction and template is not None:
        if aligned:
            tmpl = template
            ans = page if page.shape == template.shape else cv2.resize(page, (template.shape[1], template.shape[0]))
        else:
            from heatmap_analysis.alignment import align_to_template

            ans, tmpl = align_to_template(page, template)
        ink = iops.template_subtract(ans, tmpl, 25, xp=xp)
        info["mode"] = "template_subtraction"
    else:
        ink = _extract_threshold_path(page, config, xp=xp)
        info["mode"] = "threshold"
        info["warning"] = "printed content may be counted as ink"

    weights = iops.ink_weights_from_mask(ink, xp=xp)
    return weights, region, info
