"""Spatial statistics metrics for individual answer sheets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from heatmap_analysis.config import HeatmapConfig
from heatmap_analysis.gpu import to_numpy
from heatmap_analysis.heatmap import HeatmapResult
from heatmap_analysis.utils import normalized_grid_coords


@dataclass
class ImageMetrics:
    image_id: str
    ink_coverage: float
    ink_total: float
    centroid_x: float
    centroid_y: float
    top_quarter_share: float
    upper_middle_share: float
    lower_middle_share: float
    bottom_quarter_share: float
    left_quarter_share: float
    middle_left_share: float
    middle_right_share: float
    right_quarter_share: float
    left_half_share: float
    right_half_share: float
    upper_half_share: float
    lower_half_share: float
    spatial_entropy: float
    hotspot_concentration: float
    active_cell_ratio: float
    center_vertical_band_ratio: float
    left_right_balance: float
    top_bottom_balance: float
    max_cell_share: float
    effective_width: float
    effective_height: float
    dense_stroke_overlap_proxy: float
    is_blank: bool
    template_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _region_share(d_rel: np.ndarray, row_slice: slice, col_slice: slice) -> float:
    return float(np.sum(d_rel[row_slice, col_slice]))


def _normalized_entropy(d_rel: np.ndarray) -> float:
    """Spatial entropy in [0,1]: H / log(n) where H = -sum p log p."""
    p = d_rel.ravel()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -float(np.sum(p * np.log(p)))
    h_max = np.log(p.size)
    if h_max <= 0:
        return 0.0
    return float(np.clip(h / h_max, 0.0, 1.0))


def _hotspot_concentration(d_rel: np.ndarray, top_fraction: float = 0.10) -> float:
    flat = d_rel.ravel()
    n = flat.size
    k = max(int(np.ceil(n * top_fraction)), 1)
    top_vals = np.partition(flat, -k)[-k:]
    return float(np.sum(top_vals))


def _effective_extent(d_rel: np.ndarray, threshold_frac: float = 0.05) -> tuple[float, float]:
    """Effective width/height: span of cells containing threshold_frac of total mass."""
    total = float(np.sum(d_rel))
    if total <= 0:
        return 0.0, 0.0
    row_mass = np.sum(d_rel, axis=1)
    col_mass = np.sum(d_rel, axis=0)

    def span(mass: np.ndarray) -> float:
        cum = np.cumsum(mass)
        lo = np.searchsorted(cum, total * threshold_frac / 2)
        hi = np.searchsorted(cum, total * (1 - threshold_frac / 2))
        if hi <= lo:
            return 0.0
        return float((hi - lo + 1) / len(mass))

    return span(col_mass), span(row_mass)


def compute_metrics(
    image_id: str,
    hm: HeatmapResult,
    ink_weights: np.ndarray,
    config: HeatmapConfig,
    template_id: str | None = None,
) -> ImageMetrics:
    d_rel = hm.d_rel
    d_abs = hm.d_abs
    gs = hm.grid_size

    ink_arr = to_numpy(ink_weights)
    ink_total = float(np.sum(ink_arr))
    ink_coverage = ink_total / max(ink_arr.size, 1)

    xx, yy = normalized_grid_coords(gs)
    mass = d_rel if not hm.is_blank else d_abs
    total_mass = float(np.sum(mass))
    if total_mass > 0:
        centroid_x = float(np.sum(mass * xx) / total_mass)
        centroid_y = float(np.sum(mass * yy) / total_mass)
    else:
        centroid_x = centroid_y = 0.5

    # Vertical quarters
    q = gs // 4
    top_quarter_share = _region_share(d_rel, slice(0, q), slice(None))
    upper_middle_share = _region_share(d_rel, slice(q, 2 * q), slice(None))
    lower_middle_share = _region_share(d_rel, slice(2 * q, 3 * q), slice(None))
    bottom_quarter_share = _region_share(d_rel, slice(3 * q, gs), slice(None))

    # Horizontal quarters
    left_quarter_share = _region_share(d_rel, slice(None), slice(0, q))
    middle_left_share = _region_share(d_rel, slice(None), slice(q, 2 * q))
    middle_right_share = _region_share(d_rel, slice(None), slice(2 * q, 3 * q))
    right_quarter_share = _region_share(d_rel, slice(None), slice(3 * q, gs))

    mid = gs // 2
    left_half_share = _region_share(d_rel, slice(None), slice(0, mid))
    right_half_share = _region_share(d_rel, slice(None), slice(mid, gs))
    upper_half_share = _region_share(d_rel, slice(0, mid), slice(None))
    lower_half_share = _region_share(d_rel, slice(mid, gs), slice(None))

    spatial_entropy = _normalized_entropy(d_rel)
    hotspot_concentration = _hotspot_concentration(d_rel)
    active_cell_ratio = float(np.mean(d_abs > config.active_cell_threshold))

    center_cols = slice(gs // 3, 2 * gs // 3)
    side_cols = np.r_[0 : gs // 3, 2 * gs // 3 : gs]
    center_mass = float(np.sum(d_abs[:, center_cols]))
    side_mass = float(np.sum(d_abs[:, side_cols]))
    center_vertical_band_ratio = center_mass / max(side_mass / 2, 1e-12)

    left_mass = float(np.sum(d_rel[:, : mid]))
    right_mass = float(np.sum(d_rel[:, mid:]))
    top_mass = float(np.sum(d_rel[:mid, :]))
    bottom_mass = float(np.sum(d_rel[mid:, :]))
    lr_sum = left_mass + right_mass
    tb_sum = top_mass + bottom_mass
    left_right_balance = 1.0 - abs(left_mass - right_mass) / lr_sum if lr_sum > 0 else 1.0
    top_bottom_balance = 1.0 - abs(top_mass - bottom_mass) / tb_sum if tb_sum > 0 else 1.0

    max_cell_share = float(np.max(d_rel)) if not hm.is_blank else 0.0
    eff_w, eff_h = _effective_extent(d_rel)

    # dense_stroke_overlap_proxy: high local absolute density relative to coverage
    mean_abs = float(np.mean(d_abs))
    max_abs = float(np.max(d_abs))
    dense_stroke_overlap_proxy = max_abs / max(mean_abs, 1e-12) if ink_coverage > 0 else 0.0

    return ImageMetrics(
        image_id=image_id,
        ink_coverage=ink_coverage,
        ink_total=ink_total,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        top_quarter_share=top_quarter_share,
        upper_middle_share=upper_middle_share,
        lower_middle_share=lower_middle_share,
        bottom_quarter_share=bottom_quarter_share,
        left_quarter_share=left_quarter_share,
        middle_left_share=middle_left_share,
        middle_right_share=middle_right_share,
        right_quarter_share=right_quarter_share,
        left_half_share=left_half_share,
        right_half_share=right_half_share,
        upper_half_share=upper_half_share,
        lower_half_share=lower_half_share,
        spatial_entropy=spatial_entropy,
        hotspot_concentration=hotspot_concentration,
        active_cell_ratio=active_cell_ratio,
        center_vertical_band_ratio=center_vertical_band_ratio,
        left_right_balance=left_right_balance,
        top_bottom_balance=top_bottom_balance,
        max_cell_share=max_cell_share,
        effective_width=eff_w,
        effective_height=eff_h,
        dense_stroke_overlap_proxy=dense_stroke_overlap_proxy,
        is_blank=hm.is_blank,
        template_id=template_id,
    )
