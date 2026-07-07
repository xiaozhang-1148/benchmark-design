"""Quality-check figures for foreground pixel density calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot
from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.foreground_load.compute import load_grayscale_image
from benchmark_design.vision.foreground_load.models import GlobalForegroundLoadConfig, PageForegroundLoadResult
from benchmark_design.vision.foreground_load.normalization import (
    compute_darkness_from_gray,
    robust_normalize_gray,
)
from benchmark_design.vision.foreground_load.otsu import histogram_uint8, otsu_from_histogram
from benchmark_design.vision.masks import build_unified_page_masks


def _require_matplotlib():
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    return plt


@with_locked_pyplot
def _draw_calibration_histogram(
    calibration_histogram: np.ndarray,
    *,
    tau_D: float,
    output_path: Path,
) -> None:
    plt = _require_matplotlib()
    centers = (np.arange(len(calibration_histogram)) + 0.5) / len(calibration_histogram)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(centers, calibration_histogram, width=1.0 / len(calibration_histogram), color="#4472C4")
    axis.axvline(tau_D, color="#C00000", linewidth=2, label=f"tau_D={tau_D:.4f}")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Darkness S")
    axis.set_ylabel("Pixel count in calibration set")
    axis.set_title("Pooled darkness histogram (R_eff calibration set)")
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@with_locked_pyplot
def _draw_page_quality_sheet(
    *,
    gray: np.ndarray,
    normalized: np.ndarray,
    darkness: np.ndarray,
    region_mask: np.ndarray,
    tau_D: float,
    page_otsu: float,
    output_path: Path,
    title: str,
) -> None:
    plt = _require_matplotlib()
    foreground = (darkness >= tau_D) & region_mask
    raw_fg = (gray <= page_otsu) & region_mask

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = (
        (gray, "Original gray", "gray", 0, 255),
        (normalized, "Normalized G_tilde", "gray", 0, 1),
        (darkness, "Darkness S", "inferno", 0, 1),
        (raw_fg.astype(np.float32), "Raw Otsu foreground", "gray", 0, 1),
        (foreground.astype(np.float32), "Dataset-threshold foreground", "gray", 0, 1),
        (darkness * region_mask, "Darkness in R_eff", "inferno", 0, 1),
    )
    for axis, (data, panel_title, cmap, vmin, vmax) in zip(
        axes.ravel(),
        panels,
        strict=True,
    ):
        axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title, fontsize=9)
        axis.axis("off")

    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def export_foreground_pixel_density_diagnostics_figures(
    results: list[PageForegroundLoadResult],
    pages: list[PageAnnotation],
    *,
    input_dir: Path,
    figures_root: Path,
    global_config: GlobalForegroundLoadConfig,
    calibration_histogram: np.ndarray,
    sample_count: int = 5,
) -> dict[str, int]:
    output_dir = figures_root / "foreground_pixel_density" / "quality_checks"
    page_map = {page.page_id: page for page in pages}

    _draw_calibration_histogram(
        calibration_histogram,
        tau_D=global_config.tau_D,
        output_path=output_dir / "calibration_darkness_histogram.png",
    )

    candidates = [
        result
        for result in results
        if result.D_page_eff is not None and result.page_id in page_map
    ]
    candidates.sort(key=lambda item: item.D_page_eff or 0.0, reverse=True)
    written = 0
    for result in candidates[:sample_count]:
        page = page_map[result.page_id]
        image_path = input_dir / page.image_name
        if not image_path.is_file():
            continue
        gray = load_grayscale_image(image_path)
        if gray.shape != (page.image_height, page.image_width):
            try:
                import cv2
            except ImportError:
                continue
            gray = cv2.resize(
                gray,
                (page.image_width, page.image_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized = robust_normalize_gray(
            gray,
            q_low=global_config.q_low,
            q_high=global_config.q_high,
        )
        darkness = compute_darkness_from_gray(
            gray,
            q_low=global_config.q_low,
            q_high=global_config.q_high,
        )
        masks = build_unified_page_masks(
            page.blocks,
            image_width=page.image_width,
            image_height=page.image_height,
        ).foreground_masks()
        page_otsu = otsu_from_histogram(histogram_uint8(gray))
        _draw_page_quality_sheet(
            gray=gray,
            normalized=normalized,
            darkness=darkness,
            region_mask=masks.effective_union,
            tau_D=global_config.tau_D,
            page_otsu=page_otsu,
            output_path=output_dir / f"{result.page_id.replace('/', '_')}_quality.png",
            title=f"{result.page_id} D_page={result.D_page_eff:.3f} tau_D={global_config.tau_D:.4f}",
        )
        written += 1

    return {
        "calibration_histogram": 1,
        "quality_check_pages": written,
    }
