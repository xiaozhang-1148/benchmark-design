"""Continuous foreground pixel density statistics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

from benchmark_design.ocr.length_distribution import percentile
from benchmark_design.vision.flow_structure.models import PageFlowStructureResult
from benchmark_design.vision.foreground_load.models import PageForegroundLoadResult

FLOW_STRUCTURE_ORDER: tuple[str, ...] = ("Single-flow", "Columnar-flow", "Hybrid-flow")


@dataclass(frozen=True, slots=True)
class DensityDistributionStats:
    metric: str
    unit: str
    n: int
    mean: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float
    area_weighted_density: float | None

    @property
    def iqr_label(self) -> str:
        if self.n == 0:
            return "NA"
        return f"{format_density_pct(self.p25)}–{format_density_pct(self.p75)}"


@dataclass(frozen=True, slots=True)
class FlowStructureDensityStats:
    flow_structure: str
    pages: int
    mean: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float

    @property
    def iqr_label(self) -> str:
        if self.pages == 0:
            return "NA"
        return f"{format_density_pct(self.p25)}–{format_density_pct(self.p75)}"


def format_density_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def density_to_pct(value: float) -> float:
    return value * 100.0


def format_area_weighted_pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return format_density_pct(value)


@dataclass(frozen=True, slots=True)
class BlockDensityBand:
    label: str
    range_label: str
    interpretation: str
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class DiagnosticSignal:
    label: str
    count: int
    ratio: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class DensityBandSummary:
    total: int
    bands: tuple[BlockDensityBand, ...]


# Backward-compatible alias
BlockDensityBandSummary = DensityBandSummary


DENSITY_BAND_SPECS: tuple[tuple[str, str, str, float | None, float | None], ...] = (
    ("Extremely sparse", "<4%", "near-empty or very sparse answer-region candidate", None, 0.04),
    ("Sparse", "4%-6%", "low foreground occupancy", 0.04, 0.06),
    ("Low-medium", "6%-8%", "common low-to-medium foreground load", 0.06, 0.08),
    ("Medium", "8%-10%", "common medium foreground load", 0.08, 0.10),
    ("Medium-high", "10%-12%", "moderately dense foreground load", 0.10, 0.12),
    ("Dense", "12%-15%", "dense answer-region foreground load", 0.12, 0.15),
    ("Very dense", ">=15%", "high-density tail samples", 0.15, None),
)

PAGE_DENSITY_BAND_SPECS: tuple[tuple[str, str, str, float | None, float | None], ...] = (
    ("Extremely sparse", "<4%", "near-empty or very sparse annotated page-region candidate", None, 0.04),
    ("Sparse", "4%-6%", "low foreground occupancy in annotated page regions", 0.04, 0.06),
    ("Low-medium", "6%-8%", "common low-to-medium page-level foreground load", 0.06, 0.08),
    ("Medium", "8%-10%", "common medium page-level foreground load", 0.08, 0.10),
    ("Medium-high", "10%-12%", "moderately dense page-level foreground load", 0.10, 0.12),
    ("Dense", "12%-15%", "dense annotated page-region foreground load", 0.12, 0.15),
    ("Very dense", ">=15%", "high-density page-level tail samples", 0.15, None),
)

BLOCK_DENSITY_BAND_SPECS = DENSITY_BAND_SPECS


def _density_in_band(density: float, *, lower: float | None, upper: float | None) -> bool:
    if lower is not None and density < lower:
        return False
    if upper is not None and density >= upper:
        return False
    return True


def _compute_density_bands(
    densities: list[float],
    specs: tuple[tuple[str, str, str, float | None, float | None], ...],
) -> DensityBandSummary:
    total = len(densities)
    bands: list[BlockDensityBand] = []
    for label, range_label, interpretation, lower, upper in specs:
        count = sum(1 for density in densities if _density_in_band(density, lower=lower, upper=upper))
        ratio = (count / total) if total else 0.0
        bands.append(
            BlockDensityBand(
                label=label,
                range_label=range_label,
                interpretation=interpretation,
                count=count,
                ratio=ratio,
            )
        )
    return DensityBandSummary(total=total, bands=tuple(bands))


def compute_page_density_bands(results: list[PageForegroundLoadResult]) -> DensityBandSummary:
    densities = [result.D_page_eff for result in results if result.D_page_eff is not None]
    return _compute_density_bands(densities, PAGE_DENSITY_BAND_SPECS)


def compute_block_density_bands(results: list[PageForegroundLoadResult]) -> DensityBandSummary:
    densities = [
        block.D_block_i
        for result in results
        for block in result.block_results
        if block.D_block_i is not None
    ]
    return _compute_density_bands(densities, BLOCK_DENSITY_BAND_SPECS)


def compute_block_diagnostic_signals(results: list[PageForegroundLoadResult]) -> list[DiagnosticSignal]:
    from benchmark_design.vision.foreground_load.classification import EXTREME_DENSITY_TAG

    blocks = [
        block
        for result in results
        for block in result.block_results
        if block.D_block_i is not None
    ]
    total = len(blocks)
    manual_review = sum(1 for block in blocks if block.needs_manual_review)
    mask_oob = sum(1 for block in blocks if "mask_out_of_bounds" in block.review_reason.split(";"))
    saturated_low = sum(1 for block in blocks if "density_saturated_low" in block.review_reason.split(";"))
    extreme = sum(
        1
        for block in blocks
        if EXTREME_DENSITY_TAG in block.foreground_load_tags.split(";")
    )

    def _ratio(count: int) -> float:
        return (count / total) if total else 0.0

    return [
        DiagnosticSignal(
            label="Manual-review candidates",
            count=manual_review,
            ratio=_ratio(manual_review),
            interpretation="Blocks flagged by density or mask diagnostics",
        ),
        DiagnosticSignal(
            label="`mask_out_of_bounds`",
            count=mask_oob,
            ratio=_ratio(mask_oob),
            interpretation="Possible annotation boundary or clipping issue",
        ),
        DiagnosticSignal(
            label="`density_saturated_low`",
            count=saturated_low,
            ratio=_ratio(saturated_low),
            interpretation="Extremely low-density / near-empty block candidate",
        ),
        DiagnosticSignal(
            label="Extreme-density candidates",
            count=extreme,
            ratio=_ratio(extreme),
            interpretation="Samples worth inspecting in density-tail analysis",
        ),
    ]


def format_ratio_pct(ratio: float) -> str:
    return f"{ratio * 100:.2f}%"


def _distribution_stats(
    *,
    metric: str,
    unit: str,
    densities: list[float],
    total_foreground: int,
    total_area: int,
) -> DensityDistributionStats:
    if not densities:
        return DensityDistributionStats(
            metric=metric,
            unit=unit,
            n=0,
            mean=0.0,
            median=0.0,
            p10=0.0,
            p25=0.0,
            p75=0.0,
            p90=0.0,
            area_weighted_density=None,
        )
    area_weighted = (total_foreground / total_area) if total_area > 0 else None
    return DensityDistributionStats(
        metric=metric,
        unit=unit,
        n=len(densities),
        mean=mean(densities),
        median=percentile(densities, 50),
        p10=percentile(densities, 10),
        p25=percentile(densities, 25),
        p75=percentile(densities, 75),
        p90=percentile(densities, 90),
        area_weighted_density=area_weighted,
    )


def compute_page_density_stats(results: list[PageForegroundLoadResult]) -> DensityDistributionStats:
    densities = [result.D_page_eff for result in results if result.D_page_eff is not None]
    total_foreground = sum(result.F_eff for result in results if result.D_page_eff is not None)
    total_area = sum(result.R_eff_area for result in results if result.D_page_eff is not None)
    return _distribution_stats(
        metric="foreground_density_page",
        unit="page",
        densities=densities,
        total_foreground=total_foreground,
        total_area=total_area,
    )


def compute_block_density_stats(results: list[PageForegroundLoadResult]) -> DensityDistributionStats:
    densities: list[float] = []
    total_foreground = 0
    total_area = 0
    for result in results:
        for block in result.block_results:
            if block.D_block_i is None:
                continue
            densities.append(block.D_block_i)
            total_foreground += block.F_i
            total_area += block.block_mask_area
    return _distribution_stats(
        metric="foreground_density_block",
        unit="txtBlock",
        densities=densities,
        total_foreground=total_foreground,
        total_area=total_area,
    )


def compute_flow_structure_density_stats(
    fg_results: list[PageForegroundLoadResult],
    flow_results: list[PageFlowStructureResult],
) -> list[FlowStructureDensityStats]:
    flow_by_page = {result.page_id: result.flow_structure for result in flow_results}
    grouped: dict[str, list[float]] = {label: [] for label in FLOW_STRUCTURE_ORDER}
    for result in fg_results:
        flow_structure = flow_by_page.get(result.page_id, "NA")
        if flow_structure not in grouped or result.D_page_eff is None:
            continue
        grouped[flow_structure].append(result.D_page_eff)

    stats: list[FlowStructureDensityStats] = []
    for flow_structure in FLOW_STRUCTURE_ORDER:
        densities = grouped[flow_structure]
        if not densities:
            stats.append(
                FlowStructureDensityStats(
                    flow_structure=flow_structure,
                    pages=0,
                    mean=0.0,
                    median=0.0,
                    p10=0.0,
                    p25=0.0,
                    p75=0.0,
                    p90=0.0,
                )
            )
            continue
        stats.append(
            FlowStructureDensityStats(
                flow_structure=flow_structure,
                pages=len(densities),
                mean=mean(densities),
                median=percentile(densities, 50),
                p10=percentile(densities, 10),
                p25=percentile(densities, 25),
                p75=percentile(densities, 75),
                p90=percentile(densities, 90),
            )
        )
    return stats


def select_nearest_quantile_sample(
    entries: list[tuple[object, float]],
    *,
    target_percentile: float,
    exclude_review: bool = True,
    needs_review: Callable[[object], bool] | None = None,
) -> tuple[object, float] | None:
    """Pick entry whose density is closest to the corpus target percentile."""
    if not entries:
        return None
    densities = [density for _, density in entries]
    target = percentile(densities, target_percentile)
    candidates = entries
    if exclude_review and needs_review is not None:
        candidates = [entry for entry in entries if not needs_review(entry[0])]
        if not candidates:
            candidates = entries
    return min(candidates, key=lambda entry: abs(entry[1] - target))
