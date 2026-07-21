"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    dataset_dir: Path
    metadata_file: Path | None = None
    image_column: str = "image_path"
    id_column: str = "image_id"
    template_column: str = "template_id"
    template_dir: Path | None = None
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@dataclass
class PreprocessingConfig:
    use_template_subtraction: bool = False
    align_to_template: bool = True
    preserve_aspect_ratio: bool = True
    edge_mask_ratio: float = 0.02
    threshold_method: str = "otsu"  # otsu | adaptive
    min_page_pixels: int = 1000
    max_aspect_ratio: float = 3.0
    blank_ink_threshold: float = 0.0001


@dataclass
class HeatmapConfig:
    grid_size: int = 64
    gaussian_sigma: float = 1.2
    active_cell_threshold: float = 0.001
    save_raw_grid: bool = True
    save_smoothed_grid: bool = True


@dataclass
class ClusteringConfig:
    pca_variance: float = 0.95
    pca_n_components: int | None = None
    k_values: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8, 10])
    gmm_components: list[int] = field(default_factory=lambda: list(range(2, 11)))
    closest_samples: int = 3
    boundary_samples: int = 3
    random_seed: int = 42
    min_samples_for_clustering: int = 10
    separate_by_template: bool = False
    # legacy fields kept for compatibility with old clustering.py
    feature_mode: str = "relative_layout"
    algorithm: str = "kmeans"
    k_min: int = 2
    k_max: int = 10
    k_fixed: int | None = None
    bootstrap_iterations: int = 30
    use_umap: bool = False
    extra_k_outputs: list[int] = field(default_factory=list)


@dataclass
class ReportConfig:
    group_by: list[str] = field(default_factory=list)
    representative_samples: int = 3
    language: str = "zh-CN"
    colormap: str = "turbo"
    diff_colormap: str = "RdBu_r"


@dataclass
class GpuConfig:
    enabled: bool = True
    device_ids: list[int] | None = None
    num_workers: int | None = None  # defaults to number of device_ids
    min_images_for_parallel: int = 100  # below this, use single-GPU in-process mode
    preprocessing: bool = False  # GPU ink extraction (slower on small pages; optional)
    clustering: bool = True  # GPU PCA + KMeans for large datasets


@dataclass
class OutputConfig:
    output_dir: Path = Path("./hotmap")
    cache_dir: Path | None = None
    resume: bool = True


@dataclass
class AnalysisConfig:
    input: InputConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)

    @property
    def cache_dir(self) -> Path:
        if self.output.cache_dir is not None:
            return self.output.cache_dir
        return self.output.output_dir / "cache"

    def validate(self) -> None:
        if not self.input.dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir not found: {self.input.dataset_dir}")
        if self.heatmap.grid_size != 64:
            raise ValueError("grid_size must be 64")
        if self.clustering.k_min < 2:
            raise ValueError("k_min must be >= 2")
        if self.clustering.k_max < self.clustering.k_min:
            raise ValueError("k_max must be >= k_min")


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def load_config(path: Path) -> AnalysisConfig:
    """Load YAML config; paths relative to config file location."""
    config_path = path.resolve()
    base = config_path.parent
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    inp = raw.get("input", {})
    pre = raw.get("preprocessing", {})
    hm = raw.get("heatmap", {})
    cl = raw.get("clustering", {})
    rep = raw.get("report", {})
    out = raw.get("output", {})
    gpu_raw = raw.get("gpu", {})

    dataset_dir = _resolve_path(base, inp.get("dataset_dir", "./dataset"))
    assert dataset_dir is not None

    metadata_file = _resolve_path(base, inp.get("metadata_file"))
    template_dir = _resolve_path(base, inp.get("template_dir"))
    output_dir = _resolve_path(base, out.get("output_dir", "./hotmap"))
    assert output_dir is not None
    cache_dir = _resolve_path(base, out.get("cache_dir"))

    group_by = rep.get("group_by")
    if group_by is None:
        group_by_list: list[str] = []
    elif isinstance(group_by, str):
        group_by_list = [group_by]
    else:
        group_by_list = list(group_by)

    cfg = AnalysisConfig(
        input=InputConfig(
            dataset_dir=dataset_dir,
            metadata_file=metadata_file,
            image_column=inp.get("image_column", "image_path"),
            id_column=inp.get("id_column", "image_id"),
            template_column=inp.get("template_column", "template_id"),
            template_dir=template_dir,
            image_extensions=tuple(inp.get("image_extensions", [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"])),
        ),
        preprocessing=PreprocessingConfig(
            use_template_subtraction=pre.get("use_template_subtraction", False),
            align_to_template=pre.get("align_to_template", True),
            preserve_aspect_ratio=pre.get("preserve_aspect_ratio", True),
            edge_mask_ratio=float(pre.get("edge_mask_ratio", 0.02)),
            threshold_method=pre.get("threshold_method", "otsu"),
            min_page_pixels=int(pre.get("min_page_pixels", 1000)),
            max_aspect_ratio=float(pre.get("max_aspect_ratio", 3.0)),
            blank_ink_threshold=float(pre.get("blank_ink_threshold", 0.0001)),
        ),
        heatmap=HeatmapConfig(
            grid_size=int(hm.get("grid_size", 64)),
            gaussian_sigma=float(hm.get("gaussian_sigma", 1.2)),
            active_cell_threshold=float(hm.get("active_cell_threshold", 0.001)),
            save_raw_grid=hm.get("save_raw_grid", True),
            save_smoothed_grid=hm.get("save_smoothed_grid", True),
        ),
        clustering=ClusteringConfig(
            pca_variance=float(cl.get("pca_variance", 0.95)),
            pca_n_components=cl.get("pca_n_components"),
            k_values=list(cl.get("k_values", [2, 3, 4, 5, 6, 8, 10])),
            gmm_components=list(cl.get("gmm_components", list(range(2, 11)))),
            closest_samples=int(cl.get("closest_samples", 3)),
            boundary_samples=int(cl.get("boundary_samples", 3)),
            random_seed=int(cl.get("random_seed", 42)),
            min_samples_for_clustering=int(cl.get("min_samples_for_clustering", 10)),
            separate_by_template=cl.get("separate_by_template", False),
            feature_mode=cl.get("feature_mode", "relative_layout"),
            algorithm=cl.get("algorithm", "kmeans"),
            k_min=int(cl.get("k_min", 2)),
            k_max=int(cl.get("k_max", 10)),
            k_fixed=cl.get("k_fixed"),
            bootstrap_iterations=int(cl.get("bootstrap_iterations", 30)),
            use_umap=cl.get("use_umap", False),
            extra_k_outputs=list(cl.get("extra_k_outputs", [])),
        ),
        report=ReportConfig(
            group_by=group_by_list,
            representative_samples=int(rep.get("representative_samples", 3)),
            language=rep.get("language", "zh-CN"),
            colormap=rep.get("colormap", "turbo"),
            diff_colormap=rep.get("diff_colormap", "RdBu_r"),
        ),
        output=OutputConfig(
            output_dir=output_dir,
            cache_dir=cache_dir,
            resume=out.get("resume", True),
        ),
        gpu=GpuConfig(
            enabled=gpu_raw.get("enabled", True),
            device_ids=gpu_raw.get("device_ids"),
            num_workers=gpu_raw.get("num_workers"),
            min_images_for_parallel=int(gpu_raw.get("min_images_for_parallel", 100)),
            preprocessing=bool(gpu_raw.get("preprocessing", False)),
            clustering=bool(gpu_raw.get("clustering", True)),
        ),
    )
    cfg.validate()
    return cfg
