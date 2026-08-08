"""Official DeepSeek-OCR2 global+local image preprocessing.

Reproduces ``dynamic_preprocess`` / ``ImageOps.pad`` / mean-std=(0.5,) from
the model remote code. Small images (w<=768 and h<=768) use a fixed pad-to-768
local fallback so every sample has a local descriptor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import transforms


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
GLOBAL_SIZE = 1024
LOCAL_SIZE = 768
MIN_NUM = 2
MAX_NUM = 6


@dataclass(frozen=True)
class PageViewMeta:
    original_width: int
    original_height: int
    local_crop_count: int
    crop_grid_width: int
    crop_grid_height: int
    crop_order: str  # "row-major: left-to-right, top-to-bottom"
    small_image_fallback: bool
    pad_color_rgb: tuple[int, int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    global_size: int
    local_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BasicImageTransform:
    """Matches DeepSeek-OCR2 ``BasicImageTransform`` (mean/std = 0.5)."""

    def __init__(
        self,
        mean: tuple[float, float, float] = MEAN,
        std: tuple[float, float, float] = STD,
    ) -> None:
        self.mean = mean
        self.std = std
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Official ``find_closest_aspect_ratio`` from modeling_deepseekocr2.py."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = MIN_NUM,
    max_num: int = MAX_NUM,
    image_size: int = LOCAL_SIZE,
    use_thumbnail: bool = False,
) -> tuple[list[Image.Image], tuple[int, int]]:
    """Official DeepSeek-OCR2 dynamic crop (deterministic, no augmentation)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios_sorted = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios_sorted, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


def prepare_page_views(
    image: Image.Image,
    *,
    base_size: int = GLOBAL_SIZE,
    image_size: int = LOCAL_SIZE,
    min_num: int = MIN_NUM,
    max_num: int = MAX_NUM,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, PageViewMeta]:
    """
    Build global + local tensors for one page.

    Returns:
      global_t: [1, 3, base_size, base_size]
      local_t:  [m, 3, image_size, image_size]  (m >= 1 always)
      meta:     crop / fallback metadata
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = BasicImageTransform()
    pad_color = tuple(int(x * 255) for x in transform.mean)
    ow, oh = image.size
    small = ow <= image_size and oh <= image_size

    if small:
        local_pil = ImageOps.pad(image, (image_size, image_size), color=pad_color)
        local_crops = [local_pil]
        grid_w, grid_h = 1, 1
        fallback = True
    else:
        local_crops, (grid_w, grid_h) = dynamic_preprocess(
            image, min_num=min_num, max_num=max_num, image_size=image_size
        )
        fallback = False

    local_t = torch.stack([transform(c) for c in local_crops], dim=0)

    global_view = ImageOps.pad(image, (base_size, base_size), color=pad_color)
    global_t = transform(global_view).unsqueeze(0)

    meta = PageViewMeta(
        original_width=int(ow),
        original_height=int(oh),
        local_crop_count=int(local_t.shape[0]),
        crop_grid_width=int(grid_w),
        crop_grid_height=int(grid_h),
        crop_order="row-major: left-to-right, top-to-bottom",
        small_image_fallback=bool(fallback),
        pad_color_rgb=pad_color,  # type: ignore[arg-type]
        mean=MEAN,
        std=STD,
        global_size=base_size,
        local_size=image_size,
    )
    return global_t.to(dtype), local_t.to(dtype), meta


def load_image_rgb(path: str) -> Image.Image:
    """PIL open → EXIF transpose → RGB. Raises on corrupt images."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
