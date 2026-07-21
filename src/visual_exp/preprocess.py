"""Official DeepSeek-OCR-2 dynamic-resolution image preprocessing."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image, ImageOps


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 768,
) -> tuple[list[Image.Image], tuple[int, int]]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(orig_height, 1)
    target_ratios = []
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j == n:
                    target_ratios.append((i, j))
    best = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))
    target_width = image_size * best[0]
    target_height = image_size * best[1]
    resized = image.resize((target_width, target_height))
    crops = []
    for i in range(best[1]):
        for j in range(best[0]):
            box = (j * image_size, i * image_size, (j + 1) * image_size, (i + 1) * image_size)
            crops.append(resized.crop(box))
    return crops, best


class BasicImageTransform:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        from torchvision import transforms

        self.mean = mean
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img)


def prepare_image_tensors(
    image: Image.Image,
    *,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    """
    Returns:
      patches: [P, 3, image_size, image_size] (zeros if no local crops)
      global_view: [1, 3, base_size, base_size]
      spatial: [w_crops, h_crops]
      n_local_patches: number of real local patches (0 if disabled)
    """
    transform = BasicImageTransform()
    n_local = 0
    if crop_mode and (image.size[0] > 768 or image.size[1] > 768):
        crops, crop_ratio = _dynamic_preprocess(image, image_size=image_size)
        patches = torch.stack([transform(c) for c in crops], dim=0)
        spatial = [crop_ratio[0], crop_ratio[1]]
        n_local = int(patches.shape[0])
    else:
        patches = torch.zeros(1, 3, image_size, image_size)
        spatial = [1, 1]
        n_local = 0

    global_view = ImageOps.pad(
        image,
        (base_size, base_size),
        color=tuple(int(x * 255) for x in transform.mean),
    )
    global_t = transform(global_view).unsqueeze(0)
    return patches.to(torch.bfloat16), global_t.to(torch.bfloat16), spatial, n_local
