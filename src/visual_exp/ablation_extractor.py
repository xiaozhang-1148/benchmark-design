"""A0–A3 embedding extractors (crop / token / pooling variants)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .extractor import ProjectedTokenExtractor
from .preprocess import BasicImageTransform, _dynamic_preprocess

VariantId = Literal["A0", "A1", "A2", "A3"]

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "A0": {
        "description": "baseline: current flow, current tokens, current pooling",
        "crop_mode": False,
        "use_local_patches": False,
        "token_mode": "global",
        "pooling": "mean",
        "global_resize": "pad",
        "reuse_baseline": True,
    },
    "A1": {
        "description": "dynamic crop on; global tokens only; mean pooling",
        "crop_mode": True,
        "use_local_patches": False,
        "token_mode": "global",
        "pooling": "mean",
        "global_resize": "pad",
        # Global view identical to A0; local crops unused → same embedding as A0.
        "reuse_baseline": True,
    },
    "A2": {
        "description": "dynamic crop; global+local; per-view mean then equal-weight fuse",
        "crop_mode": True,
        "use_local_patches": True,
        "token_mode": "global_local",
        "pooling": "view_equal_mean",
        "global_resize": "pad",
        "reuse_baseline": False,
    },
    "A3": {
        "description": "fixed-size resize (no pad), crop off; global only; mean pooling",
        "crop_mode": False,
        "use_local_patches": False,
        "token_mode": "global",
        "pooling": "mean",
        "global_resize": "stretch",
        "reuse_baseline": False,
    },
}


def prepare_image_tensors_variant(
    image: Image.Image,
    *,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
    global_resize: str = "pad",
) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    """Like prepare_image_tensors, with optional stretch global resize for A3."""
    transform = BasicImageTransform()
    if crop_mode and (image.size[0] > 768 or image.size[1] > 768):
        crops, crop_ratio = _dynamic_preprocess(image, image_size=image_size)
        patches = torch.stack([transform(c) for c in crops], dim=0)
        spatial = [crop_ratio[0], crop_ratio[1]]
        n_local = int(patches.shape[0])
    else:
        patches = torch.zeros(1, 3, image_size, image_size)
        spatial = [1, 1]
        n_local = 0

    if global_resize == "stretch":
        global_view = image.resize((base_size, base_size), Image.Resampling.BICUBIC)
    else:
        global_view = ImageOps.pad(
            image,
            (base_size, base_size),
            color=tuple(int(x * 255) for x in transform.mean),
        )
    global_t = transform(global_view).unsqueeze(0)
    return patches.to(torch.bfloat16), global_t.to(torch.bfloat16), spatial, n_local


class AblationExtractor(ProjectedTokenExtractor):
    """ProjectedTokenExtractor with A0–A3 pooling / crop / resize variants."""

    def __init__(self, cfg: dict[str, Any], *, variant: str):
        self.variant = variant
        spec = VARIANT_SPECS[variant]
        cfg = deepcopy(cfg)
        cfg["use_local_patches"] = bool(spec["use_local_patches"])
        cfg.setdefault("preprocess", {})
        cfg["preprocess"] = dict(cfg["preprocess"])
        cfg["preprocess"]["crop_mode"] = bool(spec["crop_mode"])
        self.token_mode = str(spec["token_mode"])
        self.pooling = str(spec["pooling"])
        self.global_resize = str(spec["global_resize"])
        super().__init__(cfg)
        # Decouple crop preprocess from use_local (A1 may prepare crops but unused).
        self.crop_mode = bool(spec["crop_mode"])
        self.method_name = f"DeepSeek-OCR2 ablation {variant}"

    @torch.inference_mode()
    def embed_image(self, image: Image.Image, *, debug: bool = False) -> dict[str, Any]:
        patches, global_t, spatial, n_local = prepare_image_tensors_variant(
            image,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            global_resize=self.global_resize,
        )
        global_t = global_t.to(self.device)
        shapes: dict[str, Any] = {}

        global_f = self._encode_view(global_t)
        shapes["global"] = tuple(global_f.shape)

        if self.pooling == "view_equal_mean":
            view_means: list[torch.Tensor] = []
            g_tok = global_f.reshape(-1, global_f.shape[-1]).float()
            g_tok = g_tok[torch.isfinite(g_tok).all(dim=-1)]
            if g_tok.numel() == 0:
                raise RuntimeError("no finite global tokens")
            view_means.append(g_tok.mean(dim=0))

            if self.token_mode == "global_local" and n_local > 0:
                patches = patches.to(self.device)
                local = self._encode_view(patches[:n_local])  # [P, T, D]
                shapes["local"] = tuple(local.shape)
                local_f = local.float()
                for p in range(local_f.shape[0]):
                    tok = local_f[p].reshape(-1, local_f.shape[-1])
                    tok = tok[torch.isfinite(tok).all(dim=-1)]
                    if tok.numel() == 0:
                        continue
                    view_means.append(tok.mean(dim=0))

            pooled = torch.stack(view_means, dim=0).mean(dim=0)
            token_count = int(len(view_means))
            n_local_used = int(n_local) if self.token_mode == "global_local" else 0
        else:
            tokens = global_f.reshape(-1, global_f.shape[-1])
            n_local_used = 0
            if self.token_mode == "global_local" and n_local > 0:
                patches = patches.to(self.device)
                local = self._encode_view(patches[:n_local])
                shapes["local"] = tuple(local.shape)
                tokens = torch.cat([local.reshape(-1, local.shape[-1]), tokens], dim=0)
                n_local_used = int(n_local)
            tokens_f = tokens.float()
            finite = torch.isfinite(tokens_f).all(dim=-1)
            tokens_f = tokens_f[finite]
            if tokens_f.numel() == 0:
                raise RuntimeError("no finite projected visual tokens")
            pooled = tokens_f.mean(dim=0)
            token_count = int(tokens_f.shape[0])

        norm_before = float(torch.linalg.vector_norm(pooled).item())
        embedding = F.normalize(pooled, p=2, dim=-1).cpu().numpy().astype(np.float32)

        return {
            "embedding": embedding,
            "token_count": token_count,
            "n_local_patches": n_local_used,
            "spatial": spatial if self.crop_mode else [1, 1],
            "norm_before": norm_before,
            "embedding_dim": int(embedding.shape[0]),
            "shapes": shapes,
            "token_min": float("nan"),
            "token_max": float("nan"),
            "token_dtype": "float32",
            "variant": self.variant,
            "pooling": self.pooling,
        }


def make_extractor(cfg: dict[str, Any]) -> ProjectedTokenExtractor:
    """Factory used by extract_run (works under spawn multiprocessing)."""
    variant = cfg.get("_ablation_variant")
    if variant:
        return AblationExtractor(cfg, variant=str(variant))
    return ProjectedTokenExtractor(cfg)
