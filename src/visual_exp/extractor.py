"""Extract mean-pooled projected visual tokens (SAM → Qwen2 → Projector → mean → L2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..model_compat import patch_transformers_for_deepseek_ocr2
from .config import model_path
from .preprocess import prepare_image_tensors


class ProjectedTokenExtractor:
    """
    Official visual branch only — no LM generate().

    P1 default: global view only (256 projected tokens) → mean → L2.
    Local patches disabled so every image uses the same token recipe.
    """

    method_name = "DeepSeek-OCR2 mean-pooled projected-token embedding"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.exclude_sep = bool(cfg.get("exclude_view_separator", True))
        self.use_local = bool(cfg.get("use_local_patches", False))
        self.base_size = int(cfg["preprocess"]["base_size"])
        self.image_size = int(cfg["preprocess"]["image_size"])
        # crop_mode only matters if local patches enabled
        self.crop_mode = bool(cfg["preprocess"]["crop_mode"]) and self.use_local
        self._load()

    def _load(self) -> None:
        from transformers import AutoModel

        patch_transformers_for_deepseek_ocr2()
        path = model_path(self.cfg)
        attn = self.cfg["model"].get("attn_implementation", "eager")
        print(f"[extract] loading {path} dtype=bf16 attn={attn} use_local_patches={self.use_local}")
        self.model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            _attn_implementation=attn,
        )
        self.model.eval()
        self.model.to(self.device)
        m = self.model.model if hasattr(self.model, "model") else self.model
        self.sam = m.sam_model
        self.qwen = m.qwen2_model
        self.projector = m.projector
        self.view_seperator = getattr(m, "view_seperator", None)
        self.embed_dim = 1280
        try:
            candidate = getattr(self.projector, "layers", None)
            if isinstance(candidate, torch.nn.Linear):
                self.embed_dim = int(candidate.out_features)
            else:
                for mod in self.projector.modules():
                    if isinstance(mod, torch.nn.Linear):
                        self.embed_dim = int(mod.out_features)
                        break
        except Exception:
            self.embed_dim = 1280
        print(f"[extract] projected embed_dim≈{self.embed_dim} method={self.method_name}")

    @torch.inference_mode()
    def _encode_view(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self.sam(tensor)
        x = self.qwen(x)
        x = self.projector(x)
        return x

    @torch.inference_mode()
    def embed_image(self, image: Image.Image, *, debug: bool = False) -> dict[str, Any]:
        patches, global_t, spatial, n_local = prepare_image_tensors(
            image,
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
        )
        global_t = global_t.to(self.device)

        shapes: dict[str, Any] = {}
        # P1: only global projected tokens (fixed 256 for 1024 global view)
        global_f = self._encode_view(global_t)  # [1, T, D], T=256
        shapes["global"] = tuple(global_f.shape)
        tokens = global_f.reshape(-1, global_f.shape[-1])

        if self.use_local and n_local > 0:
            patches = patches.to(self.device)
            local = self._encode_view(patches[:n_local])
            shapes["local"] = tuple(local.shape)
            tokens = torch.cat([local.reshape(-1, local.shape[-1]), tokens], dim=0)

        shapes["tokens_before_filter"] = tuple(tokens.shape)
        tokens_f = tokens.float()
        finite = torch.isfinite(tokens_f).all(dim=-1)
        tokens_f = tokens_f[finite]
        if tokens_f.numel() == 0:
            raise RuntimeError("no finite projected visual tokens")

        pooled = tokens_f.mean(dim=0)
        norm_before = float(torch.linalg.vector_norm(pooled).item())
        embedding = F.normalize(pooled, p=2, dim=-1).cpu().numpy().astype(np.float32)

        if debug:
            print(
                f"[debug] shapes={shapes} token_count={tokens_f.shape[0]} "
                f"dtype={tokens.dtype} min={float(tokens_f.min())} max={float(tokens_f.max())} "
                f"norm_before={norm_before:.4f} emb_dim={embedding.shape[0]}"
            )

        return {
            "embedding": embedding,
            "token_count": int(tokens_f.shape[0]),
            "n_local_patches": int(n_local) if self.use_local else 0,
            "spatial": spatial if self.use_local else [1, 1],
            "norm_before": norm_before,
            "embedding_dim": int(embedding.shape[0]),
            "shapes": shapes,
            "token_min": float(tokens_f.min()),
            "token_max": float(tokens_f.max()),
            "token_dtype": str(tokens.dtype),
        }

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
