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
    Official visual branch only — no LM generate():
      image → SAM → Qwen2Decoder2Encoder (causal tokens) → MlpProjector (896→1280)
      → concat local+global projected tokens → mean pool → L2
    """

    method_name = "DeepSeek-OCR2 mean-pooled projected-token embedding"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.exclude_sep = bool(cfg.get("exclude_view_separator", True))
        self.base_size = int(cfg["preprocess"]["base_size"])
        self.image_size = int(cfg["preprocess"]["image_size"])
        self.crop_mode = bool(cfg["preprocess"]["crop_mode"])
        self._load()

    def _load(self) -> None:
        from transformers import AutoModel

        patch_transformers_for_deepseek_ocr2()
        path = model_path(self.cfg)
        attn = self.cfg["model"].get("attn_implementation", "sdpa")
        print(f"[extract] loading {path} dtype=bf16 attn={attn}")
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
            proj = self.projector
            candidate = getattr(proj, "layers", None)
            if isinstance(candidate, torch.nn.Linear):
                self.embed_dim = int(candidate.out_features)
            elif isinstance(candidate, torch.nn.Sequential):
                for mod in candidate.modules():
                    if isinstance(mod, torch.nn.Linear):
                        self.embed_dim = int(mod.out_features)
            else:
                for attr in ("proj", "linear", "fc"):
                    if hasattr(proj, attr) and isinstance(getattr(proj, attr), torch.nn.Linear):
                        self.embed_dim = int(getattr(proj, attr).out_features)
                        break
                else:
                    for mod in proj.modules():
                        if isinstance(mod, torch.nn.Linear):
                            self.embed_dim = int(mod.out_features)
                            break
        except Exception:
            self.embed_dim = 1280
        print(f"[extract] projected embed_dim≈{self.embed_dim} method={self.method_name}")

    @torch.inference_mode()
    def _encode_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """SAM → Qwen2 → Projector. Returns [B, T, D] projected tokens (BF16)."""
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
        patches = patches.to(self.device)
        global_t = global_t.to(self.device)

        parts: list[torch.Tensor] = []
        shapes: dict[str, Any] = {}

        if n_local > 0 and float(patches[:n_local].abs().sum()) > 0:
            local = self._encode_view(patches[:n_local])  # [P, T, D]
            shapes["local"] = tuple(local.shape)
            parts.append(local.reshape(-1, local.shape[-1]))

        global_f = self._encode_view(global_t)  # [1, T, D]
        shapes["global"] = tuple(global_f.shape)
        parts.append(global_f.reshape(-1, global_f.shape[-1]))

        # Do NOT include view_separator in the mean (learned non-image token).
        tokens = torch.cat(parts, dim=0)
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
            "n_local_patches": int(n_local),
            "spatial": spatial,
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
