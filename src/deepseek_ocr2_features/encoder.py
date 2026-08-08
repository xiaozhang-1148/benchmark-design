"""Load DeepSeek-OCR2 and run SAM → Qwen2 (Projector-before) only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..model_compat import patch_transformers_for_deepseek_ocr2


@dataclass
class ModulePaths:
    sam_path: str
    qwen_path: str
    projector_path: str
    sam_class: str
    qwen_class: str
    projector_class: str
    projector_in_features: int
    projector_out_features: int
    query_1024_num: int
    query_768_num: int
    query_hidden_dim: int
    language_decoder_present: bool


class NonFiniteError(RuntimeError):
    """Raised when a tensor contains NaN/Inf at a named stage."""

    def __init__(self, stage: str, tensor: torch.Tensor, *, sample_id: str = "", image_path: str = ""):
        stats = _tensor_stats(tensor)
        msg = (
            f"non-finite at stage={stage} sample_id={sample_id!r} image_path={image_path!r} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} {stats}"
        )
        super().__init__(msg)
        self.stage = stage
        self.sample_id = sample_id
        self.image_path = image_path
        self.stats = stats


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    out: dict[str, Any] = {
        "finite_frac": float(finite.float().mean().item()) if tf.numel() else 0.0,
        "numel": int(tf.numel()),
    }
    if finite.any():
        vals = tf[finite]
        out.update(
            {
                "min": float(vals.min().item()),
                "max": float(vals.max().item()),
                "mean": float(vals.mean().item()),
            }
        )
    return out


def assert_finite(tensor: torch.Tensor, stage: str, *, sample_id: str = "", image_path: str = "") -> None:
    if not torch.isfinite(tensor).all():
        raise NonFiniteError(stage, tensor, sample_id=sample_id, image_path=image_path)


class CausalFlowVisionEncoder:
    """
    Vision-only path: ``sam_model(x) → qwen2_model(x)``.

    Does not call Projector, language decoder, generate(), or infer().
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        # DeepseekOCR2ForCausalLM does not support SDPA at the LM level.
        # Qwen2 visual encoder still uses its internal SDPA path.
        self.attn_implementation = attn_implementation
        self.projector_used = False
        self.language_decoder_used = False
        self._load()

    def _load(self) -> None:
        from transformers import AutoModel

        patch_transformers_for_deepseek_ocr2()
        print(
            f"[encoder] loading {self.model_path} device={self.device} "
            f"dtype={self.dtype} attn={self.attn_implementation}"
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            _attn_implementation=self.attn_implementation,
        )
        self.model.eval()
        m = self.model.model if hasattr(self.model, "model") else self.model
        # Keep strong refs to vision modules, then drop the language tower to
        # free CPU RAM for multi-GPU workers (each would otherwise hold ~full 6GB).
        self.sam = m.sam_model
        self.qwen = m.qwen2_model
        self.projector = m.projector
        self.module_paths = self._resolve_paths(m)
        m.sam_model = None  # type: ignore[assignment]
        m.qwen2_model = None  # type: ignore[assignment]
        m.projector = None  # type: ignore[assignment]
        del self.model
        import gc

        gc.collect()
        self.sam.to(self.device)
        self.qwen.to(self.device)
        # Projector kept on CPU solely for dimension introspection; never called.
        self.projector.to("cpu")
        self.projector.eval()
        for p in self.projector.parameters():
            p.requires_grad_(False)
        print(
            f"[encoder] SAM={self.module_paths.sam_path} "
            f"Qwen2={self.module_paths.qwen_path} "
            f"Projector={self.module_paths.projector_path} "
            f"({self.module_paths.projector_in_features}→"
            f"{self.module_paths.projector_out_features}) "
            f"queries 1024={self.module_paths.query_1024_num} "
            f"768={self.module_paths.query_768_num} "
            f"hidden={self.module_paths.query_hidden_dim}"
        )

    def _resolve_paths(self, m: nn.Module) -> ModulePaths:
        proj = self.projector
        in_f, out_f = 896, 1280
        layers = getattr(proj, "layers", None)
        if isinstance(layers, nn.Linear):
            in_f = int(layers.in_features)
            out_f = int(layers.out_features)
        else:
            for mod in proj.modules():
                if isinstance(mod, nn.Linear):
                    in_f = int(mod.in_features)
                    out_f = int(mod.out_features)
                    break
        q1024 = getattr(self.qwen, "query_1024", None)
        q768 = getattr(self.qwen, "query_768", None)
        return ModulePaths(
            sam_path="model.sam_model",
            qwen_path="model.qwen2_model",
            projector_path="model.projector",
            sam_class=self.sam.__class__.__name__,
            qwen_class=self.qwen.__class__.__name__,
            projector_class=proj.__class__.__name__,
            projector_in_features=in_f,
            projector_out_features=out_f,
            query_1024_num=int(q1024.num_embeddings) if q1024 is not None else -1,
            query_768_num=int(q768.num_embeddings) if q768 is not None else -1,
            query_hidden_dim=int(q1024.embedding_dim) if q1024 is not None else -1,
            language_decoder_present=True,  # present in checkpoint, not used
        )

    @torch.inference_mode()
    def encode_views(
        self,
        images: torch.Tensor,
        *,
        expected_tokens: int | None = None,
        stage_prefix: str = "view",
        sample_id: str = "",
        image_path: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of same-resolution RGB tensors.

        Args:
          images: [B, 3, H, W] on any device; moved to encoder device/dtype.

        Returns:
          sam_out:  [B, 896, h, w]
          qwen_out: [B, T, 896] causal-flow query tokens (Projector-before)
        """
        x = images.to(device=self.device, dtype=self.dtype, non_blocking=True)
        sam_out = self.sam(x)
        assert_finite(sam_out, f"{stage_prefix}.sam", sample_id=sample_id, image_path=image_path)
        qwen_out = self.qwen(sam_out)
        assert_finite(qwen_out, f"{stage_prefix}.qwen2", sample_id=sample_id, image_path=image_path)
        if expected_tokens is not None and qwen_out.shape[1] != expected_tokens:
            raise RuntimeError(
                f"{stage_prefix}: expected {expected_tokens} tokens, got shape {tuple(qwen_out.shape)}"
            )
        if qwen_out.shape[-1] != 896:
            raise RuntimeError(
                f"{stage_prefix}: expected last dim 896 (Projector-before), got {tuple(qwen_out.shape)}"
            )
        return sam_out, qwen_out

    def close(self) -> None:
        for attr in ("sam", "qwen", "projector"):
            if hasattr(self, attr):
                delattr(self, attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
