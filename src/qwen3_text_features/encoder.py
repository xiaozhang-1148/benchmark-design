"""Load Qwen3-Embedding-0.6B and run official last-valid-token pooling (no L2)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Official Qwen3-Embedding last-valid-token pooling (supports left/right pad)."""
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


@dataclass
class ModelInfo:
    model_name: str
    model_path: str
    revision: str | None
    architecture: str
    hidden_size: int
    max_position_embeddings: int
    padding_side: str
    truncation: bool
    pooling: str
    torch_dtype: str


class Qwen3EmbeddingEncoder:
    """
    Vision-free text encoder:
      tokenize(page_text) → AutoModel → last_token_pool → FP32 (no L2, no generate).
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
        expected_name_substr: str = "Qwen3-Embedding-0.6B",
    ) -> None:
        # Prefer local NVMe cache (user already downloaded weights there).
        os.environ.setdefault("HF_HOME", "/mnt/nvme_model/.cache/huggingface")
        os.environ.pop("HF_ENDPOINT", None)

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.expected_name_substr = expected_name_substr
        self._load()

    def _load(self) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        print(
            f"[text-encoder] loading {self.model_path} device={self.device} dtype={self.dtype} "
            f"local_files_only={self.local_files_only}",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        # Enforce left pad even if tokenizer config differs.
        self.tokenizer.padding_side = "left"

        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        hidden = int(getattr(self.config, "hidden_size", -1))
        if hidden != 1024:
            raise RuntimeError(f"expected hidden_size=1024, got {hidden}")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        )
        self.model.eval()
        self.model.to(self.device)

        name_hint = str(self.model_path)
        if self.expected_name_substr not in name_hint and "Qwen3-Embedding-0.6B" not in name_hint:
            # Still allow local snapshot paths that contain the hub folder name
            hub_ok = "Qwen3-Embedding-0.6B" in name_hint or "qwen3-embedding-0.6b" in name_hint.lower()
            if not hub_ok:
                raise RuntimeError(
                    f"model path does not look like Qwen3-Embedding-0.6B: {self.model_path}"
                )

        arch = getattr(self.config, "architectures", None) or [self.model.__class__.__name__]
        self.info = ModelInfo(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_path=self.model_path,
            revision=None,
            architecture=str(arch[0] if isinstance(arch, list) else arch),
            hidden_size=hidden,
            max_position_embeddings=int(getattr(self.config, "max_position_embeddings", 0)),
            padding_side=str(self.tokenizer.padding_side),
            truncation=False,
            pooling="last_valid_token",
            torch_dtype=str(self.dtype).replace("torch.", ""),
        )
        print(
            f"[text-encoder] arch={self.info.architecture} hidden={self.info.hidden_size} "
            f"ctx={self.info.max_position_embeddings} pad={self.info.padding_side}",
            flush=True,
        )

    def count_tokens(self, text: str) -> int:
        ids = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )["input_ids"]
        return int(len(ids))

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> dict[str, Any]:
        """
        Embed a batch of page texts.

        Returns FP32 embeddings [B, 1024] on CPU, plus shapes/token counts.
        Raises on non-finite values.
        """
        if not texts:
            return {
                "embeddings": torch.zeros(0, 1024, dtype=torch.float32),
                "token_counts": [],
                "input_ids_shape": (0, 0),
                "attention_mask_shape": (0, 0),
                "last_hidden_shape": (0, 0, 1024),
            }

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Per-sample token counts from attention mask (no pad).
        token_counts = [int(x) for x in batch["attention_mask"].sum(dim=1).tolist()]

        hard = self.info.max_position_embeddings
        for i, n in enumerate(token_counts):
            if n > hard:
                raise RuntimeError(
                    f"sample batch[{i}] token_count={n} exceeds model context {hard}"
                )

        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state
        if not torch.isfinite(hidden).all():
            raise RuntimeError(
                f"non-finite last_hidden_state shape={tuple(hidden.shape)} dtype={hidden.dtype}"
            )
        if hidden.shape[-1] != 1024:
            raise RuntimeError(f"last_hidden_state last dim != 1024: {tuple(hidden.shape)}")

        pooled = last_token_pool(hidden, batch["attention_mask"])
        pooled_f32 = pooled.float()
        if not torch.isfinite(pooled_f32).all():
            raise RuntimeError(
                f"non-finite pooled embedding shape={tuple(pooled_f32.shape)} dtype={pooled_f32.dtype}"
            )
        if pooled_f32.shape != (len(texts), 1024):
            raise RuntimeError(f"pooled shape expected [{len(texts)},1024], got {tuple(pooled_f32.shape)}")

        return {
            "embeddings": pooled_f32.cpu(),
            "token_counts": token_counts,
            "input_ids_shape": tuple(batch["input_ids"].shape),
            "attention_mask_shape": tuple(batch["attention_mask"].shape),
            "last_hidden_shape": tuple(hidden.shape),
            "pooled_dtype": str(pooled_f32.dtype),
        }

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
