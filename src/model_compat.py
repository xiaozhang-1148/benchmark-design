"""Compatibility shims for DeepSeek-OCR-2 remote code under newer transformers."""

from __future__ import annotations


def patch_transformers_for_deepseek_ocr2() -> None:
    """DeepSeek remote modeling imports LlamaFlashAttention2 (removed in recent TF)."""
    try:
        import transformers.models.llama.modeling_llama as llama_mod

        if not hasattr(llama_mod, "LlamaFlashAttention2"):
            # Safe alias: remote code selects attn impl; SDPA/eager paths do not need FA2 class
            llama_mod.LlamaFlashAttention2 = getattr(llama_mod, "LlamaAttention", None)
    except Exception:
        pass
