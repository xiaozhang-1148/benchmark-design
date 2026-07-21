"""Compatibility shims for DeepSeek-OCR-2 remote code under newer transformers."""

from __future__ import annotations


def patch_transformers_for_deepseek_ocr2() -> None:
    """Patch removed TF APIs that DeepSeek remote modeling still imports."""
    try:
        import transformers.models.llama.modeling_llama as llama_mod

        if not hasattr(llama_mod, "LlamaFlashAttention2"):
            # Safe alias: remote code selects attn impl; SDPA/eager paths do not need FA2 class
            llama_mod.LlamaFlashAttention2 = getattr(llama_mod, "LlamaAttention", None)
    except Exception:
        pass

    try:
        import transformers.utils.import_utils as import_utils

        if not hasattr(import_utils, "is_torch_fx_available"):
            import_utils.is_torch_fx_available = lambda: False  # type: ignore[attr-defined]
    except Exception:
        pass
