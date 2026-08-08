"""DeepSeek-OCR2 image encoder feature extraction (Projector-before, 1792-d).

Pipeline:
  page image
    → official global(1024 pad) + local(768 dynamic crops / small fallback)
    → SAM visual tokenizer
    → Qwen2 causal-flow visual encoder
    → Projector-before 896-d query tokens
    → global mean + local mean → concat 1792-d FP32
"""

from .encoder import CausalFlowVisionEncoder
from .extract import extract_page_features, pool_global_local
from .preprocess import prepare_page_views

__all__ = [
    "CausalFlowVisionEncoder",
    "extract_page_features",
    "pool_global_local",
    "prepare_page_views",
]
