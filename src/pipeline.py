"""End-to-end pipeline: pure visual projected-token embeddings (no OCR)."""

from __future__ import annotations

from .visual_exp.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
