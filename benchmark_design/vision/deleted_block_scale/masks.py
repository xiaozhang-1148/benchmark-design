"""Mask construction for Deleted-Block Scale (re-exports unified masks)."""

from __future__ import annotations

from benchmark_design.vision.masks import AnswerDeletedMaskBundle, build_answer_deleted_masks

__all__ = ["AnswerDeletedMaskBundle", "build_answer_deleted_masks"]
