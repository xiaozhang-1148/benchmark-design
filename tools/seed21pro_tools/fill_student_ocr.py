#!/usr/bin/env python3
"""Seed 2.1 Pro tool entry. Run from repo root or seed21pro_tools/."""

from __future__ import annotations

import sys
from pathlib import Path as _P

_REPO_ROOT = _P(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.qa_latex_transcribe.fill_student_ocr import main

if __name__ == "__main__":
    raise SystemExit(main())
