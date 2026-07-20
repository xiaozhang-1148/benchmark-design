"""Shared utilities: hashing, IO, atomic writes, image helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def make_image_id(absolute_path: str, content_sha256: str | None = None) -> str:
    """Stable id: prefer content hash; fall back to path hash."""
    if content_sha256:
        return content_sha256[:16]
    return hashlib.sha256(os.path.abspath(absolute_path).encode("utf-8")).hexdigest()[:16]


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def load_image_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def safe_image_meta(path: str | Path) -> dict[str, Any]:
    """Return width/height/file_size or error fields without raising."""
    p = Path(path)
    out: dict[str, Any] = {
        "width": None,
        "height": None,
        "file_size": p.stat().st_size if p.exists() else None,
        "error_message": None,
    }
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            out["width"], out["height"] = im.size
    except Exception as e:  # noqa: BLE001
        out["error_message"] = f"{type(e).__name__}: {e}"
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_images(root: str | Path, exts: Iterable[str] | None = None, recursive: bool = True):
    root = Path(root)
    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or IMAGE_EXTS)}
    if recursive:
        paths = root.rglob("*")
    else:
        paths = root.glob("*")
    for p in sorted(paths):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p
