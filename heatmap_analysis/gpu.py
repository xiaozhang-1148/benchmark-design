"""GPU backend abstraction (CuPy) with CPU fallback."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

logger = logging.getLogger("heatmap_analysis.gpu")

ArrayModule: TypeAlias = Any

_cp = None
_gpu_available: bool | None = None


def _import_cupy() -> Any | None:
    global _cp
    if _cp is not None:
        return _cp
    try:
        import cupy as cp  # noqa: WPS433

        _cp = cp
        return cp
    except ImportError:
        return None


def is_gpu_available() -> bool:
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    cp = _import_cupy()
    if cp is None:
        _gpu_available = False
        return False
    try:
        cp.cuda.Device(0).compute_capability
        _gpu_available = True
    except Exception:
        _gpu_available = False
    return _gpu_available


def detect_device_ids(requested: list[int] | None) -> list[int]:
    if requested:
        return list(requested)
    if not is_gpu_available():
        return []
    cp = _import_cupy()
    assert cp is not None
    n = cp.cuda.runtime.getDeviceCount()
    return list(range(n))


@dataclass
class GpuContext:
    enabled: bool
    device_id: int | None = None
    xp: ArrayModule = np  # numpy or cupy

    @property
    def on_gpu(self) -> bool:
        return self.enabled and self.xp is not np


def get_xp(use_gpu: bool) -> ArrayModule:
    if use_gpu and is_gpu_available():
        return _import_cupy()
    return np


@contextmanager
def gpu_device(device_id: int | None, enabled: bool = True):
    """Bind CuPy to a specific GPU for the current thread/process."""
    if not enabled or not is_gpu_available():
        yield GpuContext(enabled=False, device_id=None, xp=np)
        return
    cp = _import_cupy()
    assert cp is not None
    dev = device_id if device_id is not None else 0
    with cp.cuda.Device(dev):
        cp.cuda.Device(dev).use()
        yield GpuContext(enabled=True, device_id=dev, xp=cp)


def to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    cp = _import_cupy()
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def init_worker_gpu(device_id: int, enabled: bool = True) -> GpuContext:
    """Initialize GPU in a worker process (call once at worker start)."""
    if enabled and is_gpu_available():
        cp = _import_cupy()
        assert cp is not None
        cp.cuda.Device(device_id).use()
        logger.debug("Worker bound to GPU %d", device_id)
        return GpuContext(enabled=True, device_id=device_id, xp=cp)
    return GpuContext(enabled=False, device_id=None, xp=np)
