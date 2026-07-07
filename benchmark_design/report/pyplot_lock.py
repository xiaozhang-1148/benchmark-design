"""Thread-safe matplotlib pyplot access for parallel export."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

R = TypeVar("R")

_PYPLOT_LOCK = threading.Lock()
_BACKEND_CONFIGURED = False


def _ensure_agg_backend() -> None:
    global _BACKEND_CONFIGURED
    if _BACKEND_CONFIGURED:
        return
    import matplotlib

    matplotlib.use("Agg")
    _BACKEND_CONFIGURED = True


@contextmanager
def locked_pyplot():
    """Serialize pyplot figure creation across worker threads."""
    _ensure_agg_backend()
    with _PYPLOT_LOCK:
        yield


def with_locked_pyplot(func: Callable[..., R]) -> Callable[..., R]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> R:
        with locked_pyplot():
            return func(*args, **kwargs)

    return wrapper
