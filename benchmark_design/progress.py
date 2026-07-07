"""Rich progress bars and thread-pool parallel helpers."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

T = TypeVar("T")
R = TypeVar("R")

_PROGRESS_DISPLAY_LOCK = threading.Lock()
_NESTED_PARALLEL_TASKS = 0
_NESTED_PARALLEL_LOCK = threading.Lock()


def _begin_parallel_section(*, show_progress: bool) -> bool:
    global _NESTED_PARALLEL_TASKS
    with _NESTED_PARALLEL_LOCK:
        # Allow one nested progress bar (e.g. vision metrics inside unified export).
        effective_progress = show_progress and _NESTED_PARALLEL_TASKS <= 1
        _NESTED_PARALLEL_TASKS += 1
    return effective_progress


def _end_parallel_section(*, had_progress: bool) -> None:
    global _NESTED_PARALLEL_TASKS
    with _NESTED_PARALLEL_LOCK:
        _NESTED_PARALLEL_TASKS -= 1
        assert _NESTED_PARALLEL_TASKS >= 0


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(32, cpu_count))


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


def _run_with_optional_progress(
    description: str,
    total: int,
    *,
    show_progress: bool,
    runner: Callable[[Callable[[], None],], None],
) -> None:
    """Run *runner(advance)* with optional Rich progress under a process-wide display lock."""
    if not show_progress:
        runner(lambda: None)
        return
    # Never block on the display lock: an outer export bar (unified 0/3) may already hold it.
    if not _PROGRESS_DISPLAY_LOCK.acquire(blocking=False):
        runner(lambda: None)
        return
    try:
        with _make_progress() as progress:
            task_id = progress.add_task(description, total=total)
            runner(lambda: progress.advance(task_id))
    finally:
        _PROGRESS_DISPLAY_LOCK.release()


def run_parallel_tasks(
    tasks: dict[str, Callable[[], R]],
    *,
    description: str = "Running export tasks",
    show_progress: bool = False,
    workers: int | None = None,
) -> dict[str, R]:
    """Run independent callables in parallel, preserving task names in the result."""
    if not tasks:
        return {}

    if len(tasks) == 1:
        name, func = next(iter(tasks.items()))
        return {name: func()}

    had_progress = _begin_parallel_section(show_progress=show_progress)
    worker_count = workers if workers is not None else min(len(tasks), default_worker_count())
    if worker_count <= 1:
        try:
            return {name: func() for name, func in tasks.items()}
        finally:
            _end_parallel_section(had_progress=had_progress)

    results: dict[str, R] = {}
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_name = {executor.submit(func): name for name, func in tasks.items()}

            def _collect(advance: Callable[[], None]) -> None:
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    results[name] = future.result()
                    advance()

            _run_with_optional_progress(
                description,
                len(future_to_name),
                show_progress=had_progress,
                runner=_collect,
            )
        return results
    finally:
        _end_parallel_section(had_progress=had_progress)


def parallel_map(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    description: str,
    show_progress: bool = False,
    workers: int | None = None,
) -> list[R]:
    """Map *func* over *items* with optional Rich progress and thread parallelism."""
    if not items:
        return []

    worker_count = workers if workers is not None else default_worker_count()
    had_progress = _begin_parallel_section(show_progress=show_progress)
    try:
        if worker_count <= 1:
            if had_progress and _PROGRESS_DISPLAY_LOCK.acquire(blocking=False):
                try:
                    with _make_progress() as progress:
                        task_id = progress.add_task(description, total=len(items))
                        results: list[R] = []
                        for item in items:
                            results.append(func(item))
                            progress.advance(task_id)
                        return results
                finally:
                    _PROGRESS_DISPLAY_LOCK.release()
            return [func(item) for item in items]

        results: list[R | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {executor.submit(func, item): index for index, item in enumerate(items)}

            def _collect(advance: Callable[[], None]) -> None:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    advance()

            _run_with_optional_progress(
                description,
                len(items),
                show_progress=had_progress,
                runner=_collect,
            )
        return results  # type: ignore[return-value]
    finally:
        _end_parallel_section(had_progress=had_progress)


def parallel_map_flatten(
    func: Callable[[T], Iterable[R]],
    items: Sequence[T],
    *,
    description: str,
    show_progress: bool = False,
    workers: int | None = None,
) -> list[R]:
    """Like :func:`parallel_map`, flattening iterable results from each item."""
    nested = parallel_map(
        func,
        items,
        description=description,
        show_progress=show_progress,
        workers=workers,
    )
    flattened: list[R] = []
    for batch in nested:
        flattened.extend(batch)
    return flattened


def track_sequence(
    items: Sequence[T],
    *,
    description: str,
    show_progress: bool = False,
):
    """Yield items from *items* with an optional Rich progress bar."""
    if not items:
        return
    if not show_progress:
        yield from items
        return
    with _PROGRESS_DISPLAY_LOCK:
        with _make_progress() as progress:
            task_id = progress.add_task(description, total=len(items))
            for item in items:
                yield item
                progress.advance(task_id)
