"""Rich progress bars and thread/process-pool parallel helpers."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, TypeVar
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

T = TypeVar("T")
R = TypeVar("R")

ExecutorKind = Literal["thread", "process"]

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


def resolve_workers(workers: int | None) -> int:
    if workers is None:
        return default_worker_count()
    return max(1, workers)


def partition_workers(
    total_workers: int | None,
    task_count: int,
    *,
    min_per_task: int = 8,
) -> int:
    """Split a worker budget across concurrent top-level tasks to reduce I/O contention."""
    if task_count <= 0:
        return default_worker_count()
    if total_workers is None:
        return default_worker_count()
    if task_count == 1:
        return max(1, total_workers)
    return max(min_per_task, total_workers // task_count)


def _chunk_sequence(items: Sequence[T], chunk_size: int) -> list[tuple[T, ...]]:
    if chunk_size <= 1:
        return [(item,) for item in items]
    chunks: list[tuple[T, ...]] = []
    for index in range(0, len(items), chunk_size):
        chunks.append(tuple(items[index : index + chunk_size]))
    return chunks


def _map_chunk(func: Callable[[T], R], chunk: tuple[T, ...]) -> list[R]:
    return [func(item) for item in chunk]


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


def _parallel_map_serial(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    description: str,
    show_progress: bool,
    chunk_size: int,
) -> list[R]:
    chunks = _chunk_sequence(items, chunk_size)
    if not show_progress:
        return [result for chunk in chunks for result in _map_chunk(func, chunk)]

    if not _PROGRESS_DISPLAY_LOCK.acquire(blocking=False):
        return [result for chunk in chunks for result in _map_chunk(func, chunk)]

    try:
        with _make_progress() as progress:
            task_id = progress.add_task(description, total=len(items))
            results: list[R] = []
            for chunk in chunks:
                results.extend(_map_chunk(func, chunk))
                progress.advance(task_id, advance=len(chunk))
            return results
    finally:
        _PROGRESS_DISPLAY_LOCK.release()


def _parallel_map_pool(
    func: Callable[[T], R],
    items: Sequence[T],
    *,
    description: str,
    show_progress: bool,
    workers: int,
    chunk_size: int,
    executor: ExecutorKind,
) -> list[R]:
    chunks = _chunk_sequence(items, chunk_size)
    if not chunks:
        return []

    pool_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    results: list[R | None] = [None] * len(items)
    with pool_cls(max_workers=workers) as pool:
        future_to_span: dict = {}
        cursor = 0
        for chunk in chunks:
            future = pool.submit(_map_chunk, func, chunk)
            span = slice(cursor, cursor + len(chunk))
            future_to_span[future] = span
            cursor += len(chunk)

        def _collect(advance: Callable[[], None]) -> None:
            for future in as_completed(future_to_span):
                span = future_to_span[future]
                chunk_results = future.result()
                results[span] = chunk_results
                for _ in range(span.stop - span.start):
                    advance()

        _run_with_optional_progress(
            description,
            len(items),
            show_progress=show_progress,
            runner=_collect,
        )
    return results  # type: ignore[return-value]


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
    chunk_size: int = 1,
    executor: ExecutorKind = "thread",
) -> list[R]:
    """Map *func* over *items* with optional Rich progress and batch parallelism."""
    if not items:
        return []

    worker_count = resolve_workers(workers)
    chunk_size = max(1, chunk_size)
    had_progress = _begin_parallel_section(show_progress=show_progress)
    try:
        if worker_count <= 1:
            return _parallel_map_serial(
                func,
                items,
                description=description,
                show_progress=had_progress,
                chunk_size=chunk_size,
            )
        return _parallel_map_pool(
            func,
            items,
            description=description,
            show_progress=had_progress,
            workers=worker_count,
            chunk_size=chunk_size,
            executor=executor,
        )
    finally:
        _end_parallel_section(had_progress=had_progress)


def parallel_map_flatten(
    func: Callable[[T], Iterable[R]],
    items: Sequence[T],
    *,
    description: str,
    show_progress: bool = False,
    workers: int | None = None,
    chunk_size: int = 1,
    executor: ExecutorKind = "thread",
) -> list[R]:
    """Like :func:`parallel_map`, flattening iterable results from each item."""
    nested = parallel_map(
        func,
        items,
        description=description,
        show_progress=show_progress,
        workers=workers,
        chunk_size=chunk_size,
        executor=executor,
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
