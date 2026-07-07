"""Tests for shared parallel helpers."""

from __future__ import annotations

from benchmark_design.progress import run_parallel_tasks


def test_run_parallel_tasks_preserves_names() -> None:
    results = run_parallel_tasks(
        {
            "a": lambda: 1,
            "b": lambda: 2,
            "c": lambda: 3,
        },
        show_progress=False,
        workers=3,
    )
    assert results == {"a": 1, "b": 2, "c": 3}
