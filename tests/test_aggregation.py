"""Tests for dataset aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from heatmap_analysis.aggregation import aggregate_stack


def test_aggregate_mean_and_usage():
    rng = np.random.default_rng(1)
    stack = rng.random((20, 8, 8))
    stats = aggregate_stack(stack, active_threshold=0.3)
    assert stats["mean"].shape == (8, 8)
    assert 0 <= stats["usage_probability"].max() <= 1
    np.testing.assert_allclose(stats["mean"], np.mean(stack, axis=0))


def test_group_difference_computation():
    a = np.ones((5, 4, 4)) * 0.8
    b = np.ones((5, 4, 4)) * 0.2
    diff = np.mean(a, axis=0) - np.mean(b, axis=0)
    assert float(diff.mean()) == pytest.approx(0.6, abs=0.01)
