"""Tests for structure co-occurrence heatmap rendering."""

from __future__ import annotations

import numpy as np

from benchmark_design.report.export_figures import _pairwise_lower_triangle_matrix


def test_pairwise_lower_triangle_matrix_masks_diagonal_and_upper_triangle() -> None:
    matrix = np.arange(16, dtype=int).reshape(4, 4)
    masked = _pairwise_lower_triangle_matrix(matrix)

    assert masked.mask[0, 0]
    assert masked.mask[1, 1]
    assert masked.mask[0, 1]
    assert masked.mask[1, 3]
    assert not masked.mask[2, 0]
    assert not masked.mask[3, 2]
    assert masked[2, 0] == 8
    assert masked[3, 1] == 13
