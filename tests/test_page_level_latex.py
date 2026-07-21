"""Tests for page-level LaTeX metrics (Chapter-6, HMER protocol)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.page_level_latex.latex_protocol import (
    detect_structures,
    length_bin_for_token_count,
    parse_expression,
    tokenize_latex,
)
from benchmark_design.page_level_latex.pipeline import run_page_level_latex_export


def test_protocol_matches_hmer_tokenization_and_bins() -> None:
    latex = r"\frac{1}{2} + x^{2}"
    parsed = parse_expression(latex)
    tokens = tokenize_latex(latex)
    assert parsed.tokens == tokens
    assert parsed.token_count == len(tokens)
    label, key = length_bin_for_token_count(parsed.token_count)
    assert key.startswith("length_")
    assert "tokens" in label
    flags = detect_structures(tokens)
    assert flags.has_frac
    assert flags.has_sup
    assert not flags.has_env


def test_sum_and_env_structure_flags() -> None:
    sum_tokens = tokenize_latex(r"\sum_{i=1}^{n} i")
    assert detect_structures(sum_tokens).has_sum
    env_tokens = tokenize_latex(r"\begin{cases} a \\ b \end{cases}")
    assert detect_structures(env_tokens).has_env


def _write_fixture_page(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    payload = {
        "image_name": "page_a.jpg",
        "blocks": [
            {
                "order": 0,
                "type": "Txtblock",
                "lines": [
                    {"order": 0, "ocr": r"\frac{1}{2}", "polygon": [[1, 1], [10, 1], [10, 5], [1, 5]]},
                    {"order": 1, "ocr": r"x^{2}+y_{1}", "polygon": [[1, 6], [20, 6], [20, 12], [1, 12]]},
                    {"order": 2, "ocr": "", "polygon": []},
                    {
                        "order": 3,
                        "ocr": r"\begin{cases} a \\ b \end{cases}",
                        "polygon": [[1, 13], [30, 13], [30, 40], [1, 40]],
                    },
                ],
            }
        ],
    }
    (input_dir / "page_a.jpg.json").write_text(json.dumps(payload), encoding="utf-8")
    Image.fromarray(np.full((80, 120), 230, dtype=np.uint8), mode="L").save(input_dir / "page_a.jpg")
    return input_dir


def test_page_level_latex_export_smoke(tmp_path: Path) -> None:
    input_dir = _write_fixture_page(tmp_path)
    output_dir = tmp_path / "page_level_latex"
    result = run_page_level_latex_export(
        input_dir,
        output_dir,
        workers=1,
        show_progress=False,
        skip_figures=False,
        strict_consistency=False,
    )
    assert (output_dir / "metrics" / "expression_latex_metrics.csv").is_file()
    assert (output_dir / "metrics" / "page_latex_metrics.csv").is_file()
    assert (output_dir / "summary" / "page_latex_protocol_audit.csv").is_file()
    assert (output_dir / "summary" / "page_latex_chapter5_consistency.csv").is_file()
    assert (output_dir / "summary" / "page_latex_max_length_distribution.csv").is_file()
    assert (output_dir / "summary" / "page_latex_similar_token_validation.csv").is_file()
    assert (output_dir / "figures" / "fig6_1_page_scale.png").is_file()
    assert (output_dir / "figures" / "fig6_5_structure_depth_joint.png").is_file()
    assert (output_dir / "figures" / "fig6_7_rare10.png").is_file()
    assert (output_dir / "plot_data" / "fig6_1_page_scale_plot_data.csv").is_file()
    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["valid_expression_count"] == 3
    assert summary["raw_expression_count"] == 4
    assert result.manifest["expression_latex_metrics"].endswith("expression_latex_metrics.csv")
