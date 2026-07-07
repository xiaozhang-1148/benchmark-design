# benchmark-design

Benchmark dataset analysis for **HMER** (handwritten math expression recognition) and **vision** (image-side) metrics.

## Project layout

```
benchmark_design/
  config/           # shared + domain config (hmer, vision)
  hmer/             # HMER domain facade (implementation in ocr/)
  ocr/              # LaTeX tokenization, structure metrics, cross-benchmark
  vision/           # image sample loading and visual metrics (in progress)
  io/               # shared dataset loaders
  report/           # export writers
    vision/         # vision output layout + export pipeline
  progress.py       # shared Rich / parallel helpers
```

| Domain | CLI | Output root | Focus |
|--------|-----|-------------|--------|
| **All** | `export` | `benchmark_export/` | HMER + Vision + dataset overview |
| HMER | `ocr …` or `hmer …` | `HMER/` | LaTeX tokens, structure, AST depth |
| Vision | `vision export` | `vision/` | flow structure, image index |

`ocr` remains the stable command namespace; `hmer` forwards to the same subcommands.

## Unified export (HMER + Vision + overview)

One command runs the full HMER export, full Vision export, and writes the dataset overview (总纲) at the top level:

```bash
python -m benchmark_design export \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./benchmark_export
```

Output layout:

```
benchmark_export/
  dataset_overview.md           # 总纲
  HMER/                         # 完整 HMER 产物（summary.md, tables/, …）
  vision/                       # 完整 Vision 产物
```

Options mirror the individual pipelines (`--skip-figures`, `--skip-cross-benchmark`, `--skip-dimensions`, vision figure skips, etc.). Use `--hmer-output` / `--vision-output` to override subdirectory names.

The unified command loads HMER and Vision inputs once (in parallel), then runs HMER export, Vision export, and dataset overview concurrently on the shared in-memory data. Within each export pipeline, independent table/figure writers also run in parallel after metrics are computed. Cross-benchmark still loads external datasets (CROHME, HME100K, MathWriting, …) when enabled — use `--skip-cross-benchmark` to skip that phase.

## Full HMER benchmark export

```bash
python -m benchmark_design ocr export \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./HMER
```

Writes a layered output tree:

```
HMER/
  summary.md              # overview report
  ocr_benchmark_summary.md
  cross_benchmark_summary.md
  metadata.json           # provenance, parse rates, manifest
  tables/                 # core summary CSVs (+ README.md)
  tables/appendix/        # detailed distributions and long listings
  cross_benchmark/        # cross-dataset comparison CSVs
  resources/              # latex_vocab.csv, token_taxonomy_map.csv
  details/                # expression-level and derived CSVs
  examples/               # audit sample CSVs
  figures/                # PNG charts
  docs/                   # tokenizer rules and taxonomy examples
```

Options:

```bash
python -m benchmark_design ocr export --skip-figures --skip-cross-benchmark
python -m benchmark_design hmer export --output ./benchmark_output
python -m benchmark_design ocr export --workers 16 --datasets CROHME2014,HME100K
```

## Vision benchmark export

```bash
python -m benchmark_design vision export \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./vision
```

Writes:

```
vision/
  vision_benchmark_summary.md
  flow_structure_summary.md
  metadata.json
  tables/sample_index.csv
  tables/flow_structure_page_metrics.csv   # flow_group + diagnostic fields
  tables/flow_group_summary.csv          # five-class primary summary
  tables/flow_structure_block_geometry.csv
  tables/foreground_pixel_density_page_metrics.csv
  tables/foreground_pixel_density_block_metrics.csv
  tables/foreground_pixel_density_region_metrics.csv
  tables/foreground_pixel_density_overall.csv
  tables/foreground_pixel_density_by_flow_structure.csv
  tables/deleted_block_scale_page_metrics.csv
  tables/deleted_block_scale_block_geometry.csv
  metadata/foreground_pixel_density_diagnostics.json
  details/flow_structure_decisions.jsonl
  foreground_pixel_density_summary.md
  deleted_block_scale_summary.md
  figures/flow_group_examples/
  figures/d_page_density_bands.png
  figures/d_block_density_bands.png
  figures/density_level_comparison.png
  figures/foreground_pixel_density/high_density_comparisons/
  figures/deleted_block_scale/r_del_histogram.png
  figures/deleted_block_scale/deleted_instance_histogram.png
  figures/deleted_block_scale/high_r_del_examples/
```

Use `--skip-flow-figures` / `--skip-foreground-load-figures` / `--skip-deleted-block-scale-figures` to skip PNG generation.
Use `--skip-dimensions` when Pillow is unavailable (flow structure only; foreground load requires images).

### Answer-Block Flow Structure only

```bash
python -m benchmark_design vision flow-structure \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./vision
```

Also writes `tables/flow_group_summary.csv`, `details/flow_structure_decisions.jsonl`,
`flow_structure_summary.md`, and optionally `figures/flow_group_examples/` mask overlays.

Classifies each page into `Single-flow`, `Columnar-flow`, `Hybrid-flow`, or `NA` using
`Txtblock` mask geometry from the benchmark JSON export (`Txtblock`, `figure`,
`deleted_text_block`, `chart` block types). Every `Hybrid-flow` row includes a non-empty
`hybrid_reason`.

Add `--skip-flow-figures` to skip overlay PNG generation.
Add `--skip-dimensions` when Pillow is unavailable (for `vision export` only).

### Effective-region foreground pixel density only

```bash
python -m benchmark_design vision foreground-load \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./vision
```

Computes **foreground pixel density** over annotated regions using:

1. grayscale `G ∈ [0,255]`
2. robust percentile normalization `G_tilde = clip((G-q_low)/(q_high-q_low), 0, 1)` with defaults `P1/P99`
3. darkness map `S = 1 - G_tilde`
4. dataset-level fixed threshold `tau_D` from pooled `S` within `R_eff` (bimodal valley → GMM intersection → pooled Otsu)
5. foreground `{p | S(p) >= tau_D}`; density = `|F| / |mask|`

Page-level density uses `R_eff = txtBlock ∪ deleted_text_block ∪ chart ∪ figure`; block-level density uses each txtBlock polygon with the same `tau_D`. Supplementary metric: **mean darkness** (`mean(S)`). **Raw Otsu Density** is exported as a non-primary baseline.

Outputs include `tables/foreground_pixel_density_region_metrics.csv` (unified page/block rows) and, for samples with density > 15%, comparison sheets under `figures/foreground_pixel_density/high_density_comparisons/`. Quality-check figures (calibration histogram, sample binarizations) are written to `figures/foreground_pixel_density/quality_checks/`.

The benchmark report treats foreground density as a **continuous** visual attribute (mean, median, IQR, P90, band histograms).
Absolute low/medium/high bins and corpus tertiles are retained only in
`metadata/foreground_pixel_density_diagnostics.json` for internal QA.

Diagnostic tags include `mask_out_of_bounds`, `saturated_low`, `saturated_high`, and
`extreme_foreground_pixel_density_candidate` (D ≥ 0.18).

Add `--skip-figures` to skip CDF / histogram / example PNG generation.

### Deleted-Block Scale only

```bash
python -m benchmark_design vision deleted-block-scale \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./vision
```

Measures the scale of visually present but task-excluded `deleted_text_block` regions relative to
answer-related page area. Valid regions include `Txtblock`, `figure`, and `chart`; only
`deleted_text_block` contributes to the deleted numerator.

For each page:

- `A_valid` = union of Txtblock, chart, and figure polygon masks
- `A_deleted` = union of deleted_text_block polygon masks
- `A_ans` = union(`A_valid`, `A_deleted`) — avoids double-counting cross-class overlap
- `R_del = |A_deleted| / |A_ans|`

Dataset-level deleted area ratio uses summed mask areas: `Σ|A_deleted| / Σ|A_ans|`.
Tail cutoffs (0.2 / 0.3 / 0.5) appear as fixed checkpoint counts in the summary table.

Add `--skip-figures` to skip histogram / instance-count / high-burden example PNG generation.

## Cross-benchmark comparison

```bash
python -m benchmark_design ocr cross-benchmark \
  --output ./HMER \
  --datasets CROHME2014,HME100K,MathWriting
```

Outputs `cross_benchmark/*.csv` (profiles, summary, length bins, tokenizer coverage, provenance) using the unified `LATEX_DICT` tokenizer. The full comparison narrative is written to `cross_benchmark_summary.md` at the output root.

## Consolidated OCR report (Tables 1–7)

```bash
python -m benchmark_design ocr report \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./HMER
```

Outputs under `HMER/`:
- `tables/` — core per-metric CSV summaries (see `tables/README.md` for core vs appendix)
- `ocr_benchmark_summary.md` — all seven tables in one Markdown document (output root)
- `cross_benchmark_summary.md` — cross-dataset comparison narrative (output root)
- `docs/metadata/ocr_benchmark_metadata.json` — generation metadata

All OCR commands support Rich progress bars (enabled by default) and parallel loading/tokenization:

```bash
python -m benchmark_design ocr report --workers 16 --output ./benchmark_output
python -m benchmark_design ocr report --no-progress --output ./benchmark_output
```

## OCR data scale

```bash
python -m benchmark_design ocr scale \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./HMER
```

## OCR expression length distribution

```bash
python -m benchmark_design ocr length --output ./benchmark_output
```

## OCR fixed length bins

```bash
python -m benchmark_design ocr bins --output ./benchmark_output
```

## OCR token taxonomy composition

```bash
python -m benchmark_design ocr taxonomy --output ./benchmark_output
```

## OCR token long-tail

```bash
python -m benchmark_design ocr longtail --output ./benchmark_output
```

## OCR structure type distribution

```bash
python -m benchmark_design ocr structure --output ./benchmark_output
```

Matrix triggers include `\begin`, `\end`, `\\`, and matrix environment names
(`cases`, `pmatrix`, `bmatrix`, ...). Matrix **Max Depth** counts nested
`\begin ... \end` blocks only: each matched pair is one layer.

## OCR structure combination complexity

```bash
python -m benchmark_design ocr structure-complexity --output ./benchmark_output
```

Counts how many distinct table-6 structure types co-occur within one expression.

## OCR LaTeX AST depth (PosFormer)

```bash
python -m benchmark_design ocr ast --output ./benchmark_output
```

PosFormer position-forest encoding depth statistics (summary section 7; `tables/ast_depth_summary.csv`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Integration tests (full benchmark / CROHME2014) run only when datasets are mounted:

```bash
pytest -m integration
```
