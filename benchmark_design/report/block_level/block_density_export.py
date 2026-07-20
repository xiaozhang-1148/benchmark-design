"""Export block-level foreground density tables."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.block_level.block_foreground_density import BlockForegroundDensityRow

BLOCK_FOREGROUND_DENSITY_COLUMNS = (
    "page_id",
    "block_id",
    "block_type",
    "mask_area",
    "foreground_pixels",
    "density",
)


def write_block_foreground_density_csv(
    rows: list[BlockForegroundDensityRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(BLOCK_FOREGROUND_DENSITY_COLUMNS)]
    for row in rows:
        lines.append(
            ",".join(
                (
                    row.page_id,
                    row.block_id,
                    row.block_type,
                    str(row.annotation_pixel_count),
                    str(row.foreground_pixel_count),
                    f"{row.foreground_density:.6f}",
                )
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
