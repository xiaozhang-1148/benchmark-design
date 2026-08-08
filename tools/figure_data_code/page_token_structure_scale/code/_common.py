"""Matplotlib helpers for this figure's render script."""

from __future__ import annotations

from pathlib import Path


def configure_fonts(plt) -> None:
    from matplotlib import font_manager

    candidates = (
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "SimHei",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    resolved: list[str] = []
    for candidate in candidates:
        if candidate in installed:
            resolved.append(candidate)
            continue
        match = next((name for name in installed if candidate.casefold() in name.casefold()), None)
        if match and match not in resolved:
            resolved.append(match)
    if not resolved:
        resolved.append("DejaVu Sans")
    resolved.append("sans-serif")
    plt.rcParams["font.family"] = resolved
    plt.rcParams["font.sans-serif"] = resolved
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "path"


def save_svg(fig, output_path: Path, *, tight: bool = True) -> Path:
    output_path = output_path.with_suffix(".svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"format": "svg", "facecolor": "white"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **kwargs)
    return output_path
