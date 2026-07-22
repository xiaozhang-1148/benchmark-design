"""L/B/D coordinate bins for expression-level structural difficulty."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark_design.ocr.structure_forest import STRUCTURE_TYPE_ORDER, compute_ast_forest_metrics

EXPRESSION_STRUCTURAL_DIFFICULTY_LABEL = "Expression-level Structural Difficulty"

L_BINS: tuple[str, ...] = ("L0", "L1", "L2")
B_BINS: tuple[str, ...] = ("B0", "B1", "B2")
D_BINS: tuple[str, ...] = ("D0", "D1", "D2")

STRUCTURAL_DIFFICULTY_TIERS: tuple[str, ...] = ("L1", "L2", "L3", "L4")

L_RANGES: dict[str, str] = {
    "L0": "<=20",
    "L1": "21-40",
    "L2": ">40",
}
B_RANGES: dict[str, str] = {
    "B0": "0-1 types",
    "B1": "2 types",
    "B2": ">=3 types",
}
D_RANGES: dict[str, str] = {
    "D0": "0-1",
    "D1": "2",
    "D2": ">=3",
}

LBD_STRUCTURE_TYPE_ORDER: tuple[str, ...] = STRUCTURE_TYPE_ORDER


@dataclass(frozen=True, slots=True)
class ExpressionLbdCoordinate:
    expression_id: str
    normalized_latex: str
    token_length: int
    structure_types: tuple[str, ...]
    structure_type_count: int
    ast_depth: int
    l_bin: str
    b_bin: str
    d_bin: str
    position_id: str
    structural_difficulty: str


@dataclass(frozen=True, slots=True)
class LbdStructuralDifficultyCount:
    structural_difficulty: str
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class LbdPositionCount:
    position_id: str
    l_bin: str
    b_bin: str
    d_bin: str
    l_range: str
    b_range: str
    d_range: str
    count: int
    ratio: float
    structural_difficulty: str


@dataclass(frozen=True, slots=True)
class LbdCoordinateMetrics:
    total_expression_count: int
    coordinates: tuple[ExpressionLbdCoordinate, ...]
    position_counts: tuple[LbdPositionCount, ...]
    structural_difficulty_counts: tuple[LbdStructuralDifficultyCount, ...]


def lbd_structure_types_present(tokens: Sequence[str]) -> tuple[str, ...]:
    """Return L/B/D structure types present in *tokens* (forest order)."""
    return compute_ast_forest_metrics(list(tokens)).present_types()


def assign_l_bin(token_length: int) -> str:
    if token_length <= 20:
        return "L0"
    if token_length <= 40:
        return "L1"
    return "L2"


def assign_b_bin(structure_type_count: int) -> str:
    if structure_type_count <= 1:
        return "B0"
    if structure_type_count == 2:
        return "B1"
    return "B2"


def assign_d_bin(ast_depth: int) -> str:
    if ast_depth <= 1:
        return "D0"
    if ast_depth == 2:
        return "D1"
    return "D2"


def lbd_bin_index(bin_label: str) -> int:
    prefix = bin_label[0]
    if prefix == "L":
        return L_BINS.index(bin_label)
    if prefix == "B":
        return B_BINS.index(bin_label)
    return D_BINS.index(bin_label)


def classify_lbd(l_bin: str, b_bin: str, d_bin: str) -> str:
    """Classify expression-level structural difficulty from L/B/D bins."""
    l_idx = int(l_bin[1])
    b_idx = int(b_bin[1])
    d_idx = int(d_bin[1])
    score = l_idx + b_idx + d_idx
    high_count = int(l_idx == 2) + int(b_idx == 2) + int(d_idx == 2)

    if l_idx == 0 and b_idx == 0 and d_idx == 0:
        return "L1"
    if high_count >= 2 and l_idx != 0 and d_idx != 0:
        return "L4"
    if score == 1:
        return "L2"
    if score == 2 and l_idx != 2 and d_idx != 2:
        return "L2"
    return "L3"


def difficulty_region(*, l_bin: str, b_bin: str, d_bin: str) -> str:
    """Backward-compatible alias for :func:`classify_lbd`."""
    return classify_lbd(l_bin, b_bin, d_bin)


def iter_all_position_ids() -> tuple[tuple[str, str, str, str], ...]:
    positions: list[tuple[str, str, str, str]] = []
    for l_bin in L_BINS:
        for b_bin in B_BINS:
            for d_bin in D_BINS:
                positions.append((f"{l_bin}{b_bin}{d_bin}", l_bin, b_bin, d_bin))
    return tuple(positions)


def assign_expression_lbd_coordinate(
    *,
    expression_id: str,
    normalized_latex: str,
    token_sequence: Sequence[str],
    token_length: int | None = None,
    ast_depth: int | None = None,
) -> ExpressionLbdCoordinate:
    tokens = list(token_sequence)
    resolved_length = token_length if token_length is not None else len(tokens)
    structure_types = lbd_structure_types_present(tokens)
    structure_type_count = len(structure_types)
    resolved_ast_depth = (
        ast_depth if ast_depth is not None else compute_ast_forest_metrics(tokens).ast_depth
    )
    l_bin = assign_l_bin(resolved_length)
    b_bin = assign_b_bin(structure_type_count)
    d_bin = assign_d_bin(resolved_ast_depth)
    structural_difficulty = classify_lbd(l_bin, b_bin, d_bin)
    return ExpressionLbdCoordinate(
        expression_id=expression_id,
        normalized_latex=normalized_latex,
        token_length=resolved_length,
        structure_types=structure_types,
        structure_type_count=structure_type_count,
        ast_depth=resolved_ast_depth,
        l_bin=l_bin,
        b_bin=b_bin,
        d_bin=d_bin,
        position_id=f"{l_bin}{b_bin}{d_bin}",
        structural_difficulty=structural_difficulty,
    )


def assign_lbd_from_feature(feature) -> ExpressionLbdCoordinate:
    return assign_expression_lbd_coordinate(
        expression_id=feature.expression_id,
        normalized_latex=feature.normalized_latex,
        token_sequence=feature.token_sequence,
        token_length=feature.token_length,
        ast_depth=feature.ast_depth,
    )


def compute_structural_difficulty_counts(
    coordinates: Sequence[ExpressionLbdCoordinate],
) -> tuple[LbdStructuralDifficultyCount, ...]:
    total = len(coordinates)
    counter = Counter(coordinate.structural_difficulty for coordinate in coordinates)
    return tuple(
        LbdStructuralDifficultyCount(
            structural_difficulty=tier,
            count=counter.get(tier, 0),
            ratio=(counter.get(tier, 0) / total) if total else 0.0,
        )
        for tier in STRUCTURAL_DIFFICULTY_TIERS
    )


def compute_lbd_coordinate_metrics(features: Sequence) -> LbdCoordinateMetrics:
    coordinates = tuple(assign_lbd_from_feature(feature) for feature in features)
    counter = Counter(coordinate.position_id for coordinate in coordinates)
    total = len(coordinates)
    position_counts: list[LbdPositionCount] = []
    for position_id, l_bin, b_bin, d_bin in iter_all_position_ids():
        count = counter.get(position_id, 0)
        structural_difficulty = classify_lbd(l_bin, b_bin, d_bin)
        position_counts.append(
            LbdPositionCount(
                position_id=position_id,
                l_bin=l_bin,
                b_bin=b_bin,
                d_bin=d_bin,
                l_range=L_RANGES[l_bin],
                b_range=B_RANGES[b_bin],
                d_range=D_RANGES[d_bin],
                count=count,
                ratio=(count / total) if total else 0.0,
                structural_difficulty=structural_difficulty,
            )
        )
    return LbdCoordinateMetrics(
        total_expression_count=total,
        coordinates=coordinates,
        position_counts=tuple(position_counts),
        structural_difficulty_counts=compute_structural_difficulty_counts(coordinates),
    )


def validate_lbd_coordinates(metrics: LbdCoordinateMetrics) -> list[str]:
    violations: list[str] = []
    total = metrics.total_expression_count
    if sum(row.count for row in metrics.position_counts) != total:
        violations.append("position counts do not sum to expression total")
    if len(metrics.position_counts) != 27:
        violations.append(f"expected 27 position rows, got {len(metrics.position_counts)}")

    for coordinate in metrics.coordinates:
        expected_position = f"{coordinate.l_bin}{coordinate.b_bin}{coordinate.d_bin}"
        if coordinate.position_id != expected_position:
            violations.append(
                f"{coordinate.expression_id}: position_id {coordinate.position_id} != {expected_position}"
            )
        if coordinate.structure_type_count == 0 and coordinate.ast_depth != 0:
            violations.append(
                f"{coordinate.expression_id}: zero structure types but ast_depth={coordinate.ast_depth}"
            )
        if coordinate.ast_depth > 0 and coordinate.structure_type_count < 1:
            violations.append(
                f"{coordinate.expression_id}: ast_depth>0 but structure_type_count=0"
            )

    return violations
