"""Parse polygon point lists from benchmark JSON annotations."""

from __future__ import annotations


def parse_polygon_points(raw_points: object) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    if not isinstance(raw_points, list):
        return ()
    for point in raw_points:
        if isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return tuple(points)
