"""Discover question groups and image↔JSON pairs under Batch02."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairedSample:
    group_id: str
    sample_id: str  # relative image path under data root
    image_path: str
    json_path: str
    image_basename: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_question_groups(data_root: str | Path) -> list[str]:
    """
    Return sorted group_id strings.

    Actual Batch02 layout (verified):
      Batch02 / {exam_id} / {question_id} / {files}

    So question directories have relative depth == 2 (not 3).
    The task brief's ``A/B/C`` depth-3 assumption does not match this tree;
    we use depth-2 leaf question dirs that contain the image files.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root not found: {root}")
    groups: list[str] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        rel = d.relative_to(root)
        if len(rel.parts) == 2:
            groups.append(str(rel).replace("\\", "/"))
    return sorted(groups)


def iter_group_pairs(data_root: str | Path, group_id: str) -> Iterator[PairedSample]:
    """Strict pairing: ``foo.jpg`` ↔ ``foo.jpg.json`` inside the group directory."""
    root = Path(data_root)
    gdir = root / Path(group_id)
    if not gdir.is_dir():
        raise FileNotFoundError(f"group dir missing: {gdir}")

    files = [p for p in gdir.iterdir() if p.is_file()]
    # Also recurse one level if nested (report elsewhere); assign to this group only
    # Spec: may recurse but still belong to this depth-2 group.
    nested = [p for p in gdir.rglob("*") if p.is_file() and p.parent != gdir]
    all_files = files + nested

    json_names = {p.name for p in all_files if p.name.endswith(".json")}
    images = sorted(
        [p for p in all_files if p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: str(p.relative_to(root)),
    )

    for img in images:
        jname = img.name + ".json"
        if jname not in json_names:
            raise RuntimeError(
                f"unpaired image (missing {jname}): {img} in group {group_id}"
            )
        # Prefer same-directory JSON
        jpath = img.with_name(jname)
        if not jpath.is_file():
            # search among all_files
            matches = [p for p in all_files if p.name == jname]
            if len(matches) != 1:
                raise RuntimeError(f"ambiguous/missing json for {img}: {matches}")
            jpath = matches[0]
        sample_id = str(img.relative_to(root)).replace("\\", "/")
        yield PairedSample(
            group_id=group_id,
            sample_id=sample_id,
            image_path=str(img.resolve()),
            json_path=str(jpath.resolve()),
            image_basename=img.name,
        )


def discover_all_pairs(data_root: str | Path) -> list[PairedSample]:
    pairs: list[PairedSample] = []
    seen_ids: set[str] = set()
    seen_basenames: set[str] = set()
    for gid in discover_question_groups(data_root):
        for s in iter_group_pairs(data_root, gid):
            if s.sample_id in seen_ids:
                raise RuntimeError(f"duplicate sample_id: {s.sample_id}")
            if s.image_basename in seen_basenames:
                raise RuntimeError(f"duplicate image basename across groups: {s.image_basename}")
            seen_ids.add(s.sample_id)
            seen_basenames.add(s.image_basename)
            pairs.append(s)
    pairs.sort(key=lambda s: (s.group_id, s.sample_id))
    return pairs
