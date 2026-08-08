"""Fill Student_1..Student_10 from raw_dataset OCR into the Seed LaTeX CSV.

Rules
-----
1. Existing CSV rows:
   - Split 题目名 ``{exam}_{question}`` → raw path ``raw_dataset/{exam}/{question}``.
   - If Question is non-empty → skip.
   - If Question is empty → take up to 10 student ``*.jpg.json`` OCR texts into
     Student_1 .. Student_10 (sorted by basename).

2. Batch01 exams under ``raw_dataset/Batch01/{exam_id}/`` (flat, no question
   subfolder): append one new CSV row per exam with empty Question/Answer and
   up to 10 student OCR texts. Skip exams whose folder name is already a 题目名.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from src.qa_latex_transcribe.run_transcribe import CSV_COLUMNS

DEFAULT_CSV = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "tempt_data/第二批数据题目_seed21pro_latex.csv"
)
DEFAULT_RAW = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "stage2_cluster_adjusted_1/raw_dataset"
)

_UUID_EXAM_RE = re.compile(
    r"^(?P<exam>\d{8}-\d{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"_(?P<qid>.+)$"
)


def split_title_to_exam_qid(title: str) -> tuple[str, str] | None:
    """Split ``exam_qid`` using UUID exam prefix (qid may contain underscores / %xx)."""
    title = title.strip()
    m = _UUID_EXAM_RE.match(title)
    if m:
        return m.group("exam"), m.group("qid")
    # fallback: last underscore
    i = title.rfind("_")
    if i <= 0 or i >= len(title) - 1:
        return None
    return title[:i], title[i + 1 :]


def resolve_raw_question_dir(raw_root: Path, exam: str, qid: str) -> Path | None:
    """Resolve question dir; try encoded qid as stored on disk."""
    direct = raw_root / exam / qid
    if direct.is_dir():
        return direct
    return None


def ocr_text_from_json(path: Path) -> str:
    """Join all block/line OCR strings in reading order (block.order, line.order)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = data.get("blocks") or []
    if not isinstance(blocks, list):
        return ""
    indexed_blocks: list[tuple[int, dict[str, Any]]] = []
    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        try:
            bo = int(block.get("order", i))
        except (TypeError, ValueError):
            bo = i
        indexed_blocks.append((bo, block))
    indexed_blocks.sort(key=lambda x: x[0])

    parts: list[str] = []
    for bo, block in indexed_blocks:
        lines = block.get("lines") or []
        if not isinstance(lines, list):
            continue
        indexed_lines: list[tuple[int, dict[str, Any]]] = []
        for j, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            try:
                lo = int(line.get("order", j))
            except (TypeError, ValueError):
                lo = j
            indexed_lines.append((lo, line))
        indexed_lines.sort(key=lambda x: x[0])
        for _, line in indexed_lines:
            ocr = line.get("ocr")
            if isinstance(ocr, str) and ocr.strip():
                parts.append(ocr)
    return "\n".join(parts).strip()


def extract_upto_10_student_ocr(folder: Path) -> list[str]:
    """Return up to 10 non-empty OCR page texts from ``*.jpg.json``, sorted by name."""
    if not folder.is_dir():
        return []
    texts: list[str] = []
    for jp in sorted(folder.glob("*.jpg.json")):
        try:
            text = ocr_text_from_json(jp)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if text:
            texts.append(text)
        if len(texts) >= 10:
            break
    return texts


def apply_students(row: dict[str, str], texts: list[str]) -> None:
    for i in range(1, 11):
        row[f"Student_{i}"] = texts[i - 1] if i - 1 < len(texts) else ""


def empty_row(title: str) -> dict[str, str]:
    return {
        "题目名": title,
        "Question": "",
        "Answer": "",
        **{f"Student_{i}": "" for i in range(1, 11)},
        "Label_level_1": "",
        "Label_level_2": "",
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            rows.append({k: (raw.get(k) or "") for k in CSV_COLUMNS})
        return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
            f.flush()
    tmp.replace(path)


def fill_existing_empty_qa(
    rows: list[dict[str, str]],
    raw_root: Path,
) -> dict[str, int]:
    """Fill students only when BOTH Question and Answer are empty.

    If either Q or A is present, clear Student_1..Student_10.
    """
    n_skip_has_qa = 0
    n_cleared_students = 0
    n_filled = 0
    n_missing_path = 0
    n_no_ocr = 0
    for row in rows:
        q = (row.get("Question") or "").strip()
        a = (row.get("Answer") or "").strip()
        if q or a:
            n_skip_has_qa += 1
            if any((row.get(f"Student_{i}") or "").strip() for i in range(1, 11)):
                apply_students(row, [])
                n_cleared_students += 1
            continue
        title = (row.get("题目名") or "").strip()
        split = split_title_to_exam_qid(title)
        if split is None:
            # Batch01-style exam-only titles: resolve under Batch01/ or exam root
            folder = (raw_root / "Batch01" / title)
            if not folder.is_dir():
                folder = raw_root / title
            if not folder.is_dir():
                n_missing_path += 1
                print(f"[miss] raw path not found for {title}", flush=True)
                continue
        else:
            exam, qid = split
            folder = resolve_raw_question_dir(raw_root, exam, qid)
            if folder is None:
                n_missing_path += 1
                print(f"[miss] raw path not found for {title} -> {exam}/{qid}", flush=True)
                continue
        texts = extract_upto_10_student_ocr(folder)
        if not texts:
            n_no_ocr += 1
        apply_students(row, texts)
        n_filled += 1
        print(f"[fill] {title} students={len(texts)} path={folder}", flush=True)
    return {
        "n_skip_has_qa": n_skip_has_qa,
        "n_cleared_students_because_qa": n_cleared_students,
        "n_filled_empty_qa": n_filled,
        "n_missing_path": n_missing_path,
        "n_no_ocr": n_no_ocr,
    }


def append_batch01(
    rows: list[dict[str, str]],
    raw_root: Path,
) -> dict[str, int]:
    batch_root = raw_root / "Batch01"
    if not batch_root.is_dir():
        raise RuntimeError(f"missing Batch01: {batch_root}")
    existing = { (r.get("题目名") or "").strip() for r in rows }
    n_added = 0
    n_skip_existing = 0
    exams = sorted([p for p in batch_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    for exam_dir in exams:
        title = exam_dir.name
        if title in existing:
            n_skip_existing += 1
            continue
        texts = extract_upto_10_student_ocr(exam_dir)
        row = empty_row(title)
        apply_students(row, texts)
        rows.append(row)
        existing.add(title)
        n_added += 1
        print(f"[batch01] +{title} students={len(texts)}", flush=True)
    return {
        "n_batch01_exams": len(exams),
        "n_batch01_added": n_added,
        "n_batch01_skip_existing": n_skip_existing,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = Path(args.csv)
    raw_root = Path(args.raw_root)
    if not csv_path.is_file():
        raise SystemExit(f"missing csv: {csv_path}")
    if not raw_root.is_dir():
        raise SystemExit(f"missing raw_root: {raw_root}")

    rows = load_csv(csv_path)
    print(f"[load] rows={len(rows)} csv={csv_path}", flush=True)

    stats1 = fill_existing_empty_qa(rows, raw_root)
    stats2 = append_batch01(rows, raw_root)
    write_csv(csv_path, rows)

    summary = {
        "csv": str(csv_path),
        "raw_root": str(raw_root),
        "n_rows_final": len(rows),
        **stats1,
        **stats2,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fill Student_1..10 OCR into LaTeX CSV")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--raw-root", default=DEFAULT_RAW)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
