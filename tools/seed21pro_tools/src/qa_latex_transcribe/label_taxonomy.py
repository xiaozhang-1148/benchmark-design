"""Per-row math taxonomy labeling via Ark Seed 2.1 Pro (fully isolated threads).

Each CSV row is submitted as its own thread-pool task. Tasks share no mutable
request state (thread-local OpenAI client). Labels are taken only from the
official L1/L2 tables in the taxonomy markdown — never invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.qa_latex_transcribe.run_transcribe import CSV_COLUMNS
from src.utils import atomic_write_json, ensure_dir

DEFAULT_CSV = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "tempt_data/第二批数据题目_seed21pro_latex.csv"
)
DEFAULT_TAXONOMY = (
    "/home/baoquan/ocr-process/benchmark-design/"
    "高中数学三级标签框架_一级与二级优化版_v0.2.md"
)
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-1-pro-260628"
THINKING_DISABLED = {"thinking": {"type": "disabled"}}

_L1_ROW_RE = re.compile(
    r"^\|\s*`(?P<code>[A-Z]+)`\s*\|\s*(?P<name>[^|]+?)\s*\|"
)
_L2_ROW_RE = re.compile(
    r"^\|\s*(?P<l1name>[^|]+?)\s*\|\s*`(?P<code>[A-Z]+\.[A-Z]+)`\s*\|\s*(?P<name>[^|]+?)\s*\|"
)

_thread_local = threading.local()
_csv_lock = threading.Lock()


@dataclass(frozen=True)
class Level1:
    code: str
    name: str


@dataclass(frozen=True)
class Level2:
    code: str
    name: str
    l1_code: str
    l1_name: str


@dataclass(frozen=True)
class Taxonomy:
    level1: dict[str, Level1]
    level2: dict[str, Level2]
    l2_by_l1: dict[str, tuple[str, ...]]
    boundary_text: str

    def format_allowed(self) -> str:
        lines = ["【一级标签（只能从下列选择）】"]
        for code, item in self.level1.items():
            lines.append(f"- {code} | {item.name}")
        lines.append("")
        lines.append("【二级标签（只能从下列选择；二级必须属于所选一级）】")
        for l1_code, l2_codes in self.l2_by_l1.items():
            l1 = self.level1[l1_code]
            lines.append(f"## {l1_code} {l1.name}")
            for c in l2_codes:
                l2 = self.level2[c]
                lines.append(f"- {c} | {l2.name}")
        if self.boundary_text.strip():
            lines.append("")
            lines.append("【易混淆边界（必须遵守）】")
            lines.append(self.boundary_text.strip())
        return "\n".join(lines)


def parse_taxonomy(path: Path) -> Taxonomy:
    text = path.read_text(encoding="utf-8")
    # Slice sections to avoid matching unrelated tables.
    m1 = re.search(r"### 2\.2 一级标签表\n(?P<body>.*?)(?=\n---\n)", text, flags=re.S)
    m2 = re.search(r"## 4\. 二级标签总表\n(?P<body>.*?)(?=\n---\n)", text, flags=re.S)
    mb = re.search(r"# 6\. 二级标签边界总览\n(?P<body>.*?)(?=\n---\n)", text, flags=re.S)
    if not m1 or not m2:
        raise RuntimeError(f"cannot locate L1/L2 tables in {path}")

    level1: dict[str, Level1] = {}
    for line in m1.group("body").splitlines():
        mm = _L1_ROW_RE.match(line.strip())
        if not mm:
            continue
        code = mm.group("code").strip()
        name = mm.group("name").strip()
        if code in {"一级编码"}:
            continue
        level1[code] = Level1(code=code, name=name)

    name_to_l1 = {v.name: k for k, v in level1.items()}
    level2: dict[str, Level2] = {}
    l2_by_l1: dict[str, list[str]] = {k: [] for k in level1}
    for line in m2.group("body").splitlines():
        mm = _L2_ROW_RE.match(line.strip())
        if not mm:
            continue
        l1_name = mm.group("l1name").strip()
        code = mm.group("code").strip()
        name = mm.group("name").strip()
        if l1_name in {"一级领域"}:
            continue
        l1_code = name_to_l1.get(l1_name)
        if l1_code is None:
            # also accept if first segment of code matches
            prefix = code.split(".", 1)[0]
            if prefix in level1:
                l1_code = prefix
            else:
                raise RuntimeError(f"unknown L1 name for L2 {code}: {l1_name}")
        level2[code] = Level2(
            code=code,
            name=name,
            l1_code=l1_code,
            l1_name=level1[l1_code].name,
        )
        l2_by_l1[l1_code].append(code)

    if len(level1) != 4 or len(level2) != 27:
        raise RuntimeError(
            f"unexpected taxonomy size: L1={len(level1)} L2={len(level2)} (expect 4/27)"
        )

    boundary = ""
    if mb:
        # keep the boundary table body as plain text for the prompt
        boundary_lines = []
        for line in mb.group("body").splitlines():
            if line.strip().startswith("|") and "---" not in line and "易混淆" not in line:
                boundary_lines.append(line.strip())
        boundary = "\n".join(boundary_lines)

    return Taxonomy(
        level1=level1,
        level2=level2,
        l2_by_l1={k: tuple(v) for k, v in l2_by_l1.items()},
        boundary_text=boundary,
    )


def _get_client(api_key: str, base_url: str) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    key = getattr(_thread_local, "key", None)
    url = getattr(_thread_local, "base_url", None)
    if client is None or key != api_key or url != base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)
        _thread_local.client = client
        _thread_local.key = api_key
        _thread_local.base_url = base_url
    return client


def build_evidence(row: dict[str, str], *, max_chars: int = 12000) -> tuple[str, str]:
    """Return (mode, evidence_text). mode in {qa, students, empty}."""
    q = (row.get("Question") or "").strip()
    a = (row.get("Answer") or "").strip()
    if q or a:
        parts = []
        if q:
            parts.append(f"【题目 Question】\n{q}")
        if a:
            parts.append(f"【答案 Answer】\n{a}")
        text = "\n\n".join(parts)
        mode = "qa"
    else:
        students = []
        for i in range(1, 11):
            s = (row.get(f"Student_{i}") or "").strip()
            if s:
                students.append(f"【学生作答 Student_{i}】\n{s}")
        if not students:
            return "empty", ""
        text = "\n\n".join(students)
        mode = "students"

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…(截断)"
    return mode, text


def build_prompt(taxonomy: Taxonomy, title: str, mode: str, evidence: str) -> str:
    allowed = taxonomy.format_allowed()
    return f"""你是高中数学知识点标注专家。请根据题目/答案或学生作答，判定其知识类别。

硬性约束：
1. Label_level_1 只能是下列一级中文名之一：预备知识 / 函数 / 几何与代数 / 概率与统计。
2. Label_level_2 只能是给定二级中文名之一，且必须隶属于所选一级。
3. 禁止自创、合并、改写类别名称；只能从给定表中选择。
4. 一道题可能涉及多个知识点：若确实考查多个互不相同的二级模块，请全部输出；
   每个知识点对应一组一级+二级；按考查重要性从高到低排序。
5. 若只有一个核心知识点，则 labels 数组只含 1 项。不要为凑数而重复或硬拆。
6. 结合数学知识内容判断，不要按题型/难度/素养归类。
7. 只输出一个 JSON 对象，不要 Markdown 围栏，不要解释。

输出 JSON schema：
{{
  "labels": [
    {{
      "Label_level_1": "一级中文名（如：函数）",
      "Label_level_2": "二级中文名（如：一元函数导数及其应用）",
      "Label_level_1_code": "一级编码（如：FUN）",
      "Label_level_2_code": "二级编码（如：FUN.DER）",
      "confidence": 0.0到1.0的数,
      "reason": "不超过40字的判定依据"
    }}
  ]
}}

题目名：{title}
证据模式：{mode}

{allowed}

【待标注内容】
{evidence}
"""


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.S | re.I)
    return m.group(1).strip() if m else t


def parse_label_json(raw: str) -> dict[str, Any]:
    text = strip_fences(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("label JSON is not an object")
    return obj


def extract_label_items(obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept multi-label ``labels`` list, or legacy single-label object."""
    labels = obj.get("labels")
    if isinstance(labels, list) and labels:
        out = [x for x in labels if isinstance(x, dict)]
        if out:
            return out
    # legacy single object
    if obj.get("Label_level_1") or obj.get("Label_level_2") or obj.get("Label_level_2_code"):
        return [obj]
    raise ValueError("missing labels array / Label_level_* fields")


def normalize_labels(obj: dict[str, Any], taxonomy: Taxonomy) -> dict[str, str]:
    """Resolve model output to official Chinese names for CSV columns."""
    raw_l1 = str(obj.get("Label_level_1", "")).strip()
    raw_l2 = str(obj.get("Label_level_2", "")).strip()
    code_l1 = str(obj.get("Label_level_1_code", "")).strip()
    code_l2 = str(obj.get("Label_level_2_code", "")).strip()

    # Strip accidental "CODE|中文" forms
    if "|" in raw_l1:
        left, right = raw_l1.split("|", 1)
        raw_l1 = right.strip() or left.strip()
    if "|" in raw_l2:
        left, right = raw_l2.split("|", 1)
        raw_l2 = right.strip() or left.strip()
    if "|" in code_l1:
        code_l1 = code_l1.split("|", 1)[0].strip()
    if "|" in code_l2:
        code_l2 = code_l2.split("|", 1)[0].strip()

    name_to_l1 = {v.name: k for k, v in taxonomy.level1.items()}
    name_to_l2 = {v.name: k for k, v in taxonomy.level2.items()}

    l1_code = ""
    if code_l1.upper() in taxonomy.level1:
        l1_code = code_l1.upper()
    elif raw_l1.upper() in taxonomy.level1:
        l1_code = raw_l1.upper()
    elif raw_l1 in name_to_l1:
        l1_code = name_to_l1[raw_l1]

    l2_code = ""
    if code_l2.upper() in taxonomy.level2:
        l2_code = code_l2.upper()
    elif raw_l2.upper() in taxonomy.level2:
        l2_code = raw_l2.upper()
    elif raw_l2 in name_to_l2:
        l2_code = name_to_l2[raw_l2]

    if not l2_code:
        raise ValueError(f"illegal Label_level_2={raw_l2!r} code={code_l2!r}")
    l2_item = taxonomy.level2[l2_code]
    if not l1_code:
        l1_code = l2_item.l1_code
    if l2_item.l1_code != l1_code:
        # Prefer L2 parent when inconsistent
        l1_code = l2_item.l1_code
    if l1_code not in taxonomy.level1:
        raise ValueError(f"illegal Label_level_1={raw_l1!r} code={code_l1!r}")

    l1_item = taxonomy.level1[l1_code]
    return {
        # CSV write format: Chinese names only
        "Label_level_1": l1_item.name,
        "Label_level_2": l2_item.name,
        "l1_code": l1_item.code,
        "l2_code": l2_item.code,
        "reason": str(obj.get("reason", ""))[:80],
        "confidence": str(obj.get("confidence", "")),
    }


def normalize_label_list(obj: dict[str, Any], taxonomy: Taxonomy) -> list[dict[str, str]]:
    items = extract_label_items(obj)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        try:
            labels = normalize_labels(item, taxonomy)
        except ValueError:
            continue
        key = (labels["Label_level_1"], labels["Label_level_2"])
        if key in seen:
            continue
        seen.add(key)
        out.append(labels)
    if not out:
        raise ValueError("no valid labels after normalization")
    return out


def call_label_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> str:
    client = _get_client(api_key, base_url)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body=THINKING_DISABLED,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"API failed after {retries} retries: {last_err}")


def label_one_row(
    row: dict[str, str],
    *,
    taxonomy: Taxonomy,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    """Fully isolated: only uses the given row snapshot + thread-local client.

    Returns ``labels`` as a list of L1/L2 dicts (length >= 1 on success).
    """
    title = (row.get("题目名") or "").strip()
    mode, evidence = build_evidence(row)
    meta: dict[str, Any] = {
        "题目名": title,
        "evidence_mode": mode,
        "status": "ok",
        "error": "",
        "n_labels": 0,
    }
    if mode == "empty":
        meta["status"] = "skip_empty"
        meta["error"] = "no Question/Answer/Student evidence"
        return {"题目名": title, "labels": [], "meta": meta}

    prompt = build_prompt(taxonomy, title, mode, evidence)
    try:
        raw = call_label_api(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        obj = parse_label_json(raw)
        labels = normalize_label_list(obj, taxonomy)
        meta["n_labels"] = len(labels)
        meta["labels"] = [
            {
                "Label_level_1": x["Label_level_1"],
                "Label_level_2": x["Label_level_2"],
                "l1_code": x["l1_code"],
                "l2_code": x["l2_code"],
                "reason": x["reason"],
                "confidence": x["confidence"],
            }
            for x in labels
        ]
        return {"题目名": title, "labels": labels, "meta": meta}
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
        return {"题目名": title, "labels": [], "meta": meta}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: (r.get(k) or "") for k in CSV_COLUMNS} for r in csv.DictReader(f)]


def write_csv_locked(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
    tmp.replace(path)


def apply_multi_label_result(
    rows: list[dict[str, str]],
    result: dict[str, Any],
    csv_path: Path,
    *,
    source_row: dict[str, str],
) -> int:
    """Replace all rows with the same 题目名 by one row per label pair.

    Returns number of CSV rows written for this title.
    """
    title = result["题目名"]
    labels: list[dict[str, str]] = list(result.get("labels") or [])
    with _csv_lock:
        # Keep non-matching rows; expand this title.
        kept = [r for r in rows if (r.get("题目名") or "").strip() != title]
        if not labels:
            # keep a single empty-label placeholder row (preserve content)
            blank = {k: source_row.get(k, "") for k in CSV_COLUMNS}
            blank["题目名"] = title
            blank["Label_level_1"] = ""
            blank["Label_level_2"] = ""
            kept.append(blank)
            n_written = 1
        else:
            n_written = 0
            for lab in labels:
                new_row = {k: source_row.get(k, "") for k in CSV_COLUMNS}
                new_row["题目名"] = title
                new_row["Label_level_1"] = lab["Label_level_1"]
                new_row["Label_level_2"] = lab["Label_level_2"]
                kept.append(new_row)
                n_written += 1
        rows.clear()
        rows.extend(kept)
        write_csv_locked(csv_path, rows)
        return n_written


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    api_key = args.api_key or os.environ.get("ARK_API_KEY") or os.environ.get("SEED_API_KEY")
    if not api_key or str(api_key).startswith("LTAI"):
        raise SystemExit(
            "缺少有效 ARK_API_KEY（方舟 API Key，UUID 格式；不要用 LTAI 开头的 AccessKey）。"
            "请: export ARK_API_KEY='...'"
        )

    csv_path = Path(args.csv)
    taxonomy_path = Path(args.taxonomy)
    work_dir = Path(args.work_dir) if args.work_dir else csv_path.parent / f"{csv_path.stem}_label_run"
    ensure_dir(work_dir / "per_row")

    taxonomy = parse_taxonomy(taxonomy_path)
    rows = load_csv(csv_path)

    # One task per unique 题目名 (first occurrence as evidence source).
    only_set = {x.strip() for x in (args.only or []) if x.strip()}
    labeled_titles: set[str] = set()
    for row in rows:
        title = (row.get("题目名") or "").strip()
        if not title:
            continue
        has = (row.get("Label_level_1") or "").strip() and (row.get("Label_level_2") or "").strip()
        if has:
            labeled_titles.add(title)

    seen_titles: set[str] = set()
    todo_sources: list[dict[str, str]] = []
    for row in rows:
        title = (row.get("题目名") or "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        if only_set and title not in only_set:
            continue
        if args.resume and title in labeled_titles and not args.force_rerun:
            continue
        todo_sources.append(dict(row))

    n_todo = len(todo_sources)
    if args.max_workers and args.max_workers > 0:
        workers = min(n_todo, int(args.max_workers))
    else:
        workers = max(1, n_todo)

    print(
        f"[init] csv_rows={len(rows)} unique_titles={len(seen_titles)} "
        f"todo={n_todo} workers={workers} (1 thread/title, isolated) "
        f"model={args.model} multi_label=on thinking=off",
        flush=True,
    )
    if n_todo == 0:
        summary = {"n_todo": 0, "note": "nothing to label", "csv": str(csv_path)}
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary

    ok = err = skip = 0
    n_expanded_rows = 0
    common = dict(
        taxonomy=taxonomy,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    def _job(source: dict[str, str]) -> dict[str, Any]:
        return label_one_row(source, **common)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="label-row") as pool:
        futs = {pool.submit(_job, src): src for src in todo_sources}
        for fut in as_completed(futs):
            source = futs[fut]
            title = (source.get("题目名") or "").strip()
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "题目名": title,
                    "labels": [],
                    "meta": {
                        "题目名": title,
                        "status": "error",
                        "error": f"future:{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "n_labels": 0,
                    },
                }
            n_written = apply_multi_label_result(
                rows, result, csv_path, source_row=source
            )
            meta = result["meta"]
            st = meta.get("status")
            if st == "ok":
                ok += 1
                n_expanded_rows += n_written
            elif st == "skip_empty":
                skip += 1
            else:
                err += 1
            labs = result.get("labels") or []
            lab_preview = "; ".join(
                f"{x['Label_level_1']}/{x['Label_level_2']}" for x in labs
            ) or "-"
            print(
                f"[{ok + err + skip}/{n_todo}] {title} status={st} "
                f"n_labels={len(labs)} csv_rows+={n_written} [{lab_preview}]",
                flush=True,
            )
            atomic_write_json(work_dir / "per_row" / f"{title}.json", meta)

    summary = {
        "csv": str(csv_path),
        "taxonomy": str(taxonomy_path),
        "model": args.model,
        "n_csv_rows_final": len(rows),
        "n_todo_titles": n_todo,
        "n_ok": ok,
        "n_error": err,
        "n_skip_empty": skip,
        "n_label_rows_written": n_expanded_rows,
        "workers": workers,
        "multi_label": True,
        "isolation": "one_thread_per_title_threadlocal_client_row_snapshot",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
        "work_dir": str(work_dir),
    }
    atomic_write_json(work_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Label CSV rows with L1/L2 taxonomy via Seed 2.1 Pro (1 thread/row, multi-label)"
    )
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    p.add_argument("--work-dir", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--model", default=os.environ.get("ARK_MODEL", DEFAULT_MODEL))
    p.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("ARK_LABEL_WORKERS", "0")),
        help="Cap concurrent threads (0 = one thread per remaining title)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=1.5)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--only", action="append", default=[])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.no_resume:
        args.resume = False
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
