"""Transcribe q_*/a_* question images to standard LaTeX via Ark Seed 2.1 Pro.

Speed / isolation design
------------------------
- Thinking is **disabled** (model capability is enough; thinking adds latency).
- Outer thread pool: one worker = one question folder (no cross-question state).
- Inner parallel: Question and Answer API calls for the same question run concurrently.
- Each worker uses a thread-local OpenAI client (no shared mutable request state).
- CSV append is lock-protected and flushed continuously.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from src.utils import atomic_write_json, ensure_dir

DEFAULT_INPUT = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "tempt_data/第二批数据题目"
)
DEFAULT_OUTPUT_CSV = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "tempt_data/第二批数据题目_seed21pro_latex.csv"
)
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seed-2-1-pro-260628"

CSV_COLUMNS = [
    "题目名",
    "Question",
    "Answer",
    *(f"Student_{i}" for i in range(1, 11)),
    "Label_level_1",
    "Label_level_2",
]

_INDEX_RE = re.compile(r"^(?P<kind>[qa])_(?P<idx>\d+)\.(?P<ext>jpe?g|png|webp|bmp)$", re.I)

PROMPT_QUESTION = """你是严谨的数学 OCR / HMER 转写器。
任务：把输入的「题目」图片按给定顺序转写为**标准 LaTeX**。

硬性要求：
1. 只输出标准 LaTeX 正文，不要 Markdown 代码围栏，不要解释、不要解题、不要补全缺失内容。
2. 多张图片按输入顺序拼接；图片之间用一个空行分隔。
3. 行内公式用 $...$，独立公式或复杂结构用 $$...$$。
4. 分数、根号、上下标、矩阵、分段等使用标准 LaTeX 命令/环境（如 \\frac, \\sqrt, \\begin{cases} ... \\end{cases}）。
5. 保留可见的中文题干文字；数学符号一律用标准 LaTeX。
6. 看不清的符号用 \\texttt{[UNK]} 占位，不要猜测。
7. 不要使用 \\left / \\right。
"""

PROMPT_ANSWER = """你是严谨的数学 OCR / HMER 转写器。
任务：把输入的「答案 / 解析」图片按给定顺序转写为**标准 LaTeX**。

硬性要求：
1. 只输出标准 LaTeX 正文，不要 Markdown 代码围栏，不要解释、不要额外解题。
2. 多张图片按输入顺序拼接；图片之间用一个空行分隔。
3. 行内公式用 $...$，独立公式或复杂结构用 $$...$$。
4. 分数、根号、上下标、矩阵、分段等使用标准 LaTeX 命令/环境。
5. 保留可见的中文文字；数学符号一律用标准 LaTeX。
6. 看不清的符号用 \\texttt{[UNK]} 占位，不要猜测。
7. 不要使用 \\left / \\right。
"""

# Disable deep thinking on Ark Seed 2.x (chat.completions extra_body).
THINKING_DISABLED = {"thinking": {"type": "disabled"}}

_thread_local = threading.local()
_csv_lock = threading.Lock()


def _get_client(api_key: str, base_url: str) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    key = getattr(_thread_local, "key", None)
    url = getattr(_thread_local, "base_url", None)
    if client is None or key != api_key or url != base_url:
        # Per-thread client: connection reuse within a thread, no cross-thread mutation.
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)
        _thread_local.client = client
        _thread_local.key = api_key
        _thread_local.base_url = base_url
    return client


def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")


def _image_to_data_url(path: Path, *, max_side: int, jpeg_quality: int) -> str:
    """Encode image; optionally downscale long side to cut upload / vision latency."""
    if max_side <= 0:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{_mime(path)};base64,{b64}"

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _sort_key(path: Path) -> tuple[int, str]:
    m = _INDEX_RE.match(path.name)
    if not m:
        return (10**9, path.name)
    return (int(m.group("idx")), path.name)


def list_kind_images(folder: Path, kind: str) -> list[Path]:
    out: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = _INDEX_RE.match(p.name)
        if m and m.group("kind").lower() == kind.lower():
            out.append(p)
    return sorted(out, key=_sort_key)


def discover_question_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.fullmatch(r"```(?:latex|tex|markdown)?\s*(.*?)\s*```", t, flags=re.DOTALL | re.I)
    return m.group(1).strip() if m else t


def call_seed_latex(
    *,
    api_key: str,
    base_url: str,
    model: str,
    images: list[Path],
    prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    max_side: int,
    jpeg_quality: int,
    disable_thinking: bool,
) -> str:
    """One isolated API call (uses thread-local client of the calling thread)."""
    client = _get_client(api_key, base_url)
    content: list[dict[str, Any]] = []
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(img, max_side=max_side, jpeg_quality=jpeg_quality)
                },
            }
        )
    names = ", ".join(p.name for p in images)
    content.append(
        {
            "type": "text",
            "text": f"{prompt}\n\n当前输入图片顺序：{names}",
        }
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if disable_thinking:
        kwargs["extra_body"] = THINKING_DISABLED

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            return strip_fences(raw)
        except Exception as exc:  # noqa: BLE001 — retry transient API failures
            last_err = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"API failed after {retries} retries: {last_err}")


def _transcribe_or_placeholder(
    *,
    kind: str,
    images: list[Path],
    folder_name: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    max_side: int,
    jpeg_quality: int,
    disable_thinking: bool,
) -> tuple[str, str]:
    """Return (text, mode). kind is 'q' or 'a'.

    Missing images → empty string (no filename / folder-name placeholder).
    """
    del folder_name  # kept in signature for call-site compatibility
    if not images:
        return "", "empty_no_images"

    prompt = PROMPT_QUESTION if kind == "q" else PROMPT_ANSWER
    text = call_seed_latex(
        api_key=api_key,
        base_url=base_url,
        model=model,
        images=images,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        retry_sleep=retry_sleep,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        disable_thinking=disable_thinking,
    )
    return text, "api"


def process_one_question(
    folder: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
    max_side: int,
    jpeg_quality: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    """One question folder only. Q and A API calls run in parallel threads."""
    title = folder.name
    q_imgs = list_kind_images(folder, "q")
    a_imgs = list_kind_images(folder, "a")

    meta: dict[str, Any] = {
        "题目名": title,
        "q_images": [p.name for p in q_imgs],
        "a_images": [p.name for p in a_imgs],
        "status": "ok",
        "error": "",
        "thinking": "disabled" if disable_thinking else "enabled",
    }

    common = dict(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        retry_sleep=retry_sleep,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        disable_thinking=disable_thinking,
    )

    try:
        # Parallel Q/A: two isolated calls, no shared buffers between them.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"qa-{title[:12]}") as inner:
            fut_q = inner.submit(
                _transcribe_or_placeholder,
                kind="q",
                images=q_imgs,
                folder_name=title,
                **common,
            )
            fut_a = inner.submit(
                _transcribe_or_placeholder,
                kind="a",
                images=a_imgs,
                folder_name=title,
                **common,
            )
            question, q_mode = fut_q.result()
            answer, a_mode = fut_a.result()

        row = {
            "题目名": title,
            "Question": question,
            "Answer": answer,
            **{f"Student_{i}": "" for i in range(1, 11)},
            "Label_level_1": "",
            "Label_level_2": "",
        }
        # Rule: if Q or A present, Student_* must stay empty.
        meta.update(
            {
                "q_mode": q_mode,
                "a_mode": a_mode,
                "question_chars": len(question),
                "answer_chars": len(answer),
                "students_cleared": bool(question.strip() or answer.strip()),
            }
        )
        return {"row": row, "meta": meta}
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        meta["traceback"] = traceback.format_exc()
        # On error: keep empty if that side has no images; otherwise leave empty too
        # (caller may re-run). Do not write filename placeholders.
        row = {
            "题目名": title,
            "Question": "" if not q_imgs else "",
            "Answer": "" if not a_imgs else "",
            **{f"Student_{i}": "" for i in range(1, 11)},
            "Label_level_1": "",
            "Label_level_2": "",
        }
        meta["q_mode"] = "error_empty"
        meta["a_mode"] = "error_empty"
        return {"row": row, "meta": meta}


def load_done_titles(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    done: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "题目名" not in reader.fieldnames:
            return set()
        for row in reader:
            title = (row.get("题目名") or "").strip()
            if title:
                done.add(title)
    return done


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        f.flush()


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    with _csv_lock:
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
            f.flush()


def upsert_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Replace existing row with same 题目名, or append if absent."""
    title = str(row.get("题目名", ""))
    with _csv_lock:
        ensure_csv_header(csv_path)
        rows: list[dict[str, str]] = []
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for old in reader:
                if (old.get("题目名") or "").strip() == title:
                    continue
                rows.append({k: old.get(k, "") for k in CSV_COLUMNS})
        rows.append({k: str(row.get(k, "")) for k in CSV_COLUMNS})
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            f.flush()
        tmp.replace(csv_path)


def blank_missing_image_fields(csv_path: Path, input_root: Path) -> dict[str, int]:
    """Set Question/Answer to empty when corresponding q_*/a_* images are absent."""
    if not csv_path.is_file():
        return {"n_rows": 0, "n_q_cleared": 0, "n_a_cleared": 0}
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n_q = n_a = 0
    out_rows: list[dict[str, str]] = []
    for row in rows:
        title = (row.get("题目名") or "").strip()
        folder = input_root / title
        has_q = bool(list_kind_images(folder, "q")) if folder.is_dir() else False
        has_a = bool(list_kind_images(folder, "a")) if folder.is_dir() else False
        new_row = {k: row.get(k, "") for k in CSV_COLUMNS}
        if not has_q and (new_row.get("Question") or ""):
            new_row["Question"] = ""
            n_q += 1
        if not has_a and (new_row.get("Answer") or ""):
            new_row["Answer"] = ""
            n_a += 1
        out_rows.append(new_row)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
        f.flush()
    tmp.replace(csv_path)
    return {"n_rows": len(out_rows), "n_q_cleared": n_q, "n_a_cleared": n_a}


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    input_root = Path(args.input_root)
    out_csv = Path(args.output_csv)
    work_dir = Path(args.work_dir) if args.work_dir else out_csv.parent / f"{out_csv.stem}_run"
    ensure_dir(work_dir)
    ensure_dir(work_dir / "per_question")
    ensure_csv_header(out_csv)

    if args.fix_empty_missing:
        cleared = blank_missing_image_fields(out_csv, input_root)
        print(f"[fix-empty-missing] {cleared}", flush=True)

    folders = discover_question_dirs(input_root)
    only = {x.strip() for x in (args.only or []) if x.strip()}
    if only:
        folders = [p for p in folders if p.name in only]
        missing_only = sorted(only - {p.name for p in folders})
        if missing_only:
            raise SystemExit(f"--only folders not found under input-root: {missing_only}")
    done = load_done_titles(out_csv) if args.resume else set()
    if args.force_rerun:
        # Re-process selected (or all) folders even if already in CSV; upsert later.
        todo = list(folders)
    else:
        todo = [p for p in folders if p.name not in done]

    if not todo:
        summary = {
            "input_root": str(input_root),
            "output_csv": str(out_csv),
            "n_todo": 0,
            "note": "nothing to process (CSV fix-only or all done)",
            "elapsed_sec": time.time() - t0,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary

    api_key = args.api_key or os.environ.get("ARK_API_KEY") or os.environ.get("SEED_API_KEY")
    if not api_key:
        raise SystemExit(
            "缺少 API Key：请设置环境变量 ARK_API_KEY，或传入 --api-key。"
            "模型页：https://ark.volcengine.com/region:cn-beijing/model/detail?name=doubao-seed-2-1-pro"
        )

    workers = max(1, int(args.workers))
    # Peak concurrent API calls ≈ workers * 2 (Q∥A). Cap if user sets --max-api-inflight.
    max_api = int(args.max_api_inflight) if args.max_api_inflight > 0 else workers * 2
    effective_workers = min(workers, max(1, (max_api + 1) // 2))

    print(
        f"[init] questions={len(folders)} done={len(done)} todo={len(todo)} "
        f"workers={effective_workers} (peak_api≈{effective_workers * 2}) "
        f"thinking={'off' if args.disable_thinking else 'on'} "
        f"model={args.model} max_side={args.max_side}",
        flush=True,
    )

    ok = err = 0
    common_kw = dict(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        disable_thinking=bool(args.disable_thinking),
    )

    def _job(folder: Path) -> dict[str, Any]:
        return process_one_question(folder, **common_kw)

    with ThreadPoolExecutor(
        max_workers=effective_workers, thread_name_prefix="qfolder"
    ) as pool:
        futs = {pool.submit(_job, folder): folder for folder in todo}
        for fut in as_completed(futs):
            folder = futs[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "row": {
                        "题目名": folder.name,
                        "Question": "",
                        "Answer": "",
                        **{f"Student_{i}": "" for i in range(1, 11)},
                        "Label_level_1": "",
                        "Label_level_2": "",
                    },
                    "meta": {
                        "题目名": folder.name,
                        "status": "error",
                        "error": f"future:{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                }
            if args.force_rerun:
                upsert_csv_row(out_csv, result["row"])
            else:
                append_csv_row(out_csv, result["row"])
            meta = result["meta"]
            if meta.get("status") == "ok":
                ok += 1
            else:
                err += 1
            print(
                f"[{ok + err}/{len(todo)}] {folder.name} status={meta.get('status')} "
                f"q={meta.get('q_mode')} a={meta.get('a_mode')}",
                flush=True,
            )
            atomic_write_json(work_dir / "per_question" / f"{folder.name}.json", meta)

    summary = {
        "input_root": str(input_root),
        "output_csv": str(out_csv),
        "work_dir": str(work_dir),
        "model": args.model,
        "base_url": args.base_url,
        "disable_thinking": bool(args.disable_thinking),
        "n_questions_total": len(folders),
        "n_already_done": len(done),
        "n_todo": len(todo),
        "n_ok": ok,
        "n_error": err,
        "workers": effective_workers,
        "peak_api_approx": effective_workers * 2,
        "max_side": args.max_side,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
    }
    atomic_write_json(work_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seed 2.1 Pro: q_*/a_* -> LaTeX CSV (multi-thread, thinking off, Q∥A)"
    )
    p.add_argument("--input-root", default=DEFAULT_INPUT)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--work-dir", default="")
    p.add_argument("--api-key", default="", help="Ark API Key (or env ARK_API_KEY)")
    p.add_argument("--base-url", default=os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--model", default=os.environ.get("ARK_MODEL", DEFAULT_MODEL))
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ARK_WORKERS", "24")),
        help="Parallel question folders (default 24). Peak API ≈ workers*2",
    )
    p.add_argument(
        "--max-api-inflight",
        type=int,
        default=0,
        help="Optional cap on concurrent API calls (0 = workers*2)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=1.5)
    p.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Downscale long image side before upload (0=original). Speeds up transfer.",
    )
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument(
        "--disable-thinking",
        action="store_true",
        default=True,
        help="Disable model thinking (default True)",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Turn thinking back on",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only process these 题目名 folder names (repeatable)",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run even if 题目名 already in CSV (upsert row)",
    )
    p.add_argument(
        "--fix-empty-missing",
        action="store_true",
        help="Clear Question/Answer in CSV when q_*/a_* images are absent",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.no_resume:
        args.resume = False
    if args.enable_thinking:
        args.disable_thinking = False
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
