"""Initial 8-candidate selection per question (random / hard / diversity)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from benchmark_design.ocr.structure_forest import compute_ast_forest_metrics
from benchmark_design.ocr.tokenizer import tokenize_greedy
from src.equal_fusion_cluster.discover import discover_question_groups
from src.utils import atomic_write_json, ensure_dir

DEFAULT_DATA = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/tempt_data/Batch02"
DEFAULT_CLUSTER = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/"
    "equal_fusion_spherical_kmeans"
)
DEFAULT_RAW = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/raw_dataset"
DEFAULT_SELECT = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set/select_dataset"
DEFAULT_AUDIT_DIR = "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/data_set"

MIN_IMAGES = 40
N_RANDOM = 4
N_HARD = 2
N_DIVERSITY = 2
RANDOM_SEED = 42
FG_THRESHOLD = 155
SPECIAL_TOKENS = (r"\sqrt", r"\frac", "^", "_", r"\begin", r"\end")


@dataclass
class SampleMetrics:
    sample_id: str
    image_path: str
    json_path: str
    image_basename: str
    fg_pixels: int
    token_count: int
    ast_node_count: int
    special_counts: dict[str, int]
    page_text_chars: int


@dataclass
class GroupSelection:
    group_id: str
    n_images: int
    status: str
    selected: dict[str, list[str]] = field(default_factory=dict)
    selected_paths: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    remaining: list[str] = field(default_factory=list)
    hard_metrics: list[dict[str, Any]] = field(default_factory=list)
    diversity: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_group_dirname(group_id: str) -> str:
    return group_id.replace("/", "__").replace("\\", "__")


def list_paired_images(group_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(group_dir.glob("*.jpg")):
        j = Path(str(img) + ".json")
        if j.is_file():
            pairs.append((img, j))
    return pairs


def foreground_pixel_count(image_path: Path, threshold: int = FG_THRESHOLD) -> int:
    """Ink = grayscale < threshold (dark strokes on light paper)."""
    with Image.open(image_path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    return int(np.count_nonzero(gray < threshold))


def count_special_tokens(tokens: list[str]) -> dict[str, int]:
    counts = {t: 0 for t in SPECIAL_TOKENS}
    for tok in tokens:
        if tok in counts:
            counts[tok] += 1
        # \begin{env} / \end{env} may appear as full tokens in some paths
        elif tok.startswith(r"\begin"):
            counts[r"\begin"] += 1
        elif tok.startswith(r"\end"):
            counts[r"\end"] += 1
    return counts


def assemble_ocr_page_text(data: dict[str, Any]) -> str:
    """Join line.ocr by block.order then line.order with ``\\n`` (OCR unmodified)."""
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("missing blocks")
    indexed_blocks = []
    for bi, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"bad block {bi}")
        indexed_blocks.append((int(block["order"]), bi, block))
    indexed_blocks.sort(key=lambda t: (t[0], t[1]))
    lines_out: list[str] = []
    for _, _, block in indexed_blocks:
        lines = block.get("lines")
        if not isinstance(lines, list):
            raise ValueError("missing lines")
        indexed_lines = []
        for li, line in enumerate(lines):
            if not isinstance(line, dict):
                raise ValueError(f"bad line {li}")
            indexed_lines.append((int(line["order"]), li, line))
        indexed_lines.sort(key=lambda t: (t[0], t[1]))
        for _, _, line in indexed_lines:
            ocr = line.get("ocr")
            if not isinstance(ocr, str):
                raise ValueError(f"ocr must be str, got {type(ocr).__name__}")
            lines_out.append(ocr)
    return "\n".join(lines_out)


def compute_sample_metrics(img: Path, jpath: Path, group_id: str, data_root: Path) -> SampleMetrics:
    rel = img.relative_to(data_root).as_posix()
    raw = json.loads(jpath.read_text(encoding="utf-8"))
    page_text = assemble_ocr_page_text(raw)
    tokens = tokenize_greedy(page_text)
    ast = compute_ast_forest_metrics(tokens)
    return SampleMetrics(
        sample_id=rel,
        image_path=str(img),
        json_path=str(jpath),
        image_basename=img.name,
        fg_pixels=foreground_pixel_count(img),
        token_count=len(tokens),
        ast_node_count=int(ast.ast_node_count),
        special_counts=count_special_tokens(tokens),
        page_text_chars=len(page_text),
    )


def select_hard_samples(metrics: list[SampleMetrics], n: int = N_HARD) -> list[SampleMetrics]:
    """
    1) sample with most foreground pixels (ties: tokens, then AST)
    2) among remaining, sample with most tokens (ties: AST, then FG)
    If n>2, continue alternating FG-priority / token-priority among remaining.
    """
    pool = list(metrics)
    chosen: list[SampleMetrics] = []
    for i in range(min(n, len(pool))):
        if i % 2 == 0:
            pool.sort(key=lambda m: (m.fg_pixels, m.token_count, m.ast_node_count), reverse=True)
        else:
            pool.sort(key=lambda m: (m.token_count, m.ast_node_count, m.fg_pixels), reverse=True)
        pick = pool.pop(0)
        chosen.append(pick)
    return chosen


def load_cluster_fused(cluster_root: Path, group_id: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gdir = cluster_root / "groups" / safe_group_dirname(group_id)
    if not gdir.is_dir():
        raise FileNotFoundError(f"missing cluster group dir: {gdir}")
    fused = np.load(gdir / "fused_features.npy")
    rows: list[dict[str, Any]] = []
    with (gdir / "clusters.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != fused.shape[0]:
        raise RuntimeError(f"{group_id}: fused N={fused.shape[0]} vs clusters.jsonl={len(rows)}")
    return fused.astype(np.float32, copy=False), rows


def select_diversity_pair(
    fused: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    exclude_basenames: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Among non-excluded rows, pick the pair with minimum cosine similarity."""
    # map basename -> index
    idxs = []
    for i, r in enumerate(rows):
        bn = Path(r["image_path"]).name
        if bn not in exclude_basenames:
            idxs.append(i)
    if len(idxs) < 2:
        raise RuntimeError(f"need >=2 candidates for diversity, got {len(idxs)}")

    z = fused[np.asarray(idxs, dtype=np.int64)]
    # ensure unit (should already be)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-12)
    sims = z @ z.T
    n = sims.shape[0]
    # ignore diagonal
    sims = sims.copy()
    np.fill_diagonal(sims, np.inf)
    flat = int(np.argmin(sims))
    a, b = divmod(flat, n)
    cos = float(sims[a, b])
    ia, ib = idxs[a], idxs[b]
    picked = [rows[ia], rows[ib]]
    meta = {
        "cosine_similarity": cos,
        "sample_ids": [picked[0]["sample_id"], picked[1]["sample_id"]],
        "image_basenames": [Path(picked[0]["image_path"]).name, Path(picked[1]["image_path"]).name],
        "method": "min_pairwise_cosine_on_fused_features",
        "n_candidates": len(idxs),
    }
    return picked, meta


def select_random(
    pool: list[SampleMetrics],
    n: int,
    *,
    seed: int = RANDOM_SEED,
) -> list[SampleMetrics]:
    if n <= 0 or not pool:
        return []
    rng = np.random.default_rng(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    take = order[: min(n, len(pool))]
    return [pool[i] for i in take]


def link_or_copy(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_pair(img: Path, jpath: Path, dest_dir: Path) -> dict[str, str]:
    ensure_dir(dest_dir)
    mode_i = link_or_copy(img, dest_dir / img.name)
    mode_j = link_or_copy(jpath, dest_dir / jpath.name)
    return {
        "image": str(dest_dir / img.name),
        "json": str(dest_dir / jpath.name),
        "image_mode": mode_i,
        "json_mode": mode_j,
    }


def process_group(
    group_id: str,
    *,
    data_root: Path,
    cluster_root: Path,
    raw_root: Path,
    select_root: Path,
    min_images: int = MIN_IMAGES,
) -> GroupSelection:
    gdir = data_root / group_id
    pairs = list_paired_images(gdir)
    n = len(pairs)
    sel = GroupSelection(group_id=group_id, n_images=n, status="pending")

    if n < min_images:
        sel.status = "skipped_too_few_images"
        sel.notes.append(f"n_images={n} < {min_images}")
        return sel

    # metrics for all samples (needed for hard + audit)
    metrics = [compute_sample_metrics(img, jp, group_id, data_root) for img, jp in pairs]
    by_bn = {Path(m.image_path).name: m for m in metrics}

    hard = select_hard_samples(metrics, N_HARD)
    hard_bns = {Path(m.image_path).name for m in hard}
    sel.hard_metrics = [
        {
            "rank": i + 1,
            "sample_id": m.sample_id,
            "image_basename": Path(m.image_path).name,
            "fg_pixels": m.fg_pixels,
            "token_count": m.token_count,
            "ast_node_count": m.ast_node_count,
            "special_counts": m.special_counts,
            "selection_rule": "max_fg_then_ties" if i == 0 else "max_tokens_then_ties",
        }
        for i, m in enumerate(hard)
    ]

    fused, rows = load_cluster_fused(cluster_root, group_id)
    # align cluster rows to local images by basename
    div_rows, div_meta = select_diversity_pair(fused, rows, exclude_basenames=hard_bns)
    sel.diversity = div_meta
    div_bns = {Path(r["image_path"]).name for r in div_rows}
    # ensure diversity basenames exist in this question dir
    missing_div = [bn for bn in div_bns if bn not in by_bn]
    if missing_div:
        raise RuntimeError(f"{group_id}: diversity basenames not in question dir: {missing_div}")

    used = set(hard_bns) | set(div_bns)
    remain_metrics = [m for m in metrics if Path(m.image_path).name not in used]
    randoms = select_random(remain_metrics, N_RANDOM, seed=RANDOM_SEED)
    rand_bns = {Path(m.image_path).name for m in randoms}
    used |= rand_bns

    if len(hard) < N_HARD or len(div_rows) < N_DIVERSITY or len(randoms) < N_RANDOM:
        sel.notes.append(
            f"shortfall hard={len(hard)}/{N_HARD} diversity={len(div_rows)}/{N_DIVERSITY} "
            f"random={len(randoms)}/{N_RANDOM}"
        )

    selected_map = {
        "hard": [m.sample_id for m in hard],
        "diversity": [r["sample_id"] for r in div_rows],
        "random": [m.sample_id for m in randoms],
    }
    sel.selected = selected_map

    # materialize select + raw
    dest_q_sel = select_root / group_id
    dest_q_raw = raw_root / group_id
    ensure_dir(dest_q_sel)
    ensure_dir(dest_q_raw)

    selected_paths: dict[str, list[dict[str, str]]] = {"hard": [], "diversity": [], "random": []}
    for m in hard:
        selected_paths["hard"].append(materialize_pair(Path(m.image_path), Path(m.json_path), dest_q_sel))
    for r in div_rows:
        bn = Path(r["image_path"]).name
        m = by_bn[bn]
        selected_paths["diversity"].append(materialize_pair(Path(m.image_path), Path(m.json_path), dest_q_sel))
    for m in randoms:
        selected_paths["random"].append(materialize_pair(Path(m.image_path), Path(m.json_path), dest_q_sel))
    sel.selected_paths = selected_paths

    remaining_ids = []
    for m in metrics:
        bn = Path(m.image_path).name
        if bn in used:
            continue
        materialize_pair(Path(m.image_path), Path(m.json_path), dest_q_raw)
        remaining_ids.append(m.sample_id)
    sel.remaining = remaining_ids
    sel.status = "success"
    return sel


def build_selected_by_type_record(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact per-question map of the 8 samples by selection type."""
    questions: list[dict[str, Any]] = []
    for a in audits:
        if a.get("status") != "success":
            continue
        selected = a.get("selected") or {}
        hard = list(selected.get("hard") or [])
        diversity = list(selected.get("diversity") or [])
        random_s = list(selected.get("random") or [])
        questions.append(
            {
                "group_id": a["group_id"],
                "n_images": a.get("n_images"),
                "samples": {
                    "随机": random_s,
                    "难度": hard,
                    "多样性": diversity,
                },
                # English aliases for downstream scripts
                "samples_en": {
                    "random": random_s,
                    "hard": hard,
                    "diversity": diversity,
                },
                "n_selected": len(hard) + len(diversity) + len(random_s),
            }
        )
    return {
        "description": "每道合格题目抽取的 8 个样本分类：随机 / 难度 / 多样性",
        "counts_per_type": {"随机": N_RANDOM, "难度": N_HARD, "多样性": N_DIVERSITY},
        "n_questions": len(questions),
        "questions": questions,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select 8 initial candidates per qualifying question")
    p.add_argument("--data-root", default=DEFAULT_DATA)
    p.add_argument("--cluster-root", default=DEFAULT_CLUSTER)
    p.add_argument("--raw-out", default=DEFAULT_RAW)
    p.add_argument("--select-out", default=DEFAULT_SELECT)
    p.add_argument("--audit-dir", default=DEFAULT_AUDIT_DIR)
    p.add_argument("--min-images", type=int, default=MIN_IMAGES)
    p.add_argument("--limit", type=int, default=0, help="process at most N qualifying groups (0=all)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    data_root = Path(args.data_root)
    cluster_root = Path(args.cluster_root)
    raw_out = Path(args.raw_out)
    select_out = Path(args.select_out)
    audit_dir = Path(args.audit_dir)

    # clean outputs for a fresh run
    for d in (raw_out, select_out):
        if d.exists():
            shutil.rmtree(d)
        ensure_dir(d)
    ensure_dir(audit_dir)

    t0 = time.time()
    groups = discover_question_groups(data_root)
    print(f"[init] groups_discovered={len(groups)} data_root={data_root}", flush=True)

    unqualified: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    n_skip = 0
    n_select_files = 0
    n_raw_files = 0

    for gi, gid in enumerate(groups, 1):
        gdir = data_root / gid
        n_img = len(list(gdir.glob("*.jpg")))
        if n_img < args.min_images:
            row = {"group_id": gid, "n_images": n_img, "reason": f"n_images<{args.min_images}"}
            unqualified.append(row)
            n_skip += 1
            print(f"[{gi}/{len(groups)}] SKIP {gid} N={n_img}", flush=True)
            continue

        if args.limit and n_ok >= args.limit:
            break

        print(f"[{gi}/{len(groups)}] PROCESS {gid} N={n_img}", flush=True)
        try:
            result = process_group(
                gid,
                data_root=data_root,
                cluster_root=cluster_root,
                raw_root=raw_out,
                select_root=select_out,
                min_images=args.min_images,
            )
            audits.append(result.to_dict())
            if result.status == "success":
                n_ok += 1
                n_select_files += sum(len(v) for v in result.selected_paths.values())
                n_raw_files += len(result.remaining)
                print(
                    f"  -> selected hard={len(result.selected.get('hard', []))} "
                    f"div={len(result.selected.get('diversity', []))} "
                    f"rand={len(result.selected.get('random', []))} "
                    f"remaining={len(result.remaining)} "
                    f"div_cos={result.diversity.get('cosine_similarity')}",
                    flush=True,
                )
            else:
                n_skip += 1
                unqualified.append(
                    {"group_id": gid, "n_images": result.n_images, "reason": result.status, "notes": result.notes}
                )
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            err = {"group_id": gid, "n_images": n_img, "error": f"{type(e).__name__}: {e}"}
            audits.append({"group_id": gid, "status": "failed", **err})
            print(f"  -> FAILED {err['error']}", flush=True)

    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
        "data_root": str(data_root),
        "cluster_root": str(cluster_root),
        "raw_out": str(raw_out),
        "select_out": str(select_out),
        "min_images": args.min_images,
        "n_groups_total": len(groups),
        "n_qualified_processed": n_ok,
        "n_unqualified_skipped": n_skip,
        "n_failed": n_fail,
        "n_select_sample_pairs": n_select_files,
        "n_raw_sample_pairs": n_raw_files,
        "selection_spec": {
            "random": {"n": N_RANDOM, "seed": RANDOM_SEED},
            "hard": {
                "n": N_HARD,
                "fg_threshold": FG_THRESHOLD,
                "rule": "1) max FG (tie tokens, AST); 2) max tokens (tie AST, FG)",
                "special_tokens_recorded": list(SPECIAL_TOKENS),
            },
            "diversity": {
                "n": N_DIVERSITY,
                "rule": "min pairwise cosine on fused_features among non-hard samples",
                "cluster_root": str(cluster_root),
            },
        },
    }

    atomic_write_json(audit_dir / "unqualified_questions.json", unqualified)
    atomic_write_json(audit_dir / "selection_audit.json", audits)
    atomic_write_json(audit_dir / "selection_summary.json", summary)
    by_type = build_selected_by_type_record(audits)
    atomic_write_json(audit_dir / "selected_samples_by_type.json", by_type)

    print("\n========== 不合格题目清单 ==========", flush=True)
    print(json.dumps(unqualified, ensure_ascii=False, indent=2), flush=True)
    print("\n========== 抽取审计摘要 ==========", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\n========== 每题抽取样本分类（随机/难度/多样性） ==========", flush=True)
    print(json.dumps(by_type, ensure_ascii=False, indent=2), flush=True)
    print("\n========== 完整抽取审计（逐题） ==========", flush=True)
    print(json.dumps(audits, ensure_ascii=False, indent=2), flush=True)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
