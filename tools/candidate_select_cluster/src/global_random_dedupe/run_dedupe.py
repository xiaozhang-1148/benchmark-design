"""Global-cluster random dedupe (stage2 → adjusted_2).

Rule (matches ``stage2_cluster_adjusted_2`` manifests):
  - Only ``post_selection_source == random`` samples are candidates for removal.
  - Group by **global** ``cluster_id`` (not per-question clusters).
  - Keep at most one random per global cluster: max ``cosine_to_centroid``,
    ties broken by lexicographically smaller ``sample_id``.
  - Keep all non-random samples (hard / diverse / replacement) unchanged.

This is distinct from per-question stage2 if-else (``run_stage2.py`` / τ = ceil(K/2)).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ASSIGNMENTS = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "stage2_cluster_adjusted_2/manifests/select_assignments_reconstructed.csv"
)
DEFAULT_REF_MANIFESTS = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "stage2_cluster_adjusted_2/manifests"
)
DEFAULT_OUT = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
    "stage2_cluster_adjusted_2"
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(path)


def normalize_cluster_id(raw: str | int) -> str:
    s = str(raw).strip()
    if s.startswith("cluster_"):
        return s
    return f"cluster_{int(s)}"


def load_assignments_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            src = (r.get("post_selection_source") or r.get("selection_source") or "").strip()
            if not src:
                raise RuntimeError(f"missing post_selection_source for {r.get('sample_id')}")
            sid = (r.get("sample_id") or "").strip()
            if not sid:
                raise RuntimeError("assignment row missing sample_id")
            cos = float(r["cosine_to_centroid"])
            rows.append(
                {
                    "exam_id": (r.get("exam_id") or "").strip(),
                    "question_id": (r.get("question_id") or "").strip(),
                    "sample_id": sid,
                    "basename": (r.get("basename") or Path(sid).name).strip(),
                    "post_selection_source": src,
                    "cluster_id": normalize_cluster_id(r["cluster_id"]),
                    "cluster_size_raw": (r.get("cluster_size_raw") or "").strip(),
                    "cosine_to_centroid": cos,
                    "triggered_replacement": (r.get("triggered_replacement") or "").strip(),
                    "image_path": (r.get("image_path") or "").strip(),
                }
            )
    if not rows:
        raise RuntimeError(f"empty assignments: {path}")
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample_id in assignments")
    return rows


def reconstruct_assignments_from_adjusted2_manifests(manifests_dir: Path) -> list[dict[str, Any]]:
    """Build assignment rows from kept_samples.csv + removed_samples.csv."""
    kept_p = manifests_dir / "kept_samples.csv"
    rem_p = manifests_dir / "removed_samples.csv"
    if not kept_p.is_file() or not rem_p.is_file():
        raise FileNotFoundError(f"need kept/removed under {manifests_dir}")
    rows: list[dict[str, Any]] = []
    for p in (kept_p, rem_p):
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "exam_id": r.get("exam_id", ""),
                        "question_id": r.get("question_id", ""),
                        "sample_id": r["sample_id"],
                        "basename": r.get("basename") or Path(r["sample_id"]).name,
                        "post_selection_source": r["post_selection_source"],
                        "cluster_id": normalize_cluster_id(r["cluster_id"]),
                        "cluster_size_raw": r.get("cluster_size_raw", ""),
                        "cosine_to_centroid": float(r["cosine_to_centroid"]),
                        "triggered_replacement": r.get("triggered_replacement", ""),
                        "image_path": "",
                    }
                )
    return rows


def dedupe_randoms_global(
    assignments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kept_rows, removed_rows, per_cluster_decisions)."""
    non_random = [r for r in assignments if r["post_selection_source"] != "random"]
    randoms = [r for r in assignments if r["post_selection_source"] == "random"]

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in randoms:
        by_cluster[r["cluster_id"]].append(r)

    kept_random: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for cid in sorted(by_cluster, key=lambda c: int(c.split("_", 1)[1])):
        members = list(by_cluster[cid])
        members.sort(key=lambda r: (-r["cosine_to_centroid"], r["sample_id"]))
        best = members[0]
        max_cos = best["cosine_to_centroid"]
        n_ties = sum(1 for r in members if abs(r["cosine_to_centroid"] - max_cos) <= 1e-12)
        kept_random.append(best)
        for rank, r in enumerate(members, start=1):
            if r["sample_id"] == best["sample_id"]:
                continue
            removed.append(
                {
                    **r,
                    "decision": "remove",
                    "decision_reason": (
                        f"random_duplicate_in_{cid}; kept={best['sample_id']} "
                        f"(cos={best['cosine_to_centroid']:.10f}) > this "
                        f"(cos={r['cosine_to_centroid']:.10f})"
                    ),
                    "randoms_in_cluster_before": str(len(members)),
                    "rank_in_cluster_by_cosine": str(rank),
                    "kept_sample_id": best["sample_id"],
                    "kept_cosine_to_centroid": f"{best['cosine_to_centroid']:.16g}",
                }
            )
        decisions.append(
            {
                "cluster_id": cid,
                "n_random_before": len(members),
                "n_random_kept": 1,
                "n_random_removed": len(members) - 1,
                "kept_sample_id": best["sample_id"],
                "kept_cosine_to_centroid": f"{best['cosine_to_centroid']:.16g}",
                "n_cosine_ties_at_max": n_ties,
                "removed_sample_ids": json.dumps(
                    [r["sample_id"] for r in members if r["sample_id"] != best["sample_id"]],
                    ensure_ascii=False,
                ),
            }
        )

    kept: list[dict[str, Any]] = []
    for r in non_random:
        kept.append(
            {
                **r,
                "decision": "keep",
                "decision_reason": f"non_random_source={r['post_selection_source']}",
                "randoms_in_cluster_before": "",
                "rank_in_cluster_by_cosine": "",
            }
        )
    for r in kept_random:
        members = by_cluster[r["cluster_id"]]
        members_sorted = sorted(members, key=lambda x: (-x["cosine_to_centroid"], x["sample_id"]))
        rank = next(i for i, x in enumerate(members_sorted, start=1) if x["sample_id"] == r["sample_id"])
        kept.append(
            {
                **r,
                "decision": "keep",
                "decision_reason": (
                    f"random_keep_max_cosine_in_{r['cluster_id']}; "
                    f"n_random_before={len(members)}"
                ),
                "randoms_in_cluster_before": str(len(members)),
                "rank_in_cluster_by_cosine": str(rank),
            }
        )

    kept.sort(key=lambda r: r["sample_id"])
    removed.sort(key=lambda r: r["sample_id"])
    return kept, removed, decisions


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


def resolve_image_path(row: dict[str, Any], select_root: Path) -> Path:
    if row.get("image_path"):
        p = Path(row["image_path"])
        if p.is_file():
            return p
    # sample_id is typically exam/qid/basename or exam/basename
    sid = row["sample_id"]
    cand = select_root / sid
    if cand.is_file():
        return cand
    # try basename search under exam/question
    bn = row["basename"]
    exam = row.get("exam_id") or ""
    qid = row.get("question_id") or ""
    if exam and qid:
        p = select_root / exam / qid / bn
        if p.is_file():
            return p
    if exam:
        p = select_root / exam / bn
        if p.is_file():
            return p
    matches = list(select_root.rglob(bn))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve image for sample_id={sid} under {select_root}")


def materialize_kept(
    kept: list[dict[str, Any]],
    *,
    select_in: Path,
    select_out: Path,
) -> list[dict[str, str]]:
    copy_log: list[dict[str, str]] = []
    for r in kept:
        src = resolve_image_path(r, select_in)
        # preserve relative layout under select_out when possible
        try:
            rel = src.relative_to(select_in)
        except ValueError:
            rel = Path(r["sample_id"])
        dst = select_out / rel
        mode = link_or_copy(src, dst)
        jsrc = Path(str(src) + ".json")
        if not jsrc.is_file():
            raise FileNotFoundError(jsrc)
        link_or_copy(jsrc, Path(str(dst) + ".json"))
        copy_log.append(
            {
                "sample_id": r["sample_id"],
                "src": str(src),
                "dst": str(dst),
                "mode": mode,
            }
        )
    return copy_log


def verify_against_reference(
    kept: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    ref_manifests: Path,
) -> list[str]:
    errors: list[str] = []
    ref_kept = {r["sample_id"] for r in csv.DictReader((ref_manifests / "kept_samples.csv").open(encoding="utf-8"))}
    ref_rem = {r["sample_id"] for r in csv.DictReader((ref_manifests / "removed_samples.csv").open(encoding="utf-8"))}
    got_kept = {r["sample_id"] for r in kept}
    got_rem = {r["sample_id"] for r in removed}
    if got_kept != ref_kept:
        errors.append(
            f"kept mismatch: extra={sorted(got_kept - ref_kept)[:5]} "
            f"missing={sorted(ref_kept - got_kept)[:5]} "
            f"(|Δ|={len(got_kept ^ ref_kept)})"
        )
    if got_rem != ref_rem:
        errors.append(
            f"removed mismatch: extra={sorted(got_rem - ref_rem)[:5]} "
            f"missing={sorted(ref_rem - got_rem)[:5]} "
            f"(|Δ|={len(got_rem ^ ref_rem)})"
        )
    return errors


KEPT_FIELDS = [
    "exam_id",
    "question_id",
    "sample_id",
    "basename",
    "post_selection_source",
    "cluster_id",
    "cluster_size_raw",
    "cosine_to_centroid",
    "triggered_replacement",
    "decision",
    "decision_reason",
    "randoms_in_cluster_before",
    "rank_in_cluster_by_cosine",
]
REMOVED_FIELDS = KEPT_FIELDS + ["kept_sample_id", "kept_cosine_to_centroid"]
DECISION_FIELDS = [
    "cluster_id",
    "n_random_before",
    "n_random_kept",
    "n_random_removed",
    "kept_sample_id",
    "kept_cosine_to_centroid",
    "n_cosine_ties_at_max",
    "removed_sample_ids",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Global-cluster random dedupe → adjusted_2")
    p.add_argument(
        "--assignments-csv",
        default=None,
        help="CSV with sample_id, post_selection_source, cluster_id, cosine_to_centroid, ...",
    )
    p.add_argument(
        "--from-adjusted2-manifests",
        default=None,
        help="Reconstruct assignments from kept_samples.csv + removed_samples.csv",
    )
    p.add_argument("--select-in", default=None, help="Input select_dataset root (for --materialize)")
    p.add_argument("--output-root", default=DEFAULT_OUT)
    p.add_argument("--dry-run", action="store_true", help="Write manifests only (no select copy)")
    p.add_argument("--materialize", action="store_true", help="Copy/hardlink kept pairs into output")
    p.add_argument(
        "--verify-against-manifests",
        default=None,
        help="Compare kept/removed sample_id sets to reference manifests dir",
    )
    p.add_argument(
        "--export-assignments",
        default=None,
        help="Write reconstructed/normalized assignments CSV to this path and exit",
    )
    args = p.parse_args(argv)

    t0 = time.time()
    if args.from_adjusted2_manifests:
        assignments = reconstruct_assignments_from_adjusted2_manifests(Path(args.from_adjusted2_manifests))
    elif args.assignments_csv:
        assignments = load_assignments_csv(Path(args.assignments_csv))
    else:
        # default: reconstruct from gold adjusted_2 manifests if present
        ref = Path(DEFAULT_REF_MANIFESTS)
        if (ref / "kept_samples.csv").is_file():
            print(f"[info] reconstructing assignments from {ref}", flush=True)
            assignments = reconstruct_assignments_from_adjusted2_manifests(ref)
        else:
            print("FATAL: provide --assignments-csv or --from-adjusted2-manifests", flush=True)
            return 2

    if args.export_assignments:
        out_p = Path(args.export_assignments)
        fields = [
            "exam_id",
            "question_id",
            "sample_id",
            "basename",
            "post_selection_source",
            "cluster_id",
            "cluster_size_raw",
            "cosine_to_centroid",
            "triggered_replacement",
            "image_path",
        ]
        write_csv(out_p, assignments, fields)
        print(f"wrote assignments n={len(assignments)} -> {out_p}", flush=True)
        return 0

    kept, removed, decisions = dedupe_randoms_global(assignments)
    src_counts = Counter(r["post_selection_source"] for r in assignments)
    kept_counts = Counter(r["post_selection_source"] for r in kept)

    print(
        f"[dedupe] in={len(assignments)} random_in={src_counts.get('random', 0)} "
        f"kept={len(kept)} removed={len(removed)} "
        f"random_kept={kept_counts.get('random', 0)}",
        flush=True,
    )

    verify_errors: list[str] = []
    ref_dir = Path(args.verify_against_manifests) if args.verify_against_manifests else None
    if ref_dir is None and Path(DEFAULT_REF_MANIFESTS).is_dir():
        ref_dir = Path(DEFAULT_REF_MANIFESTS)
    if ref_dir is not None and (ref_dir / "kept_samples.csv").is_file():
        verify_errors = verify_against_reference(kept, removed, ref_dir)
        if verify_errors:
            for e in verify_errors:
                print(f"VERIFY FAIL: {e}", flush=True)
            return 3
        print(f"[verify] OK against {ref_dir}", flush=True)

    out_root = Path(args.output_root)
    man = out_root / "manifests"
    # format cosine as float string like reference
    kept_out = [{**r, "cosine_to_centroid": r["cosine_to_centroid"]} for r in kept]
    rem_out = [{**r, "cosine_to_centroid": r["cosine_to_centroid"]} for r in removed]
    write_csv(man / "kept_samples.csv", kept_out, KEPT_FIELDS)
    write_csv(man / "removed_samples.csv", rem_out, REMOVED_FIELDS)
    write_csv(man / "per_cluster_random_decisions.csv", decisions, DECISION_FIELDS)
    atomic_write_json(man / "per_cluster_random_decisions.json", decisions)
    atomic_write_json(man / "removed_sample_ids.json", [r["sample_id"] for r in removed])

    copy_log: list[dict[str, str]] = []
    if args.materialize:
        if not args.select_in:
            print("FATAL: --materialize requires --select-in", flush=True)
            return 2
        select_in = Path(args.select_in)
        select_out = out_root / "select_dataset"
        if select_out.exists():
            shutil.rmtree(select_out)
        copy_log = materialize_kept(kept, select_in=select_in, select_out=select_out)
        write_csv(man / "copy_log.csv", copy_log, ["sample_id", "src", "dst", "mode"])
        n_jpg = len(list(select_out.rglob("*.jpg")))
        if n_jpg != len(kept):
            print(f"FATAL: materialized jpg={n_jpg} != kept={len(kept)}", flush=True)
            return 4
    elif args.dry_run:
        print("[dry-run] manifests written; select not materialized", flush=True)

    summary = {
        "status": "success",
        "rule": (
            "For post_selection_source==random, at most one sample per global cluster_id; "
            "keep max cosine_to_centroid (tie-break: smallest sample_id). "
            "Keep all non-random samples."
        ),
        "n_input": len(assignments),
        "n_random_input": int(src_counts.get("random", 0)),
        "n_non_random_input": len(assignments) - int(src_counts.get("random", 0)),
        "n_kept": len(kept),
        "n_removed": len(removed),
        "n_random_kept": int(kept_counts.get("random", 0)),
        "n_random_removed": int(src_counts.get("random", 0)) - int(kept_counts.get("random", 0)),
        "n_global_clusters_with_random": len(decisions),
        "source_counts_input": dict(src_counts),
        "source_counts_kept": dict(kept_counts),
        "n_output_jpg": len(copy_log) if copy_log else None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - t0,
    }
    atomic_write_json(man / "summary.json", summary)
    readme = out_root / "流程说明.md"
    readme.write_text(
        "\n".join(
            [
                "# stage2_cluster_adjusted_2：按全局簇去重 random 样本",
                "",
                "## 规则",
                "",
                "- 仅处理 `post_selection_source == random`；",
                "- 按**全局** `cluster_id` 分组；",
                "- 每个全局簇最多保留 1 个 random（余弦最大；并列取更小 `sample_id`）；",
                "- `hard` / `diverse` / `replacement` 全部保留。",
                "",
                "## 数量",
                "",
                f"- 输入：{summary['n_input']}（random {summary['n_random_input']}）",
                f"- 保留：{summary['n_kept']}（random {summary['n_random_kept']}）",
                f"- 移除：{summary['n_removed']}",
                "",
                f"实现：`tools/candidate_select_cluster/global_random_dedupe.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
