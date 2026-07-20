"""End-to-end page-level HMER multilabel stratified split pipeline."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from benchmark_design.page_level_latex_split.audit import (
    AcceptanceResult,
    CandidateSeedScore,
    SplitAcceptanceError,
    assignment_to_manifest_frame,
    manifest_hash,
    run_acceptance_checks,
    select_best_candidate,
    write_candidate_scores,
    write_split_lists,
)
from benchmark_design.page_level_latex_split.config import SplitConfig, load_split_config
from benchmark_design.page_level_latex_split.io_validate import SplitInputBundle, file_sha256, load_and_validate_inputs
from benchmark_design.page_level_latex_split.labels import PageLabels, build_page_labels, labels_to_frame
from benchmark_design.page_level_latex_split.refine import (
    global_swap_refine,
    residual_vocab_repair,
    vocabulary_audit,
)
from benchmark_design.page_level_latex_split.report import export_reports
from benchmark_design.page_level_latex_split.stratify import StratifyState, iterative_multilabel_assign
from benchmark_design.page_level_latex_split.vocab_cover import (
    VocabIndex,
    build_train_vocab_cover,
    check_vocab_feasibility,
    deduct_locked_train_quotas,
)

_CONSOLE = Console()
ALGORITHM_VERSION = "page_level_latex_split_v3"


def _make_progress(*, show_progress: bool) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_CONSOLE,
        transient=False,
        disable=not show_progress,
    )


def _write_status(output_root: Path, payload: dict) -> None:
    path = output_root / "split_status.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class VocabFeasibilityError(RuntimeError):
    """Raised when train vocabulary cover cannot satisfy hard constraints."""


@dataclass(frozen=True, slots=True)
class SplitPipelineResult:
    output_root: Path
    selected_seed: int
    manifest_sha256: str
    acceptance: AcceptanceResult
    artifact_manifest: dict[str, str]


@dataclass(frozen=True, slots=True)
class _SeedRunResult:
    seed: int
    state: StratifyState
    assignment: dict[str, str]
    score: CandidateSeedScore
    cover_page_count: int


def _run_one_seed(
    bundle: SplitInputBundle,
    config: SplitConfig,
    seed: int,
    page_labels: list[PageLabels],
    vocab_index: VocabIndex,
    *,
    progress: Progress | None = None,
    seed_task_id: int | None = None,
) -> _SeedRunResult:
    """Vocab-first candidate seed: cover lock -> stratify -> residual repair -> coverage swap."""
    started = time.perf_counter()
    if progress is not None and seed_task_id is not None:
        progress.update(seed_task_id, description=f"Seed {seed}: train vocab cover")

    cover = build_train_vocab_cover(vocab_index, page_labels, config, seed=seed)
    adjusted = deduct_locked_train_quotas(page_labels, cover.locked_assignment, config)
    feasibility = check_vocab_feasibility(
        vocab_index,
        page_count=len(page_labels),
        train_ratio=config.train_ratio,
    )
    if cover.cover_page_count > feasibility.train_capacity:
        raise VocabFeasibilityError(
            f"seed={seed}: vocab cover requires {cover.cover_page_count} train pages "
            f"but capacity is {feasibility.train_capacity}"
        )

    if progress is not None and seed_task_id is not None:
        progress.update(
            seed_task_id,
            description=f"Seed {seed}: stratify ({cover.cover_page_count} locked train pages)",
        )

    state = iterative_multilabel_assign(
        page_labels,
        config,
        seed=seed,
        locked_assignment=cover.locked_assignment,
        adjusted_quotas=adjusted.quotas,
        remaining_capacity=adjusted.remaining_capacity,
        initial_filled=adjusted.initial_filled,
    )

    if progress is not None and seed_task_id is not None:
        progress.update(seed_task_id, description=f"Seed {seed}: residual vocab repair")

    assignment, _ = residual_vocab_repair(
        state.assignment,
        page_labels,
        bundle.token_counts,
        bundle.features,
        config,
        state.quotas,
        seed=seed,
        progress=progress,
    )

    if progress is not None and seed_task_id is not None:
        progress.update(seed_task_id, description=f"Seed {seed}: coverage-preserving swap")

    assignment = global_swap_refine(
        assignment,
        page_labels,
        bundle.features,
        bundle.token_counts,
        state.quotas,
        config,
        seed=seed,
        progress=progress,
    )

    score = CandidateSeedScore.from_assignment(
        seed,
        assignment,
        page_labels,
        state.quotas,
        bundle.token_counts,
        bundle.features,
        elapsed_seconds=time.perf_counter() - started,
    )
    return _SeedRunResult(
        seed=seed,
        state=state,
        assignment=assignment,
        score=score,
        cover_page_count=cover.cover_page_count,
    )


def _seed_worker(payload: tuple) -> tuple[int, dict[str, str], dict, CandidateSeedScore, int]:
    """Process-pool entry point; returns picklable payload for StratifyState rehydration."""
    (
        inputs_dir,
        config_path,
        seed,
        features_path,
        token_counts_path,
        manifest_path,
        dataset_version,
        input_hashes,
    ) = payload
    config = load_split_config(Path(config_path))
    bundle = SplitInputBundle(
        manifest=__import__("pandas").read_csv(manifest_path),
        features=__import__("pandas").read_csv(features_path),
        token_counts=__import__("pandas").read_csv(token_counts_path),
        dataset_version=dataset_version,
        input_hashes=input_hashes,
        inputs_dir=Path(inputs_dir),
    )
    page_labels = build_page_labels(bundle.features, config)
    vocab_index = VocabIndex.from_token_counts(bundle.token_counts)
    result = _run_one_seed(
        bundle,
        config,
        seed,
        page_labels,
        vocab_index,
        progress=None,
        seed_task_id=None,
    )
    state_payload = {
        "assignment": result.state.assignment,
        "remaining_capacity": dict(result.state.remaining_capacity),
        "formal_labels": list(result.state.formal_labels),
        "quotas": {
            label: {
                "label": q.label,
                "support": q.support,
                "formal": q.formal,
                "reason": q.reason,
                "targets": dict(q.targets),
                "expected_raw": dict(q.expected_raw),
            }
            for label, q in result.state.quotas.items()
        },
    }
    return (
        result.seed,
        result.assignment,
        state_payload,
        result.score,
        result.cover_page_count,
    )


def _rehydrate_state(state_payload: dict) -> StratifyState:
    from benchmark_design.page_level_latex_split.stratify import LabelQuota

    quotas = {
        label: LabelQuota(
            label=q["label"],
            support=q["support"],
            formal=q["formal"],
            reason=q["reason"],
            targets=q["targets"],
            expected_raw=q["expected_raw"],
        )
        for label, q in state_payload["quotas"].items()
    }
    return StratifyState(
        assignment=dict(state_payload["assignment"]),
        remaining_capacity=dict(state_payload["remaining_capacity"]),
        filled={label: __import__("collections").Counter() for label in quotas},
        formal_labels=list(state_payload["formal_labels"]),
        quotas=quotas,
    )


def _write_unseen_tokens(vocab, output_root: Path) -> Path:
    records = []
    for token in vocab.val_unseen:
        records.append({"token": token, "split": "val"})
    for token in vocab.test_unseen:
        records.append({"token": token, "split": "test"})
    path = output_root / "vocabulary_unseen_tokens.csv"
    if records:
        __import__("pandas").DataFrame.from_records(records).to_csv(path, index=False)
    else:
        path.write_text("token,split\n", encoding="utf-8")
    return path


def _export_diagnostics(
    *,
    output_root: Path,
    split_manifest,
    labels_frame,
    page_labels,
    assignment,
    state: StratifyState,
    vocab,
    config: SplitConfig,
    bundle: SplitInputBundle,
    seed_scores: list[CandidateSeedScore],
    selected: CandidateSeedScore,
    skip_figures: bool,
) -> dict[str, str]:
    manifest_name = "diagnostic_split_manifest.csv"
    split_manifest_path = output_root / manifest_name
    labels_path = output_root / "page_stratification_labels.csv"
    split_manifest.to_csv(split_manifest_path, index=False)
    labels_frame.to_csv(labels_path, index=False)
    candidate_scores_path = write_candidate_scores(seed_scores, output_root)
    unseen_path = _write_unseen_tokens(vocab, output_root)
    report_manifest = export_reports(
        manifest=split_manifest,
        features=bundle.features,
        labels_frame=labels_frame,
        page_labels=page_labels,
        assignment=assignment,
        quotas=state.quotas,
        vocab=vocab,
        config=config,
        output_root=output_root,
        skip_figures=skip_figures,
    )
    return {
        manifest_name: split_manifest_path.name,
        "page_stratification_labels": labels_path.name,
        "candidate_scores": candidate_scores_path.name,
        "vocabulary_unseen_tokens": unseen_path.name,
        **report_manifest,
        "selected_seed": selected.seed,
    }


def run_page_level_latex_split(
    inputs_dir: Path,
    output_root: Path,
    *,
    config_path: Path | None = None,
    config: SplitConfig | None = None,
    skip_figures: bool = False,
    show_progress: bool = True,
    workers: int | None = None,
    allow_failed_acceptance: bool = False,
) -> SplitPipelineResult:
    started = time.perf_counter()
    if config is None:
        if config_path is None:
            raise ValueError("config or config_path is required")
        config = load_split_config(config_path)

    cpu = os.cpu_count() or 1
    seed_workers = workers if workers is not None else min(len(config.candidate_seeds), cpu)
    seed_workers = max(1, min(seed_workers, len(config.candidate_seeds)))

    with _make_progress(show_progress=show_progress) as progress:
        validate_task = progress.add_task("Validate split inputs", total=1)
        bundle = load_and_validate_inputs(inputs_dir, config)
        progress.update(
            validate_task,
            advance=1,
            description=(
                f"Validated {len(bundle.manifest)} pages / {len(bundle.token_counts)} token rows"
            ),
        )

        output_root.mkdir(parents=True, exist_ok=True)
        _write_status(
            output_root,
            {
                "phase": "started",
                "page_count": len(bundle.manifest),
                "candidate_seeds": list(config.candidate_seeds),
                "tie_break": config.tie_break,
                "algorithm_version": ALGORITHM_VERSION,
                "seed_workers": seed_workers,
                "started_at": datetime.now(tz=UTC).isoformat(),
            },
        )

        labels_task = progress.add_task("Build stratification labels", total=1)
        page_labels = build_page_labels(bundle.features, config)
        vocab_index = VocabIndex.from_token_counts(bundle.token_counts)
        feasibility = check_vocab_feasibility(
            vocab_index,
            page_count=len(page_labels),
            train_ratio=config.train_ratio,
        )
        if not feasibility.feasible:
            raise VocabFeasibilityError(
                f"vocabulary feasibility failed: min_cover={feasibility.min_cover_pages} "
                f"train_capacity={feasibility.train_capacity} "
                f"impossible_tokens={feasibility.impossible_tokens[:10]}"
            )
        progress.update(
            labels_task,
            advance=1,
            description=(
                f"Built labels for {len(page_labels)} pages; "
                f"vocab cover lower bound={feasibility.min_cover_pages} pages"
            ),
        )

        seed_total = len(config.candidate_seeds)
        seeds_task = progress.add_task(
            f"Candidate stratified splits (0/{seed_total})",
            total=seed_total,
        )

        seed_scores: list[CandidateSeedScore] = []
        seed_payloads: dict[int, tuple] = {}

        if seed_workers <= 1 or seed_total == 1:
            for index, seed in enumerate(config.candidate_seeds, start=1):
                result = _run_one_seed(
                    bundle,
                    config,
                    seed,
                    page_labels,
                    vocab_index,
                    progress=progress,
                    seed_task_id=seeds_task,
                )
                seed_scores.append(result.score)
                seed_payloads[seed] = (page_labels, result.state, result.assignment)
                progress.update(
                    seeds_task,
                    completed=index,
                    description=(
                        f"Candidate stratified splits ({index}/{seed_total}, "
                        f"seed={seed}, unseen={result.score.vocab_unseen_total}, "
                        f"max_dev={result.score.max_formal_deviation:.4f})"
                    ),
                )
                _write_status(
                    output_root,
                    {
                        "phase": "candidate_seed_done",
                        "completed_seeds": index,
                        "total_seeds": seed_total,
                        "last_seed": seed,
                        "last_vocab_unseen": result.score.vocab_unseen_total,
                        "last_max_dev": result.score.max_formal_deviation,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
        else:
            worker_payload = (
                str(inputs_dir.resolve()),
                str(config_path.resolve() if config_path else ""),
                None,
                str((inputs_dir / "page_hmer_features.csv").resolve()),
                str((inputs_dir / "page_token_counts.csv").resolve()),
                str((inputs_dir / "dataset_manifest.csv").resolve()),
                bundle.dataset_version,
                bundle.input_hashes,
            )
            if not config_path:
                raise ValueError("config_path is required for parallel seed execution")
            completed = 0
            with ProcessPoolExecutor(
                max_workers=seed_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(
                        _seed_worker,
                        (
                            worker_payload[0],
                            worker_payload[1],
                            seed,
                            worker_payload[3],
                            worker_payload[4],
                            worker_payload[5],
                            worker_payload[6],
                            worker_payload[7],
                        ),
                    ): seed
                    for seed in config.candidate_seeds
                }
                for future in as_completed(futures):
                    seed, assignment, state_payload, score, cover_pages = future.result()
                    completed += 1
                    seed_scores.append(score)
                    seed_payloads[seed] = (page_labels, _rehydrate_state(state_payload), assignment)
                    progress.update(
                        seeds_task,
                        completed=completed,
                        description=(
                            f"Candidate stratified splits ({completed}/{seed_total}, "
                            f"seed={seed}, unseen={score.vocab_unseen_total}, "
                            f"cover={cover_pages})"
                        ),
                    )

        selected = select_best_candidate(seed_scores)
        write_candidate_scores(seed_scores, output_root)
        progress.update(
            seeds_task,
            description=(
                f"Selected seed={selected.seed} "
                f"(unseen={selected.vocab_unseen_total}, max_dev={selected.max_formal_deviation:.4f})"
            ),
        )
        _write_status(
            output_root,
            {
                "phase": "seed_selected",
                "selected_seed": selected.seed,
                "selected_vocab_unseen": selected.vocab_unseen_total,
                "selected_max_dev": selected.max_formal_deviation,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

        finalize_task = progress.add_task("Finalize split artifacts", total=4)
        page_labels, state, assignment = seed_payloads[selected.seed]

        progress.update(finalize_task, description="Vocabulary audit + acceptance checks")
        vocab = vocabulary_audit(bundle.token_counts, assignment, bundle.features)
        low_support = [label for label, q in state.quotas.items() if not q.formal]
        acceptance = run_acceptance_checks(
            assignment,
            page_labels,
            config,
            state.quotas,
            vocab,
            low_support,
        )
        progress.advance(finalize_task)

        split_manifest = assignment_to_manifest_frame(assignment, bundle.manifest, page_labels)
        labels_frame = labels_to_frame(page_labels, bundle.features)
        labels_frame = labels_frame.merge(
            split_manifest.loc[:, ["page_id", "split"]],
            on="page_id",
            how="left",
        )
        low_set = set(low_support)
        labels_frame["has_low_support_label"] = labels_frame["labels"].map(
            lambda text: int(any(part in low_set for part in str(text).split(";") if part))
        )

        progress.update(finalize_task, description="Write diagnostic tables and figures")
        diagnostic_manifest = _export_diagnostics(
            output_root=output_root,
            split_manifest=split_manifest,
            labels_frame=labels_frame,
            page_labels=page_labels,
            assignment=assignment,
            state=state,
            vocab=vocab,
            config=config,
            bundle=bundle,
            seed_scores=seed_scores,
            selected=selected,
            skip_figures=skip_figures,
        )
        progress.advance(finalize_task)

        if not acceptance.passed and not allow_failed_acceptance:
            reject_path = output_root / "split_rejected.json"
            reject_path.write_text(
                json.dumps(
                    {
                        "selected_seed": selected.seed,
                        "candidate_scores": [
                            {
                                "seed": score.seed,
                                "unseen": score.vocab_unseen_total,
                                "max_deviation": score.max_formal_deviation,
                                "joint_tv": score.joint_tv,
                                "mean_deviation": score.mean_formal_deviation,
                                "status": score.status,
                                "elapsed_s": score.elapsed_seconds,
                            }
                            for score in seed_scores
                        ],
                        "acceptance": {
                            "passed": acceptance.passed,
                            "checks": acceptance.checks,
                            "messages": list(acceptance.messages),
                        },
                        "diagnostic_artifacts": diagnostic_manifest,
                        "official_lists_written": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            progress.advance(finalize_task)
            progress.update(finalize_task, description="Finalize split artifacts (rejected)")
            _write_status(
                output_root,
                {
                    "phase": "rejected",
                    "selected_seed": selected.seed,
                    "acceptance_passed": False,
                    "vocab_unseen": len(vocab.val_unseen) + len(vocab.test_unseen),
                },
            )
            raise SplitAcceptanceError(
                "Split failed acceptance checks; official train/val/test lists were not written. "
                f"Diagnostic tables/figures are under {output_root.resolve()}. "
                f"See {reject_path.resolve()}"
            )

        progress.update(finalize_task, description="Write official train/val/test lists")
        official_manifest_path = output_root / "split_manifest.csv"
        split_manifest.to_csv(official_manifest_path, index=False)
        list_paths = write_split_lists(split_manifest, output_root)
        progress.advance(finalize_task)

        progress.update(finalize_task, description="Write split_metadata.json")
        man_hash = manifest_hash(split_manifest)
        elapsed = time.perf_counter() - started
        metadata = {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "dataset_version": bundle.dataset_version,
            "algorithm_version": ALGORITHM_VERSION,
            "tie_break": config.tie_break,
            "random_seed": config.random_seed,
            "selected_seed": selected.seed,
            "candidate_seeds": list(config.candidate_seeds),
            "candidate_scores": [
                {
                    "seed": score.seed,
                    "unseen": score.vocab_unseen_total,
                    "max_deviation": score.max_formal_deviation,
                    "joint_tv": score.joint_tv,
                    "mean_deviation": score.mean_formal_deviation,
                    "status": score.status,
                    "elapsed_s": score.elapsed_seconds,
                }
                for score in seed_scores
            ],
            "input_hashes": bundle.input_hashes,
            "split_manifest_sha256": man_hash,
            "manifest_hash_definition": "sha256(csv of page_id,split sorted by page_id)",
            "output_hashes": {
                "split_manifest.csv": file_sha256(official_manifest_path),
                "page_stratification_labels.csv": file_sha256(output_root / "page_stratification_labels.csv"),
            },
            "config": config.raw,
            "acceptance": {
                "passed": acceptance.passed,
                "checks": acceptance.checks,
                "messages": list(acceptance.messages),
            },
            "vocab_coverage": {
                "require": config.require_train_vocab_coverage,
                "val_unseen": len(vocab.val_unseen),
                "test_unseen": len(vocab.test_unseen),
            },
            "elapsed_seconds": elapsed,
            "page_count": len(split_manifest),
            "formal_label_count": sum(1 for q in state.quotas.values() if q.formal),
            "low_support_label_count": len(low_support),
        }
        meta_path = output_root / "split_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        progress.advance(finalize_task)
        progress.update(finalize_task, description="Finalize split artifacts done")
        _write_status(
            output_root,
            {
                "phase": "completed",
                "selected_seed": selected.seed,
                "manifest_sha256": man_hash,
                "acceptance_passed": acceptance.passed,
                "elapsed_seconds": elapsed,
            },
        )

    artifact_manifest = {
        "split_manifest": official_manifest_path.name,
        "page_stratification_labels": "page_stratification_labels.csv",
        "split_metadata": meta_path.name,
        "train": list_paths["train"].name,
        "val": list_paths["val"].name,
        "test": list_paths["test"].name,
        **diagnostic_manifest,
    }
    return SplitPipelineResult(
        output_root=output_root,
        selected_seed=selected.seed,
        manifest_sha256=man_hash,
        acceptance=acceptance,
        artifact_manifest=artifact_manifest,
    )
