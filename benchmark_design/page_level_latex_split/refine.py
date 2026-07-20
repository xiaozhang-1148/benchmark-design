"""Vocabulary coverage repair and global page-swap refinement."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from benchmark_design.page_level_latex_split.config import SplitConfig
from benchmark_design.page_level_latex_split.labels import PageLabels
from benchmark_design.page_level_latex_split.refine_state import RefinementState
from benchmark_design.page_level_latex_split.stratify import (
    SPLITS,
    LabelQuota,
    relative_deviation,
)
from benchmark_design.page_level_latex_split.tie_break import sort_key_for_page

if TYPE_CHECKING:
    from rich.progress import Progress


@dataclass(frozen=True, slots=True)
class VocabAudit:
    overall_size: int
    train_size: int
    val_size: int
    test_size: int
    val_unseen: tuple[str, ...]
    test_unseen: tuple[str, ...]
    rare8_by_split: dict[str, int]
    anomalies: tuple[dict[str, str], ...]


def build_split_vocabularies(
    token_counts: pd.DataFrame,
    assignment: dict[str, str],
    *,
    page_tokens: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    vocabs: dict[str, set[str]] = {split: set() for split in SPLITS}
    if page_tokens is None:
        if token_counts.empty:
            return vocabs
        for row in token_counts.itertuples(index=False):
            page_id = str(row.page_id)
            split = assignment.get(page_id)
            if split is None:
                continue
            vocabs[split].add(str(row.token))
        return vocabs
    for page_id, split in assignment.items():
        tokens = page_tokens.get(page_id)
        if tokens:
            vocabs[split].update(tokens)
    return vocabs


def vocabulary_audit(
    token_counts: pd.DataFrame,
    assignment: dict[str, str],
    features: pd.DataFrame,
) -> VocabAudit:
    vocabs = build_split_vocabularies(token_counts, assignment)
    overall = set().union(*vocabs.values()) if any(vocabs.values()) else set()
    val_unseen = sorted(vocabs["val"] - vocabs["train"])
    test_unseen = sorted(vocabs["test"] - vocabs["train"])

    token_pages: dict[str, list[str]] = defaultdict(list)
    if not token_counts.empty:
        for row in token_counts.itertuples(index=False):
            token_pages[str(row.token)].append(str(row.page_id))

    anomalies: list[dict[str, str]] = []
    for token in val_unseen:
        pages = [pid for pid in token_pages[token] if assignment.get(pid) == "val"]
        anomalies.append({"token": token, "split": "val", "page_ids": ";".join(sorted(pages))})
    for token in test_unseen:
        pages = [pid for pid in token_pages[token] if assignment.get(pid) == "test"]
        anomalies.append({"token": token, "split": "test", "page_ids": ";".join(sorted(pages))})

    rare8_by_split = {"train": 0, "val": 0, "test": 0}
    for row in features.itertuples(index=False):
        split = assignment.get(str(row.page_id))
        if split and int(row.has_rare8) == 1:
            rare8_by_split[split] += 1

    return VocabAudit(
        overall_size=len(overall),
        train_size=len(vocabs["train"]),
        val_size=len(vocabs["val"]),
        test_size=len(vocabs["test"]),
        val_unseen=tuple(val_unseen),
        test_unseen=tuple(test_unseen),
        rare8_by_split=rare8_by_split,
        anomalies=tuple(anomalies),
    )


def _label_similarity(state: RefinementState, donor: str, recv: str) -> int:
    donor_labels = set(state.by_id[donor].labels)
    recv_labels = set(state.by_id[recv].labels)
    return len(donor_labels & recv_labels)


def _quota_harm(state: RefinementState, donor: str, recv: str) -> float:
    donor_split = state.assignment[donor]
    recv_split = state.assignment[recv]
    donor_labels = state.by_id[donor].labels
    recv_labels = state.by_id[recv].labels
    harm = 0.0
    for label, quota in state.quotas.items():
        if not quota.formal:
            continue
        for split in SPLITS:
            count = state.observed[label][split]
            if label in donor_labels:
                if split == donor_split:
                    count -= 1
                elif split == recv_split:
                    count += 1
            if label in recv_labels:
                if split == recv_split:
                    count -= 1
                elif split == donor_split:
                    count += 1
            harm += relative_deviation(count, quota.targets[split])
    return harm


def _rank_train_recv_candidates(
    state: RefinementState,
    donor: str,
    config: SplitConfig,
    seed: int,
    *,
    top_k: int,
    allow_critical: bool,
) -> list[str]:
    safe = [
        page_id
        for page_id, split in state.assignment.items()
        if split == "train" and state.is_safe_to_remove_from_train(page_id)
    ]
    pool = safe
    if allow_critical and not pool:
        pool = [page_id for page_id, split in state.assignment.items() if split == "train"]

    ranked = sorted(
        pool,
        key=lambda recv: (
            -_label_similarity(state, donor, recv),
            _quota_harm(state, donor, recv),
            sort_key_for_page(config, seed, recv),
        ),
    )
    return ranked[:top_k]


def residual_vocab_repair(
    assignment: dict[str, str],
    page_labels: list[PageLabels],
    token_counts: pd.DataFrame,
    features: pd.DataFrame,
    config: SplitConfig,
    quotas: dict[str, LabelQuota],
    *,
    seed: int,
    show_progress: bool = True,
    progress: Progress | None = None,
) -> tuple[dict[str, str], VocabAudit]:
    del show_progress
    if not config.require_train_vocab_coverage:
        return dict(assignment), vocabulary_audit(token_counts, assignment, features)

    state = RefinementState.from_assignment(
        assignment,
        page_labels,
        features,
        token_counts,
        quotas,
    )
    initial_unseen = state.unseen_count()
    max_iters = max(initial_unseen, 1)
    top_k = config.repair_top_k_train

    task_id = None
    if progress is not None:
        task_id = progress.add_task(
            f"Residual vocab repair (unseen={initial_unseen})",
            total=max_iters,
        )

    for iteration in range(max_iters):
        unseen_n = state.unseen_count()
        if task_id is not None and progress is not None:
            progress.update(
                task_id,
                completed=min(iteration, max_iters),
                description=f"Residual vocab repair (unseen={unseen_n}, iter={iteration + 1})",
            )
        if unseen_n == 0:
            break

        unseen = state.unseen_token_set()
        donor_fixable: dict[str, set[str]] = {}
        for page_id, split in state.assignment.items():
            if split not in {"val", "test"}:
                continue
            fixable = state.page_tokens.get(page_id, set()) & unseen
            if fixable:
                donor_fixable[page_id] = fixable

        if not donor_fixable:
            break

        donors = sorted(
            donor_fixable,
            key=lambda page_id: (
                -len(donor_fixable[page_id]),
                sort_key_for_page(config, seed, page_id),
            ),
        )

        best_move: tuple[int, tuple[int | str, ...], tuple[int | str, ...], str, str] | None = None
        for allow_critical in (False, True):
            for donor in donors:
                recvs = _rank_train_recv_candidates(
                    state,
                    donor,
                    config,
                    seed,
                    top_k=top_k,
                    allow_critical=allow_critical,
                )
                for recv in recvs:
                    delta = state.vocab_move_delta(donor, recv)
                    if delta >= 0:
                        continue
                    candidate = (
                        delta,
                        sort_key_for_page(config, seed, recv),
                        sort_key_for_page(config, seed, donor),
                        donor,
                        recv,
                    )
                    if best_move is None or candidate < best_move:
                        best_move = candidate
            if best_move is not None:
                break

        if best_move is None:
            break
        _, _, _, donor, recv = best_move
        state.apply_vocab_move(donor, recv)

    if task_id is not None and progress is not None:
        progress.update(
            task_id,
            completed=max_iters,
            description=f"Residual vocab repair done (unseen={state.unseen_count()})",
        )
        progress.remove_task(task_id)

    return state.assignment, vocabulary_audit(token_counts, state.assignment, features)


def repair_vocab_coverage(
    assignment: dict[str, str],
    page_labels: list[PageLabels],
    token_counts: pd.DataFrame,
    features: pd.DataFrame,
    config: SplitConfig,
    quotas: dict[str, LabelQuota],
    *,
    seed: int,
    show_progress: bool = True,
    progress: Progress | None = None,
    workers: int = 1,
) -> tuple[dict[str, str], VocabAudit]:
    del workers
    return residual_vocab_repair(
        assignment,
        page_labels,
        token_counts,
        features,
        config,
        quotas,
        seed=seed,
        show_progress=show_progress,
        progress=progress,
    )


def global_swap_refine(
    assignment: dict[str, str],
    page_labels: list[PageLabels],
    features: pd.DataFrame,
    token_counts: pd.DataFrame,
    quotas: dict[str, LabelQuota],
    config: SplitConfig,
    *,
    seed: int,
    show_progress: bool = True,
    progress: Progress | None = None,
    workers: int = 1,
) -> dict[str, str]:
    """Coverage-preserving swaps guided by formal-label deficits."""
    del show_progress, workers

    state = RefinementState.from_assignment(
        assignment,
        page_labels,
        features,
        token_counts,
        quotas,
    )
    by_id = state.by_id
    best = state.lexicographic_objective()
    max_iters = config.global_swap_max_iterations
    task_id = None
    if progress is not None:
        task_id = progress.add_task(
            f"Global swap (max_dev={best[2]:.3f})",
            total=max_iters,
        )

    for iteration in range(max_iters):
        if task_id is not None and progress is not None:
            progress.update(
                task_id,
                completed=iteration + 1,
                description=(
                    f"Global swap iter={iteration + 1} "
                    f"unseen={best[1]} max_dev={best[2]:.3f}"
                ),
            )
        if best[0] == 0 and best[2] <= config.max_relative_deviation_tolerance:
            break

        worst: list[tuple[float, str, str]] = []
        for label, quota in quotas.items():
            if not quota.formal:
                continue
            for split in SPLITS:
                dev = relative_deviation(state.observed[label][split], quota.targets[split])
                worst.append((dev, label, split))
        worst.sort(reverse=True)
        if not worst or (worst[0][0] <= config.max_relative_deviation_tolerance and best[1] == 0):
            break

        improved = False
        for _dev, label, over_or_under_split in worst[:12]:
            target = quotas[label].targets[over_or_under_split]
            filled = state.observed[label][over_or_under_split]
            if filled > target:
                donor_split = over_or_under_split
                recv_splits = [
                    split_name
                    for split_name in SPLITS
                    if split_name != donor_split
                    and state.observed[label][split_name] < quotas[label].targets[split_name]
                ]
            elif filled < target:
                recv_splits = [over_or_under_split]
                donor_split = None
                for split_name in SPLITS:
                    if (
                        split_name != over_or_under_split
                        and state.observed[label][split_name] > quotas[label].targets[split_name]
                    ):
                        donor_split = split_name
                        break
                if donor_split is None:
                    continue
            else:
                continue

            donors = sorted(
                page_id
                for page_id, split in state.assignment.items()
                if split == donor_split and label in by_id[page_id].labels
            )[:40]
            recvs_pool: list[str] = []
            for recv_split in recv_splits:
                recvs_pool.extend(
                    sorted(
                        page_id
                        for page_id, split in state.assignment.items()
                        if split == recv_split and label not in by_id[page_id].labels
                    )
                )
            if not recvs_pool:
                recvs_pool = sorted(
                    page_id for page_id, split in state.assignment.items() if split in set(recv_splits)
                )
            recvs_pool = recvs_pool[:40]

            pairs = [
                (donor, recv)
                for donor in donors
                for recv in recvs_pool
                if state.assignment[donor] != state.assignment[recv]
            ]
            if not pairs:
                continue

            scored: list[tuple[tuple[int, int, float, float, float], tuple, str, str]] = []
            for donor, recv in pairs:
                score = state.score_swap(donor, recv)
                if score is None:
                    continue
                scored.append(
                    (
                        score,
                        (
                            sort_key_for_page(config, seed, donor),
                            sort_key_for_page(config, seed, recv),
                        ),
                        donor,
                        recv,
                    )
                )
            if not scored:
                continue
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            score, _, donor, recv = scored[0]
            if score < best:
                state.apply_swap(donor, recv)
                best = score
                improved = True
                break
        if not improved:
            break

    if task_id is not None and progress is not None:
        progress.update(
            task_id,
            completed=max_iters,
            description=f"Global swap done (max_dev={best[2]:.3f})",
        )
        progress.remove_task(task_id)

    return state.assignment
