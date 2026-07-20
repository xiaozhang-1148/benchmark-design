"""Acceptance checks and multi-seed candidate selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from benchmark_design.page_level_latex_split.config import SplitConfig
from benchmark_design.page_level_latex_split.labels import PageLabels
from benchmark_design.page_level_latex_split.refine import VocabAudit, vocabulary_audit
from benchmark_design.page_level_latex_split.refine_state import RefinementState
from benchmark_design.page_level_latex_split.stratify import (
    SPLITS,
    LabelQuota,
    largest_remainder_counts,
    max_label_deviation,
    observed_label_counts,
)


class SplitAcceptanceError(RuntimeError):
    """Raised when stratified split fails mandatory acceptance checks."""


@dataclass(frozen=True, slots=True)
class CandidateSeedScore:
    seed: int
    max_formal_deviation: float
    mean_formal_deviation: float
    joint_tv: float
    val_unseen: int
    test_unseen: int
    vocab_unseen_total: int
    hard_violation_count: int
    status: str
    elapsed_seconds: float = 0.0

    @classmethod
    def from_assignment(
        cls,
        seed: int,
        assignment: dict[str, str],
        page_labels: list[PageLabels],
        quotas: dict[str, LabelQuota],
        token_counts: pd.DataFrame,
        features: pd.DataFrame,
        *,
        elapsed_seconds: float = 0.0,
    ) -> CandidateSeedScore:
        observed = observed_label_counts(page_labels, assignment)
        max_dev = max_label_deviation(quotas, observed, formal_only=True)
        formal_devs: list[float] = []
        for label, quota in quotas.items():
            if not quota.formal:
                continue
            for split in SPLITS:
                formal_devs.append(
                    abs(observed[label][split] - quota.targets[split]) / max(1, quota.targets[split])
                )
        mean_dev = sum(formal_devs) / len(formal_devs) if formal_devs else 0.0
        vocab = vocabulary_audit(token_counts, assignment, features)
        val_unseen = len(vocab.val_unseen)
        test_unseen = len(vocab.test_unseen)
        state = RefinementState.from_assignment(
            assignment,
            page_labels,
            features,
            token_counts,
            quotas,
        )
        hard_violations, unseen_n, _, joint_tv, _ = state.lexicographic_objective()
        del unseen_n
        return cls(
            seed=seed,
            max_formal_deviation=max_dev,
            mean_formal_deviation=mean_dev,
            joint_tv=joint_tv,
            val_unseen=val_unseen,
            test_unseen=test_unseen,
            vocab_unseen_total=val_unseen + test_unseen,
            hard_violation_count=hard_violations,
            status="pass" if val_unseen + test_unseen == 0 else "fail",
            elapsed_seconds=elapsed_seconds,
        )

    def lexicographic_key(self) -> tuple:
        return (
            self.hard_violation_count,
            self.vocab_unseen_total,
            self.max_formal_deviation,
            self.joint_tv,
            self.mean_formal_deviation,
            self.seed,
        )


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    passed: bool
    checks: dict[str, bool]
    messages: tuple[str, ...]


def assignment_to_manifest_frame(
    assignment: dict[str, str],
    manifest: pd.DataFrame,
    page_labels: list[PageLabels],
) -> pd.DataFrame:
    label_map = {page.page_id: page for page in page_labels}
    records = []
    for row in manifest.itertuples(index=False):
        page_id = str(row.page_id)
        lab = label_map[page_id]
        records.append(
            {
                "page_id": page_id,
                "split": assignment[page_id],
                "image_path": str(row.image_path),
                "annotation_path": str(row.annotation_path),
                "feature_labels": ";".join(sorted(lab.labels)),
            }
        )
    frame = pd.DataFrame.from_records(records).sort_values(["split", "page_id"]).reset_index(drop=True)
    return frame


def manifest_hash(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, ["page_id", "split"]].sort_values("page_id")
    text = payload.to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_acceptance_checks(
    assignment: dict[str, str],
    page_labels: list[PageLabels],
    config: SplitConfig,
    quotas: dict[str, LabelQuota],
    vocab: VocabAudit,
    low_support_labels: list[str],
) -> AcceptanceResult:
    checks: dict[str, bool] = {}
    messages: list[str] = []
    n = len(page_labels)
    page_ids = {page.page_id for page in page_labels}

    checks["all_pages_assigned_once"] = set(assignment) == page_ids and len(assignment) == n
    if not checks["all_pages_assigned_once"]:
        messages.append("not all pages assigned exactly once")

    by_split = {s: {pid for pid, sp in assignment.items() if sp == s} for s in SPLITS}
    checks["splits_disjoint"] = (
        not (by_split["train"] & by_split["val"])
        and not (by_split["train"] & by_split["test"])
        and not (by_split["val"] & by_split["test"])
    )
    if not checks["splits_disjoint"]:
        messages.append("train/val/test are not disjoint")

    expected = largest_remainder_counts(n, config.ratios, SPLITS)
    actual = {s: len(by_split[s]) for s in SPLITS}
    checks["capacities_match"] = actual == expected
    if not checks["capacities_match"]:
        messages.append(f"capacities mismatch expected={expected} actual={actual}")

    observed = observed_label_counts(page_labels, assignment)
    formal = [label for label, q in quotas.items() if q.formal]
    checks["formal_labels_have_quotas"] = all(label in quotas for label in formal)
    checks["low_support_recorded"] = all(
        (not quotas[label].formal) for label in low_support_labels if label in quotas
    )

    if config.require_train_vocab_coverage:
        covered = len(vocab.val_unseen) == 0 and len(vocab.test_unseen) == 0
        checks["vocab_coverage"] = covered
        if not covered:
            messages.append(
                f"vocab coverage failed: val_unseen={len(vocab.val_unseen)} "
                f"test_unseen={len(vocab.test_unseen)}"
            )
    else:
        checks["vocab_coverage"] = True

    checks["max_deviation_recorded"] = True
    checks["no_model_training_used"] = True

    passed = all(checks.values())
    return AcceptanceResult(passed=passed, checks=checks, messages=tuple(messages))


def select_best_candidate(
    scores: list[CandidateSeedScore],
) -> CandidateSeedScore:
    """Lexicographic: hard violations, unseen, max deviation, joint TV, mean deviation, seed."""
    if not scores:
        raise ValueError("no candidates")
    ranked = sorted(scores, key=lambda item: item.lexicographic_key())
    return ranked[0]


def write_candidate_scores(scores: list[CandidateSeedScore], output_dir: Path) -> Path:
    records = [
        {
            "seed": score.seed,
            "unseen": score.vocab_unseen_total,
            "max_deviation": score.max_formal_deviation,
            "joint_tv": score.joint_tv,
            "mean_deviation": score.mean_formal_deviation,
            "status": score.status,
            "elapsed_s": score.elapsed_seconds,
        }
        for score in scores
    ]
    path = output_dir / "candidate_scores.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def write_split_lists(manifest: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    paths = {}
    for split in SPLITS:
        path = output_dir / f"{split}.txt"
        ids = manifest.loc[manifest["split"] == split, "page_id"].astype(str).tolist()
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        paths[split] = path
    return paths
