"""Incremental assignment state for fast vocab repair and swap scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from benchmark_design.page_level_latex_split.labels import PageLabels
from benchmark_design.page_level_latex_split.stratify import (
    SPLITS,
    LabelQuota,
    max_label_deviation,
    relative_deviation,
)


def page_tokens_index(token_counts: pd.DataFrame) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    if token_counts.empty:
        return dict(index)
    for row in token_counts.itertuples(index=False):
        index[str(row.page_id)].add(str(row.token))
    return dict(index)


def structure_keys(features: pd.DataFrame) -> dict[str, tuple[int, int]]:
    keys: dict[str, tuple[int, int]] = {}
    for row in features.itertuples(index=False):
        page_id = str(row.page_id)
        keys[page_id] = (int(row.structure_type_count), min(int(row.max_ast_depth), 5))
    return keys


@dataclass
class RefinementState:
    """Mutable split assignment with cached counters for O(token) scoring."""

    assignment: dict[str, str]
    page_tokens: dict[str, set[str]]
    by_id: dict[str, PageLabels]
    page_structure_key: dict[str, tuple[int, int]]
    quotas: dict[str, LabelQuota]
    train_token_page_count: Counter[str]
    split_vocabs: dict[str, set[str]]
    observed: dict[str, Counter[str]]
    joint_overall: Counter[tuple[int, int]]
    joint_by_split: dict[str, Counter[tuple[int, int]]]
    _unseen_n: int = 0

    @classmethod
    def from_assignment(
        cls,
        assignment: dict[str, str],
        page_labels: list[PageLabels],
        features: pd.DataFrame,
        token_counts: pd.DataFrame,
        quotas: dict[str, LabelQuota],
    ) -> RefinementState:
        page_tokens = page_tokens_index(token_counts)
        by_id = {page.page_id: page for page in page_labels}
        page_structure_key = structure_keys(features)

        train_token_page_count: Counter[str] = Counter()
        split_vocabs: dict[str, set[str]] = {split: set() for split in SPLITS}
        observed: dict[str, Counter[str]] = defaultdict(Counter)
        joint_overall: Counter[tuple[int, int]] = Counter()
        joint_by_split: dict[str, Counter[tuple[int, int]]] = {split: Counter() for split in SPLITS}

        for page_id, split in assignment.items():
            tokens = page_tokens.get(page_id, set())
            split_vocabs[split].update(tokens)
            if split == "train":
                for token in tokens:
                    train_token_page_count[token] += 1
            for label in by_id[page_id].labels:
                observed[label][split] += 1
            key = page_structure_key[page_id]
            joint_overall[key] += 1
            joint_by_split[split][key] += 1

        holdout = split_vocabs["val"] | split_vocabs["test"]
        unseen_n = sum(1 for token in holdout if train_token_page_count.get(token, 0) <= 0)

        return cls(
            assignment=dict(assignment),
            page_tokens=page_tokens,
            by_id=by_id,
            page_structure_key=page_structure_key,
            quotas=quotas,
            train_token_page_count=train_token_page_count,
            split_vocabs=split_vocabs,
            observed=dict(observed),
            joint_overall=joint_overall,
            joint_by_split=joint_by_split,
            _unseen_n=unseen_n,
        )

    @property
    def n_pages(self) -> int:
        return len(self.assignment)

    def unseen_count(self) -> int:
        return self._unseen_n

    def unseen_token_set(self) -> set[str]:
        holdout = self._holdout_vocab()
        return {token for token in holdout if self.train_token_page_count.get(token, 0) <= 0}

    def structure_joint_tv_distance(
        self,
        by_split: dict[str, Counter[tuple[int, int]]] | None = None,
    ) -> float:
        """Total-variation-style distance for structure_count x depth joint distribution."""
        by_split = by_split or self.joint_by_split
        n = max(1, self.n_pages)
        ratios = {
            split: sum(1 for value in self.assignment.values() if value == split) / n for split in SPLITS
        }
        tv = 0.0
        for key, count in self.joint_overall.items():
            overall_p = count / n
            for split in SPLITS:
                target = overall_p * ratios[split]
                observed = by_split[split][key] / n
                tv += abs(observed - target)
        return 0.5 * tv

    def vocab_move_delta(self, donor: str, recv: str) -> int:
        """Local unseen-token delta for moving donor to train and recv to donor's split."""
        unseen = self.unseen_token_set()
        donor_tokens = self.page_tokens.get(donor, set())
        recv_tokens = self.page_tokens.get(recv, set())
        fixed = len(unseen & donor_tokens)
        created = sum(
            1
            for token in recv_tokens
            if self.train_token_page_count.get(token, 0) == 1 and token not in donor_tokens
        )
        return created - fixed

    def is_safe_to_remove_from_train(self, page_id: str) -> bool:
        return all(
            self.train_token_page_count.get(token, 0) > 1 for token in self.page_tokens.get(page_id, set())
        )

    def lexicographic_objective(
        self,
        *,
        observed: dict[str, Counter[str]] | None = None,
        by_split: dict[str, Counter[tuple[int, int]]] | None = None,
    ) -> tuple[int, int, float, float, float]:
        observed = observed or self.observed
        by_split = by_split or self.joint_by_split
        hard_violations = 0
        if self.unseen_count() > 0:
            hard_violations += 1
        max_dev, mean_dev = self._label_objective(observed)
        joint_tv = self.structure_joint_tv_distance(by_split)
        return (
            hard_violations,
            self.unseen_count(),
            max_dev,
            joint_tv,
            mean_dev,
        )

    def _unseen_delta(self, token: str, delta: int, holdout: set[str]) -> int:
        if token not in holdout or delta == 0:
            return 0
        old = self.train_token_page_count.get(token, 0)
        new = old + delta
        if old <= 0 < new:
            return -1
        if old > 0 >= new:
            return 1
        return 0

    def _holdout_vocab(self) -> set[str]:
        return self.split_vocabs["val"] | self.split_vocabs["test"]

    def _joint_cost_from(self, by_split: dict[str, Counter[tuple[int, int]]]) -> float:
        n = self.n_pages
        ratios = {split: sum(1 for value in self.assignment.values() if value == split) / n for split in SPLITS}
        cost = 0.0
        for key, count in self.joint_overall.items():
            for split in SPLITS:
                target = count * ratios[split]
                cost += abs(by_split[split][key] - target) / max(1.0, target)
        return cost

    def _label_objective(self, observed: dict[str, Counter[str]]) -> tuple[float, float]:
        max_dev = max_label_deviation(self.quotas, observed, formal_only=True)
        formal_devs: list[float] = []
        for label, quota in self.quotas.items():
            if not quota.formal:
                continue
            for split in SPLITS:
                formal_devs.append(relative_deviation(observed[label][split], quota.targets[split]))
        mean_dev = sum(formal_devs) / len(formal_devs) if formal_devs else 0.0
        return max_dev, mean_dev

    def objective(self) -> tuple[float, float, float, float]:
        max_dev, mean_dev = self._label_objective(self.observed)
        return (
            float(self.unseen_count()),
            max_dev,
            mean_dev,
            self._joint_cost_from(self.joint_by_split),
        )

    def _recompute_unseen(self) -> None:
        holdout = self._holdout_vocab()
        self._unseen_n = sum(1 for token in holdout if self.train_token_page_count.get(token, 0) <= 0)

    def unseen_after_vocab_move(self, donor: str, recv: str) -> int:
        """Score moving donor to train and recv to donor's original split."""
        holdout_before = self._holdout_vocab()
        donor_tokens = self.page_tokens.get(donor, set())
        recv_tokens = self.page_tokens.get(recv, set())
        counts = self.train_token_page_count
        unseen = self._unseen_n

        for token in donor_tokens:
            if token in holdout_before and counts.get(token, 0) == 0:
                unseen -= 1

        for token in recv_tokens:
            old_count = counts.get(token, 0)
            new_count = old_count - 1
            if token in holdout_before:
                if old_count > 0 and new_count <= 0:
                    unseen += 1
            elif new_count <= 0:
                unseen += 1

        return unseen

    def unseen_after_swap(self, donor: str, recv: str) -> int:
        donor_split = self.assignment[donor]
        recv_split = self.assignment[recv]
        if donor_split == recv_split:
            return self.unseen_count()

        holdout = self._holdout_vocab()
        delta = 0
        for token in self.page_tokens.get(donor, set()):
            token_delta = (1 if recv_split == "train" else 0) - (1 if donor_split == "train" else 0)
            delta += self._unseen_delta(token, token_delta, holdout)
        for token in self.page_tokens.get(recv, set()):
            token_delta = (1 if donor_split == "train" else 0) - (1 if recv_split == "train" else 0)
            delta += self._unseen_delta(token, token_delta, holdout)
        return self._unseen_n + delta

    def _simulated_joint_by_split_after_swap(
        self,
        donor: str,
        recv: str,
    ) -> dict[str, Counter[tuple[int, int]]]:
        donor_split = self.assignment[donor]
        recv_split = self.assignment[recv]
        donor_key = self.page_structure_key[donor]
        recv_key = self.page_structure_key[recv]
        by_split = {split: Counter(counts) for split, counts in self.joint_by_split.items()}
        by_split[donor_split][donor_key] -= 1
        by_split[recv_split][donor_key] += 1
        by_split[recv_split][recv_key] -= 1
        by_split[donor_split][recv_key] += 1
        return by_split

    def score_swap(self, donor: str, recv: str) -> tuple[int, int, float, float, float] | None:
        """Lexicographic swap score; None when hard constraints would be violated."""
        if self.unseen_after_swap(donor, recv) > 0:
            return None

        donor_split = self.assignment[donor]
        recv_split = self.assignment[recv]
        donor_labels = self.by_id[donor].labels
        recv_labels = self.by_id[recv].labels

        simulated_observed: dict[str, Counter[str]] = {
            label: Counter(counts) for label, counts in self.observed.items()
        }
        for label in self.quotas:
            if label in donor_labels:
                simulated_observed[label][donor_split] -= 1
                simulated_observed[label][recv_split] += 1
            if label in recv_labels:
                simulated_observed[label][recv_split] -= 1
                simulated_observed[label][donor_split] += 1

        simulated_joint = self._simulated_joint_by_split_after_swap(donor, recv)
        return self.lexicographic_objective(observed=simulated_observed, by_split=simulated_joint)

    def _token_in_holdout(self, token: str) -> bool:
        return token in self.split_vocabs["val"] or token in self.split_vocabs["test"]

    def _move_page(self, page_id: str, from_split: str, to_split: str) -> None:
        if from_split == to_split:
            return
        tokens = self.page_tokens.get(page_id, set())
        labels = self.by_id[page_id].labels
        key = self.page_structure_key[page_id]

        snapshots = {
            token: (self._token_in_holdout(token), self.train_token_page_count.get(token, 0))
            for token in tokens
        }

        for token in tokens:
            self.split_vocabs[from_split].discard(token)
            self.split_vocabs[to_split].add(token)
            if from_split == "train":
                self.train_token_page_count[token] -= 1
                if self.train_token_page_count[token] <= 0:
                    del self.train_token_page_count[token]
            if to_split == "train":
                self.train_token_page_count[token] += 1

        for token, (in_holdout_before, train_before) in snapshots.items():
            in_holdout_after = self._token_in_holdout(token)
            train_after = self.train_token_page_count.get(token, 0)
            was_unseen = in_holdout_before and train_before <= 0
            now_unseen = in_holdout_after and train_after <= 0
            if was_unseen and not now_unseen:
                self._unseen_n -= 1
            elif not was_unseen and now_unseen:
                self._unseen_n += 1

        for label in labels:
            self.observed[label][from_split] -= 1
            self.observed[label][to_split] += 1

        self.joint_by_split[from_split][key] -= 1
        if self.joint_by_split[from_split][key] == 0:
            del self.joint_by_split[from_split][key]
        self.joint_by_split[to_split][key] += 1

        self.assignment[page_id] = to_split

    def apply_vocab_move(self, donor: str, recv: str) -> None:
        donor_split = self.assignment[donor]
        self._move_page(donor, donor_split, "train")
        self._move_page(recv, "train", donor_split)

    def apply_swap(self, donor: str, recv: str) -> None:
        donor_split = self.assignment[donor]
        recv_split = self.assignment[recv]
        if donor_split == recv_split:
            return
        self._move_page(donor, donor_split, recv_split)
        self._move_page(recv, recv_split, donor_split)
