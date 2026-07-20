"""Capacity, quotas, low-support filtering, and iterative multilabel stratification."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from benchmark_design.page_level_latex_split.config import SplitConfig
from benchmark_design.page_level_latex_split.labels import PageLabels
from benchmark_design.page_level_latex_split.tie_break import sort_key_for_page

SPLITS = ("train", "val", "test")


def largest_remainder_counts(total: int, ratios: dict[str, float], order: tuple[str, ...] = SPLITS) -> dict[str, int]:
    """Allocate integer capacities with fixed largest-remainder rule."""
    raw = {name: ratios[name] * total for name in order}
    floors = {name: int(math.floor(value)) for name, value in raw.items()}
    assigned = sum(floors.values())
    remainder = total - assigned
    frac_order = sorted(
        order,
        key=lambda name: (-(raw[name] - floors[name]), order.index(name)),
    )
    counts = dict(floors)
    for name in frac_order[:remainder]:
        counts[name] += 1
    if sum(counts.values()) != total:
        raise RuntimeError("largest remainder failed to conserve total")
    return counts


@dataclass(frozen=True, slots=True)
class LabelQuota:
    label: str
    support: int
    formal: bool
    reason: str
    targets: dict[str, int]  # split -> quota
    expected_raw: dict[str, float]


@dataclass
class StratifyState:
    assignment: dict[str, str] = field(default_factory=dict)  # page_id -> split
    remaining_capacity: dict[str, int] = field(default_factory=dict)
    filled: dict[str, Counter] = field(default_factory=dict)  # label -> Counter(split)
    formal_labels: list[str] = field(default_factory=list)
    quotas: dict[str, LabelQuota] = field(default_factory=dict)


def compute_label_support(page_labels: list[PageLabels]) -> Counter[str]:
    support: Counter[str] = Counter()
    for page in page_labels:
        for label in page.labels:
            support[label] += 1
    return support


def classify_formal_labels(
    support: Counter[str],
    capacities: dict[str, int],
    config: SplitConfig,
) -> dict[str, LabelQuota]:
    del capacities  # Capacity is enforced during assignment; quotas use label support.
    ratios = config.ratios
    quotas: dict[str, LabelQuota] = {}
    for label, count in sorted(support.items()):
        expected_raw = {split: ratios[split] * count for split in SPLITS}
        targets = largest_remainder_counts(count, ratios, SPLITS)
        formal = True
        reasons: list[str] = []
        if count < config.min_support_pages:
            formal = False
            reasons.append(f"support<{config.min_support_pages}")
        if any(targets[s] < config.min_expected_per_split for s in SPLITS):
            formal = False
            reasons.append("expected_per_split_too_small")
        elif any(expected_raw[s] < config.min_expected_per_split for s in ("val", "test")):
            formal = False
            reasons.append("expected_per_split_too_small")
        quotas[label] = LabelQuota(
            label=label,
            support=count,
            formal=formal,
            reason=";".join(reasons) if reasons else "ok",
            targets=targets,
            expected_raw=expected_raw,
        )
    return quotas


def relative_deviation(observed: int, target: int) -> float:
    return abs(observed - target) / max(1, target)


def _split_score(
    split: str,
    page: PageLabels,
    state: StratifyState,
    active_label: str,
) -> tuple:
    """Higher is better. Prefer largest related-label deficit among splits with capacity."""
    if state.remaining_capacity.get(split, 0) <= 0:
        return (-10**18, split)
    deficits = []
    for label in page.labels:
        quota = state.quotas.get(label)
        if quota is None or not quota.formal:
            continue
        target = quota.targets[split]
        filled = state.filled[label][split]
        deficits.append(target - filled)
    primary = max(deficits) if deficits else 0
    # Prefer closing the active label deficit.
    active_quota = state.quotas[active_label]
    active_deficit = active_quota.targets[split] - state.filled[active_label][split]
    return (active_deficit, primary, -SPLITS.index(split))


def iterative_multilabel_assign(
    page_labels: list[PageLabels],
    config: SplitConfig,
    *,
    seed: int,
    locked_assignment: dict[str, str] | None = None,
    adjusted_quotas: dict[str, LabelQuota] | None = None,
    remaining_capacity: dict[str, int] | None = None,
    initial_filled: dict[str, Counter[str]] | None = None,
) -> StratifyState:
    n = len(page_labels)
    capacities = largest_remainder_counts(n, config.ratios, SPLITS)
    support = compute_label_support(page_labels)
    quotas = adjusted_quotas or classify_formal_labels(support, capacities, config)
    formal_labels = [label for label, q in quotas.items() if q.formal]
    locked = dict(locked_assignment or {})

    if remaining_capacity is not None:
        rem_cap = dict(remaining_capacity)
    elif locked:
        rem_cap = dict(capacities)
        for split in SPLITS:
            rem_cap[split] = capacities[split] - sum(1 for sp in locked.values() if sp == split)
    else:
        rem_cap = dict(capacities)

    filled: dict[str, Counter[str]] = {}
    for label in quotas:
        filled[label] = Counter(initial_filled.get(label, Counter()) if initial_filled else Counter())

    state = StratifyState(
        assignment=dict(locked),
        remaining_capacity=rem_cap,
        filled=filled,
        formal_labels=sorted(formal_labels),
        quotas=quotas,
    )

    unassigned = {
        page.page_id: page for page in page_labels if page.page_id not in locked
    }

    # Process formal labels rare-first (remaining unmet support of unassigned pages).
    def remaining_support(label: str) -> int:
        return sum(1 for page in unassigned.values() if label in page.labels)

    pending_labels = set(formal_labels)
    while pending_labels and unassigned:
        # Choose unmet label with smallest remaining support among those still short.
        candidates = []
        for label in pending_labels:
            rem = remaining_support(label)
            if rem <= 0:
                continue
            # Still need any split quota?
            need = False
            for split in SPLITS:
                if state.filled[label][split] < state.quotas[label].targets[split]:
                    if state.remaining_capacity[split] > 0:
                        need = True
                        break
            if need:
                candidates.append((rem, label))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        _rem, active = candidates[0]

        carriers = [page for page in unassigned.values() if active in page.labels]
        carriers.sort(key=lambda page: sort_key_for_page(config, seed, page.page_id))

        for page in carriers:
            if page.page_id not in unassigned:
                continue
            scored = []
            for split in SPLITS:
                score = _split_score(split, page, state, active)
                scored.append((score, split))
            scored.sort(reverse=True)
            best_score, best_split = scored[0]
            if best_score[0] <= -10**17:
                continue
            # Assign
            state.assignment[page.page_id] = best_split
            state.remaining_capacity[best_split] -= 1
            for label in page.labels:
                state.filled[label][best_split] += 1
            del unassigned[page.page_id]

        # Drop active if quotas met or no carriers left.
        if remaining_support(active) == 0 or all(
            state.filled[active][s] >= state.quotas[active].targets[s]
            or state.remaining_capacity[s] <= 0
            for s in SPLITS
        ):
            pending_labels.discard(active)

    # Assign remaining pages: fill residual capacity, rare labels first among leftovers.
    leftovers = sorted(
        unassigned.values(),
        key=lambda page: (len(page.labels), *sort_key_for_page(config, seed, page.page_id)),
    )
    for page in leftovers:
        # Prefer split with most remaining capacity; ties by split order then rng.
        options = [s for s in SPLITS if state.remaining_capacity[s] > 0]
        if not options:
            raise RuntimeError("no capacity left for remaining pages")
        options.sort(key=lambda s: (-state.remaining_capacity[s], SPLITS.index(s)))
        best_split = options[0]
        state.assignment[page.page_id] = best_split
        state.remaining_capacity[best_split] -= 1
        for label in page.labels:
            state.filled[label][best_split] += 1
        del unassigned[page.page_id]

    if unassigned:
        raise RuntimeError(f"failed to assign pages: {list(unassigned)[:5]}")
    if any(state.remaining_capacity[s] != 0 for s in SPLITS):
        raise RuntimeError(f"capacity not exhausted: {state.remaining_capacity}")
    return state


def observed_label_counts(
    page_labels: list[PageLabels],
    assignment: dict[str, str],
) -> dict[str, Counter[str]]:
    observed: dict[str, Counter[str]] = defaultdict(Counter)
    by_id = {page.page_id: page for page in page_labels}
    for page_id, split in assignment.items():
        for label in by_id[page_id].labels:
            observed[label][split] += 1
    return observed


def max_label_deviation(
    quotas: dict[str, LabelQuota],
    observed: dict[str, Counter[str]],
    *,
    formal_only: bool = True,
) -> float:
    worst = 0.0
    for label, quota in quotas.items():
        if formal_only and not quota.formal:
            continue
        for split in SPLITS:
            worst = max(worst, relative_deviation(observed[label][split], quota.targets[split]))
    return worst
