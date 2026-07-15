"""Deterministic A -> B -> A reference benchmark for rank migration.

The benchmark deliberately avoids a model download.  Each of four sites owns an
orthogonal diagonal basis.  A task's teacher is a rank-``k`` matrix at one hidden
site and the student can keep only ``k`` active rank-one factors in total (except
for the explicit ``extra_capacity`` control).

This is a mechanism test, not evidence that neural networks naturally organize
knowledge into these sites.  ``ConservedAllocator`` is intentionally small so a
backend using real LoRA modules can replace it without changing the experiment
or trajectory schema.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .benchmark_artifacts import write_benchmark_artifacts

CONDITIONS = (
    "vault",
    "recycle",
    "static",
    "fixed_split",
    "random",
    "oracle",
    "extra_capacity",
)
PHASES = ("learn_a", "learn_b", "return_a")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the cheap reference experiment."""

    seeds: tuple[int, ...] = tuple(range(10))
    site_count: int = 4
    task_rank: int = 2
    phase_steps: int = 8
    learning_rate: float = 0.5
    transfer_per_step: int = 1
    score_threshold: float = 0.9
    control_margin: float = 0.05
    bootstrap_resamples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 73

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique for paired fixture comparisons")
        if self.site_count < 2:
            raise ValueError("site_count must be at least 2")
        if self.task_rank < 1 or self.phase_steps < 1:
            raise ValueError("task_rank and phase_steps must be positive")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.transfer_per_step < 1:
            raise ValueError("transfer_per_step must be positive")
        if not 0.0 < self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in (0, 1]")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class Task:
    """A diagonal rank-k teacher located at one site."""

    name: str
    site: int
    coefficients: tuple[float, ...]
    targets: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Transfer:
    """One conserved rank-unit transfer."""

    donor_site: int
    donor_component: int
    target_site: int
    target_component: int
    parked: bool
    restored: bool
    reason: str


class Student:
    """Rank-slot student state in the orthogonal teacher basis."""

    def __init__(self, site_count: int, task_rank: int) -> None:
        self.site_count = site_count
        self.task_rank = task_rank
        self.weights = [[0.0] * task_rank for _ in range(site_count)]
        self.active = [set() for _ in range(site_count)]
        self.vault: dict[tuple[int, int], float] = {}

    def activate_site(self, site: int) -> None:
        self.active[site].update(range(self.task_rank))

    @property
    def active_rank(self) -> int:
        return sum(len(components) for components in self.active)

    @property
    def vaulted_rank(self) -> int:
        return len(self.vault)

    def rank_map(self) -> dict[str, int]:
        return {
            f"site_{site}": len(self.active[site])
            for site in range(self.site_count)
        }

    def score(self, task: Task, *, include_vault: bool = False) -> float:
        error = 0.0
        scale = 0.0
        for component, target in enumerate(task.coefficients):
            if component in self.active[task.site]:
                prediction = self.weights[task.site][component]
            elif include_vault:
                prediction = self.vault.get((task.site, component), 0.0)
            else:
                prediction = 0.0
            error += (prediction - target) ** 2
            scale += target**2
        return max(0.0, min(1.0, 1.0 - error / scale))

    def train(self, task: Task, learning_rate: float) -> None:
        for component in self.active[task.site]:
            target = task.coefficients[component]
            value = self.weights[task.site][component]
            self.weights[task.site][component] = value + learning_rate * (
                target - value
            )


class Allocator(Protocol):
    """Injection seam for a conserved-rank backend."""

    def move_toward(
        self,
        student: Student,
        *,
        target_site: int,
        max_transfers: int,
        park_released: bool,
        restore_target: bool,
        reason: str,
    ) -> list[Transfer]: ...

    def move_random(
        self,
        student: Student,
        *,
        max_transfers: int,
        rng: random.Random,
        donor_pool: list[tuple[int, int]],
        forbidden_targets: set[tuple[int, int]],
        reason: str,
    ) -> list[Transfer]: ...

    def move_by_signal(
        self,
        student: Student,
        *,
        recipient_signals: dict[tuple[int, int], float],
        max_transfers: int,
        park_released: bool,
        restore_target: bool,
        reason: str,
    ) -> list[Transfer]: ...


class ConservedAllocator:
    """Reference one-in/one-out allocator used by the toy benchmark."""

    @staticmethod
    def _apply(
        student: Student,
        donor: tuple[int, int],
        target: tuple[int, int],
        *,
        park_released: bool,
        restore_target: bool,
        reason: str,
    ) -> Transfer:
        before = student.active_rank
        donor_site, donor_component = donor
        target_site, target_component = target
        donor_value = student.weights[donor_site][donor_component]
        parked = park_released and abs(donor_value) > 1e-15
        if parked:
            student.vault[(donor_site, donor_component)] = donor_value
        else:
            student.vault.pop((donor_site, donor_component), None)
        student.active[donor_site].remove(donor_component)
        student.weights[donor_site][donor_component] = 0.0

        key = (target_site, target_component)
        restored = restore_target and key in student.vault
        if restored:
            student.weights[target_site][target_component] = student.vault.pop(key)
        else:
            student.vault.pop(key, None)
            student.weights[target_site][target_component] = 0.0
        student.active[target_site].add(target_component)
        if student.active_rank != before:
            raise RuntimeError("rank transfer violated the active-rank invariant")
        return Transfer(
            donor_site,
            donor_component,
            target_site,
            target_component,
            parked,
            restored,
            reason,
        )

    def move_toward(
        self,
        student: Student,
        *,
        target_site: int,
        max_transfers: int,
        park_released: bool,
        restore_target: bool,
        reason: str,
    ) -> list[Transfer]:
        events: list[Transfer] = []
        for _ in range(max_transfers):
            inactive = sorted(set(range(student.task_rank)) - student.active[target_site])
            if not inactive:
                break
            if restore_target:
                inactive.sort(
                    key=lambda component: (
                        (target_site, component) not in student.vault,
                        component,
                    )
                )
            donors = [
                (site, component)
                for site in range(student.site_count)
                if site != target_site
                for component in sorted(student.active[site])
            ]
            if not donors:
                break
            donor = min(
                donors,
                key=lambda item: (
                    abs(student.weights[item[0]][item[1]]),
                    item[0],
                    item[1],
                ),
            )
            events.append(
                self._apply(
                    student,
                    donor,
                    (target_site, inactive[0]),
                    park_released=park_released,
                    restore_target=restore_target,
                    reason=reason,
                )
            )
        return events

    def move_random(
        self,
        student: Student,
        *,
        max_transfers: int,
        rng: random.Random,
        donor_pool: list[tuple[int, int]],
        forbidden_targets: set[tuple[int, int]],
        reason: str,
    ) -> list[Transfer]:
        """Move the same donor capacity toward shuffled, non-reversing targets."""

        events: list[Transfer] = []
        for _ in range(max_transfers):
            donors = [
                item
                for item in donor_pool
                if item[1] in student.active[item[0]]
            ]
            if not donors:
                break
            donor = min(
                donors,
                key=lambda item: (
                    abs(student.weights[item[0]][item[1]]),
                    item[0],
                    item[1],
                ),
            )
            targets = [
                (site, component)
                for site in range(student.site_count)
                for component in range(student.task_rank)
                if component not in student.active[site]
                and (site, component) not in forbidden_targets
                and site != donor[0]
            ]
            if not targets:
                break
            target_site, target_component = rng.choice(sorted(targets))
            events.append(
                self._apply(
                    student,
                    donor,
                    (target_site, target_component),
                    park_released=False,
                    restore_target=False,
                    reason=reason,
                )
            )
            donor_pool.remove(donor)
        return events

    def move_by_signal(
        self,
        student: Student,
        *,
        recipient_signals: dict[tuple[int, int], float],
        max_transfers: int,
        park_released: bool,
        restore_target: bool,
        reason: str,
    ) -> list[Transfer]:
        """Transfer capacity toward a supplied counterfactual demand proxy.

        The caller supplies counterfactual gradient magnitudes, not task or site
        identifiers.  This is the reference seam a neural implementation must
        replace with measured gradient, validation-loss, or probe-activation gain.
        """

        events: list[Transfer] = []
        for _ in range(max_transfers):
            candidates = [
                (signal, site, component)
                for (site, component), signal in recipient_signals.items()
                if component not in student.active[site] and signal > 0.0
            ]
            if not candidates:
                break
            _, target_site, target_component = max(
                candidates,
                key=lambda item: (item[0], -item[1], -item[2]),
            )
            donors = [
                (site, component)
                for site in range(student.site_count)
                for component in sorted(student.active[site])
                if site != target_site
            ]
            if not donors:
                break
            donor = min(
                donors,
                key=lambda item: (
                    abs(student.weights[item[0]][item[1]]),
                    item[0],
                    item[1],
                ),
            )
            events.append(
                self._apply(
                    student,
                    donor,
                    (target_site, target_component),
                    park_released=park_released,
                    restore_target=restore_target,
                    reason=reason,
                )
            )
        return events


def _tasks(seed: int, config: BenchmarkConfig) -> tuple[Task, Task]:
    rng = random.Random(seed)
    site_a, site_b = rng.sample(range(config.site_count), 2)

    def coefficients() -> tuple[float, ...]:
        return tuple(
            (1.0 if rng.random() >= 0.5 else -1.0) * (1.0 + 0.2 * rng.random())
            for _ in range(config.task_rank)
        )

    def task(name: str, site: int) -> Task:
        values = coefficients()
        targets = tuple(
            values if candidate == site else (0.0,) * config.task_rank
            for candidate in range(config.site_count)
        )
        return Task(name, site, values, targets)

    return task("A", site_a), task("B", site_b)


def _counterfactual_gradient_signals(
    student: Student,
    task: Task,
) -> dict[tuple[int, int], float]:
    """Return counterfactual component-gradient magnitudes for the current data.

    The calculation consumes the dense target matrix rather than ``task.site``.
    In this orthogonal fixture it is nevertheless an unusually clean, component-
    level counterfactual gradient and should be treated as a routing upper bound.
    A learned-model backend must estimate a noisier analogue from a real loss.
    """

    signals: dict[tuple[int, int], float] = {}
    for site in range(student.site_count):
        for component in range(student.task_rank):
            if component in student.active[site]:
                signals[(site, component)] = 0.0
                continue
            target = task.targets[site][component]
            prediction = student.weights[site][component]
            signals[(site, component)] = abs(target - prediction)
    return signals


def _allocation_step(
    condition: str,
    phase: str,
    phase_step: int,
    task: Task,
    student: Student,
    allocator: Allocator,
    rng: random.Random,
    config: BenchmarkConfig,
    random_donor_pool: list[tuple[int, int]],
    random_forbidden_targets: set[tuple[int, int]],
) -> list[Transfer]:
    if phase == "learn_a" or condition in {"static", "fixed_split", "extra_capacity"}:
        return []
    if condition == "oracle":
        if phase_step:
            return []
        return allocator.move_toward(
            student,
            target_site=task.site,
            max_transfers=config.task_rank,
            park_released=False,
            restore_target=False,
            reason="oracle_phase_switch",
        )
    if condition == "vault":
        transfers = config.task_rank if phase == "return_a" and phase_step == 0 else config.transfer_per_step
        return allocator.move_by_signal(
            student,
            recipient_signals=_counterfactual_gradient_signals(student, task),
            max_transfers=transfers,
            park_released=True,
            restore_target=phase == "return_a",
            reason=(
                "counterfactual_gradient_then_wake"
                if phase == "return_a"
                else "counterfactual_gradient"
            ),
        )
    if condition == "recycle":
        return allocator.move_by_signal(
            student,
            recipient_signals=_counterfactual_gradient_signals(student, task),
            max_transfers=config.transfer_per_step,
            park_released=False,
            restore_target=False,
            reason="counterfactual_gradient_recycle",
        )
    if condition == "random":
        if phase_step * config.transfer_per_step >= config.task_rank:
            return []
        return allocator.move_random(
            student,
            max_transfers=min(
                config.transfer_per_step,
                config.task_rank - phase_step * config.transfer_per_step,
            ),
            rng=rng,
            donor_pool=random_donor_pool,
            forbidden_targets=random_forbidden_targets,
            reason="same_timing_shuffled_recipient_control",
        )
    raise ValueError(f"unknown condition: {condition}")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired vectors must be non-empty and equal length")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    bootstrap = sorted(
        _mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(resamples)
    )
    tail = (1.0 - confidence_level) / 2.0
    return {
        "method": "paired_fixture_seed_bootstrap",
        "pairs": len(differences),
        "mean_difference": _mean(differences),
        "confidence_level": confidence_level,
        "ci_lower": _percentile(bootstrap, tail),
        "ci_upper": _percentile(bootstrap, 1.0 - tail),
        "probability_left_better": sum(value > 0.0 for value in bootstrap) / resamples,
        "resamples": resamples,
        "seed": seed,
    }


def _steps_to_threshold(rows: Sequence[dict[str, Any]], field: str, threshold: float) -> int | None:
    if rows and rows[0][field.replace("score", "score_pre_update")] >= threshold:
        return 0
    for index, row in enumerate(rows, 1):
        if row[field] >= threshold:
            return index
    return None


def _scratch_reference(
    task: Task,
    *,
    starting_site: int,
    config: BenchmarkConfig,
    allocator: Allocator,
) -> dict[str, Any]:
    """Learn A without prior A state under the recycle condition's same schedule."""

    student = Student(config.site_count, config.task_rank)
    student.activate_site(starting_site)
    scores = []
    immediate_score = 0.0
    for step in range(config.phase_steps):
        allocator.move_by_signal(
            student,
            recipient_signals=_counterfactual_gradient_signals(student, task),
            max_transfers=config.transfer_per_step,
            park_released=False,
            restore_target=False,
            reason="scratch_counterfactual_gradient",
        )
        if step == 0:
            immediate_score = student.score(task)
        student.train(task, config.learning_rate)
        scores.append(student.score(task))
    return {
        "immediate_score": immediate_score,
        "score_auc": _mean(scores),
        "final_score": scores[-1],
        "steps_to_threshold": next(
            (index for index, score in enumerate(scores, 1) if score >= config.score_threshold),
            None,
        ),
    }


def _run_condition(
    seed: int,
    condition: str,
    config: BenchmarkConfig,
    allocator: Allocator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_a, task_b = _tasks(seed, config)
    student = Student(config.site_count, config.task_rank)
    budget = config.task_rank
    if condition == "fixed_split":
        a_rank = (config.task_rank + 1) // 2
        b_rank = config.task_rank - a_rank
        student.active[task_a.site].update(range(a_rank))
        student.active[task_b.site].update(range(b_rank))
    else:
        student.activate_site(task_a.site)
    if condition == "extra_capacity":
        student.activate_site(task_b.site)
        budget *= 2
    rng = random.Random(seed * 1009 + CONDITIONS.index(condition) * 9176)
    trajectory: list[dict[str, Any]] = []
    global_step = 0
    for phase, task in zip(PHASES, (task_a, task_b, task_a), strict=True):
        random_donor_pool = [
            (site, component)
            for site in range(student.site_count)
            for component in sorted(student.active[site])
        ]
        random_forbidden_targets = set(random_donor_pool)
        for phase_step in range(config.phase_steps):
            transfers = _allocation_step(
                condition,
                phase,
                phase_step,
                task,
                student,
                allocator,
                rng,
                config,
                random_donor_pool,
                random_forbidden_targets,
            )
            a_pre = student.score(task_a)
            b_pre = student.score(task_b)
            student.train(task, config.learning_rate)
            rank_map = student.rank_map()
            row = {
                "seed": seed,
                "condition": condition,
                "phase": phase,
                "phase_step": phase_step,
                "global_step": global_step,
                "task": task.name,
                "task_site": task.site,
                "rank_map": rank_map,
                "active_rank": student.active_rank,
                "active_rank_budget": budget,
                "budget_ok": student.active_rank == budget,
                "vaulted_rank": student.vaulted_rank,
                "resident_rank": student.active_rank + student.vaulted_rank,
                "target_rank_coverage": rank_map[f"site_{task.site}"] / config.task_rank,
                "a_score_pre_update": a_pre,
                "b_score_pre_update": b_pre,
                "a_score": student.score(task_a),
                "b_score": student.score(task_b),
                "a_latent_score": student.score(task_a, include_vault=True),
                "b_latent_score": student.score(task_b, include_vault=True),
                "transfers": [asdict(event) for event in transfers],
            }
            trajectory.append(row)
            global_step += 1

    by_phase = {
        phase: [row for row in trajectory if row["phase"] == phase]
        for phase in PHASES
    }
    b_rows = by_phase["learn_b"]
    return_rows = by_phase["return_a"]
    scratch = _scratch_reference(
        task_a,
        starting_site=task_b.site,
        config=config,
        allocator=allocator,
    )
    summary = {
        "seed": seed,
        "condition": condition,
        "task_a_site": task_a.site,
        "task_b_site": task_b.site,
        "active_rank_budget": budget,
        "budget_invariant": all(row["budget_ok"] for row in trajectory),
        "max_resident_rank": max(row["resident_rank"] for row in trajectory),
        "b_migrated_rank": b_rows[-1]["rank_map"][f"site_{task_b.site}"],
        "b_final_alignment": b_rows[-1]["target_rank_coverage"],
        "b_score_auc": _mean([row["b_score"] for row in b_rows]),
        "b_final_score": b_rows[-1]["b_score"],
        "a_score_end_b": b_rows[-1]["a_score"],
        "a_latent_score_end_b": b_rows[-1]["a_latent_score"],
        "a_return_immediate_score": return_rows[0]["a_score_pre_update"],
        "a_return_score_auc": _mean([row["a_score"] for row in return_rows]),
        "a_return_final_score": return_rows[-1]["a_score"],
        "a_return_steps_to_threshold": _steps_to_threshold(
            return_rows, "a_score", config.score_threshold
        ),
        "scratch_return_score_auc": scratch["score_auc"],
        "relearning_advantage_over_scratch": _mean(
            [row["a_score"] for row in return_rows]
        )
        - scratch["score_auc"],
    }
    return trajectory, summary


def _aggregate(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = (
        "b_migrated_rank",
        "b_final_alignment",
        "b_score_auc",
        "b_final_score",
        "a_score_end_b",
        "a_latent_score_end_b",
        "a_return_immediate_score",
        "a_return_score_auc",
        "a_return_final_score",
        "relearning_advantage_over_scratch",
        "max_resident_rank",
    )
    aggregates = []
    for condition in CONDITIONS:
        selected = [run for run in runs if run["condition"] == condition]
        row: dict[str, Any] = {
            "condition": condition,
            "seeds": len(selected),
            "budget_pass_rate": _mean([float(run["budget_invariant"]) for run in selected]),
        }
        row.update(
            {
                f"{field}_mean": _mean([float(run[field]) for run in selected])
                for field in numeric
            }
        )
        aggregates.append(row)
    return aggregates


def _gates(
    aggregate: Sequence[dict[str, Any]],
    runs: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    rows = {row["condition"]: row for row in aggregate}
    by_condition = {
        condition: sorted(
            (run for run in runs if run["condition"] == condition),
            key=lambda run: run["seed"],
        )
        for condition in CONDITIONS
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for left in ("vault", "recycle"):
        for right in ("static", "fixed_split", "random"):
            key = f"{left}_vs_{right}_b_score_auc"
            comparisons[key] = _paired_bootstrap(
                [run["b_score_auc"] for run in by_condition[left]],
                [run["b_score_auc"] for run in by_condition[right]],
                resamples=config.bootstrap_resamples,
                confidence_level=config.confidence_level,
                seed=config.bootstrap_seed + len(comparisons),
            )
    budget_passed = all(row["budget_pass_rate"] == 1.0 for row in aggregate)
    migration_passed = all(
        rows[name]["b_final_alignment_mean"] >= 0.99
        for name in ("vault", "recycle", "oracle")
    )
    beats_static = all(
        comparisons[f"{name}_vs_static_b_score_auc"]["ci_lower"]
        > config.control_margin
        for name in ("vault", "recycle")
    )
    beats_random = all(
        comparisons[f"{name}_vs_random_b_score_auc"]["ci_lower"]
        > config.control_margin
        for name in ("vault", "recycle")
    )
    beats_fixed_split = all(
        comparisons[f"{name}_vs_fixed_split_b_score_auc"]["ci_lower"]
        > config.control_margin
        for name in ("vault", "recycle")
    )
    vault_wake = (
        rows["vault"]["a_score_end_b_mean"] < 0.1
        and rows["vault"]["a_latent_score_end_b_mean"] >= config.score_threshold
        and rows["vault"]["a_return_immediate_score_mean"] >= config.score_threshold
    )
    recycle_erases = rows["recycle"]["a_return_immediate_score_mean"] < 0.1
    recycle_matches_scratch = (
        abs(rows["recycle"]["relearning_advantage_over_scratch_mean"]) < 1e-12
    )
    extra_control = (
        rows["extra_capacity"]["a_score_end_b_mean"] >= config.score_threshold
        and rows["extra_capacity"]["b_final_score_mean"] >= config.score_threshold
    )
    criteria = [
        {"id": "active_budget_invariant", "passed": budget_passed},
        {"id": "rank_reaches_new_task_site", "passed": migration_passed},
        {"id": "migration_beats_static", "passed": beats_static},
        {"id": "migration_beats_future_aware_fixed_split", "passed": beats_fixed_split},
        {"id": "migration_beats_same_timing_random", "passed": beats_random},
        {"id": "vault_is_inaccessible_then_cue_wakeable", "passed": vault_wake},
        {"id": "recycle_has_no_immediate_memory", "passed": recycle_erases},
        {"id": "recycle_matches_never_a_schedule", "passed": recycle_matches_scratch},
        {"id": "extra_capacity_can_retain_both_tasks", "passed": extra_control},
    ]
    passed = all(item["passed"] for item in criteria)
    return {
        "status": (
            "counterfactual_reference_mechanics_passed"
            if passed
            else "counterfactual_reference_gate_failed"
        ),
        "passed": passed,
        "criteria": criteria,
        "paired_comparisons": comparisons,
        "kill_criteria": [
            {"id": "active_budget_violation", "triggered": not budget_passed},
            {"id": "no_advantage_over_static", "triggered": not beats_static},
            {"id": "random_transfer_equivalent", "triggered": not beats_random},
            {"id": "fixed_split_equivalent", "triggered": not beats_fixed_split},
            {"id": "rank_did_not_migrate", "triggered": not migration_passed},
        ],
    }


def run_benchmark(
    config: BenchmarkConfig | None = None,
    *,
    allocator: Allocator | None = None,
) -> dict[str, Any]:
    """Run all declared conditions and return trajectories plus qualified verdict."""

    resolved = config or BenchmarkConfig()
    backend = allocator or ConservedAllocator()
    trajectories: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    tasks = []
    for seed in resolved.seeds:
        task_a, task_b = _tasks(seed, resolved)
        tasks.append(
            {
                "seed": seed,
                "task_a": asdict(task_a),
                "task_b": asdict(task_b),
                "teacher_matrix_rank": resolved.task_rank,
            }
        )
        for condition in CONDITIONS:
            rows, summary = _run_condition(seed, condition, resolved, backend)
            trajectories.extend(rows)
            runs.append(summary)
    aggregate = _aggregate(runs)
    gates = _gates(aggregate, runs, resolved)
    return {
        "kind": "capacity_migration_reference_benchmark",
        "schema_version": 1,
        "evidence_status": gates["status"],
        "config": {**asdict(resolved), "seeds": list(resolved.seeds)},
        "tasks": tasks,
        "runs": runs,
        "aggregate": aggregate,
        "gates": gates,
        "claim_boundary": (
            "Passing verifies effective active-rank accounting, paired transfers, "
            "and cue-triggered dormant-factor restoration in an orthogonal synthetic "
            "fixture with idealized component-level counterfactual gradients. That "
            "signal is a routing upper bound and does not show that a trained neural "
            "model can discover the same sites. Vault rows also report resident rank: parked "
            "factors are stored information, not erased parameters. The fixed student "
            "preallocates every possible slot, so this is not parameter-memory conservation."
        ),
        "integration_seam": (
            "LoRAManager currently supplies mask-based actuation and norm-product donor "
            "heuristics, but it does not accept the benchmark's inactive-component demand "
            "signal. The learned MLX bridge must add a recipient-demand seam derived from "
            "real gradients, loss ablations, or probe activations before claiming that it "
            "reproduces this fixture. Preserve the trajectory schema and active-rank invariant."
        ),
        "trajectory": trajectories,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the compact, evidence-qualified benchmark summary."""

    lines = [
        "# Capacity Migration Reference Benchmark",
        "",
        f"**Verdict:** `{report['evidence_status']}`",
        "",
        report["claim_boundary"],
        "",
        "| condition | B AUC | B alignment | A after B | A cue return pre-update | budget pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"| {row['condition']} | {row['b_score_auc_mean']:.3f} | "
            f"{row['b_final_alignment_mean']:.3f} | {row['a_score_end_b_mean']:.3f} | "
            f"{row['a_return_immediate_score_mean']:.3f} | {row['budget_pass_rate']:.3f} |"
        )
    lines.extend(["", "## Declared gates", ""])
    for criterion in report["gates"]["criteria"]:
        lines.append(f"- [{'x' if criterion['passed'] else ' '}] `{criterion['id']}`")
    lines.extend(["", "## Paired fixture-seed B-acquisition comparisons", ""])
    for name, comparison in report["gates"]["paired_comparisons"].items():
        lines.append(
            f"- `{name}`: mean={comparison['mean_difference']:.3f}, "
            f"{comparison['confidence_level']:.0%} CI "
            f"[{comparison['ci_lower']:.3f}, {comparison['ci_upper']:.3f}]"
        )
    lines.extend(["", "## Integration seam", "", report["integration_seam"], ""])
    return "\n".join(lines)


def write_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write JSONL trajectory and JSON/CSV/Markdown summaries."""

    return write_benchmark_artifacts(
        report,
        output_dir,
        markdown=render_markdown(report),
    )


__all__ = [
    "Allocator",
    "BenchmarkConfig",
    "CONDITIONS",
    "ConservedAllocator",
    "render_markdown",
    "run_benchmark",
    "write_artifacts",
]
