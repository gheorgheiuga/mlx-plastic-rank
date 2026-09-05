"""Calibrate whether one-step loss lookahead predicts realized rank-transfer utility."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx

from .learned_capacity_migration import (
    _build_trial,
    _candidate_slots,
    _checkpoint_fingerprint,
    _commit_recycled_transfer,
    _loss_fn,
    _make_task,
    _probe_loss_guided_transfer,
    _Protocol,
    _require_finite,
    _score,
    _train_step,
)

PROTOCOL_NAME = "loss_lookahead_calibration_v1"
DEVELOPMENT_SEEDS = (0,)
EVIDENCE_SEEDS = tuple(range(11, 21))


@dataclass(frozen=True, slots=True)
class LookaheadCalibrationConfig:
    """Select the frozen smoke or evidence partition for the calibration test."""

    mode: Literal["smoke", "development", "evidence"] = "smoke"
    seeds: tuple[int, ...] | None = None

    def resolved_seeds(self) -> tuple[int, ...]:
        if self.mode not in {"smoke", "development", "evidence"}:
            raise ValueError(f"unsupported calibration mode: {self.mode}")
        default = EVIDENCE_SEEDS if self.mode == "evidence" else DEVELOPMENT_SEEDS
        seeds = default if self.seeds is None else tuple(int(seed) for seed in self.seeds)
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("seeds must be non-empty and unique")
        if self.mode in {"smoke", "development"} and seeds != DEVELOPMENT_SEEDS:
            raise ValueError(f"{self.mode} mode is restricted to development seed 0")
        if self.mode == "evidence" and seeds != EVIDENCE_SEEDS:
            raise ValueError(
                "evidence mode requires frozen seeds 11 through 20 in canonical order"
            )
        return seeds


@dataclass(frozen=True, slots=True)
class _CalibrationProtocol:
    branch_horizon: int = 12
    bootstrap_resamples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 91_337


def _restore_checkpoint(manager, params, gates) -> None:
    manager.set_trainable_parameters(params)
    adapters = dict(manager.iter_adapters())
    for name, active in gates.items():
        adapters[name].set_active_components(active)
    mx.eval(*params)


def _prepare_a_checkpoint(seed: int, protocol: _Protocol):
    model, manager = _build_trial(seed, protocol)
    task_a_site, task_b_site = random.Random(seed * 7_919 + 41).sample(
        range(protocol.site_count), 2
    )
    task_a = _make_task(
        name="A",
        site=task_a_site,
        seed=seed * 17 + 1,
        protocol=protocol,
        model=model,
    )
    task_b = _make_task(
        name="B",
        site=task_b_site,
        seed=seed * 17 + 2,
        protocol=protocol,
        model=model,
    )
    params = manager.trainable_parameters()
    for step in range(protocol.phase_steps):
        probe_loss = _loss_fn(
            manager,
            model,
            task_a.probe_features,
            task_a.probe_targets,
            task_a.loss_mask,
        )
        if step % protocol.allocation_interval == 0:
            proposal = _probe_loss_guided_transfer(
                manager,
                probe_loss,
                params,
                protocol,
            )
            if proposal is not None:
                params, _ = _commit_recycled_transfer(
                    manager,
                    params,
                    proposal,
                    protocol,
                    seed=seed * 100_000 + step * 102,
                )
        train_loss = _loss_fn(
            manager,
            model,
            task_a.train_features,
            task_a.train_targets,
            task_a.loss_mask,
        )
        params, _ = _train_step(
            manager,
            train_loss,
            params,
            protocol.learning_rate,
        )
    gates = {
        name: adapter.active_component_indices
        for name, adapter in manager.iter_adapters()
    }
    checkpoint = list(params)
    mx.eval(*checkpoint)
    return model, manager, task_a, task_b, checkpoint, gates


def _candidate_id(donor: tuple[str, int], recipient: tuple[str, int]) -> str:
    return f"{donor[0]}:{donor[1]}->{recipient[0]}:{recipient[1]}"


def _score_candidates(manager, loss, params, protocol: _Protocol) -> list[dict[str, Any]]:
    entries = list(manager.iter_adapters())
    base_params = list(params)
    base_gates = {
        name: adapter.active_component_indices for name, adapter in entries
    }

    def restore() -> None:
        _restore_checkpoint(manager, base_params, base_gates)
        if sum(manager.active_rank_state().values()) != protocol.active_rank_budget:
            raise RuntimeError("candidate probe failed to restore the active-rank budget")

    baseline, baseline_grads = mx.value_and_grad(loss)(base_params)
    no_swap_params = [
        param - protocol.probe_learning_rate * grad
        for param, grad in zip(base_params, baseline_grads, strict=True)
    ]
    no_swap_after = loss(no_swap_params)
    mx.eval(baseline, baseline_grads, no_swap_params, no_swap_after)
    _require_finite("calibration baseline", baseline, no_swap_after)
    _require_finite("calibration baseline gradients", *baseline_grads)
    _require_finite("calibration baseline parameters", *no_swap_params)
    baseline_value = float(baseline.item())
    no_swap_after_value = float(no_swap_after.item())
    restore()

    candidates: list[dict[str, Any]] = []
    donors, recipients = _candidate_slots(manager, protocol)
    try:
        for donor in donors:
            for recipient in recipients:
                if donor[0] == recipient[0]:
                    continue
                restore()
                try:
                    manager.transfer_conserved_rank(
                        donor=donor,
                        recipient=recipient,
                        total_active_rank=protocol.active_rank_budget,
                        min_rank=protocol.min_rank,
                    )
                    swapped_loss, grads = mx.value_and_grad(loss)(base_params)
                    virtual_params = [
                        param - protocol.probe_learning_rate * grad
                        for param, grad in zip(base_params, grads, strict=True)
                    ]
                    after_loss = loss(virtual_params)
                    mx.eval(swapped_loss, grads, virtual_params, after_loss)
                    _require_finite(
                        "calibration candidate",
                        swapped_loss,
                        after_loss,
                        *grads,
                        *virtual_params,
                    )
                    after_value = float(after_loss.item())
                    candidates.append(
                        {
                            "candidate_id": _candidate_id(donor, recipient),
                            "donor_slot": donor,
                            "recipient_slot": recipient,
                            "probe_base_loss": baseline_value,
                            "probe_no_swap_after_loss": no_swap_after_value,
                            "probe_swapped_loss": float(swapped_loss.item()),
                            "probe_after_loss": after_value,
                            "predicted_loss_gain": no_swap_after_value - after_value,
                        }
                    )
                finally:
                    restore()
    finally:
        restore()
    return sorted(candidates, key=lambda row: row["candidate_id"])


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[ordered[offset]] = average
        position = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires aligned vectors of length >= 2")
    return _pearson(_rankdata(left), _rankdata(right))


def _train_branch(
    manager,
    model,
    task_b,
    params,
    protocol: _Protocol,
    calibration: _CalibrationProtocol,
) -> tuple[list[mx.array], list[float]]:
    scores: list[float] = []
    for _ in range(calibration.branch_horizon):
        train_loss = _loss_fn(
            manager,
            model,
            task_b.train_features,
            task_b.train_targets,
            task_b.loss_mask,
        )
        params, _ = _train_step(
            manager,
            train_loss,
            params,
            protocol.learning_rate,
        )
        scores.append(_score(model, task_b))
    return params, scores


def _best_candidate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row[key],
            row["recipient_slot"],
            row["donor_slot"],
        ),
    )


def _evaluate_seed(
    seed: int,
    protocol: _Protocol,
    calibration: _CalibrationProtocol,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, manager, task_a, task_b, checkpoint, gates = _prepare_a_checkpoint(
        seed, protocol
    )
    checkpoint_fingerprint = _checkpoint_fingerprint(manager, checkpoint)
    b_loss = _loss_fn(
        manager,
        model,
        task_b.probe_features,
        task_b.probe_targets,
        task_b.loss_mask,
    )
    b_candidates = _score_candidates(manager, b_loss, checkpoint, protocol)
    a_loss = _loss_fn(
        manager,
        model,
        task_a.probe_features,
        task_a.probe_targets,
        task_a.loss_mask,
    )
    a_candidates = _score_candidates(manager, a_loss, checkpoint, protocol)
    if [row["candidate_id"] for row in b_candidates] != [
        row["candidate_id"] for row in a_candidates
    ]:
        raise RuntimeError("A and B probes produced different legal candidate sets")
    a_gain_by_id = {
        row["candidate_id"]: row["predicted_loss_gain"] for row in a_candidates
    }
    for row in b_candidates:
        row["a_predicted_loss_gain"] = a_gain_by_id[row["candidate_id"]]

    predicted_best = _best_candidate(b_candidates, "predicted_loss_gain")
    predicted_worst = min(
        b_candidates,
        key=lambda row: (
            row["predicted_loss_gain"],
            row["recipient_slot"],
            row["donor_slot"],
        ),
    )
    wrong_task_best = _best_candidate(b_candidates, "a_predicted_loss_gain")
    random_candidate = random.Random(seed * 104_729 + protocol.phase_steps).choice(
        b_candidates
    )
    selectors = {
        "predicted_best": predicted_best["candidate_id"],
        "predicted_worst": predicted_worst["candidate_id"],
        "wrong_task_best": wrong_task_best["candidate_id"],
        "prediction_independent_random": random_candidate["candidate_id"],
    }

    _restore_checkpoint(manager, checkpoint, gates)
    static_start_fingerprint = _checkpoint_fingerprint(manager, checkpoint)
    static_params, static_scores = _train_branch(
        manager,
        model,
        task_b,
        list(checkpoint),
        protocol,
        calibration,
    )
    del static_params
    static_final = static_scores[-1]
    branches: list[dict[str, Any]] = [
        {
            "seed": seed,
            "branch": "static",
            "candidate_id": None,
            "start_checkpoint_fingerprint": static_start_fingerprint,
            "candidate_count": len(b_candidates),
            "predicted_loss_gain": 0.0,
            "a_predicted_loss_gain": 0.0,
            "b_score_start": _score_after_restore(
                manager, model, task_b, checkpoint, gates
            ),
            "b_score_trajectory": static_scores,
            "b_score_final": static_final,
            "realized_gain_vs_static": 0.0,
            "budget_ok": True,
            "selected_roles": ["static"],
            "finite": True,
        }
    ]
    branch_failures: list[dict[str, Any]] = []
    for candidate in b_candidates:
        _restore_checkpoint(manager, checkpoint, gates)
        branch_start_fingerprint = _checkpoint_fingerprint(manager, checkpoint)
        proposal = {
            "donor_slot": candidate["donor_slot"],
            "recipient_slot": candidate["recipient_slot"],
            "evidence_source": "exhaustive_first_b_opportunity_branch",
            "predicted_loss_gain": candidate["predicted_loss_gain"],
            "a_predicted_loss_gain": candidate["a_predicted_loss_gain"],
        }
        try:
            branch_params, event = _commit_recycled_transfer(
                manager,
                list(checkpoint),
                proposal,
                protocol,
                seed=seed * 100_000 + protocol.phase_steps * 102,
            )
            budget_after_transfer = sum(manager.active_rank_state().values())
            branch_params, scores = _train_branch(
                manager,
                model,
                task_b,
                branch_params,
                protocol,
                calibration,
            )
            del branch_params
            final_score = scores[-1]
            selected_roles = [
                role for role, candidate_id in selectors.items()
                if candidate_id == candidate["candidate_id"]
            ]
            branches.append(
                {
                    "seed": seed,
                    "branch": "legal_transfer",
                    "candidate_id": candidate["candidate_id"],
                    "donor": list(candidate["donor_slot"]),
                    "recipient": list(candidate["recipient_slot"]),
                    "start_checkpoint_fingerprint": branch_start_fingerprint,
                    "candidate_count": len(b_candidates),
                    "predicted_loss_gain": candidate["predicted_loss_gain"],
                    "a_predicted_loss_gain": candidate["a_predicted_loss_gain"],
                    "probe_no_swap_after_loss": candidate[
                        "probe_no_swap_after_loss"
                    ],
                    "probe_after_loss": candidate["probe_after_loss"],
                    "b_score_start": branches[0]["b_score_start"],
                    "b_score_trajectory": scores,
                    "b_score_final": final_score,
                    "realized_gain_vs_static": final_score - static_final,
                    "budget_after_transfer": budget_after_transfer,
                    "budget_ok": (
                        budget_after_transfer == protocol.active_rank_budget
                    ),
                    "strict_recycle_verified": event[
                        "recycled_slot_reset_verified"
                    ],
                    "selected_roles": selected_roles,
                    "finite": True,
                }
            )
        except (FloatingPointError, RuntimeError) as exc:
            branch_failures.append(
                {
                    "seed": seed,
                    "candidate_id": candidate["candidate_id"],
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    transfer_rows = [row for row in branches if row["branch"] == "legal_transfer"]
    predicted = [row["predicted_loss_gain"] for row in transfer_rows]
    realized = [row["realized_gain_vs_static"] for row in transfer_rows]
    by_id = {row["candidate_id"]: row for row in transfer_rows}
    selected_gains = {
        role: by_id[candidate_id]["realized_gain_vs_static"]
        for role, candidate_id in selectors.items()
        if candidate_id in by_id
    }
    complete = not branch_failures and len(transfer_rows) == len(b_candidates)
    return branches, {
        "seed": seed,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "all_branch_checkpoints_match": all(
            row["start_checkpoint_fingerprint"] == checkpoint_fingerprint
            for row in branches
        ),
        "candidate_count": len(b_candidates),
        "finite_candidate_count": len(transfer_rows),
        "complete": complete,
        "branch_failures": branch_failures,
        "budget_invariant": all(row["budget_ok"] for row in branches),
        "strict_recycle_invariant": all(
            row.get("strict_recycle_verified", True) for row in branches
        ),
        "spearman_predicted_vs_realized": (
            _spearman(predicted, realized) if len(transfer_rows) >= 2 else None
        ),
        "selectors": selectors,
        "selected_realized_gains": selected_gains,
        "static_b_score_final": static_final,
        "predicted_best_b_score_final": (
            by_id[selectors["predicted_best"]]["b_score_final"]
            if selectors["predicted_best"] in by_id else None
        ),
        "predicted_best_equals_predicted_worst": (
            selectors["predicted_best"] == selectors["predicted_worst"]
        ),
        "predicted_best_equals_wrong_task_best": (
            selectors["predicted_best"] == selectors["wrong_task_best"]
        ),
    }


def _score_after_restore(manager, model, task, params, gates) -> float:
    _restore_checkpoint(manager, params, gates)
    return _score(model, task)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean(
    values: list[float],
    protocol: _CalibrationProtocol,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_lower": None, "ci_upper": None}
    rng = random.Random(protocol.bootstrap_seed + seed_offset)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(protocol.bootstrap_resamples)
    ]
    tail = (1.0 - protocol.confidence_level) / 2.0
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "ci_lower": _percentile(means, tail),
        "ci_upper": _percentile(means, 1.0 - tail),
        "confidence_level": protocol.confidence_level,
        "bootstrap_resamples": protocol.bootstrap_resamples,
        "bootstrap_seed": protocol.bootstrap_seed + seed_offset,
    }


def _comparison_values(seed_rows: list[dict[str, Any]], left: str, right: str):
    values = []
    for row in seed_rows:
        selected = row["selected_realized_gains"]
        left_value = 0.0 if left == "static" else selected.get(left)
        right_value = 0.0 if right == "static" else selected.get(right)
        if left_value is not None and right_value is not None:
            values.append(left_value - right_value)
    return values


def run_loss_lookahead_calibration(
    config: LookaheadCalibrationConfig | None = None,
) -> dict[str, Any]:
    resolved = config or LookaheadCalibrationConfig()
    seeds = resolved.resolved_seeds()
    protocol = _Protocol()
    calibration = _CalibrationProtocol()
    branches: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            seed_branches, seed_summary = _evaluate_seed(
                seed, protocol, calibration
            )
        except (FloatingPointError, RuntimeError) as exc:
            failures.append(
                {
                    "seed": seed,
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        branches.extend(seed_branches)
        seed_rows.append(seed_summary)
        failures.extend(seed_summary["branch_failures"])

    correlations = [
        row["spearman_predicted_vs_realized"]
        for row in seed_rows
        if row["spearman_predicted_vs_realized"] is not None
    ]
    comparisons = {
        "predicted_best_vs_static": _bootstrap_mean(
            _comparison_values(seed_rows, "predicted_best", "static"),
            calibration,
            seed_offset=1,
        ),
        "predicted_best_vs_predicted_worst": _bootstrap_mean(
            _comparison_values(
                seed_rows, "predicted_best", "predicted_worst"
            ),
            calibration,
            seed_offset=2,
        ),
        "predicted_best_vs_wrong_task_best": _bootstrap_mean(
            _comparison_values(seed_rows, "predicted_best", "wrong_task_best"),
            calibration,
            seed_offset=3,
        ),
        "predicted_best_vs_prediction_independent_random": _bootstrap_mean(
            _comparison_values(
                seed_rows,
                "predicted_best",
                "prediction_independent_random",
            ),
            calibration,
            seed_offset=4,
        ),
    }
    correlation = _bootstrap_mean(correlations, calibration, seed_offset=0)
    complete = (
        not failures
        and len(seed_rows) == len(seeds)
        and all(row["complete"] for row in seed_rows)
    )
    invariants = {
        "complete_finite_seed_branch_matrix": complete,
        "all_branch_checkpoints_match": bool(seed_rows) and all(
            row["all_branch_checkpoints_match"] for row in seed_rows
        ),
        "active_rank_budget_conserved": bool(seed_rows) and all(
            row["budget_invariant"] for row in seed_rows
        ),
        "strict_recycle_verified": bool(seed_rows) and all(
            row["strict_recycle_invariant"] for row in seed_rows
        ),
    }
    criteria = [
        {
            "id": "ten_frozen_evidence_seeds",
            "passed": resolved.mode == "evidence" and len(seed_rows) == 10,
        },
        {"id": "complete_finite_seed_branch_matrix", "passed": complete},
        {
            "id": "paired_checkpoint_and_budget_invariants",
            "passed": all(invariants.values()),
        },
        {
            "id": "predicted_gain_ranks_realized_gain",
            "passed": correlation["n"] == len(seeds)
            and correlation["ci_lower"] is not None
            and correlation["ci_lower"] > 0.0,
        },
    ]
    for comparison_id in (
        "predicted_best_vs_static",
        "predicted_best_vs_predicted_worst",
        "predicted_best_vs_wrong_task_best",
    ):
        result = comparisons[comparison_id]
        criteria.append(
            {
                "id": comparison_id,
                "passed": result["n"] == len(seeds)
                and result["ci_lower"] is not None
                and result["ci_lower"] > 0.0,
            }
        )
    passed = all(item["passed"] for item in criteria)
    status = (
        "lookahead_calibration_gate_passed"
        if passed
        else (
            "lookahead_calibration_gate_failed"
            if resolved.mode == "evidence"
            else "lookahead_calibration_development_only"
        )
    )
    roles = (
        "predicted_best",
        "predicted_worst",
        "wrong_task_best",
        "prediction_independent_random",
    )
    aggregate = []
    for role in roles:
        values = [
            row["selected_realized_gains"][role]
            for row in seed_rows
            if role in row["selected_realized_gains"]
        ]
        estimate = _bootstrap_mean(
            values,
            calibration,
            seed_offset=20 + roles.index(role),
        )
        aggregate.append({"role": role, **estimate})
    aggregate.append(
        {
            "role": "static",
            "n": len(seed_rows),
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "confidence_level": calibration.confidence_level,
            "bootstrap_resamples": calibration.bootstrap_resamples,
            "bootstrap_seed": None,
        }
    )
    return {
        "kind": "loss_lookahead_calibration",
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "mode": resolved.mode,
        "evidence_status": status,
        "config": {
            "learned_fixture": asdict(protocol),
            "calibration": asdict(calibration),
            "seeds": list(seeds),
        },
        "seed_split": {
            "frozen": True,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "evidence_seeds": list(EVIDENCE_SEEDS),
            "selected_partition": (
                "evidence" if resolved.mode == "evidence" else "development"
            ),
            "selected_seeds": list(seeds),
        },
        "metric_semantics": {
            "prediction": (
                "one_virtual_update_probe_loss_gain_after_gate_only_shadow_swap"
            ),
            "realization": (
                "B_eval_score_after_one_strict_recycle_transfer_and_12_updates_"
                "minus_static_matched_checkpoint_score"
            ),
            "experimental_unit": "fixture_seed",
            "candidate_branches_are_independent_replicates": False,
        },
        "invariants": invariants,
        "failures": failures,
        "seeds": seed_rows,
        "correlation": correlation,
        "comparisons": comparisons,
        "aggregate": aggregate,
        "gates": {"passed": passed, "criteria": criteria},
        "claim_boundary": (
            "This test calibrates one transfer signal at the first B opportunity "
            "in the tiny synthetic MLX fixture. It cannot promote full capacity "
            "migration, cue-triggered wake, physical-memory conservation, large-model "
            "behavior, or Pop's theorem."
        ),
        "branches": branches,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Loss-Lookahead Calibration",
        "",
        f"**Decision:** `{report['evidence_status']}`",
        "",
        report["claim_boundary"],
        "",
        "## Frozen gate",
        "",
    ]
    for criterion in report["gates"]["criteria"]:
        marker = "PASS" if criterion["passed"] else "FAIL"
        lines.append(f"- {marker}: `{criterion['id']}`")
    correlation = report["correlation"]
    lines.extend(
        [
            "",
            "## Primary result",
            "",
            (
                "- Mean seed-level Spearman(predicted gain, realized gain): "
                f"{correlation['mean']} (95% CI [{correlation['ci_lower']}, "
                f"{correlation['ci_upper']}], n={correlation['n']})."
            ),
            "",
            "## Paired selected-branch comparisons",
            "",
            "| comparison | n | mean difference | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in report["comparisons"].items():
        lines.append(
            f"| {name} | {result['n']} | {result['mean']} | "
            f"[{result['ci_lower']}, {result['ci_upper']}] |"
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- `{failure}`")
    return "\n".join(lines) + "\n"


def write_artifacts(
    report: dict[str, Any],
    output_dir: Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the frozen protocol, raw branches, diagnostics, and interpretation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol_json": output_dir / "protocol.json",
        "provenance_json": output_dir / "provenance.json",
        "raw_results_jsonl": output_dir / "raw_results.jsonl",
        "summary_csv": output_dir / "summary.csv",
        "summary_json": output_dir / "summary.json",
        "diagnostics_csv": output_dir / "diagnostics.csv",
        "interpretation_markdown": output_dir / "interpretation.md",
    }
    protocol_payload = {
        "protocol": report["protocol"],
        "config": report["config"],
        "seed_split": report["seed_split"],
        "metric_semantics": report["metric_semantics"],
        "gates": [item["id"] for item in report["gates"]["criteria"]],
    }
    paths["protocol_json"].write_text(
        json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n"
    )
    paths["provenance_json"].write_text(
        json.dumps(provenance or {}, indent=2, sort_keys=True) + "\n"
    )
    with paths["raw_results_jsonl"].open("w", encoding="utf-8") as handle:
        for row in report["branches"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with paths["summary_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["aggregate"][0]))
        writer.writeheader()
        writer.writerows(report["aggregate"])
    summary = {key: value for key, value in report.items() if key != "branches"}
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic_fields = [
        "seed",
        "checkpoint_fingerprint",
        "all_branch_checkpoints_match",
        "candidate_count",
        "finite_candidate_count",
        "complete",
        "budget_invariant",
        "strict_recycle_invariant",
        "spearman_predicted_vs_realized",
        "static_b_score_final",
        "predicted_best_b_score_final",
        "predicted_best_equals_predicted_worst",
        "predicted_best_equals_wrong_task_best",
    ]
    with paths["diagnostics_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        for row in report["seeds"]:
            writer.writerow({field: row[field] for field in diagnostic_fields})
    paths["interpretation_markdown"].write_text(
        render_markdown(report), encoding="utf-8"
    )
    return paths


__all__ = [
    "DEVELOPMENT_SEEDS",
    "EVIDENCE_SEEDS",
    "LookaheadCalibrationConfig",
    "PROTOCOL_NAME",
    "render_markdown",
    "run_loss_lookahead_calibration",
    "write_artifacts",
]
