"""Frozen two-transfer test of a multi-batch rank-allocation controller."""

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
    _candidate_slots,
    _checkpoint_fingerprint,
    _commit_recycled_transfer,
    _loss_fn,
    _oracle_proposal,
    _Protocol,
    _require_finite,
    _score,
    _train_step,
)
from .loss_lookahead_calibration import (
    _candidate_id,
    _prepare_a_checkpoint,
    _restore_checkpoint,
)

PROTOCOL_NAME = "multibatch_controller_v2"
DEVELOPMENT_SEEDS = (0,)
EVIDENCE_SEEDS = tuple(range(21, 31))
CONDITIONS = (
    "b_horizon3",
    "static",
    "fixed_random",
    "b_exact_one_step",
    "a_horizon3_wrong_task",
    "site_oracle",
)
MATCHED_TRANSFER_CONDITIONS = (
    "b_horizon3",
    "fixed_random",
    "b_exact_one_step",
    "a_horizon3_wrong_task",
)


@dataclass(frozen=True, slots=True)
class MultiBatchControllerConfig:
    """Select the frozen development or evidence seed partition."""

    mode: Literal["smoke", "development", "evidence"] = "smoke"
    seeds: tuple[int, ...] | None = None

    def resolved_seeds(self) -> tuple[int, ...]:
        if self.mode not in {"smoke", "development", "evidence"}:
            raise ValueError(f"unsupported multi-batch mode: {self.mode}")
        default = EVIDENCE_SEEDS if self.mode == "evidence" else DEVELOPMENT_SEEDS
        seeds = default if self.seeds is None else tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.mode in {"smoke", "development"} and seeds != DEVELOPMENT_SEEDS:
            raise ValueError(f"{self.mode} mode is restricted to development seed 0")
        if self.mode == "evidence" and seeds != EVIDENCE_SEEDS:
            raise ValueError(
                "evidence mode requires frozen seeds 21 through 30 in canonical order"
            )
        return seeds


@dataclass(frozen=True, slots=True)
class _ControllerProtocol:
    b_steps: int = 24
    transfer_steps: tuple[int, ...] = (0, 12)
    rollout_horizon: int = 3
    microbatch_size: int = 8
    bootstrap_resamples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 104_729


def _legal_candidates(manager, protocol: _Protocol) -> list[dict[str, Any]]:
    donors, recipients = _candidate_slots(manager, protocol)
    rows = [
        {
            "candidate_id": _candidate_id(donor, recipient),
            "donor_slot": donor,
            "recipient_slot": recipient,
        }
        for donor in donors
        for recipient in recipients
        if donor[0] != recipient[0]
    ]
    return sorted(rows, key=lambda row: row["candidate_id"])


def _terminal_probe_loss(manager, model, task, params) -> float:
    probe = _loss_fn(
        manager,
        model,
        task.probe_features,
        task.probe_targets,
        task.loss_mask,
    )
    value = probe(params)
    mx.eval(value)
    _require_finite("multi-batch terminal probe loss", value)
    return float(value.item())


def _rollout_updates(
    manager,
    model,
    task,
    params,
    learned_protocol: _Protocol,
    controller_protocol: _ControllerProtocol,
    *,
    horizon: int,
):
    updated = list(params)
    for rollout_step in range(horizon):
        start = rollout_step * controller_protocol.microbatch_size
        end = start + controller_protocol.microbatch_size
        loss = _loss_fn(
            manager,
            model,
            task.train_features[start:end],
            task.train_targets[start:end],
            task.loss_mask,
        )
        updated, _ = _train_step(
            manager,
            loss,
            updated,
            learned_protocol.learning_rate,
        )
    return updated


def _select_rollout_candidate(
    manager,
    model,
    task,
    params,
    learned_protocol: _Protocol,
    controller_protocol: _ControllerProtocol,
    *,
    seed: int,
    actual_step: int,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_params = list(params)
    base_gates = {
        name: adapter.active_component_indices
        for name, adapter in manager.iter_adapters()
    }
    base_fingerprint = _checkpoint_fingerprint(manager, base_params)
    commit_seed = seed * 1_000_000 + actual_step * 1_000 + 17

    def restore() -> None:
        _restore_checkpoint(manager, base_params, base_gates)
        if _checkpoint_fingerprint(manager, base_params) != base_fingerprint:
            raise RuntimeError("shadow rollout failed to restore the exact checkpoint")

    restore()
    baseline_params = _rollout_updates(
        manager,
        model,
        task,
        base_params,
        learned_protocol,
        controller_protocol,
        horizon=horizon,
    )
    baseline_terminal_loss = _terminal_probe_loss(
        manager, model, task, baseline_params
    )
    restore()

    candidates = _legal_candidates(manager, learned_protocol)
    if not candidates:
        raise RuntimeError("multi-batch selector has no legal transfer candidate")
    scored: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            restore()
            proposal = {
                **candidate,
                "evidence_source": f"strict_recycle_horizon_{horizon}_rollout",
            }
            shadow_params, event = _commit_recycled_transfer(
                manager,
                list(base_params),
                proposal,
                learned_protocol,
                seed=commit_seed,
            )
            if not event["recycled_slot_reset_verified"]:
                raise RuntimeError("shadow rollout did not verify strict recycle")
            shadow_params = _rollout_updates(
                manager,
                model,
                task,
                shadow_params,
                learned_protocol,
                controller_protocol,
                horizon=horizon,
            )
            terminal_loss = _terminal_probe_loss(
                manager, model, task, shadow_params
            )
            scored.append(
                {
                    **candidate,
                    "terminal_probe_loss": terminal_loss,
                    "predicted_terminal_loss_gain": (
                        baseline_terminal_loss - terminal_loss
                    ),
                }
            )
    finally:
        restore()
    selected = max(
        scored,
        key=lambda row: (
            row["predicted_terminal_loss_gain"],
            row["recipient_slot"],
            row["donor_slot"],
        ),
    )
    proposal = {
        "donor_slot": selected["donor_slot"],
        "recipient_slot": selected["recipient_slot"],
        "evidence_source": f"strict_recycle_horizon_{horizon}_rollout",
        "predicted_terminal_loss_gain": selected[
            "predicted_terminal_loss_gain"
        ],
        "candidate_count": len(scored),
    }
    return proposal, {
        "selection_task": task.name,
        "horizon": horizon,
        "candidate_count": len(scored),
        "virtual_gradient_evaluations": (len(scored) + 1) * horizon,
        "baseline_terminal_probe_loss": baseline_terminal_loss,
        "selected_terminal_probe_loss": selected["terminal_probe_loss"],
        "predicted_terminal_loss_gain": selected[
            "predicted_terminal_loss_gain"
        ],
        "shadow_restore_verified": True,
        "selector_observations": [
            "train_features",
            "train_targets",
            "probe_features",
            "probe_targets",
            "parameters",
            "active_masks",
        ],
        "task_site_metadata_observed": False,
    }


def _random_proposal(
    manager,
    learned_protocol: _Protocol,
    *,
    seed: int,
    actual_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _legal_candidates(manager, learned_protocol)
    if not candidates:
        raise RuntimeError("fixed-random control has no legal transfer candidate")
    selected = random.Random(seed * 104_729 + actual_step).choice(candidates)
    return {
        "donor_slot": selected["donor_slot"],
        "recipient_slot": selected["recipient_slot"],
        "evidence_source": "fixed_cadence_prediction_independent_random",
        "predicted_terminal_loss_gain": None,
        "candidate_count": len(candidates),
    }, {
        "selection_task": None,
        "horizon": 0,
        "candidate_count": len(candidates),
        "virtual_gradient_evaluations": 0,
        "baseline_terminal_probe_loss": None,
        "selected_terminal_probe_loss": None,
        "predicted_terminal_loss_gain": None,
        "shadow_restore_verified": True,
        "selector_observations": ["legal_candidates", "seed"],
        "task_site_metadata_observed": False,
    }


def _run_condition(
    seed: int,
    condition: str,
    model,
    manager,
    task_a,
    task_b,
    checkpoint,
    gates,
    learned_protocol: _Protocol,
    controller_protocol: _ControllerProtocol,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported multi-batch condition: {condition}")
    _restore_checkpoint(manager, checkpoint, gates)
    params = list(checkpoint)
    start_fingerprint = _checkpoint_fingerprint(manager, params)
    target_name = f"blocks.{task_b.site}.attn.q_proj"
    rows: list[dict[str, Any]] = []
    actual_events: list[dict[str, Any]] = []
    total_virtual_gradients = 0
    for step in range(controller_protocol.b_steps):
        event = None
        selection = None
        if step in controller_protocol.transfer_steps and condition != "static":
            if condition == "b_horizon3":
                proposal, selection = _select_rollout_candidate(
                    manager,
                    model,
                    task_b,
                    params,
                    learned_protocol,
                    controller_protocol,
                    seed=seed,
                    actual_step=step,
                    horizon=controller_protocol.rollout_horizon,
                )
            elif condition == "b_exact_one_step":
                proposal, selection = _select_rollout_candidate(
                    manager,
                    model,
                    task_b,
                    params,
                    learned_protocol,
                    controller_protocol,
                    seed=seed,
                    actual_step=step,
                    horizon=1,
                )
            elif condition == "a_horizon3_wrong_task":
                proposal, selection = _select_rollout_candidate(
                    manager,
                    model,
                    task_a,
                    params,
                    learned_protocol,
                    controller_protocol,
                    seed=seed,
                    actual_step=step,
                    horizon=controller_protocol.rollout_horizon,
                )
            elif condition == "fixed_random":
                proposal, selection = _random_proposal(
                    manager,
                    learned_protocol,
                    seed=seed,
                    actual_step=step,
                )
            else:
                proposal = _oracle_proposal(manager, task_b, learned_protocol)
                if proposal is not None:
                    proposal["evidence_source"] = (
                        "hidden_B_site_structural_control"
                    )
                selection = {
                    "selection_task": "B",
                    "horizon": 0,
                    "candidate_count": 0,
                    "virtual_gradient_evaluations": 0,
                    "baseline_terminal_probe_loss": None,
                    "selected_terminal_probe_loss": None,
                    "predicted_terminal_loss_gain": None,
                    "shadow_restore_verified": True,
                    "selector_observations": ["hidden_task_site"],
                    "task_site_metadata_observed": True,
                }
            if proposal is not None:
                params, event = _commit_recycled_transfer(
                    manager,
                    params,
                    proposal,
                    learned_protocol,
                    seed=seed * 1_000_000 + step * 1_000 + 17,
                )
                event["selection"] = selection
                actual_events.append(event)
                total_virtual_gradients += int(
                    selection["virtual_gradient_evaluations"]
                )
        loss = _loss_fn(
            manager,
            model,
            task_b.train_features,
            task_b.train_targets,
            task_b.loss_mask,
        )
        params, train_loss = _train_step(
            manager,
            loss,
            params,
            learned_protocol.learning_rate,
        )
        rank_map = manager.active_rank_state()
        rows.append(
            {
                "seed": seed,
                "condition": condition,
                "step": step,
                "active_rank": sum(rank_map.values()),
                "active_rank_budget": learned_protocol.active_rank_budget,
                "budget_ok": (
                    sum(rank_map.values()) == learned_protocol.active_rank_budget
                ),
                "rank_map": rank_map,
                "b_target_rank_coverage": (
                    rank_map[target_name] / learned_protocol.target_rank
                ),
                "a_score": _score(model, task_a),
                "b_score": _score(model, task_b),
                "train_loss": train_loss,
                "transfer": event,
            }
        )
    return rows, {
        "seed": seed,
        "condition": condition,
        "start_checkpoint_fingerprint": start_fingerprint,
        "b_score_auc": sum(row["b_score"] for row in rows) / len(rows),
        "b_final_score": rows[-1]["b_score"],
        "b_final_alignment": rows[-1]["b_target_rank_coverage"],
        "a_score_end_b": rows[-1]["a_score"],
        "transfer_count": len(actual_events),
        "virtual_gradient_evaluations": total_virtual_gradients,
        "budget_invariant": all(row["budget_ok"] for row in rows),
        "strict_recycle_invariant": all(
            event["recycled_slot_reset_verified"] for event in actual_events
        ),
        "shadow_restore_invariant": all(
            event["selection"]["shadow_restore_verified"]
            for event in actual_events
        ),
        "task_site_metadata_observed": any(
            event["selection"]["task_site_metadata_observed"]
            for event in actual_events
        ),
        "selection_candidate_counts": [
            event["selection"]["candidate_count"] for event in actual_events
        ],
    }


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
    protocol: _ControllerProtocol,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_lower": None, "ci_upper": None}
    rng = random.Random(protocol.bootstrap_seed + seed_offset)
    bootstrapped = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(protocol.bootstrap_resamples)
    ]
    tail = (1.0 - protocol.confidence_level) / 2.0
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "ci_lower": _percentile(bootstrapped, tail),
        "ci_upper": _percentile(bootstrapped, 1.0 - tail),
        "confidence_level": protocol.confidence_level,
        "bootstrap_resamples": protocol.bootstrap_resamples,
        "bootstrap_seed": protocol.bootstrap_seed + seed_offset,
    }


def _paired_comparison(
    runs: list[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    protocol: _ControllerProtocol,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    left_rows = {
        row["seed"]: row for row in runs if row["condition"] == left
    }
    right_rows = {
        row["seed"]: row for row in runs if row["condition"] == right
    }
    paired_seeds = sorted(set(left_rows) & set(right_rows))
    result = _bootstrap_mean(
        [
            left_rows[seed][metric] - right_rows[seed][metric]
            for seed in paired_seeds
        ],
        protocol,
        seed_offset=seed_offset,
    )
    return {
        **result,
        "left": left,
        "right": right,
        "metric": metric,
        "paired_seeds": paired_seeds,
    }


def run_multibatch_controller(
    config: MultiBatchControllerConfig | None = None,
) -> dict[str, Any]:
    resolved = config or MultiBatchControllerConfig()
    seeds = resolved.resolved_seeds()
    learned_protocol = _Protocol()
    controller_protocol = _ControllerProtocol()
    trajectory: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seed_fixtures: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            model, manager, task_a, task_b, checkpoint, gates = (
                _prepare_a_checkpoint(seed, learned_protocol)
            )
        except (FloatingPointError, RuntimeError) as exc:
            failures.append(
                {
                    "seed": seed,
                    "condition": "a_checkpoint",
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        seed_fixtures.append(
            {
                "seed": seed,
                "a_site": task_a.site,
                "b_site": task_b.site,
                "a_checkpoint_fingerprint": _checkpoint_fingerprint(
                    manager, checkpoint
                ),
            }
        )
        for condition in CONDITIONS:
            try:
                rows, summary = _run_condition(
                    seed,
                    condition,
                    model,
                    manager,
                    task_a,
                    task_b,
                    checkpoint,
                    gates,
                    learned_protocol,
                    controller_protocol,
                )
            except (FloatingPointError, RuntimeError) as exc:
                failures.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "failure_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            trajectory.extend(rows)
            runs.append(summary)

    comparisons = {}
    controls = (
        "static",
        "fixed_random",
        "b_exact_one_step",
        "a_horizon3_wrong_task",
    )
    for offset, control in enumerate(controls, start=1):
        comparisons[f"b_horizon3_vs_{control}_b_score_auc"] = _paired_comparison(
            runs,
            "b_horizon3",
            control,
            "b_score_auc",
            controller_protocol,
            seed_offset=offset,
        )
    comparisons["b_horizon3_vs_fixed_random_b_final_alignment"] = (
        _paired_comparison(
            runs,
            "b_horizon3",
            "fixed_random",
            "b_final_alignment",
            controller_protocol,
            seed_offset=5,
        )
    )
    expected_runs = len(seeds) * len(CONDITIONS)
    complete = not failures and len(runs) == expected_runs
    fixture_by_seed = {
        row["seed"]: row["a_checkpoint_fingerprint"] for row in seed_fixtures
    }
    checkpoint_invariant = bool(runs) and all(
        row["start_checkpoint_fingerprint"] == fixture_by_seed[row["seed"]]
        for row in runs
    )
    matched_transfers = all(
        row["transfer_count"] == len(controller_protocol.transfer_steps)
        for row in runs
        if row["condition"] in MATCHED_TRANSFER_CONDITIONS
    )
    budget_invariant = bool(runs) and all(row["budget_invariant"] for row in runs)
    recycle_invariant = bool(runs) and all(
        row["strict_recycle_invariant"] for row in runs
    )
    shadow_invariant = bool(runs) and all(
        row["shadow_restore_invariant"] for row in runs
    )
    treatment_no_site_leakage = all(
        not row["task_site_metadata_observed"]
        for row in runs
        if row["condition"] != "site_oracle"
    )
    invariants = {
        "complete_finite_seed_condition_matrix": complete,
        "shared_a_checkpoint": checkpoint_invariant,
        "matched_two_transfer_schedule": matched_transfers,
        "active_rank_budget_conserved": budget_invariant,
        "strict_recycle_verified": recycle_invariant,
        "shadow_restore_verified": shadow_invariant,
        "non_oracle_site_metadata_hidden": treatment_no_site_leakage,
    }
    run_by_condition = {
        condition: [row for row in runs if row["condition"] == condition]
        for condition in CONDITIONS
    }
    site_oracle_full_alignment = bool(run_by_condition["site_oracle"]) and all(
        row["b_final_alignment"] >= 1.0
        for row in run_by_condition["site_oracle"]
    )
    criteria = [
        {
            "id": "ten_frozen_evidence_seeds",
            "passed": resolved.mode == "evidence"
            and len(run_by_condition["b_horizon3"]) == 10,
        },
        {
            "id": "complete_valid_control_matrix",
            "passed": all(invariants.values()),
        },
    ]
    for control in controls:
        comparison_id = f"b_horizon3_vs_{control}_b_score_auc"
        comparison = comparisons[comparison_id]
        criteria.append(
            {
                "id": comparison_id,
                "passed": comparison["n"] == len(seeds)
                and comparison["ci_lower"] is not None
                and comparison["ci_lower"] > 0.0,
            }
        )
    alignment = comparisons[
        "b_horizon3_vs_fixed_random_b_final_alignment"
    ]
    criteria.extend(
        [
            {
                "id": "b_horizon3_localizes_more_than_fixed_random",
                "passed": alignment["n"] == len(seeds)
                and alignment["ci_lower"] is not None
                and alignment["ci_lower"] > 0.0,
            },
            {
                "id": "site_oracle_reaches_full_b_alignment",
                "passed": site_oracle_full_alignment,
            },
        ]
    )
    passed = all(item["passed"] for item in criteria)
    evidence_status = (
        "multibatch_controller_gate_passed"
        if passed
        else (
            "multibatch_controller_gate_failed"
            if resolved.mode == "evidence"
            else "multibatch_controller_development_only"
        )
    )
    aggregate = []
    for offset, condition in enumerate(CONDITIONS, start=20):
        condition_runs = run_by_condition[condition]
        auc = _bootstrap_mean(
            [row["b_score_auc"] for row in condition_runs],
            controller_protocol,
            seed_offset=offset,
        )
        aggregate.append(
            {
                "condition": condition,
                "n": auc["n"],
                "b_score_auc_mean": auc["mean"],
                "b_score_auc_ci_lower": auc["ci_lower"],
                "b_score_auc_ci_upper": auc["ci_upper"],
                "b_final_score_mean": (
                    sum(row["b_final_score"] for row in condition_runs)
                    / len(condition_runs)
                    if condition_runs else None
                ),
                "b_final_alignment_mean": (
                    sum(row["b_final_alignment"] for row in condition_runs)
                    / len(condition_runs)
                    if condition_runs else None
                ),
                "a_score_end_b_mean": (
                    sum(row["a_score_end_b"] for row in condition_runs)
                    / len(condition_runs)
                    if condition_runs else None
                ),
                "transfer_count_mean": (
                    sum(row["transfer_count"] for row in condition_runs)
                    / len(condition_runs)
                    if condition_runs else None
                ),
                "virtual_gradient_evaluations_mean": (
                    sum(
                        row["virtual_gradient_evaluations"]
                        for row in condition_runs
                    )
                    / len(condition_runs)
                    if condition_runs else None
                ),
            }
        )
    return {
        "kind": "multibatch_controller",
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "mode": resolved.mode,
        "evidence_status": evidence_status,
        "config": {
            "learned_fixture": asdict(learned_protocol),
            "controller": asdict(controller_protocol),
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
        "measurement_semantics": {
            "primary_metric": "mean_B_eval_score_across_24_post_update_steps",
            "localization_metric": "final_hidden_B_site_active_rank_divided_by_3",
            "experimental_unit": "fixture_seed",
            "steps_or_candidates_are_independent_replicates": False,
            "actual_training_budget_matched": True,
            "controller_selection_compute_matched": False,
            "compute_efficiency_claimed": False,
            "supervised_task_labels_used_for_selection": True,
        },
        "conditions": list(CONDITIONS),
        "fixtures": seed_fixtures,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "invariants": invariants,
        "failures": failures,
        "gates": {"passed": passed, "criteria": criteria},
        "claim_boundary": (
            "This protocol tests two supervised transfer decisions in the tiny "
            "synthetic MLX fixture. Even a passed gate would justify only iteration "
            "to a separate full V2; it cannot establish return-A behavior, unlabeled "
            "wake, physical-memory conservation, large-model behavior, compute "
            "efficiency, or Pop's theorem."
        ),
        "trajectory": trajectory,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Multi-Batch Controller V2",
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
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| condition | n | B AUC | B final | B alignment | transfers | virtual grads |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["aggregate"]:
        lines.append(
            f"| {row['condition']} | {row['n']} | {row['b_score_auc_mean']} | "
            f"{row['b_final_score_mean']} | {row['b_final_alignment_mean']} | "
            f"{row['transfer_count_mean']} | "
            f"{row['virtual_gradient_evaluations_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| comparison | n | mean difference | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, comparison in report["comparisons"].items():
        lines.append(
            f"| {name} | {comparison['n']} | {comparison['mean']} | "
            f"[{comparison['ci_lower']}, {comparison['ci_upper']}] |"
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
    """Write protocol, provenance, trajectory, summaries, and diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "protocol_json": output_dir / "protocol.json",
        "provenance_json": output_dir / "provenance.json",
        "trajectory_jsonl": output_dir / "trajectory.jsonl",
        "summary_json": output_dir / "summary.json",
        "summary_csv": output_dir / "summary.csv",
        "diagnostics_csv": output_dir / "diagnostics.csv",
        "interpretation_markdown": output_dir / "interpretation.md",
    }
    protocol_payload = {
        "protocol": report["protocol"],
        "config": report["config"],
        "seed_split": report["seed_split"],
        "measurement_semantics": report["measurement_semantics"],
        "conditions": report["conditions"],
        "gates": [item["id"] for item in report["gates"]["criteria"]],
    }
    paths["protocol_json"].write_text(
        json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["provenance_json"].write_text(
        json.dumps(provenance or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["trajectory_jsonl"].open("w", encoding="utf-8") as handle:
        for row in report["trajectory"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "trajectory"}
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["summary_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["aggregate"][0]))
        writer.writeheader()
        writer.writerows(report["aggregate"])
    diagnostic_fields = [
        "seed",
        "condition",
        "start_checkpoint_fingerprint",
        "b_score_auc",
        "b_final_score",
        "b_final_alignment",
        "a_score_end_b",
        "transfer_count",
        "virtual_gradient_evaluations",
        "budget_invariant",
        "strict_recycle_invariant",
        "shadow_restore_invariant",
        "task_site_metadata_observed",
    ]
    with paths["diagnostics_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fields)
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({field: row[field] for field in diagnostic_fields})
    paths["interpretation_markdown"].write_text(
        render_markdown(report), encoding="utf-8"
    )
    return paths


__all__ = [
    "CONDITIONS",
    "DEVELOPMENT_SEEDS",
    "EVIDENCE_SEEDS",
    "MultiBatchControllerConfig",
    "PROTOCOL_NAME",
    "render_markdown",
    "run_multibatch_controller",
    "write_artifacts",
]
