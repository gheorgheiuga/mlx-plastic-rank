"""Development runner for the declared gradient-agreement admission protocol."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .gradient_agreement import (
    AuditedTrial,
    SelectionBatch,
    array_identity,
    select_gradient,
    select_one_step,
)
from .learned_capacity_migration import _make_task, _score, _TaskBatch

SPEC_PATH = Path(__file__).resolve().parents[1] / "codex/research/gradient-agreement/protocol.json"
CONDITIONS = (
    "agreement", "static", "fixed_random", "exact_one_step", "gradient_energy",
    "wrong_task_agreement", "site_oracle", "future_fixed_split", "joint_capacity",
)
COMMON = CONDITIONS[:7]
TWO_TRANSFERS = {"agreement", "fixed_random", "exact_one_step", "gradient_energy",
                 "wrong_task_agreement"}


def load_spec() -> dict[str, Any]:
    """Read and check the declaration against the supported development fixture."""
    spec = json.loads(SPEC_PATH.read_text())
    f = spec["fixture"]
    expected = {
        "site_count": 4, "hidden_size": 6, "target_rank": 3, "max_rank": 4,
        "min_rank": 1, "active_rank_budget": 6, "physical_rank": 16, "alpha": 8,
        "initialization": "component-v1", "initial_active_ranks": [2, 2, 1, 1],
        "router_epsilon": .05, "router_temperature": 1.0, "router_scale": 2.0,
        "train_examples": 32, "probe_examples": 16, "eval_examples": 32,
        "a_steps": 72, "a_allocation_interval": 12, "b_steps": 24,
        "b_transfer_steps": [0, 12], "selection_train_row_ranges": [[0, 8], [8, 16], [16, 24]],
    }
    if any(f.get(key) != value for key, value in expected.items()):
        raise ValueError("declaration differs from implemented fixture")
    if tuple(spec["conditions"]) != CONDITIONS or tuple(spec["common_a_checkpoint_conditions"]) != COMMON:
        raise ValueError("declaration differs from implemented conditions")
    numerics = spec["numerics"]
    expected_numerics = {
        "optimizer": "sgd", "learning_rate": 1.5, "virtual_learning_rate": 1.5,
        "momentum": 0, "global_gradient_norm_clip": 1.0, "norm_epsilon": 1e-12,
        "clip_actual_and_virtual_updates": True, "clip_analytic_selection_scores": False,
        "master_dtype": "float32", "adapter_dtype": "float16", "seed_specific_rescue": False,
    }
    if numerics != expected_numerics:
        raise ValueError("declaration differs from implemented numerical policy")
    if spec["seeds"]["development"] != [31, 32, 33, 34, 35]:
        raise ValueError("development partition changed")
    return spec


def resolved_seeds(mode: str) -> tuple[int, ...]:
    """Keep reserved evidence inaccessible until a later successful freeze."""
    if mode == "smoke":
        return (0,)
    if mode == "development":
        return (31, 32, 33, 34, 35)
    raise ValueError("evidence mode is disabled; development and a source freeze must pass")


def selection(task: _TaskBatch, split: str = "train") -> SelectionBatch:
    if split not in {"train", "probe"}:
        raise ValueError("evaluation data cannot be passed to selectors")
    return SelectionBatch(getattr(task, f"{split}_features"),
                          getattr(task, f"{split}_targets"), task.loss_mask)


def _tasks(trial: AuditedTrial, seed: int) -> tuple[_TaskBatch, _TaskBatch]:
    sites = random.Random(seed * 7_919 + 41).sample(range(4), 2)
    tasks = [_make_task(name=name, site=site, seed=seed * 17 + offset,
                        protocol=trial.protocol, model=trial.model)
             for name, site, offset in zip(("A", "B"), sites, (1, 2), strict=True)]
    return tasks[0], tasks[1]


def _input_arrays(trial: AuditedTrial, a: _TaskBatch, b: _TaskBatch) -> dict[str, mx.array]:
    arrays = {"router": trial.model.router_matrix}
    for i, block in enumerate(trial.model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj"):
            layer = getattr(block.self_attn, name)
            arrays[f"frozen_{i}_{name}"] = getattr(layer, "base", layer).weight
    for task in (a, b):
        for split in ("train", "probe", "eval"):
            for kind in ("features", "targets", "routes"):
                arrays[f"{task.name}_{split}_{kind}"] = getattr(task, f"{split}_{kind}")
        arrays[f"{task.name}_mask"] = task.loss_mask
        arrays[f"{task.name}_transform"] = task.transform
    for i, param in enumerate(trial.params):
        arrays[f"initial_master_{i}"] = param
    return arrays


def _identity(arrays: dict[str, mx.array]) -> dict[str, str]:
    return {name: array_identity(array) for name, array in arrays.items()}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize(runs: list[dict[str, Any]], failures: list[dict[str, Any]],
              seeds: tuple[int, ...], spec: dict[str, Any]) -> dict[str, Any]:
    """Enforce a complete finite matrix; development never grants admission."""
    keys = [(r["seed"], r["condition"]) for r in runs]
    expected = {(seed, condition) for seed in seeds for condition in CONDITIONS}
    required = ("b_auc", "b_final", "a_readiness", "a_final", "b_coverage")
    complete = (len(keys) == len(set(keys)) and set(keys) == expected
                and not failures and all(math.isfinite(r[k]) for r in runs for k in required))
    pairing = complete
    if complete:
        for seed in seeds:
            rows = [r for r in runs if r["seed"] == seed]
            pairing &= len({r["input_identity"] for r in rows}) == 1
            pairing &= len({r["initial_bank_identity"] for r in rows}) == 1
            pairing &= len({r["start_checkpoint"] for r in rows if r["condition"] in COMMON}) == 1
    aggregate: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rows = [r for r in runs if r["condition"] == condition]
        aggregate.append({"condition": condition, "completed_seeds": len(rows), **{
            k: float(np.mean([r[k] for r in rows])) if rows else None for k in required
        }})
    by_name = {r["condition"]: r for r in aggregate}
    readiness = complete and all(by_name[c]["a_readiness"] >= .8
                                 for c in ("agreement", "future_fixed_split"))
    solvability = complete and all(by_name["joint_capacity"][k] >= .8
                                  for k in ("a_final", "b_final"))
    mechanics = complete and all(
        r["full_factor_audit"] and r["actual_b_updates"] == 24
        and (r["transfer_count"] == 2 if r["condition"] in TWO_TRANSFERS else
             0 <= r["transfer_count"] <= 2 if r["condition"] == "site_oracle" else
             r["transfer_count"] == 0)
        and (r["b_coverage"] == 1 if r["condition"] == "site_oracle" else True)
        for r in runs
    )
    checks = {"complete_finite_matrix": complete, "checkpoint_and_input_pairing": bool(pairing),
              "mechanics": mechanics, "a_readiness": readiness, "joint_solvability": solvability}
    # No inferential or comparative efficacy gates on development seeds.
    return {"protocol_id": spec["protocol_id"], "seeds": list(seeds), "runs": runs,
            "failures": failures, "aggregate": aggregate, "checks": checks,
            "development_valid": all(checks.values()), "admission_passed": False,
            "evidence_enabled": False, "paired_intervals": None,
            "interpretation": "development_only" if all(checks.values()) else "park_before_evidence"}


def run_development(
    mode: str, *, emit: Callable[[str, dict[str, Any]], None] | None = None,
    save_inputs: Callable[[int, dict[str, mx.array]], None] | None = None,
    time_limit_seconds: float = 1800,
) -> dict[str, Any]:
    """Run smoke or the fixed development partition, preserving each partial row."""
    seeds = resolved_seeds(mode)
    spec = load_spec()
    if not math.isfinite(time_limit_seconds) or not 0 < time_limit_seconds <= 1800:
        raise ValueError("development time limit must be positive and at most 1800 seconds")
    sink = emit or (lambda kind, row: None)
    deadline = time.monotonic() + time_limit_seconds
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    context: dict[str, Any] = {}

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise TimeoutError("development benchmark time cap reached")

    def train_a(trial: AuditedTrial, task: _TaskBatch, *, common: bool) -> None:
        for step in range(72):
            context.update(phase="learn_a", step=step)
            if common and step % 12 == 0:
                probe = selection(task, "probe")
                choice, diagnostics = select_one_step(trial, probe, probe, 0, inherited_a=True)
                event = None if choice is None else trial.commit(
                    *choice, reset_seed=context["seed"] * 100_000 + step * 102,
                )
                sink("events", {**context, "event": event, "selection": diagnostics})
            update = trial.update(selection(task))
            sink("trajectory", {**context, **update, "checkpoint": trial.fingerprint(),
                                "active_ranks": trial.manager.active_rank_state()})

    def train_b(trial: AuditedTrial, task_a: _TaskBatch, task_b: _TaskBatch,
                condition: str, identities: dict[str, str]) -> dict[str, Any]:
        trial.audit()
        start = trial.fingerprint()
        a_readiness = _score(trial.model, task_a)
        before_work = dict(trial.work)
        count = 0
        scores = []
        for step in range(24):
            context.update(phase="learn_b", step=step)
            check_time()
            if step in (0, 12) and condition not in {"static", "future_fixed_split", "joint_capacity"}:
                reset_seed = context["seed"] * 1_000_000 + step * 1_000 + 17
                choice = None
                diagnostics: dict[str, Any] = {}
                if condition in {"agreement", "gradient_energy", "wrong_task_agreement"}:
                    batch = selection(task_a if condition == "wrong_task_agreement" else task_b)
                    donor, recipient, diagnostics = select_gradient(
                        trial, batch, energy=condition == "gradient_energy",
                    )
                    choice = donor, recipient
                elif condition == "exact_one_step":
                    choice, diagnostics = select_one_step(
                        trial, selection(task_b).slice(0, 8), selection(task_b, "probe"), reset_seed,
                    )
                elif condition == "fixed_random":
                    rng = random.Random(context["seed"] * 1_000_003 + step * 101 + 29)
                    choice = rng.choice(trial.candidates())
                elif condition == "site_oracle":
                    target = f"blocks.{task_b.site}.attn.q_proj"
                    if dict(trial.entries)[target].active_rank < 3:
                        choice = next(pair for pair in trial.candidates() if pair[1][0] == target)
                event = None if choice is None else trial.commit(*choice, reset_seed=reset_seed)
                count += int(event is not None)
                sink("events", {**context, "event": event, "selection": diagnostics,
                                "checkpoint": trial.fingerprint()})
            update = trial.update(selection(task_b))
            scores.append(_score(trial.model, task_b))
            sink("trajectory", {**context, **update, "b_score": scores[-1],
                                "a_score": _score(trial.model, task_a),
                                "checkpoint": trial.fingerprint(),
                                "active_ranks": trial.manager.active_rank_state()})
        return {"seed": context["seed"], "condition": condition, **identities,
                "start_checkpoint": start, "end_checkpoint": trial.fingerprint(),
                "b_auc": float(np.mean(scores)), "b_final": scores[-1],
                "a_readiness": a_readiness, "a_final": _score(trial.model, task_a),
                "b_coverage": min(trial.manager.active_rank_state()[f"blocks.{task_b.site}.attn.q_proj"] / 3, 1),
                "transfer_count": count, "full_factor_audit": True,
                "actual_b_updates": trial.work["actual_updates"] - before_work["actual_updates"],
                "selection_work": {k: v - before_work[k] for k, v in trial.work.items()
                                   if k != "max_preclip_norm"},
                "max_preclip_norm_cumulative": trial.work["max_preclip_norm"],
                "storage": trial.storage()}

    for seed in seeds:
        context = {"seed": seed, "condition": "common_a", "phase": "initialize", "step": -1}
        try:
            check_time()
            trial = AuditedTrial(seed)
            trial.check_time = check_time
            task_a, task_b = _tasks(trial, seed)
            arrays = _input_arrays(trial, task_a, task_b)
            input_hashes = _identity(arrays)
            identities = {"input_identity": _digest(input_hashes),
                          "initial_bank_identity": _digest([array_identity(p) for p in trial.params])}
            if save_inputs:
                save_inputs(seed, arrays)
            sink("inputs", {"seed": seed, "array_hashes": input_hashes, **identities})
            train_a(trial, task_a, common=True)
            common = trial.snapshot()
            common_work = dict(trial.work)
            sink("preparations", {**context, "checkpoint": trial.fingerprint(), "work": common_work})
        except Exception as exc:
            failure = {**context, "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            sink("failures", failure)
            if isinstance(exc, TimeoutError):
                break
            continue
        for condition in CONDITIONS:
            context = {"seed": seed, "condition": condition, "phase": "initialize", "step": -1}
            started = time.monotonic()
            try:
                check_time()
                if condition in COMMON:
                    trial.restore(common)
                    trial.work = dict(common_work)
                    a, b = task_a, task_b
                else:
                    trial = AuditedTrial(seed)
                    trial.check_time = check_time
                    a, b = _tasks(trial, seed)
                    if _identity(_input_arrays(trial, a, b)) != input_hashes:
                        raise RuntimeError("future-aware control initial bank or data mismatch")
                    rank = 2 if condition == "future_fixed_split" else 3
                    trial.budget = 6 if condition == "future_fixed_split" else 8
                    for name, adapter in trial.entries:
                        site = int(name.split(".")[1])
                        adapter.set_active_rank(rank if site in (a.site, b.site) else 1)
                    trial.audit()
                    train_a(trial, a, common=False)
                    sink("preparations", {**context, "checkpoint": trial.fingerprint(),
                                          "work": dict(trial.work)})
                row = train_b(trial, a, b, condition, identities)
                row["seconds_including_separate_a_training"] = time.monotonic() - started
                runs.append(row)
                sink("runs", row)
            except Exception as exc:
                failure = {**context, "error_type": type(exc).__name__, "error": str(exc)}
                failures.append(failure)
                sink("failures", failure)
                if isinstance(exc, TimeoutError):
                    break
        if time.monotonic() > deadline:
            break
    return summarize(runs, failures, seeds, spec)
