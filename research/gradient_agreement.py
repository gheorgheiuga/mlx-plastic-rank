"""Task-blind gradient selectors and audited state for the admission fixture.

The inherited synthetic model and recycle actuator are reused without changing
their historical defaults. This module never receives held-out task metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .learned_capacity_migration import (
    _candidate_slots,
    _commit_recycled_transfer,
    _loss_fn,
    _Protocol,
    _require_finite,
    _TinyDenseRoutedModel,
)
from .rank_manager import ResearchLoRAManager as LoRAManager

Slot = tuple[str, int]


def array_identity(array: mx.array) -> str:
    """Bind shape, dtype and exact materialized bytes."""
    value = np.asarray(array)
    digest = hashlib.sha256(f"{value.dtype}:{value.shape}:".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SelectionBatch:
    """Only supervised selection observations; no eval arrays or hidden site."""

    features: mx.array
    targets: mx.array
    mask: mx.array

    def slice(self, start: int, end: int) -> SelectionBatch:
        return SelectionBatch(self.features[start:end], self.targets[start:end], self.mask)


@dataclass
class Checkpoint:
    params: list[mx.array]
    gates: dict[str, tuple[int, ...]]
    clean_rows: dict[Slot, mx.array]
    resets: dict[Slot, int | None]


def clipped_sgd(
    params: list[mx.array], gradients: list[mx.array], learning_rate: float = 1.5,
) -> tuple[list[mx.array], float]:
    """Apply the declared global norm-1 clip, rejecting non-finite inputs."""
    _require_finite("SGD inputs", *params, *gradients)
    # Scale before squaring, so a large but finite gradient cannot overflow norm.
    peak = max(float(mx.max(mx.abs(g)).item()) for g in gradients)
    norm = 0.0 if peak == 0 else peak * math.sqrt(sum(
        float(mx.sum((g / peak) ** 2).item()) for g in gradients
    ))
    multiplier = min(1.0, 1.0 / max(norm, 1e-12))
    updated = [p - learning_rate * (g * multiplier)
               for p, g in zip(params, gradients, strict=True)]
    mx.eval(*updated)
    _require_finite("SGD outputs", *updated)
    return updated, norm


class AuditedTrial:
    """Own real parameters, clean factor baselines and reversible shadow state."""

    def __init__(self, seed: int, protocol: _Protocol | None = None) -> None:
        # Reserve evidence tasks even for callers bypassing the CLI.
        if seed not in (0, 31, 32, 33, 34, 35):
            raise ValueError("only smoke/development seeds are enabled")
        self.protocol = protocol or _Protocol()
        self.model = _TinyDenseRoutedModel(seed, self.protocol)
        self.manager = LoRAManager(self.model)
        adapters = self.manager.initialize_adapters(
            targets=["attn.q_proj"], rank=4, alpha=8, seed=seed,
            allowed_ranks=(1, 2, 3, 4), initial_active_rank=1,
            initialization="component-v1",
        )
        for site, rank in enumerate((2, 2, 1, 1)):
            adapters[f"blocks.{site}.attn.q_proj"].set_active_rank(rank)
        self.params = self.manager.trainable_parameters()
        self.entries = list(self.manager.iter_adapters())
        self.positions = {name: index for index, (name, _) in enumerate(self.entries)}
        self.clean_rows = {(name, c): self.params[2 * i + 1][c]
                           for i, (name, adapter) in enumerate(self.entries)
                           for c in range(adapter.rank)}
        self.resets: dict[Slot, int | None] = dict.fromkeys(self.clean_rows)
        self.budget = 6
        self.check_time: Callable[[], None] = lambda: None
        self.work: dict[str, Any] = {
            "actual_updates": 0, "virtual_updates": 0,
            "actual_clipped_updates": 0, "virtual_clipped_updates": 0,
            "max_preclip_norm": 0.0, "dense_batch_gradients": 0,
            "donor_loss_evaluations": 0, "terminal_loss_evaluations": 0,
        }
        self.audit()

    def snapshot(self) -> Checkpoint:
        return Checkpoint(list(self.params), {
            name: tuple(adapter.active_component_indices) for name, adapter in self.entries
        }, dict(self.clean_rows), dict(self.resets))

    def restore(self, checkpoint: Checkpoint) -> None:
        self.params = list(checkpoint.params)
        self.clean_rows = dict(checkpoint.clean_rows)
        self.resets = dict(checkpoint.resets)
        self.manager.set_trainable_parameters(self.params)
        for name, adapter in self.entries:
            adapter.set_active_components(checkpoint.gates[name])
        self.audit()

    def fingerprint(self) -> str:
        state = {
            "masters": [array_identity(p) for p in self.params],
            "materialized": [array_identity(a) for _, adapter in self.entries
                             for a in (adapter.A, adapter.B)],
            "gates": {name: list(adapter.active_component_indices)
                      for name, adapter in self.entries},
            "clean_rows": [(name, c, array_identity(row), self.resets[name, c])
                           for (name, c), row in sorted(self.clean_rows.items())],
        }
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def audit(self, *, inactive: bool = True) -> None:
        if sum(self.manager.active_rank_state().values()) != self.budget:
            raise RuntimeError("active rank budget violated")
        _require_finite("master parameters", *self.params)
        for i, (name, adapter) in enumerate(self.entries):
            if not 1 <= adapter.active_rank <= 4:
                raise RuntimeError("site rank bounds violated")
            _require_finite("materialized factors", adapter.A, adapter.B)
            for master, actual in zip(self.params[2 * i:2 * i + 2],
                                      (adapter.A, adapter.B), strict=True):
                if not bool(mx.array_equal(master.astype(actual.dtype), actual).item()):
                    raise RuntimeError("materialized factors differ from masters")
            if inactive:
                for c in set(range(adapter.rank)) - set(adapter.active_component_indices):
                    if not bool(mx.all(mx.equal(self.params[2 * i][:, c], 0)).item()):
                        raise RuntimeError(f"learned inactive output factor: {name}:{c}")
                    if not bool(mx.array_equal(self.params[2 * i + 1][c],
                                               self.clean_rows[name, c]).item()):
                        raise RuntimeError(f"learned inactive input factor: {name}:{c}")

    def loss(self, batch: SelectionBatch):
        return _loss_fn(self.manager, self.model, batch.features, batch.targets, batch.mask)

    def update(self, batch: SelectionBatch, *, virtual: bool = False,
               legacy_shadow: bool = False) -> dict[str, Any]:
        self.check_time()
        value, grads = mx.value_and_grad(self.loss(batch))(self.params)
        _require_finite("update loss and gradients", value, *grads)
        self.params, norm = clipped_sgd(self.params, grads)
        self.manager.set_trainable_parameters(self.params)
        self.audit(inactive=not legacy_shadow)
        kind = "virtual" if virtual else "actual"
        self.work[f"{kind}_updates"] += 1
        self.work[f"{kind}_clipped_updates"] += int(norm > 1.0)
        self.work["max_preclip_norm"] = max(self.work["max_preclip_norm"], norm)
        return {"loss": float(value.item()), "preclip_norm": norm, "clipped": norm > 1.0}

    def value(self, batch: SelectionBatch) -> float:
        self.check_time()
        value = self.loss(batch)(self.params)
        _require_finite("selection loss", value)
        self.work["terminal_loss_evaluations"] += 1
        return float(value.item())

    def candidates(self) -> list[tuple[Slot, Slot]]:
        donors, recipients = _candidate_slots(self.manager, self.protocol)
        return sorted((d, r) for d in donors for r in recipients if d[0] != r[0])

    def commit(self, donor: Slot, recipient: Slot, reset_seed: int) -> dict[str, Any]:
        self.audit()
        checkpoint = self.snapshot()
        try:
            self.params, event = _commit_recycled_transfer(
                self.manager, self.params,
                {"donor_slot": donor, "recipient_slot": recipient},
                self.protocol, seed=reset_seed,
            )
            i = self.positions[donor[0]]
            self.clean_rows[donor] = self.params[2 * i + 1][donor[1]]
            self.resets[donor] = reset_seed
            self.audit()
            event.update(donor_slot=donor, recipient_slot=recipient,
                         reset_seed=reset_seed, full_factor_audit=True)
            return event
        except Exception:
            self.restore(checkpoint)
            raise

    def storage(self) -> dict[str, int]:
        return {
            "active_rank": self.budget, "physical_rank": 16,
            "materialized_factor_bytes": sum(a.nbytes for _, ad in self.entries
                                             for a in (ad.A, ad.B)),
            "master_bytes": sum(p.nbytes for p in self.params),
            "optimizer_state_bytes": 0,
            "inactive_factor_bytes": (16 - self.budget) * 12 * 2,
            "inactive_master_bytes": (16 - self.budget) * 12 * 4,
            "clean_baseline_payload_bytes": sum(r.nbytes for r in self.clean_rows.values()),
        }


def prospective_gradient(batch: SelectionBatch, residual: mx.array,
                         route: mx.array, input_row: mx.array, scale: float) -> mx.array:
    """Differentiate an ungated zero output-factor at donor-removed predictions."""
    dense = (2.0 / (batch.features.shape[0] * mx.sum(batch.mask))) * (
        (residual * batch.mask).T @ (route * batch.features)
    )
    return scale * (dense @ input_row)


def agreement_score(gradients: mx.array, *, energy: bool = False) -> float:
    """Aggregate three batches, retaining negative cross-batch agreement."""
    _require_finite("prospective gradients", gradients)
    k = gradients.shape[0]
    if k < 2:
        raise ValueError("agreement requires at least two batches")
    square_sum = mx.sum(gradients ** 2)
    score = square_sum / k if energy else (
        mx.sum(mx.sum(gradients, axis=0) ** 2) - square_sum
    ) / (k * (k - 1))
    _require_finite("agreement score", score)
    return float(score.item())


def select_gradient(trial: AuditedTrial, batch: SelectionBatch, *, energy: bool = False
                    ) -> tuple[Slot, Slot, dict[str, Any]]:
    """Choose removal-cost donor and agreement/energy recipient without updates."""
    trial.check_time()
    before = trial.fingerprint()
    pairs = trial.candidates()
    if not pairs:
        raise RuntimeError("no legal transfer")
    # Use the materialized fp16 factors used by the real model, evaluated in fp32.
    subbatches = [batch.slice(start, start + 8) for start in (0, 8, 16)]
    if any(b.features.shape[0] != 8 for b in subbatches):
        raise ValueError("selection requires three disjoint eight-example batches")
    predictions = [trial.model(b.features) for b in subbatches]
    routes = [trial.model.routes(b.features) for b in subbatches]
    donors = sorted({d for d, _ in pairs})
    costs = []
    contributions: dict[Slot, list[mx.array]] = {}
    for donor in donors:
        i = trial.positions[donor[0]]
        adapter = trial.entries[i][1]
        a = adapter.A[:, donor[1]].astype(mx.float32)
        b_row = adapter.B[donor[1]].astype(mx.float32)
        pieces = [r[:, i:i + 1] * (b.features @ b_row)[:, None] * a[None, :] * adapter.scale
                  for b, r in zip(subbatches, routes, strict=True)]
        contributions[donor] = pieces
        cost = mx.mean(mx.stack([
            mx.sum((((p - part - b.targets) ** 2) - (p - b.targets) ** 2) * b.mask)
            / (8 * mx.sum(b.mask))
            for p, part, b in zip(predictions, pieces, subbatches, strict=True)
        ]))
        _require_finite("donor removal cost", cost)
        costs.append((float(cost.item()), donor))
    _, donor = min(costs)
    dense_gradients = [mx.stack([
        (2.0 / (8 * mx.sum(b.mask))) * (
            ((p - part - b.targets) * b.mask).T @ (r[:, i:i + 1] * b.features)
        ) for i in range(4)
    ]) for b, p, part, r in zip(subbatches, predictions, contributions[donor],
                                routes, strict=True)]
    scores = []
    for recipient in sorted({r for d, r in pairs if d == donor}):
        i = trial.positions[recipient[0]]
        adapter = trial.entries[i][1]
        row = adapter.B[recipient[1]].astype(mx.float32)
        gradients = mx.stack([adapter.scale * (g[i] @ row) for g in dense_gradients])
        scores.append((agreement_score(gradients, energy=energy), recipient))
    # Stable ascending slot ties, even when all scores are negative.
    score, recipient = min(scores, key=lambda value: (-value[0], value[1]))
    trial.work["dense_batch_gradients"] += 3
    trial.work["donor_loss_evaluations"] += 3 * len(donors)
    trial.audit()
    if before != trial.fingerprint():
        raise RuntimeError("read-only selector mutated state")
    return donor, recipient, {
        "donor_costs": [{"slot": d, "cost": v} for v, d in costs],
        "recipient_scores": [{"slot": r, "score": v} for v, r in scores],
        "selected_score": score, "state_restored": True,
        "virtual_updates": 0, "dense_batch_gradients": 3,
        "donor_loss_evaluations": 3 * len(donors),
        "temporary_array_payload_bytes": sum(a.nbytes for a in predictions + routes)
        + sum(a.nbytes for parts in contributions.values() for a in parts)
        + sum(g.nbytes for g in dense_gradients) + gradients.nbytes,
        "workspace_measurement": "partial_live_array_payload_not_peak_allocation",
    }


def select_one_step(trial: AuditedTrial, train: SelectionBatch, probe: SelectionBatch,
                    reset_seed: int, *, inherited_a: bool = False
                    ) -> tuple[tuple[Slot, Slot] | None, dict[str, Any]]:
    """Exhaustive clipped one-step comparator, or the inherited A gate-only probe."""
    base = trial.snapshot()
    fingerprint = trial.fingerprint()
    candidates = trial.candidates()
    scored = []
    try:
        trial.update(train, virtual=True)
        baseline = trial.value(probe)
        trial.restore(base)
        for donor, recipient in candidates:
            trial.check_time()
            if inherited_a:
                trial.manager.transfer_conserved_rank(
                    donor=donor, recipient=recipient, total_active_rank=6, min_rank=1,
                )
            else:
                trial.commit(donor, recipient, reset_seed)
            trial.update(train, virtual=True, legacy_shadow=inherited_a)
            terminal = trial.value(probe)
            scored.append((baseline - terminal, donor, recipient))
            trial.restore(base)
    finally:
        trial.restore(base)
        if trial.fingerprint() != fingerprint:
            raise RuntimeError("one-step branch failed exact restoration")
    diagnostics = {
        "candidate_count": len(scored), "virtual_updates": len(scored) + 1,
        "baseline_terminal_loss": baseline, "state_restored": True,
        "candidates": [{"gain": gain, "donor": d, "recipient": r} for gain, d, r in scored],
    }
    if inherited_a:
        eligible = [row for row in scored if row[0] > 1e-8]
        if not eligible:
            return None, diagnostics
        gain, donor, recipient = max(eligible, key=lambda row: (row[0], row[2], row[1]))
    else:
        if not scored:
            raise RuntimeError("no legal one-step candidate")
        gain, donor, recipient = min(scored, key=lambda row: (-row[0], row[1], row[2]))
    diagnostics["selected_gain"] = gain
    return (donor, recipient), diagnostics
