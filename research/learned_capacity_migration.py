"""Learned MLX capacity-migration benchmark with loss-only allocation probes."""

from __future__ import annotations

import hashlib
import json
import math
import random
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
import mlx.nn as nn

from .benchmark_artifacts import write_benchmark_artifacts
from .rank_manager import ResearchLoRAManager as LoRAManager

CORE_CONDITIONS = (
    "guided_recycle",
    "static",
    "random",
    "fixed_split",
    "extra_capacity",
)
EVIDENCE_CONDITIONS = CORE_CONDITIONS + (
    "guided_vault",
    "oracle",
    "never_a",
)
DEVELOPMENT_SEEDS = (0,)
CONFIRMATORY_SEEDS = tuple(range(1, 11))


@dataclass(frozen=True, slots=True)
class LearnedMigrationConfig:
    """Select the canonical tiny-MLX protocol or its non-promotional smoke run."""

    protocol: str = "tiny_mlx_dense_v1"
    mode: Literal["smoke", "development", "evidence"] = "smoke"
    seeds: tuple[int, ...] | None = None

    def resolved_seeds(self) -> tuple[int, ...]:
        if self.protocol != "tiny_mlx_dense_v1":
            raise ValueError(f"unsupported learned migration protocol: {self.protocol}")
        if self.mode not in {"smoke", "development", "evidence"}:
            raise ValueError(f"unsupported learned migration mode: {self.mode}")
        seeds = self.seeds
        if seeds is None:
            seeds = (
                CONFIRMATORY_SEEDS
                if self.mode == "evidence"
                else DEVELOPMENT_SEEDS
            )
        seeds = tuple(int(seed) for seed in seeds)
        if not seeds:
            raise ValueError("seeds must not be empty")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if self.mode in {"smoke", "development"} and seeds != DEVELOPMENT_SEEDS:
            raise ValueError(
                f"{self.mode} mode is restricted to development seed 0 only"
            )
        if self.mode == "evidence" and seeds != CONFIRMATORY_SEEDS:
            raise ValueError(
                "evidence mode requires the frozen confirmatory seeds 1 through 10 "
                "in canonical order"
            )
        return seeds


@dataclass(frozen=True)
class _Protocol:
    site_count: int = 4
    hidden_size: int = 6
    target_rank: int = 3
    max_rank: int = 4
    active_rank_budget: int = 6
    min_rank: int = 1
    phase_steps: int = 72
    learning_rate: float = 1.5
    probe_learning_rate: float = 1.5
    allocation_interval: int = 12
    train_examples: int = 32
    probe_examples: int = 16
    eval_examples: int = 32
    router_epsilon: float = 0.05
    router_temperature: float = 1.0
    router_scale: float = 2.0
    score_threshold: float = 0.8
    bootstrap_resamples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 73_271


@dataclass(frozen=True)
class _TaskBatch:
    name: str
    site: int
    train_features: mx.array
    train_routes: mx.array
    train_targets: mx.array
    probe_features: mx.array
    probe_routes: mx.array
    probe_targets: mx.array
    eval_features: mx.array
    eval_routes: mx.array
    eval_targets: mx.array
    loss_mask: mx.array
    transform: mx.array
    output_head: tuple[int, int]
    transform_fingerprint: str


class _TinyAttention:
    def __init__(self, hidden_size: int) -> None:
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_proj.weight = mx.zeros_like(self.q_proj.weight)
        self.k_proj.weight = mx.zeros_like(self.k_proj.weight)
        self.v_proj.weight = mx.zeros_like(self.v_proj.weight)


class _TinyBlock:
    def __init__(self, hidden_size: int) -> None:
        self.self_attn = _TinyAttention(hidden_size)


def _orthonormal_rows(
    *,
    seed: int,
    count: int,
    width: int,
) -> tuple[tuple[float, ...], ...]:
    """Build a deterministic orthonormal row set without another dependency."""

    if count > width:
        raise ValueError("cannot construct more orthonormal rows than their width")
    rng = random.Random(seed)
    rows: list[tuple[float, ...]] = []
    while len(rows) < count:
        vector = [rng.gauss(0.0, 1.0) for _ in range(width)]
        for basis in rows:
            projection = sum(value * axis for value, axis in zip(vector, basis, strict=True))
            vector = [
                value - projection * axis
                for value, axis in zip(vector, basis, strict=True)
            ]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 1e-8:
            rows.append(tuple(value / norm for value in vector))
    return tuple(rows)


class _TinyDenseRoutedModel:
    """Frozen input-routed substrate whose q-projections use real MLX LoRA."""

    def __init__(self, seed: int, protocol: _Protocol) -> None:
        site_count = protocol.site_count
        hidden_size = protocol.hidden_size
        self.model = types.SimpleNamespace(
            layers=[_TinyBlock(hidden_size) for _ in range(site_count)]
        )
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.model_type = "gemma"
        self.site_count = site_count
        self.router_epsilon = protocol.router_epsilon
        self.router_temperature = protocol.router_temperature
        self.router_directions = _orthonormal_rows(
            seed=seed * 65_537 + 17,
            count=site_count,
            width=hidden_size,
        )
        self.router_matrix = mx.array(
            [
                [protocol.router_scale * value for value in direction]
                for direction in self.router_directions
            ],
            dtype=mx.float32,
        )
        mx.eval(self.router_matrix)

    def routes(self, features: mx.array) -> mx.array:
        """Return dense, nonzero routes derived only from the input features."""

        logits = mx.matmul(features, mx.transpose(self.router_matrix))
        probabilities = mx.softmax(logits / self.router_temperature, axis=-1)
        remaining_mass = 1.0 - self.site_count * self.router_epsilon
        return self.router_epsilon + remaining_mass * probabilities

    def __call__(self, features: mx.array) -> mx.array:
        routes = self.routes(features)
        output = mx.zeros_like(features)
        for site, block in enumerate(self.model.layers):
            site_output = block.self_attn.q_proj(features)
            output = output + routes[:, site : site + 1] * site_output
        return output


def _make_task(
    *,
    name: str,
    site: int,
    seed: int,
    protocol: _Protocol,
    model: _TinyDenseRoutedModel,
) -> _TaskBatch:
    rng = random.Random(seed)
    input_basis = mx.array(
        _orthonormal_rows(
            seed=seed * 131 + 5,
            count=protocol.target_rank,
            width=protocol.hidden_size,
        ),
        dtype=mx.float32,
    )
    if protocol.hidden_size != 2 * protocol.target_rank:
        raise RuntimeError("dense task fixture requires two disjoint rank-sized heads")
    head_start = 0 if name == "A" else protocol.target_rank
    output_basis = mx.array(
        [
            [
                1.0 if index == head_start + component else 0.0
                for index in range(protocol.hidden_size)
            ]
            for component in range(protocol.target_rank)
        ],
        dtype=mx.float32,
    )
    singular_values = mx.array(
        [1.25 - 0.2 * index for index in range(protocol.target_rank)],
        dtype=mx.float32,
    )
    transform = mx.matmul(
        mx.transpose(input_basis) * singular_values[None, :],
        output_basis,
    )
    mx.eval(input_basis, output_basis, singular_values, transform)
    transform_fingerprint = hashlib.sha256(
        json.dumps(transform.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    loss_mask = mx.array(
        [
            1.0 if head_start <= index < head_start + protocol.target_rank else 0.0
            for index in range(protocol.hidden_size)
        ],
        dtype=mx.float32,
    )

    def build(count: int) -> tuple[mx.array, mx.array, mx.array]:
        features = [
            [rng.gauss(0.0, 1.0) for _ in range(protocol.hidden_size)]
            for _ in range(count)
        ]
        feature_array = mx.array(features, dtype=mx.float32)
        routes = model.routes(feature_array)
        transformed = mx.matmul(feature_array, transform)
        targets = routes[:, site : site + 1] * transformed
        mx.eval(feature_array, routes, targets)
        return feature_array, routes, targets

    train_features, train_routes, train_targets = build(protocol.train_examples)
    probe_features, probe_routes, probe_targets = build(protocol.probe_examples)
    eval_features, eval_routes, eval_targets = build(protocol.eval_examples)
    return _TaskBatch(
        name=name,
        site=site,
        train_features=train_features,
        train_routes=train_routes,
        train_targets=train_targets,
        probe_features=probe_features,
        probe_routes=probe_routes,
        probe_targets=probe_targets,
        eval_features=eval_features,
        eval_routes=eval_routes,
        eval_targets=eval_targets,
        loss_mask=loss_mask,
        transform=transform,
        output_head=(head_start, head_start + protocol.target_rank),
        transform_fingerprint=transform_fingerprint,
    )


def _build_trial(
    seed: int,
    protocol: _Protocol,
) -> tuple[_TinyDenseRoutedModel, LoRAManager]:
    model = _TinyDenseRoutedModel(seed, protocol)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=protocol.max_rank,
        alpha=2.0 * protocol.max_rank,
        seed=seed,
        allowed_ranks=tuple(range(1, protocol.max_rank + 1)),
        initial_active_rank=protocol.min_rank,
    )
    initial_ranks = (2, 2, 1, 1)
    if len(initial_ranks) != protocol.site_count:
        raise RuntimeError("tiny MLX protocol initial allocation does not match site count")
    for site, active in enumerate(initial_ranks):
        adapters[f"blocks.{site}.attn.q_proj"].set_active_components(range(active))
    if sum(manager.active_rank_state().values()) != protocol.active_rank_budget:
        raise RuntimeError("tiny MLX trial initial allocation violates the active-rank budget")
    return model, manager


def _loss_fn(
    manager: LoRAManager,
    model: _TinyDenseRoutedModel,
    features: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
):
    def calculate(param_arrays: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(param_arrays)
        predictions = model(features)
        squared_error = (predictions - targets) ** 2 * loss_mask[None, :]
        return mx.sum(squared_error) / (features.shape[0] * mx.sum(loss_mask))

    return calculate


def _require_finite(label: str, *arrays: mx.array) -> None:
    for array in arrays:
        finite = mx.all(mx.isfinite(array))
        mx.eval(finite)
        if not bool(finite.item()):
            raise FloatingPointError(f"non-finite value in {label}")


def _score(model: _TinyDenseRoutedModel, task: _TaskBatch) -> float:
    predictions = model(task.eval_features)
    masked_error = (predictions - task.eval_targets) ** 2 * task.loss_mask[None, :]
    masked_target = task.eval_targets**2 * task.loss_mask[None, :]
    denominator = task.eval_features.shape[0] * mx.sum(task.loss_mask)
    loss = mx.sum(masked_error) / denominator
    baseline = mx.sum(masked_target) / denominator
    mx.eval(loss, baseline)
    _require_finite("evaluation score", loss, baseline)
    return 1.0 - float(loss.item()) / max(float(baseline.item()), 1e-12)


def _candidate_slots(
    manager: LoRAManager,
    protocol: _Protocol,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    donors: list[tuple[str, int]] = []
    recipients: list[tuple[str, int]] = []
    for name, adapter in manager.iter_adapters():
        if not name.endswith("attn.q_proj"):
            continue
        active = set(adapter.active_component_indices)
        if adapter.active_rank > protocol.min_rank:
            donors.extend((name, index) for index in sorted(active))
        if adapter.active_rank < adapter.rank:
            recipients.extend(
                (name, index)
                for index in range(adapter.rank)
                if index not in active
            )
    return donors, recipients


def _probe_loss_guided_transfer(
    manager: LoRAManager,
    loss,
    params: list[mx.array],
    protocol: _Protocol,
    *,
    selection: Literal["guided", "random"] = "guided",
    rng: random.Random | None = None,
    forced_donor: tuple[str, int] | None = None,
    forbidden_recipients: set[tuple[str, int]] | None = None,
) -> dict[str, Any] | None:
    entries = list(manager.iter_adapters())
    adapters = dict(entries)
    base_params = list(params)
    base_gates = {
        name: adapter.active_component_indices
        for name, adapter in entries
    }
    mx.eval(*base_params)

    def restore() -> None:
        manager.set_trainable_parameters(base_params)
        for name, indices in base_gates.items():
            adapters[name].set_active_components(indices)
        if sum(manager.active_rank_state().values()) != protocol.active_rank_budget:
            raise RuntimeError("loss probe failed to restore the active-rank budget")

    candidates: list[dict[str, Any]] = []
    baseline_value = 0.0
    no_swap_after_value = 0.0
    try:
        baseline, baseline_grads = mx.value_and_grad(loss)(base_params)
        no_swap_params = [
            param - protocol.probe_learning_rate * grad
            for param, grad in zip(base_params, baseline_grads, strict=True)
        ]
        no_swap_after = loss(no_swap_params)
        mx.eval(baseline, baseline_grads, no_swap_params, no_swap_after)
        _require_finite("probe baseline", baseline)
        _require_finite("probe baseline gradients", *baseline_grads)
        _require_finite("probe no-swap parameters", *no_swap_params)
        _require_finite("probe no-swap loss", no_swap_after)
        baseline_value = float(baseline.item())
        no_swap_after_value = float(no_swap_after.item())
        restore()
        donors, recipients = _candidate_slots(manager, protocol)
        for donor_name, donor_index in donors:
            for recipient_name, recipient_index in recipients:
                if donor_name == recipient_name:
                    continue
                restore()
                try:
                    manager.transfer_conserved_rank(
                        donor=(donor_name, donor_index),
                        recipient=(recipient_name, recipient_index),
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
                    _require_finite("probe candidate loss", swapped_loss, after_loss)
                    _require_finite("probe candidate gradients", *grads)
                    _require_finite("probe candidate parameters", *virtual_params)
                    after_value = float(after_loss.item())
                    candidates.append(
                        {
                            "donor": (donor_name, donor_index),
                            "recipient": (recipient_name, recipient_index),
                            "probe_swapped_loss": float(swapped_loss.item()),
                            "probe_after_loss": after_value,
                            "predicted_loss_gain": no_swap_after_value - after_value,
                        }
                    )
                finally:
                    restore()
    finally:
        restore()

    if selection == "guided":
        eligible = [
            candidate
            for candidate in candidates
            if math.isfinite(candidate["predicted_loss_gain"])
            and candidate["predicted_loss_gain"] > 1e-8
        ]
        if not eligible:
            return None
        best = max(
            eligible,
            key=lambda candidate: (
                candidate["predicted_loss_gain"],
                candidate["recipient"],
                candidate["donor"],
            ),
        )
        evidence_source = "rank_conserving_loss_lookahead"
    else:
        if rng is None:
            raise ValueError("random shadow-swap selection requires rng")
        forbidden = forbidden_recipients or set()
        eligible = [
            candidate
            for candidate in candidates
            if (forced_donor is None or candidate["donor"] == forced_donor)
            and candidate["recipient"] not in forbidden
        ]
        if not eligible:
            return None
        best = rng.choice(
            sorted(
                eligible,
                key=lambda candidate: (candidate["recipient"], candidate["donor"]),
            )
        )
        evidence_source = "same_timing_random_legal_transfer"
    return {
        "donor_slot": best["donor"],
        "recipient_slot": best["recipient"],
        "evidence_source": evidence_source,
        "probe_base_loss": baseline_value,
        "probe_no_swap_after_loss": no_swap_after_value,
        "probe_swapped_loss": best["probe_swapped_loss"],
        "probe_after_loss": best["probe_after_loss"],
        "predicted_loss_gain": best["predicted_loss_gain"],
        "candidate_count": len(candidates),
    }


def _commit_recycled_transfer(
    manager: LoRAManager,
    params: list[mx.array],
    proposal: dict[str, Any],
    protocol: _Protocol,
    *,
    seed: int,
) -> tuple[list[mx.array], dict[str, Any]]:
    """Commit one gate transfer and strict donor recycle as one transaction."""

    entries = list(manager.iter_adapters())
    adapters = dict(entries)
    positions = {name: index for index, (name, _) in enumerate(entries)}
    base_params = list(params)
    base_gates = {
        name: adapter.active_component_indices for name, adapter in entries
    }
    donor_name, donor_component = proposal["donor_slot"]
    recipient_name, recipient_component = proposal["recipient_slot"]
    donor_adapter = adapters[donor_name]
    donor_position = positions[donor_name]
    recipient_position = positions[recipient_name]
    donor_a_index = 2 * donor_position
    donor_b_index = donor_a_index + 1
    recipient_a_index = 2 * recipient_position

    recipient_column = base_params[recipient_a_index][:, recipient_component]
    mx.eval(recipient_column)
    if not bool(mx.all(mx.equal(recipient_column, 0.0)).item()):
        raise RuntimeError("strict recycle cannot activate a learned inactive recipient")

    try:
        event = manager.transfer_conserved_rank(
            donor=(donor_name, donor_component),
            recipient=(recipient_name, recipient_component),
            total_active_rank=protocol.active_rank_budget,
            min_rank=protocol.min_rank,
        )
        selector = mx.array(
            [
                0.0 if index == donor_component else 1.0
                for index in range(donor_adapter.rank)
            ],
            dtype=base_params[donor_a_index].dtype,
        )
        replacement = 1.0 - selector
        rng = random.Random(seed)
        row = mx.array(
            [
                rng.gauss(0.0, 1.0) / math.sqrt(donor_adapter.input_dim)
                for _ in range(donor_adapter.input_dim)
            ],
            dtype=base_params[donor_b_index].dtype,
        )
        updated = list(base_params)
        updated[donor_a_index] = base_params[donor_a_index] * selector[None, :]
        updated[donor_b_index] = (
            base_params[donor_b_index] * selector[:, None]
            + replacement[:, None] * row[None, :]
        )
        manager.set_trainable_parameters(updated)
        mx.eval(*updated)
        if sum(manager.active_rank_state().values()) != protocol.active_rank_budget:
            raise RuntimeError("strict recycle violated the active-rank budget")
        released_a_column = updated[donor_a_index][:, donor_component]
        released_b_row = updated[donor_b_index][donor_component, :]
        mx.eval(released_a_column, released_b_row, row)
        released_a_column_zero = bool(mx.all(mx.equal(released_a_column, 0.0)).item())
        released_b_row_matches_deterministic_replacement = bool(
            mx.array_equal(released_b_row, row).item()
        )
        recycled_slot_reset_verified = (
            released_a_column_zero
            and released_b_row_matches_deterministic_replacement
        )
        if not recycled_slot_reset_verified:
            raise RuntimeError("strict recycle did not reset the released factor slot")
        non_donor_master_parameters_exact = all(
            bool(mx.array_equal(before, after).item())
            for index, (before, after) in enumerate(zip(base_params, updated, strict=True))
            if index not in {donor_a_index, donor_b_index}
        )
        if not non_donor_master_parameters_exact:
            raise RuntimeError("strict recycle changed an unrelated master parameter")
    except Exception:
        manager.set_trainable_parameters(base_params)
        for name, indices in base_gates.items():
            adapters[name].set_active_components(indices)
        mx.eval(*base_params)
        raise

    event.update(
        {
            key: value
            for key, value in proposal.items()
            if key not in {"donor_slot", "recipient_slot"}
        }
    )
    event.update(
        {
            "non_donor_master_parameters_exact": non_donor_master_parameters_exact,
            "released_a_column_zero": released_a_column_zero,
            "released_b_row_matches_deterministic_replacement": (
                released_b_row_matches_deterministic_replacement
            ),
            "recycled_slot_reset_verified": recycled_slot_reset_verified,
            "storage_semantics": "strict_recycle",
            "inactive_a_column_state_retained": False,
            "historical_erasure_claimed": False,
        }
    )
    return updated, event


def _commit_vault_transfer(
    manager: LoRAManager,
    params: list[mx.array],
    proposal: dict[str, Any],
    protocol: _Protocol,
) -> tuple[list[mx.array], dict[str, Any]]:
    """Move an active gate while deliberately retaining the donor master state."""

    entries = list(manager.iter_adapters())
    adapters = dict(entries)
    positions = {name: index for index, (name, _) in enumerate(entries)}
    base_gates = {
        name: adapter.active_component_indices for name, adapter in entries
    }
    donor_name, donor_component = proposal["donor_slot"]
    donor_a = params[2 * positions[donor_name]][:, donor_component]
    mx.eval(donor_a)
    retained_learned_state = bool(mx.any(mx.abs(donor_a) > 1e-8).item())
    try:
        event = manager.transfer_conserved_rank(
            donor=proposal["donor_slot"],
            recipient=proposal["recipient_slot"],
            total_active_rank=protocol.active_rank_budget,
            min_rank=protocol.min_rank,
        )
        if sum(manager.active_rank_state().values()) != protocol.active_rank_budget:
            raise RuntimeError("vault transfer violated the active-rank budget")
    except Exception:
        for name, indices in base_gates.items():
            adapters[name].set_active_components(indices)
        manager.set_trainable_parameters(params)
        mx.eval(*params)
        raise
    event.update(
        {
            key: value
            for key, value in proposal.items()
            if key not in {"donor_slot", "recipient_slot"}
        }
    )
    event.update(
        {
            "non_donor_master_parameters_exact": True,
            "released_a_column_zero": None,
            "released_b_row_matches_deterministic_replacement": None,
            "recycled_slot_reset_verified": False,
            "storage_semantics": "vault",
            "inactive_a_column_state_retained": retained_learned_state,
            "historical_erasure_claimed": False,
        }
    )
    return params, event


def _oracle_proposal(
    manager: LoRAManager,
    task: _TaskBatch,
    protocol: _Protocol,
) -> dict[str, Any] | None:
    """Construct the explicit hidden-site upper-bound transfer."""

    entries = list(manager.iter_adapters())
    target_name = f"blocks.{task.site}.attn.q_proj"
    adapters = dict(entries)
    target = adapters[target_name]
    if target.active_rank >= protocol.target_rank:
        return None
    target_active = set(target.active_component_indices)
    recipient_component = next(
        component
        for component in range(target.rank)
        if component not in target_active
    )
    donor_candidates = [
        (adapter.active_rank, name, component)
        for name, adapter in entries
        if name != target_name and adapter.active_rank > protocol.min_rank
        for component in adapter.active_component_indices
    ]
    if not donor_candidates:
        return None
    _, donor_name, donor_component = max(donor_candidates)
    return {
        "donor_slot": (donor_name, donor_component),
        "recipient_slot": (target_name, recipient_component),
        "evidence_source": "hidden_site_oracle_upper_bound",
        "probe_base_loss": None,
        "probe_no_swap_after_loss": None,
        "probe_swapped_loss": None,
        "probe_after_loss": None,
        "predicted_loss_gain": None,
        "candidate_count": 0,
    }


def _train_step(
    manager: LoRAManager,
    loss,
    params: list[mx.array],
    learning_rate: float,
) -> tuple[list[mx.array], float]:
    value, grads = mx.value_and_grad(loss)(params)
    updated = [
        param - learning_rate * grad
        for param, grad in zip(params, grads, strict=True)
    ]
    manager.set_trainable_parameters(updated)
    mx.eval(value, *updated)
    _require_finite("training loss", value)
    _require_finite("training gradients", *grads)
    _require_finite("training parameters", *updated)
    return updated, float(value.item())


def _checkpoint_fingerprint(
    manager: LoRAManager,
    params: list[mx.array],
) -> str:
    entries = list(manager.iter_adapters())
    mx.eval(*params)
    payload = []
    for position, (name, adapter) in enumerate(entries):
        master_a = params[2 * position]
        master_b = params[2 * position + 1]
        payload.append(
            {
                "name": name,
                "active": list(adapter.active_component_indices),
                "A": {
                    "dtype": str(master_a.dtype),
                    "shape": list(master_a.shape),
                    "values": master_a.tolist(),
                },
                "B": {
                    "dtype": str(master_b.dtype),
                    "shape": list(master_b.shape),
                    "values": master_b.tolist(),
                },
            }
        )
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _a_column_learned_inactive_rank_lower_bound(
    manager: LoRAManager,
    params: list[mx.array],
) -> int:
    """Count inactive factors with learned A columns, not B-only dormant state."""

    learned = 0
    for position, (_, adapter) in enumerate(manager.iter_adapters()):
        active = set(adapter.active_component_indices)
        master_a = params[2 * position]
        for component in range(adapter.rank):
            if component in active:
                continue
            magnitude = mx.max(mx.abs(master_a[:, component]))
            mx.eval(magnitude)
            if float(magnitude.item()) > 1e-8:
                learned += 1
    return learned


def _numeric_rank(rows: list[list[float]], *, tolerance: float = 1e-6) -> int:
    matrix = [list(map(float, row)) for row in rows]
    if not matrix:
        return 0
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = max(
            range(pivot_row, row_count),
            key=lambda row: abs(matrix[row][column]),
            default=pivot_row,
        )
        if abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        pivot_value = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot_value for value in matrix[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if abs(factor) <= tolerance:
                continue
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    matrix[row], matrix[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _joint_analytic_error(task_a: _TaskBatch, task_b: _TaskBatch) -> float:
    maximum = 0.0
    for task in (task_a, task_b):
        for features, routes, targets in (
            (task.train_features, task.train_routes, task.train_targets),
            (task.probe_features, task.probe_routes, task.probe_targets),
            (task.eval_features, task.eval_routes, task.eval_targets),
        ):
            prediction = (
                routes[:, task_a.site : task_a.site + 1]
                * mx.matmul(features, task_a.transform)
                + routes[:, task_b.site : task_b.site + 1]
                * mx.matmul(features, task_b.transform)
            )
            error = mx.max(
                mx.abs(prediction - targets) * task.loss_mask[None, :]
            )
            mx.eval(error)
            maximum = max(maximum, float(error.item()))
    return maximum


def _run_condition(
    seed: int,
    condition: str,
    protocol: _Protocol,
    *,
    replay: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if condition not in EVIDENCE_CONDITIONS:
        raise ValueError(f"unsupported learned migration condition: {condition}")
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
    if condition in {"fixed_split", "extra_capacity"}:
        for name, adapter in manager.iter_adapters():
            site = int(name.split(".")[1])
            task_rank = 2 if condition == "fixed_split" else protocol.target_rank
            active = task_rank if site in {task_a_site, task_b_site} else protocol.min_rank
            adapter.set_active_components(range(active))
    condition_budget = (
        protocol.active_rank_budget + 2
        if condition == "extra_capacity"
        else protocol.active_rank_budget
    )
    if sum(manager.active_rank_state().values()) != condition_budget:
        raise RuntimeError(f"{condition} initial allocation violates its active-rank budget")
    params = manager.trainable_parameters()
    adapter_entries = list(manager.iter_adapters())
    physical_rank = sum(adapter.rank for _, adapter in adapter_entries)
    physical_fp16_parameter_bytes = sum(
        2 * (int(adapter.A.size) + int(adapter.B.size))
        for _, adapter in adapter_entries
    )
    master_parameter_bytes = sum(4 * int(param.size) for param in params)
    factor_scalar_count = (
        adapter_entries[0][1].output_dim + adapter_entries[0][1].input_dim
    )
    fp16_factor_bytes = 2 * factor_scalar_count
    master_factor_bytes = 4 * factor_scalar_count
    trajectory: list[dict[str, Any]] = []
    global_step = 0
    recycled_rank_cumulative = 0
    end_a_checkpoint_fingerprint = ""
    phase_tasks = (
        ("learn_a", task_b if condition == "never_a" else task_a),
        ("learn_b", task_b),
        ("return_a", task_a),
    )
    for phase, task in phase_tasks:
        for phase_step in range(protocol.phase_steps):
            if trajectory:
                a_before_probe = trajectory[-1]["a_score"]
                b_before_probe = trajectory[-1]["b_score"]
            else:
                a_before_probe = _score(model, task_a)
                b_before_probe = _score(model, task_b)
            rank_before_probe = manager.active_rank_state()
            probe_loss = _loss_fn(
                manager,
                model,
                task.probe_features,
                task.probe_targets,
                task.loss_mask,
            )
            event = None
            guided_opportunity = (
                (
                    condition in {"guided_recycle", "guided_vault", "never_a"}
                    or (phase == "learn_a" and condition in {"static", "random"})
                )
                and global_step % protocol.allocation_interval == 0
            )
            random_opportunity = (
                condition == "random"
                and phase != "learn_a"
                and replay is not None
                and global_step in replay
            )
            oracle_opportunity = (
                condition == "oracle"
                and global_step % protocol.allocation_interval == 0
            )
            if guided_opportunity:
                event = _probe_loss_guided_transfer(
                    manager,
                    probe_loss,
                    params,
                    protocol,
                )
            elif oracle_opportunity:
                event = _oracle_proposal(manager, task, protocol)
            elif random_opportunity:
                assert replay is not None
                replay_event = replay[global_step]
                guided_donor = (
                    replay_event["donor"],
                    int(replay_event["donor_component_index"]),
                )
                guided_recipient = (
                    replay_event["recipient"],
                    int(replay_event["recipient_component_index"]),
                )
                event = _probe_loss_guided_transfer(
                    manager,
                    probe_loss,
                    params,
                    protocol,
                    selection="random",
                    rng=random.Random(seed * 104_729 + global_step),
                )
                if event is None:
                    raise RuntimeError(
                        "random control could not replay a guided transfer opportunity"
                    )
                event.update(
                    {
                        "replay_guided_donor": list(guided_donor),
                        "replay_guided_recipient": list(guided_recipient),
                    }
                )
            transfers: list[dict[str, Any]] = []
            if event is not None:
                if condition == "guided_vault":
                    params, event = _commit_vault_transfer(
                        manager,
                        params,
                        event,
                        protocol,
                    )
                else:
                    params, event = _commit_recycled_transfer(
                        manager,
                        params,
                        event,
                        protocol,
                        seed=seed * 100_000 + global_step * 101 + len(trajectory),
                    )
                    recycled_rank_cumulative += int(event["rank_units"])
                transfers.append(event)
            loss = _loss_fn(
                manager,
                model,
                task.train_features,
                task.train_targets,
                task.loss_mask,
            )
            a_pre_update = _score(model, task_a)
            b_pre_update = _score(model, task_b)
            params, train_loss = _train_step(
                manager,
                loss,
                params,
                protocol.learning_rate,
            )
            rank_map = manager.active_rank_state()
            learned_inactive_rank_lower_bound = (
                _a_column_learned_inactive_rank_lower_bound(manager, params)
            )
            target_name = f"blocks.{task.site}.attn.q_proj"
            row = {
                "seed": seed,
                "condition": condition,
                "phase": phase,
                "phase_step": phase_step,
                "global_step": global_step,
                "task": task.name,
                "rank_map": rank_map,
                "active_rank": sum(rank_map.values()),
                "active_rank_budget": condition_budget,
                "budget_ok": sum(rank_map.values()) == condition_budget,
                "physical_rank": physical_rank,
                "physical_fp16_parameter_bytes": physical_fp16_parameter_bytes,
                "physical_parameter_bytes": physical_fp16_parameter_bytes,
                "physical_float32_master_parameter_bytes": master_parameter_bytes,
                "master_parameter_bytes": master_parameter_bytes,
                "optimizer_state_bytes": 0,
                "active_fp16_factor_bytes": (
                    sum(rank_map.values()) * fp16_factor_bytes
                ),
                "active_master_factor_bytes": (
                    sum(rank_map.values()) * master_factor_bytes
                ),
                "active_factor_bytes": (
                    sum(rank_map.values()) * master_factor_bytes
                ),
                "learned_inactive_rank_a_column_lower_bound": (
                    learned_inactive_rank_lower_bound
                ),
                "learned_inactive_fp16_factor_bytes_lower_bound": (
                    learned_inactive_rank_lower_bound * fp16_factor_bytes
                ),
                "learned_inactive_master_factor_bytes_lower_bound": (
                    learned_inactive_rank_lower_bound * master_factor_bytes
                ),
                "learned_inactive_rank": learned_inactive_rank_lower_bound,
                "learned_inactive_bytes": (
                    learned_inactive_rank_lower_bound * master_factor_bytes
                ),
                "resident_rank_a_column_lower_bound": (
                    sum(rank_map.values()) + learned_inactive_rank_lower_bound
                ),
                "resident_declared_rank": (
                    sum(rank_map.values()) + learned_inactive_rank_lower_bound
                ),
                "recycled_rank_cumulative": recycled_rank_cumulative,
                "target_rank_coverage_before_probe": (
                    rank_before_probe[target_name] / protocol.target_rank
                ),
                "target_rank_coverage": rank_map[target_name] / protocol.target_rank,
                "a_score_before_probe": a_before_probe,
                "b_score_before_probe": b_before_probe,
                "a_score_post_supervised_probe_pre_update": a_pre_update,
                "b_score_post_supervised_probe_pre_update": b_pre_update,
                "a_score_pre_update": a_pre_update,
                "b_score_pre_update": b_pre_update,
                "a_score": _score(model, task_a),
                "b_score": _score(model, task_b),
                "train_loss": train_loss,
                "transfers": transfers,
            }
            trajectory.append(row)
            global_step += 1
        if phase == "learn_a":
            end_a_checkpoint_fingerprint = _checkpoint_fingerprint(manager, params)
    b_rows = [row for row in trajectory if row["phase"] == "learn_b"]
    return_rows = [row for row in trajectory if row["phase"] == "return_a"]
    b_target_name = f"blocks.{task_b.site}.attn.q_proj"
    route_arrays = (
        task_a.train_routes,
        task_a.probe_routes,
        task_a.eval_routes,
        task_b.train_routes,
        task_b.probe_routes,
        task_b.eval_routes,
    )
    route_minima = [float(mx.min(routes).item()) for routes in route_arrays]
    route_maxima = [float(mx.max(routes).item()) for routes in route_arrays]
    route_design_ranks = [
        _numeric_rank(cast(list[list[float]], routes.tolist())) for routes in route_arrays
    ]
    return trajectory, {
        "seed": seed,
        "condition": condition,
        "active_rank_budget": condition_budget,
        "end_a_checkpoint_fingerprint": end_a_checkpoint_fingerprint,
        "budget_invariant": all(row["budget_ok"] for row in trajectory),
        "b_score_auc": sum(row["b_score"] for row in b_rows) / len(b_rows),
        "b_final_score": b_rows[-1]["b_score"],
        "b_final_alignment": b_rows[-1]["target_rank_coverage"],
        "b_migrated_rank": b_rows[-1]["rank_map"][b_target_name],
        "a_score_end_b": b_rows[-1]["a_score"],
        "a_return_post_supervised_probe_pre_update_score": return_rows[0][
            "a_score_post_supervised_probe_pre_update"
        ],
        "a_return_immediate_score": return_rows[0]["a_score_pre_update"],
        "a_return_score_auc": sum(row["a_score"] for row in return_rows)
        / len(return_rows),
        "a_return_final_score": return_rows[-1]["a_score"],
        "transfer_count": sum(len(row["transfers"]) for row in trajectory),
        "max_learned_inactive_rank_a_column_lower_bound": max(
            row["learned_inactive_rank_a_column_lower_bound"] for row in trajectory
        ),
        "max_learned_inactive_rank": max(
            row["learned_inactive_rank_a_column_lower_bound"] for row in trajectory
        ),
        "max_resident_rank_a_column_lower_bound": max(
            row["resident_rank_a_column_lower_bound"] for row in trajectory
        ),
        "max_resident_declared_rank": max(
            row["resident_declared_rank"] for row in trajectory
        ),
        "physical_rank": physical_rank,
        "physical_fp16_parameter_bytes": physical_fp16_parameter_bytes,
        "physical_parameter_bytes": physical_fp16_parameter_bytes,
        "physical_float32_master_parameter_bytes": master_parameter_bytes,
        "master_parameter_bytes": master_parameter_bytes,
        "optimizer_state_bytes": 0,
        "fixture": {
            "task_a_site": task_a_site,
            "task_b_site": task_b_site,
            "task_a_transform_fingerprint": task_a.transform_fingerprint,
            "task_b_transform_fingerprint": task_b.transform_fingerprint,
            "route_minimum": min(route_minima),
            "route_maximum": max(route_maxima),
            "minimum_route_design_rank": min(route_design_ranks),
            "task_a_transform_rank": _numeric_rank(cast(list[list[float]], task_a.transform.tolist())),
            "task_b_transform_rank": _numeric_rank(cast(list[list[float]], task_b.transform.tolist())),
            "task_a_output_head": list(task_a.output_head),
            "task_b_output_head": list(task_b.output_head),
            "joint_sufficient_analytic_max_abs_error": _joint_analytic_error(
                task_a, task_b
            ),
        },
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric vector")
    return sum(values) / len(values)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def _paired_bootstrap(
    left: list[float],
    right: list[float],
    *,
    protocol: _Protocol,
    seed_offset: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("paired control metrics must be non-empty and aligned")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(protocol.bootstrap_seed + seed_offset)
    bootstrap = sorted(
        _mean([differences[rng.randrange(len(differences))] for _ in differences])
        for _ in range(protocol.bootstrap_resamples)
    )
    tail = (1.0 - protocol.confidence_level) / 2.0
    return {
        "method": "paired_fixture_seed_bootstrap",
        "pairs": len(differences),
        "mean_difference": _mean(differences),
        "confidence_level": protocol.confidence_level,
        "ci_lower": _percentile(bootstrap, tail),
        "ci_upper": _percentile(bootstrap, 1.0 - tail),
        "probability_left_better": sum(value > 0.0 for value in bootstrap)
        / len(bootstrap),
        "resamples": protocol.bootstrap_resamples,
        "seed": protocol.bootstrap_seed + seed_offset,
    }


def _aggregate(
    runs: list[dict[str, Any]],
    conditions: tuple[str, ...],
) -> list[dict[str, Any]]:
    numeric = (
        "b_migrated_rank",
        "b_final_alignment",
        "b_score_auc",
        "b_final_score",
        "a_score_end_b",
        "a_return_post_supervised_probe_pre_update_score",
        "a_return_immediate_score",
        "a_return_score_auc",
        "a_return_final_score",
        "transfer_count",
        "max_learned_inactive_rank_a_column_lower_bound",
        "max_learned_inactive_rank",
        "max_resident_rank_a_column_lower_bound",
        "max_resident_declared_rank",
        "physical_rank",
        "physical_fp16_parameter_bytes",
        "physical_parameter_bytes",
        "physical_float32_master_parameter_bytes",
        "master_parameter_bytes",
        "optimizer_state_bytes",
    )
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        selected = [run for run in runs if run["condition"] == condition]
        if not selected:
            continue
        row: dict[str, Any] = {
            "condition": condition,
            "seeds": len(selected),
            "active_rank_budget": selected[0]["active_rank_budget"],
            "budget_pass_rate": _mean(
                [float(run["budget_invariant"]) for run in selected]
            ),
        }
        row.update(
            {
                f"{field}_mean": _mean([float(run[field]) for run in selected])
                for field in numeric
            }
        )
        rows.append(row)
    return rows


def _event_windows(
    trajectory: list[dict[str, Any]],
    seeds: tuple[int, ...],
    protocol: _Protocol,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for seed in seeds:
        guided = {
            row["global_step"]: row
            for row in trajectory
            if row["seed"] == seed and row["condition"] == "guided_recycle"
        }
        static = {
            row["global_step"]: row
            for row in trajectory
            if row["seed"] == seed and row["condition"] == "static"
        }
        if not guided or not static:
            continue
        first_b_transfer = next(
            (
                row
                for row in guided.values()
                if row["phase"] == "learn_b" and row["transfers"]
            ),
            None,
        )
        if first_b_transfer is None:
            continue
        rank_move_step = int(first_b_transfer["global_step"])
        measurement_step = rank_move_step + protocol.allocation_interval - 1
        guided_measurement = guided[measurement_step]
        static_before = static[rank_move_step]
        static_measurement = static[measurement_step]
        windows.append(
            {
                "seed": seed,
                "event_semantics": "first_b_phase_loss_guided_transfer",
                "rank_move_step": rank_move_step,
                "measurement_step": measurement_step,
                "window_steps": protocol.allocation_interval,
                "pre_transfer_score_gap": first_b_transfer[
                    "b_score_before_probe"
                ]
                - static_before["b_score_before_probe"],
                "post_window_score_advantage": guided_measurement["b_score"]
                - static_measurement["b_score"],
                "guided_post_window_b_score": guided_measurement["b_score"],
                "static_post_window_b_score": static_measurement["b_score"],
                "transfer": first_b_transfer["transfers"][0],
            }
        )
    return windows


def _promotion_gates(
    runs: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    *,
    configured_conditions: tuple[str, ...],
    seeds: tuple[int, ...],
    protocol: _Protocol,
    integrity: dict[str, Any],
    event_windows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    by_condition = {
        condition: sorted(
            (run for run in runs if run["condition"] == condition),
            key=lambda run: run["seed"],
        )
        for condition in configured_conditions
    }
    aggregate_by_condition = {row["condition"]: row for row in aggregate}

    def paired_condition_metric(
        left_condition: str,
        right_condition: str,
        field: str,
        seed_offset: int,
    ) -> dict[str, Any]:
        left_by_seed = {
            run["seed"]: run[field] for run in by_condition.get(left_condition, [])
        }
        right_by_seed = {
            run["seed"]: run[field] for run in by_condition.get(right_condition, [])
        }
        paired_seeds = sorted(set(left_by_seed) & set(right_by_seed))
        if not paired_seeds:
            return {
                "method": "paired_fixture_seed_bootstrap",
                "pairs": 0,
                "mean_difference": 0.0,
                "confidence_level": protocol.confidence_level,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "probability_left_better": 0.0,
                "resamples": protocol.bootstrap_resamples,
                "seed": protocol.bootstrap_seed + seed_offset,
            }
        return _paired_bootstrap(
            [float(left_by_seed[seed]) for seed in paired_seeds],
            [float(right_by_seed[seed]) for seed in paired_seeds],
            protocol=protocol,
            seed_offset=seed_offset,
        )

    comparisons = {}
    for offset, control in enumerate(("static", "fixed_split", "random")):
        comparisons[f"guided_recycle_vs_{control}_b_score_auc"] = (
            paired_condition_metric(
                "guided_recycle",
                control,
                "b_score_auc",
                offset,
            )
        )
    if event_windows:
        comparisons["guided_event_window_vs_static"] = _paired_bootstrap(
            [window["post_window_score_advantage"] for window in event_windows],
            [0.0 for _ in event_windows],
            protocol=protocol,
            seed_offset=3,
        )
    else:
        comparisons["guided_event_window_vs_static"] = {
            "method": "paired_fixture_seed_bootstrap",
            "pairs": 0,
            "mean_difference": 0.0,
            "confidence_level": protocol.confidence_level,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "probability_left_better": 0.0,
            "resamples": protocol.bootstrap_resamples,
            "seed": protocol.bootstrap_seed + 3,
        }
    comparisons["guided_recycle_vs_random_b_final_alignment"] = (
        paired_condition_metric(
            "guided_recycle",
            "random",
            "b_final_alignment",
            4,
        )
    )
    enough_seeds = seeds == CONFIRMATORY_SEEDS
    expected_seed_condition_pairs = {
        (seed, condition)
        for seed in seeds
        for condition in configured_conditions
    }
    actual_seed_condition_pairs = {
        (int(run["seed"]), str(run["condition"])) for run in runs
    }
    complete_matrix = (
        not failures
        and len(runs) == len(expected_seed_condition_pairs)
        and actual_seed_condition_pairs == expected_seed_condition_pairs
    )
    budget_passed = all(row["budget_pass_rate"] == 1.0 for row in aggregate)
    comparisons_passed = all(
        comparisons[f"guided_recycle_vs_{control}_b_score_auc"]["ci_lower"]
        > 0.0
        and comparisons[f"guided_recycle_vs_{control}_b_score_auc"]["pairs"]
        == len(seeds)
        for control in ("static", "fixed_split", "random")
    )
    guided_row = aggregate_by_condition.get("guided_recycle")
    guided_alignment = (
        guided_row["b_final_alignment_mean"] if guided_row is not None else 0.0
    )
    alignment_comparison = comparisons[
        "guided_recycle_vs_random_b_final_alignment"
    ]
    migration_localized = (
        alignment_comparison["ci_lower"] > 0.0
        and alignment_comparison["pairs"] == len(seeds)
    )
    event_ordering_passed = (
        all(abs(window["pre_transfer_score_gap"]) < 1e-6 for window in event_windows)
        and all(
            window["rank_move_step"] < window["measurement_step"]
            for window in event_windows
        )
        and comparisons["guided_event_window_vs_static"]["ci_lower"] > 0.0
        and comparisons["guided_event_window_vs_static"]["pairs"] == len(seeds)
    )
    joint = aggregate_by_condition.get("extra_capacity")
    joint_learnable = joint is not None and (
        joint["a_score_end_b_mean"] >= protocol.score_threshold
        and joint["b_final_score_mean"] >= protocol.score_threshold
    )
    declared_control_conditions_present = all(
        condition in aggregate_by_condition
        for condition in ("guided_vault", "oracle", "never_a")
    )
    vault_post_probe_access_has_inactive_a_column_evidence = False
    recycle_post_probe_access_is_low = False
    never_a_is_scratch_like = False
    oracle_bounds_localization = False
    if declared_control_conditions_present:
        vault = aggregate_by_condition["guided_vault"]
        recycle = aggregate_by_condition["guided_recycle"]
        never_a = aggregate_by_condition["never_a"]
        oracle = aggregate_by_condition["oracle"]
        vault_post_probe_access_has_inactive_a_column_evidence = (
            vault["a_score_end_b_mean"] < 0.2
            and vault[
                "a_return_post_supervised_probe_pre_update_score_mean"
            ]
            >= 0.5
            and vault["a_return_post_supervised_probe_pre_update_score_mean"]
            > recycle["a_return_post_supervised_probe_pre_update_score_mean"]
            + 0.3
            and vault[
                "max_learned_inactive_rank_a_column_lower_bound_mean"
            ]
            > 0.0
        )
        recycle_post_probe_access_is_low = (
            recycle["a_return_post_supervised_probe_pre_update_score_mean"] < 0.2
            and recycle[
                "max_learned_inactive_rank_a_column_lower_bound_mean"
            ]
            == 0.0
        )
        never_a_is_scratch_like = (
            never_a["a_return_post_supervised_probe_pre_update_score_mean"] < 0.2
        )
        oracle_bounds_localization = (
            oracle["b_final_alignment_mean"] >= guided_alignment
        )
    criteria = [
        {"id": "at_least_ten_confirmatory_seeds", "passed": enough_seeds},
        {"id": "complete_finite_seed_condition_matrix", "passed": complete_matrix},
        {"id": "active_rank_budget_conserved", "passed": budget_passed},
        {
            "id": "guided_rank_localizes_b_more_than_random",
            "passed": migration_localized,
        },
        {
            "id": "guided_b_acquisition_beats_matched_controls",
            "passed": comparisons_passed,
        },
        {
            "id": "rank_move_precedes_matched_b_advantage",
            "passed": event_ordering_passed,
        },
        {
            "id": "strict_recycle_slot_reset_is_verified",
            "passed": integrity["strict_recycle_slot_reset_verified"],
        },
        {"id": "joint_sufficient_capacity_learns_both", "passed": joint_learnable},
        {
            "id": "declared_control_conditions_present",
            "passed": declared_control_conditions_present,
        },
        {
            "id": "vault_post_supervised_probe_access_has_inactive_a_column_evidence",
            "passed": vault_post_probe_access_has_inactive_a_column_evidence,
        },
        {
            "id": "strict_recycle_post_supervised_probe_access_is_low",
            "passed": recycle_post_probe_access_is_low,
        },
        {"id": "never_a_reference_is_scratch_like", "passed": never_a_is_scratch_like},
        {"id": "oracle_bounds_learned_localization", "passed": oracle_bounds_localization},
        {"id": "unlabeled_cue_wake_measured", "passed": False},
    ]
    passed = all(item["passed"] for item in criteria)
    return {
        "status": (
            "learned_dense_capacity_migration_gate_passed"
            if passed
            else "learned_dense_capacity_migration_gate_failed"
        ),
        "passed": passed,
        "criteria": criteria,
        "paired_comparisons": comparisons,
        "kill_criteria": [
            {"id": "active_budget_violation", "triggered": not budget_passed},
            {
                "id": "random_transfer_equivalent",
                "triggered": not comparisons[
                    "guided_recycle_vs_random_b_score_auc"
                ]["ci_lower"]
                > 0.0,
            },
            {
                "id": "fixed_or_static_allocation_equivalent",
                "triggered": not all(
                    comparisons[key]["ci_lower"] > 0.0
                    for key in (
                        "guided_recycle_vs_static_b_score_auc",
                        "guided_recycle_vs_fixed_split_b_score_auc",
                    )
                ),
            },
            {"id": "rank_did_not_localize", "triggered": not migration_localized},
            {
                "id": "rank_move_did_not_precede_performance",
                "triggered": not event_ordering_passed,
            },
        ],
    }


def run_learned_capacity_migration(
    config: LearnedMigrationConfig | None = None,
) -> dict[str, Any]:
    """Run the canonical local MLX learned-demand protocol."""

    resolved = config or LearnedMigrationConfig()
    seeds = resolved.resolved_seeds()
    protocol = _Protocol()
    conditions = (
        CORE_CONDITIONS if resolved.mode == "smoke" else EVIDENCE_CONDITIONS
    )
    trajectories: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    guided_fixtures: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        guided_rows: list[dict[str, Any]] | None = None
        try:
            guided_rows, guided_summary = _run_condition(
                seed,
                "guided_recycle",
                protocol,
            )
        except FloatingPointError as exc:
            failures.append(
                {
                    "seed": seed,
                    "condition": "guided_recycle",
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            trajectories.extend(guided_rows)
            runs.append(guided_summary)
            guided_fixtures.append(guided_summary["fixture"])
        replay = (
            {
                row["global_step"]: row["transfers"][0]
                for row in guided_rows
                if row["phase"] != "learn_a" and row["transfers"]
            }
            if guided_rows is not None
            else None
        )
        for condition in conditions:
            if condition == "guided_recycle":
                continue
            if condition == "random" and replay is None:
                failures.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "failure_type": "DependencyFailure",
                        "message": "guided replay was unavailable after a numerical failure",
                    }
                )
                continue
            try:
                rows, summary = _run_condition(
                    seed,
                    condition,
                    protocol,
                    replay=replay,
                )
            except FloatingPointError as exc:
                failures.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "failure_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            trajectories.extend(rows)
            runs.append(summary)
    committed_transfers = [
        event
        for row in trajectories
        for event in row["transfers"]
    ]
    strict_recycle_transfers = [
        event
        for event in committed_transfers
        if event["storage_semantics"] == "strict_recycle"
    ]
    strict_recycle_slot_reset_verified = bool(strict_recycle_transfers) and all(
        event["non_donor_master_parameters_exact"]
        and event["recycled_slot_reset_verified"]
        and event["released_a_column_zero"]
        and event["released_b_row_matches_deterministic_replacement"]
        for event in strict_recycle_transfers
    )
    integrity = {
        "strict_recycle_slot_reset_verified": strict_recycle_slot_reset_verified,
        "historical_erasure_claimed": False,
        "scope": (
            "Verifies deterministic replacement of the released factor slot and "
            "exact preservation of unrelated float32 master parameters only."
        ),
    }
    aggregate = _aggregate(runs, conditions)
    event_windows = _event_windows(trajectories, seeds, protocol)
    gates = _promotion_gates(
        runs,
        aggregate,
        configured_conditions=conditions,
        seeds=seeds,
        protocol=protocol,
        integrity=integrity,
        event_windows=event_windows,
        failures=failures,
    )
    if resolved.mode == "smoke":
        evidence_status = "learned_mlx_smoke_only"
        claim_boundary = (
            "This smoke run uses real MLX loss and gradients with rank-conserving "
            "shadow swaps over a frozen, input-derived dense router. It is not an "
            "evidence run and cannot promote the thesis."
        )
    elif resolved.mode == "development":
        evidence_status = "learned_mlx_development_only"
        claim_boundary = (
            "This development run exercises the full declared control matrix on "
            "development seed 0. It is not a confirmatory evidence run and cannot "
            "promote the thesis."
        )
    else:
        evidence_status = gates["status"]
        claim_boundary = (
            "This confirmatory synthetic MLX run uses exhaustive supervised loss "
            "lookahead to reallocate effective active rank. Its verdict is governed "
            "by the complete seed/control matrix; it does not establish physical-memory "
            "conservation, autonomous self-reorganization, human-like forgetting, or a "
            "large-model result."
        )
    return {
        "kind": "learned_mlx_capacity_migration",
        "schema_version": 2,
        "protocol": resolved.protocol,
        "mode": resolved.mode,
        "evidence_status": evidence_status,
        "config": {**asdict(protocol), "seeds": list(seeds)},
        "seed_split": {
            "frozen": True,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
            "selected_partition": (
                "confirmatory" if resolved.mode == "evidence" else "development"
            ),
            "selected_seeds": list(seeds),
        },
        "conditions": list(conditions),
        "failures": failures,
        "fixture": {
            "router": {
                "kind": "frozen_input_dependent_dense",
                "accepts_explicit_routes": False,
                "epsilon": protocol.router_epsilon,
                "minimum_weight_observed": min(
                    item["route_minimum"] for item in guided_fixtures
                ),
                "maximum_weight_observed": max(
                    item["route_maximum"] for item in guided_fixtures
                ),
                "minimum_route_design_rank": min(
                    item["minimum_route_design_rank"] for item in guided_fixtures
                ),
            },
            "task_transforms_distinct": all(
                item["task_a_transform_fingerprint"]
                != item["task_b_transform_fingerprint"]
                for item in guided_fixtures
            ),
            "site_metadata_reaches_allocator": False,
            "allocator_observations": ["loss", "parameters", "active_masks"],
            "task_transform_ranks": {
                "A": min(
                    item["task_a_transform_rank"] for item in guided_fixtures
                ),
                "B": min(
                    item["task_b_transform_rank"] for item in guided_fixtures
                ),
            },
            "task_output_heads": {
                "A": guided_fixtures[0]["task_a_output_head"],
                "B": guided_fixtures[0]["task_b_output_head"],
            },
            "joint_sufficient_analytic_max_abs_error": max(
                item["joint_sufficient_analytic_max_abs_error"]
                for item in guided_fixtures
            ),
        },
        "integrity": integrity,
        "measurement_semantics": {
            "canonical_a_return_metric": (
                "a_return_post_supervised_probe_pre_update_score"
            ),
            "deprecated_alias": "a_return_immediate_score",
            "supervised_probe_precedes_measurement": True,
            "parameter_update_precedes_measurement": False,
            "cue_triggered_wake_tested": False,
            "unlabeled_cue_wake_measured": False,
        },
        "event_window_semantics": {
            "event": "first_b_phase_loss_guided_transfer",
            "phase": "learn_b",
            "directionality_claimed": False,
        },
        "capacity_accounting": {
            "conserves_effective_active_rank": True,
            "conserves_physical_bytes": False,
            "physical_rank_is_preallocated": True,
            "optimizer": "stateless_sgd",
            "optimizer_state_bytes": 0,
            "master_parameters_reported_separately": True,
            "active_byte_fields": {
                "fp16": "active_fp16_factor_bytes",
                "float32_master": "active_master_factor_bytes",
            },
            "inactive_byte_lower_bound_fields": {
                "fp16": "learned_inactive_fp16_factor_bytes_lower_bound",
                "float32_master": (
                    "learned_inactive_master_factor_bytes_lower_bound"
                ),
            },
            "dormant_ledger": {
                "status": "provisional_a_column_lower_bound",
                "detection": "inactive_A_columns_with_magnitude_above_1e-8",
                "detects_b_only_learned_state": False,
            },
            "deprecated_aliases": {
                "physical_parameter_bytes": "physical_fp16_parameter_bytes",
                "master_parameter_bytes": (
                    "physical_float32_master_parameter_bytes"
                ),
                "active_factor_bytes": "active_master_factor_bytes",
                "learned_inactive_rank": (
                    "learned_inactive_rank_a_column_lower_bound"
                ),
                "learned_inactive_bytes": (
                    "learned_inactive_master_factor_bytes_lower_bound"
                ),
                "resident_declared_rank": (
                    "resident_rank_a_column_lower_bound"
                ),
            },
            "scope": (
                "The core condition conserves effective active rank only. All physical "
                "LoRA slots and float32 training masters remain resident and are reported "
                "separately."
            ),
        },
        "runs": runs,
        "aggregate": aggregate,
        "gates": gates,
        "event_windows": event_windows,
        "claim_boundary": claim_boundary,
        "trajectory": trajectories,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the learned benchmark's compact evidence boundary and metrics."""

    lines = [
        "# Learned MLX Capacity Migration Benchmark",
        "",
        f"**Verdict:** `{report['evidence_status']}`",
        "",
        report["claim_boundary"],
        "",
        "The allocator uses real MLX loss and gradient lookahead; hidden task-site "
        "metadata is retained for diagnostics only.",
        "",
        "Return-A access is measured after the supervised allocation probe and before "
        "the parameter update. Unlabeled cue-triggered wake is not measured.",
        "",
        "Event windows begin at the first loss-guided transfer observed during the "
        "B-training phase; no transfer directionality is claimed.",
        "",
        "The inactive-state ledger is a provisional lower bound from learned inactive "
        "A columns and does not detect B-only state.",
        "",
        "| condition | n | active R | B AUC | B final | A after B | A return post-probe / pre-update | inactive A-col R lower bound | budget pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"| {row['condition']} | {row['seeds']} | "
            f"{row['active_rank_budget']} | "
            f"{row['b_score_auc_mean']:.3f} | {row['b_final_score_mean']:.3f} | "
            f"{row['a_score_end_b_mean']:.3f} | "
            f"{row['a_return_post_supervised_probe_pre_update_score_mean']:.3f} | "
            f"{row['max_learned_inactive_rank_a_column_lower_bound_mean']:.3f} | "
            f"{row['budget_pass_rate']:.3f} |"
        )
    lines.extend(["", "## Declared gates", ""])
    for criterion in report["gates"]["criteria"]:
        marker = "x" if criterion["passed"] else " "
        lines.append(f"- [{marker}] `{criterion['id']}`")
    if report["failures"]:
        lines.extend(["", "## Numerical failures", ""])
        for failure in report["failures"]:
            lines.append(
                f"- seed {failure['seed']} / `{failure['condition']}`: "
                f"`{failure['failure_type']}` — {failure['message']}"
            )
    lines.extend(["", "## Paired seed comparisons", ""])
    for name, comparison in report["gates"]["paired_comparisons"].items():
        lines.append(
            f"- `{name}`: pairs={comparison['pairs']}, "
            f"mean={comparison['mean_difference']:.3f}, "
            f"{comparison['confidence_level']:.0%} CI "
            f"[{comparison['ci_lower']:.3f}, {comparison['ci_upper']:.3f}]"
        )
    lines.extend(
        [
            "",
            "## Capacity boundary",
            "",
            report["capacity_accounting"]["scope"],
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write the standard four-file learned benchmark artifact set."""

    return write_benchmark_artifacts(
        report,
        output_dir,
        markdown=render_markdown(report),
    )


__all__ = [
    "LearnedMigrationConfig",
    "render_markdown",
    "run_learned_capacity_migration",
    "write_artifacts",
]
