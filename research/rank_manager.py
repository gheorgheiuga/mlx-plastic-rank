"""Historical allocation policies, outside the installed pack lifecycle."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mlx.core as mx
import numpy as np

from mlx_plastic_rank.packs.inspection import ALLOWED_RANKS
from mlx_plastic_rank.packs.lora import SliceLoRA
from mlx_plastic_rank.packs.manager import (
    LoRAManager,
    PackApplicationError,
    _get_nested_attr,
    _linear_weight_array,
)
from mlx_plastic_rank.rank_select import choose_rank


class ResearchLoRAManager(LoRAManager):
    """Preserve legacy initialization and parked rank-controller behavior."""

    def initialize_adapters(self, *args, initialization="legacy", **kwargs):
        return super().initialize_adapters(*args, initialization=initialization, **kwargs)

    def compute_auto_ranks(
        self,
        targets: List[str],
        *,
        strategy: str,
        target_compression: float,
        eps: float = 1e-6,
        allowed_ranks: tuple[int, ...] | None = None,
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        if strategy not in {"stable", "gram_energy", "theorem"}:
            raise PackApplicationError(f"Unsupported rank strategy '{strategy}'")
        if not targets:
            raise PackApplicationError("No targets specified for auto rank selection")

        block0 = self._blocks()[0]
        rank_map: Dict[str, int] = {}
        alpha_map: Dict[str, float] = {}
        residuals: Dict[str, float] = {}

        allowed = sorted(allowed_ranks or ALLOWED_RANKS)

        for target in targets:
            spec = self._target_specs.get(target)
            if spec is None:
                raise PackApplicationError(f"Target '{target}' not supported for model type '{self.model_type}'")
            linear = _get_nested_attr(block0, spec.wrapper_attr)
            weight = _linear_weight_array(linear)
            mx.eval(weight)
            slice_weight = weight[spec.start : spec.end, :].astype(mx.float32)
            if int(slice_weight.shape[0]) == 0 or int(slice_weight.shape[1]) == 0:
                raise PackApplicationError(f"Unable to compute rank for empty slice '{target}'")
            mat_for_rank = slice_weight
            if strategy in {"gram_energy", "theorem"}:
                mat_for_rank = mx.matmul(slice_weight, mx.transpose(slice_weight))
            mx.eval(mat_for_rank)
            rank, residual = choose_rank(mat_for_rank, target_compression, strategy=strategy, eps=eps)
            if rank <= 0:
                rank = allowed[0]
            selected = allowed[-1]
            for candidate in allowed:
                if candidate >= rank:
                    selected = candidate
                    break
            rank_map[target] = selected
            alpha_map[target] = 2.0 * selected
            residuals[target] = float(residual)
        return rank_map, alpha_map, residuals


    @staticmethod
    def _adapter_rank_signal(adapter: SliceLoRA) -> float:
        if adapter.alpha == 0.0:
            return 0.0
        active_indices = adapter.active_component_indices
        if not active_indices:
            return 0.0
        indices = mx.array(active_indices, dtype=mx.int32)
        A = mx.take(adapter.A, indices, axis=1).astype(mx.float32)
        B = mx.take(adapter.B, indices, axis=0).astype(mx.float32)
        if int(A.shape[0]) == 0 or int(A.shape[1]) == 0 or int(B.shape[0]) == 0 or int(B.shape[1]) == 0:
            return 0.0
        column_norms = mx.sqrt(mx.sum(A * A, axis=0))
        row_norms = mx.sqrt(mx.sum(B * B, axis=1))
        utilities = column_norms * row_norms
        return float(mx.sum(utilities).item())


    @staticmethod
    def _active_rank_choices(
        adapter: SliceLoRA,
        allowed_ranks: tuple[int, ...],
        min_rank: int,
    ) -> list[int]:
        allowed = sorted(rank for rank in allowed_ranks if rank <= adapter.rank)
        if not allowed:
            return [adapter.rank]
        floor = min_rank
        candidates = [rank for rank in allowed if rank >= floor]
        if candidates:
            return candidates
        return [allowed[0]]


    def _conserved_rank_adapters(self, target_suffix: str) -> list[SliceLoRA]:
        adapters = sorted(
            (
                adapter
                for adapter in self._adapter_registry.values()
                if adapter.name.endswith(target_suffix)
            ),
            key=lambda adapter: adapter.name,
        )
        if not adapters:
            raise PackApplicationError(
                f"No adapters match conserved-rank target suffix '{target_suffix}'"
            )
        shapes = {
            (adapter.output_dim, adapter.input_dim, adapter.rank)
            for adapter in adapters
        }
        if len(shapes) != 1:
            raise PackApplicationError(
                "Conserved rank currently requires equal-shaped adapters; "
                f"found {sorted(shapes)} for suffix '{target_suffix}'"
            )
        return adapters


    def active_rank_state(self, *, target_suffix: str = "attn.q_proj") -> dict[str, int]:
        """Return the deterministic active-rank allocation for one target family."""

        return {
            adapter.name: adapter.active_rank
            for adapter in self._conserved_rank_adapters(target_suffix)
        }


    def transfer_conserved_rank(
        self,
        *,
        donor: tuple[str, int],
        recipient: tuple[str, int],
        total_active_rank: int,
        min_rank: int = 1,
        target_suffix: str = "attn.q_proj",
    ) -> dict[str, Any]:
        """Atomically exchange one active factor slot without reordering arrays."""

        if min_rank < 1:
            raise PackApplicationError("min_rank must be at least 1")
        adapters = self._conserved_rank_adapters(target_suffix)
        by_name = {adapter.name: adapter for adapter in adapters}
        donor_name, donor_index = donor
        recipient_name, recipient_index = recipient
        if donor_name == recipient_name:
            raise PackApplicationError("Conserved transfer requires different adapters")
        if donor_name not in by_name:
            raise PackApplicationError(f"Unknown donor adapter '{donor_name}'")
        if recipient_name not in by_name:
            raise PackApplicationError(f"Unknown recipient adapter '{recipient_name}'")

        donor_adapter = by_name[donor_name]
        recipient_adapter = by_name[recipient_name]
        if donor_index < 0 or donor_index >= donor_adapter.rank:
            raise PackApplicationError(
                f"Donor component index {donor_index} is outside '{donor_name}'"
            )
        if recipient_index < 0 or recipient_index >= recipient_adapter.rank:
            raise PackApplicationError(
                f"Recipient component index {recipient_index} is outside '{recipient_name}'"
            )

        state_before = {adapter.name: adapter.active_rank for adapter in adapters}
        budget_before = sum(state_before.values())
        if budget_before != total_active_rank:
            raise PackApplicationError(
                "Conserved-rank budget mismatch: "
                f"expected {total_active_rank} active ranks, found {budget_before}"
            )
        below_floor = {
            name: rank for name, rank in state_before.items() if rank < min_rank
        }
        if below_floor:
            raise PackApplicationError(
                f"Current allocation violates min_rank={min_rank}: {below_floor}"
            )
        donor_active = donor_adapter.active_component_indices
        recipient_active = recipient_adapter.active_component_indices
        if donor_index not in donor_active:
            raise PackApplicationError(
                f"Donor component {donor_name}[{donor_index}] is not active"
            )
        if recipient_index in recipient_active:
            raise PackApplicationError(
                f"Recipient component {recipient_name}[{recipient_index}] is already active"
            )
        if donor_adapter.active_rank <= min_rank:
            raise PackApplicationError(
                f"Donor '{donor_name}' cannot fall below min_rank={min_rank}"
            )

        snapshots = {
            donor_name: donor_active,
            recipient_name: recipient_active,
        }
        try:
            donor_adapter.set_active_components(
                index for index in donor_active if index != donor_index
            )
            recipient_adapter.set_active_components((*recipient_active, recipient_index))
            state_after = {adapter.name: adapter.active_rank for adapter in adapters}
            budget_after = sum(state_after.values())
            if budget_after != total_active_rank:
                raise RuntimeError(
                    "Conserved-rank invariant violated: "
                    f"expected {total_active_rank}, found {budget_after}"
                )
            below_floor = {
                name: rank for name, rank in state_after.items() if rank < min_rank
            }
            if below_floor:
                raise RuntimeError(
                    f"Conserved-rank transfer violates min_rank={min_rank}: {below_floor}"
                )
        except Exception:
            donor_adapter.set_active_components(snapshots[donor_name])
            recipient_adapter.set_active_components(snapshots[recipient_name])
            raise

        return {
            "action": "transfer",
            "donor": donor_name,
            "recipient": recipient_name,
            "rank_units": 1,
            "donor_from_rank": state_before[donor_name],
            "donor_to_rank": state_after[donor_name],
            "recipient_from_rank": state_before[recipient_name],
            "recipient_to_rank": state_after[recipient_name],
            "donor_component_index": donor_index,
            "recipient_component_index": recipient_index,
            "budget_before": budget_before,
            "budget_after": budget_after,
            "active_rank_before": state_before,
            "active_rank_after": state_after,
        }


    def adjust_conserved_ranks(
        self,
        *,
        total_active_rank: int,
        min_rank: int = 1,
        max_transfers: int = 1,
        seed: int = 0,
        target_suffix: str = "attn.q_proj",
    ) -> list[dict[str, Any]]:
        """Move rank while preserving the caller-declared active-rank count.

        Each event transfers exactly one rank unit from the adapter with the
        lowest marginal active-component utility to the eligible adapter with
        the highest mean active-component utility. The seed is used only to
        resolve equal-score candidates, so repeated runs are deterministic.

        The norm-product score is an actuation heuristic over already-active
        factors, not a counterfactual recipient-demand measure. It cannot by
        itself establish where a new task needs rank, and it is not invariant
        to general rotations of equivalent LoRA factors. Alpha-zero adapters
        are treated as having zero functional utility.

        This invariant does not conserve physical bytes, optimizer state,
        resident learned factors, or information.

        Unit transfers may create ranks that pack profiles cannot serialize.
        Rebalance to profile-allowed ranks before exporting a runtime state.
        """

        if total_active_rank < 0:
            raise PackApplicationError("total_active_rank must be non-negative")
        if min_rank < 1:
            raise PackApplicationError(
                "min_rank must be at least 1 so every adapter retains "
                "a live path for self-discovery"
            )
        if max_transfers < 0:
            raise PackApplicationError("max_transfers must be non-negative")

        adapters = self._conserved_rank_adapters(target_suffix)
        if any(min_rank > adapter.rank for adapter in adapters):
            raise PackApplicationError("min_rank cannot exceed an adapter's maximum rank")

        initial_state = {adapter.name: adapter.active_rank for adapter in adapters}
        initial_total = sum(initial_state.values())
        if initial_total != total_active_rank:
            raise PackApplicationError(
                "Conserved-rank budget mismatch: "
                f"expected {total_active_rank} active ranks, found {initial_total}"
            )
        if any(rank < min_rank for rank in initial_state.values()):
            raise PackApplicationError(
                f"Current allocation violates min_rank={min_rank}: {initial_state}"
            )

        rng = np.random.default_rng(seed)
        tie_breakers = {
            adapter.name: float(rng.random())
            for adapter in adapters
        }
        component_tie_breakers = {
            (adapter.name, index): float(rng.random())
            for adapter in adapters
            for index in range(adapter.rank)
        }
        events: list[dict[str, Any]] = []
        for _ in range(max_transfers):
            recipients = [adapter for adapter in adapters if adapter.active_rank < adapter.rank]
            if not recipients:
                break

            signals = {
                adapter.name: self._adapter_rank_signal(adapter)
                for adapter in adapters
            }
            recipient_means = {
                adapter.name: (
                    signals[adapter.name] / adapter.active_rank
                    if adapter.active_rank > 0
                    else 0.0
                )
                for adapter in recipients
            }
            recipient = max(
                recipients,
                key=lambda adapter: (
                    recipient_means[adapter.name],
                    tie_breakers[adapter.name],
                    adapter.name,
                ),
            )

            donors = [
                adapter
                for adapter in adapters
                if adapter.name != recipient.name and adapter.active_rank > min_rank
            ]
            if not donors:
                break
            utilities = {
                adapter.name: (
                    tuple(0.0 for _ in range(adapter.rank))
                    if adapter.alpha == 0.0
                    else adapter.component_utilities()
                )
                for adapter in adapters
            }
            donor_components = {
                adapter.name: min(
                    adapter.active_component_indices,
                    key=lambda index: (
                        utilities[adapter.name][index],
                        component_tie_breakers[(adapter.name, index)],
                        index,
                    ),
                )
                for adapter in donors
            }
            donor_marginals = {
                adapter.name: utilities[adapter.name][donor_components[adapter.name]]
                for adapter in donors
            }
            donor = min(
                donors,
                key=lambda adapter: (
                    donor_marginals[adapter.name],
                    tie_breakers[adapter.name],
                    adapter.name,
                ),
            )
            donor_marginal = donor_marginals[donor.name]
            recipient_mean = recipient_means[recipient.name]
            if recipient_mean <= donor_marginal:
                break

            recipient_inactive = [
                index
                for index in range(recipient.rank)
                if index not in recipient.active_component_indices
            ]
            recipient_component = max(
                recipient_inactive,
                key=lambda index: (
                    utilities[recipient.name][index],
                    component_tie_breakers[(recipient.name, index)],
                    -index,
                ),
            )
            donor_component = donor_components[donor.name]

            event = self.transfer_conserved_rank(
                donor=(donor.name, donor_component),
                recipient=(recipient.name, recipient_component),
                total_active_rank=total_active_rank,
                min_rank=min_rank,
                target_suffix=target_suffix,
            )
            event.update(
                {
                    "donor_signal": signals[donor.name],
                    "recipient_signal": signals[recipient.name],
                    "donor_marginal_utility": donor_marginal,
                    "recipient_mean_utility": recipient_mean,
                    "recipient_component_utility": utilities[recipient.name][recipient_component],
                    "seed": seed,
                }
            )
            events.append(event)
        return events


    def adjust_dynamic_ranks(
        self,
        *,
        allowed_ranks: tuple[int, ...],
        min_rank: int,
        grow_threshold: float,
        prune_threshold: float,
    ) -> list[dict[str, Any]]:
        """Grow or shrink gated active ranks based on per-adapter signal."""

        gated_adapters = [
            adapter
            for adapter in self._adapter_registry.values()
            if adapter.gates is not None
        ]
        if not gated_adapters:
            return []

        signals = {adapter.name: self._adapter_rank_signal(adapter) for adapter in gated_adapters}
        global_signal = max(signals.values(), default=0.0)
        if global_signal <= 0.0:
            return []

        events: list[dict[str, Any]] = []
        grow_bar = max(0.0, grow_threshold) * global_signal
        prune_bar = max(0.0, prune_threshold) * global_signal
        for adapter in gated_adapters:
            choices = self._active_rank_choices(adapter, allowed_ranks, min_rank)
            active = adapter.active_rank
            if active not in choices:
                choices = sorted(set(choices + [active]))
            idx = choices.index(active)
            signal = signals[adapter.name]
            next_rank = active
            action = "keep"
            if signal <= prune_bar and idx > 0:
                next_rank = choices[idx - 1]
                action = "shrink"
            elif signal >= grow_bar and idx < len(choices) - 1:
                next_rank = choices[idx + 1]
                action = "grow"

            if next_rank != active:
                adapter.set_active_rank(next_rank)
                event = {
                    "adapter": adapter.name,
                    "action": action,
                    "from_rank": active,
                    "to_rank": next_rank,
                    "max_rank": adapter.rank,
                    "signal": signal,
                    "global_signal": global_signal,
                }
                events.append(event)
        return events
