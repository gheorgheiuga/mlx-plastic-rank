import mlx.core as mx
import pytest
from test_manager_adapters import SeparateModel, _eye_rank

from mlx_plastic_rank.packs.manager import PackApplicationError
from research.rank_manager import ResearchLoRAManager as LoRAManager


def test_dynamic_rank_adjusts_only_high_signal_adapters():
    model = SeparateModel(layers=2, hidden=8)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )
    high = adapters["blocks.0.attn.q_proj"]
    low = adapters["blocks.1.attn.q_proj"]
    high.A = mx.ones_like(high.A)
    low.A = mx.zeros_like(low.A)

    events = manager.adjust_dynamic_ranks(
        allowed_ranks=(2, 4),
        min_rank=2,
        grow_threshold=0.25,
        prune_threshold=0.03,
    )

    assert high.active_rank == 4
    assert low.active_rank == 2
    assert len(events) == 1
    assert events[0]["adapter"] == "blocks.0.attn.q_proj"
    assert events[0]["action"] == "grow"
    assert events[0]["from_rank"] == 2
    assert events[0]["to_rank"] == 4
    assert events[0]["max_rank"] == 4
    assert events[0]["signal"] == pytest.approx(events[0]["global_signal"])


def test_conserved_rank_transfer_preserves_exact_global_budget():
    model = SeparateModel(layers=3, hidden=8)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )

    def set_utilities(adapter, utilities):
        A = mx.zeros_like(adapter.A)
        B = mx.zeros_like(adapter.B)
        for index, utility in enumerate(utilities):
            A[index, index] = utility
            B[index, index] = 1.0
        adapter.A = A
        adapter.B = B

    set_utilities(adapters["blocks.0.attn.q_proj"], (10.0, 9.0, 8.0, 7.0))
    set_utilities(adapters["blocks.1.attn.q_proj"], (2.0, 1.0, 0.5, 0.25))
    set_utilities(adapters["blocks.2.attn.q_proj"], (6.0, 5.0, 4.0, 3.0))
    arrays_before = {
        name: (adapter.A, adapter.B)
        for name, adapter in adapters.items()
    }

    before = manager.active_rank_state(target_suffix="attn.q_proj")
    events = manager.adjust_conserved_ranks(
        total_active_rank=6,
        min_rank=1,
        max_transfers=1,
        seed=17,
        target_suffix="attn.q_proj",
    )
    after = manager.active_rank_state(target_suffix="attn.q_proj")

    assert before == {
        "blocks.0.attn.q_proj": 2,
        "blocks.1.attn.q_proj": 2,
        "blocks.2.attn.q_proj": 2,
    }
    assert after == {
        "blocks.0.attn.q_proj": 3,
        "blocks.1.attn.q_proj": 1,
        "blocks.2.attn.q_proj": 2,
    }
    assert sum(before.values()) == sum(after.values()) == 6
    assert adapters["blocks.0.attn.q_proj"].active_component_indices == (0, 1, 2)
    assert adapters["blocks.1.attn.q_proj"].active_component_indices == (0,)
    for name, adapter in adapters.items():
        assert adapter.A is arrays_before[name][0]
        assert adapter.B is arrays_before[name][1]
    assert events == [
        {
            "action": "transfer",
            "donor": "blocks.1.attn.q_proj",
            "recipient": "blocks.0.attn.q_proj",
            "rank_units": 1,
            "donor_from_rank": 2,
            "donor_to_rank": 1,
            "recipient_from_rank": 2,
            "recipient_to_rank": 3,
            "donor_signal": pytest.approx(3.0),
            "recipient_signal": pytest.approx(19.0),
            "donor_marginal_utility": pytest.approx(1.0),
            "recipient_mean_utility": pytest.approx(9.5),
            "donor_component_index": 1,
            "recipient_component_index": 2,
            "recipient_component_utility": pytest.approx(8.0),
            "budget_before": 6,
            "budget_after": 6,
            "active_rank_before": before,
            "active_rank_after": after,
            "seed": 17,
        }
    ]


def test_explicit_conserved_transfer_moves_requested_slot_without_reordering_arrays():
    manager = LoRAManager(SeparateModel(layers=3, hidden=8))
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        allowed_ranks=(1, 2, 3, 4),
        initial_active_rank=2,
    )
    arrays_before = {
        name: (adapter.A, adapter.B)
        for name, adapter in adapters.items()
    }

    event = manager.transfer_conserved_rank(
        donor=("blocks.1.attn.q_proj", 1),
        recipient=("blocks.0.attn.q_proj", 2),
        total_active_rank=6,
        min_rank=1,
    )

    assert manager.active_rank_state() == {
        "blocks.0.attn.q_proj": 3,
        "blocks.1.attn.q_proj": 1,
        "blocks.2.attn.q_proj": 2,
    }
    assert event["action"] == "transfer"
    assert event["donor"] == "blocks.1.attn.q_proj"
    assert event["donor_component_index"] == 1
    assert event["recipient"] == "blocks.0.attn.q_proj"
    assert event["recipient_component_index"] == 2
    assert event["budget_before"] == event["budget_after"] == 6
    for name, adapter in adapters.items():
        assert adapter.A is arrays_before[name][0]
        assert adapter.B is arrays_before[name][1]


def test_explicit_conserved_transfer_rejects_preexisting_floor_violation():
    manager = LoRAManager(SeparateModel(layers=4, hidden=8))
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        allowed_ranks=(1, 2, 3, 4),
        initial_active_rank=1,
    )
    adapters["blocks.0.attn.q_proj"].set_active_components((0, 1, 2))
    gates_before = {
        name: adapter.active_component_indices for name, adapter in adapters.items()
    }

    with pytest.raises(PackApplicationError, match="violates min_rank=2"):
        manager.transfer_conserved_rank(
            donor=("blocks.0.attn.q_proj", 2),
            recipient=("blocks.1.attn.q_proj", 1),
            total_active_rank=6,
            min_rank=2,
        )

    assert {
        name: adapter.active_component_indices for name, adapter in adapters.items()
    } == gates_before


def test_conserved_rank_treats_alpha_zero_adapter_as_zero_effect():
    model = SeparateModel(layers=3, hidden=8)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )

    def set_utilities(adapter, utilities):
        A = mx.zeros_like(adapter.A)
        B = mx.zeros_like(adapter.B)
        for index, utility in enumerate(utilities):
            A[index, index] = utility
            B[index, index] = 1.0
        adapter.A = A
        adapter.B = B

    zero_effect = adapters["blocks.0.attn.q_proj"]
    zero_effect.alpha = 0.0
    set_utilities(zero_effect, (100.0, 90.0, 80.0, 70.0))
    set_utilities(adapters["blocks.1.attn.q_proj"], (2.0, 1.0, 0.5, 0.25))
    set_utilities(adapters["blocks.2.attn.q_proj"], (10.0, 9.0, 8.0, 7.0))

    events = manager.adjust_conserved_ranks(
        total_active_rank=6,
        min_rank=1,
        max_transfers=1,
        seed=17,
    )

    assert events[0]["donor"] == "blocks.0.attn.q_proj"
    assert events[0]["recipient"] == "blocks.2.attn.q_proj"
    assert events[0]["donor_signal"] == 0.0
    assert events[0]["donor_marginal_utility"] == 0.0


def test_conserved_rank_rejects_zero_minimum_needed_for_self_discovery():
    model = SeparateModel(layers=2, hidden=8)
    manager = LoRAManager(model)
    manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )

    with pytest.raises(PackApplicationError, match="min_rank must be at least 1.*self-discovery"):
        manager.adjust_conserved_ranks(total_active_rank=4, min_rank=0)


def test_conserved_rank_tie_breaking_is_seed_deterministic():
    def run_once(seed):
        manager = LoRAManager(SeparateModel(layers=3, hidden=8))
        adapters = manager.initialize_adapters(
            targets=["attn.q_proj"],
            rank=4,
            alpha=8.0,
            seed=0,
            initial_active_rank=2,
        )
        for block in (0, 1):
            adapter = adapters[f"blocks.{block}.attn.q_proj"]
            adapter.A = mx.eye(8, 4, dtype=mx.float16) * 4.0
            adapter.B = mx.eye(4, 8, dtype=mx.float16)
        donor = adapters["blocks.2.attn.q_proj"]
        donor.A = mx.eye(8, 4, dtype=mx.float16) * 0.25
        donor.B = mx.eye(4, 8, dtype=mx.float16)

        events = manager.adjust_conserved_ranks(
            total_active_rank=6,
            min_rank=1,
            max_transfers=1,
            seed=seed,
        )
        return events[0]["donor"], events[0]["recipient"], manager.active_rank_state()

    assert run_once(23) == run_once(23)


def test_runtime_conserved_rank_requires_rebalancing_before_pack_export(tmp_path):
    manager = LoRAManager(SeparateModel(layers=2, hidden=8))
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )
    adapters["blocks.0.attn.q_proj"].A = mx.ones((8, 4), dtype=mx.float16) * 4.0
    adapters["blocks.1.attn.q_proj"].A = mx.ones((8, 4), dtype=mx.float16) * 0.25
    manager.adjust_conserved_ranks(total_active_rank=4, max_transfers=1, seed=5)

    with pytest.raises(
        PackApplicationError,
        match="Runtime active-rank allocation.*rebalance.*before export",
    ):
        manager.export_active_pack("runtime-odd-ranks", tmp_path)


def test_compute_auto_ranks_with_theorem():
    model = SeparateModel(hidden=4)
    block = model.model.layers[0]
    block.self_attn.q_proj.weight = _eye_rank(4, 4)
    block.self_attn.k_proj.weight = _eye_rank(4, 2)
    block.self_attn.v_proj.weight = _eye_rank(4, 2)

    manager = LoRAManager(model)
    targets = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
    ranks, alphas, residuals = manager.compute_auto_ranks(
        targets,
        strategy="theorem",
        target_compression=0.9,
    )
    assert ranks["attn.q_proj"] == 4
    assert ranks["attn.k_proj"] == 2
    assert ranks["attn.v_proj"] == 2
    assert alphas["attn.q_proj"] == pytest.approx(8.0)
    assert residuals["attn.q_proj"] <= 1e-6


def test_compute_auto_ranks_with_gram_energy():
    model = SeparateModel(hidden=4)
    block = model.model.layers[0]
    block.self_attn.q_proj.weight = _eye_rank(4, 4)
    block.self_attn.k_proj.weight = _eye_rank(4, 2)
    block.self_attn.v_proj.weight = _eye_rank(4, 2)

    manager = LoRAManager(model)
    ranks, alphas, residuals = manager.compute_auto_ranks(
        ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        strategy="gram_energy",
        target_compression=0.9,
    )

    assert ranks == {"attn.q_proj": 4, "attn.k_proj": 2, "attn.v_proj": 2}
    assert alphas["attn.q_proj"] == pytest.approx(8.0)
    assert residuals["attn.q_proj"] <= 1e-6

