import csv
import json

import mlx.core as mx
import pytest

import research.learned_capacity_migration as migration
from research.learned_capacity_migration import (
    CORE_CONDITIONS,
    LearnedMigrationConfig,
    _aggregate,
    _build_trial,
    _checkpoint_fingerprint,
    _probe_loss_guided_transfer,
    _promotion_gates,
    _Protocol,
    run_learned_capacity_migration,
    write_artifacts,
)
from research.learned_capacity_migration_benchmark import main


@pytest.fixture(scope="module")
def smoke_report():
    return run_learned_capacity_migration(
        LearnedMigrationConfig(mode="smoke", seeds=(0,))
    )


@pytest.fixture(scope="module")
def development_control_report():
    return run_learned_capacity_migration(
        LearnedMigrationConfig(mode="development", seeds=(0,))
    )


def test_seed_split_is_frozen_and_mode_specific():
    assert LearnedMigrationConfig(mode="smoke").resolved_seeds() == (0,)
    assert LearnedMigrationConfig(mode="development").resolved_seeds() == (0,)
    assert LearnedMigrationConfig(mode="evidence").resolved_seeds() == tuple(
        range(1, 11)
    )
    assert LearnedMigrationConfig(
        mode="evidence", seeds=tuple(range(1, 11))
    ).resolved_seeds() == tuple(range(1, 11))

    with pytest.raises(ValueError, match="development seed 0 only"):
        LearnedMigrationConfig(mode="smoke", seeds=(1,)).resolved_seeds()
    with pytest.raises(ValueError, match="development seed 0 only"):
        LearnedMigrationConfig(mode="development", seeds=(1,)).resolved_seeds()
    with pytest.raises(ValueError, match="frozen confirmatory seeds 1 through 10"):
        LearnedMigrationConfig(mode="evidence", seeds=(0,)).resolved_seeds()
    with pytest.raises(ValueError, match="frozen confirmatory seeds 1 through 10"):
        LearnedMigrationConfig(mode="evidence", seeds=tuple(range(1, 10))).resolved_seeds()


def test_fixture_uses_input_derived_dense_routing_without_site_metadata(smoke_report):
    report = smoke_report
    fixture = report["fixture"]
    router = fixture["router"]

    assert report["protocol"] == "tiny_mlx_dense_v1"
    assert router["kind"] == "frozen_input_dependent_dense"
    assert router["accepts_explicit_routes"] is False
    assert router["minimum_weight_observed"] >= router["epsilon"] - 1e-6
    assert router["maximum_weight_observed"] < 0.9
    assert router["minimum_route_design_rank"] == 4
    assert fixture["task_transforms_distinct"] is True
    assert fixture["task_transform_ranks"] == {"A": 3, "B": 3}
    assert fixture["task_output_heads"] == {"A": [0, 3], "B": [3, 6]}
    assert fixture["joint_sufficient_analytic_max_abs_error"] < 1e-6
    assert fixture["site_metadata_reaches_allocator"] is False
    assert fixture["allocator_observations"] == ["loss", "parameters", "active_masks"]


def test_smoke_moves_rank_using_loss_only_shadow_swaps(smoke_report):
    report = smoke_report
    b_rows = [
        row
        for row in report["trajectory"]
        if row["condition"] == "guided_recycle" and row["phase"] == "learn_b"
    ]
    transfers = [event for row in b_rows for event in row["transfers"]]

    assert report["evidence_status"] == "learned_mlx_smoke_only"
    assert b_rows
    assert all(row["active_rank"] == row["active_rank_budget"] for row in b_rows)
    assert (
        b_rows[-1]["target_rank_coverage"]
        > b_rows[0]["target_rank_coverage_before_probe"]
    )
    assert transfers
    assert all(event["evidence_source"] == "rank_conserving_loss_lookahead" for event in transfers)
    assert all("task_site" not in event for event in transfers)


def test_static_control_branches_from_the_same_learned_a_checkpoint(smoke_report):
    report = smoke_report
    runs = {row["condition"]: row for row in report["runs"]}
    static_b_rows = [
        row
        for row in report["trajectory"]
        if row["condition"] == "static" and row["phase"] == "learn_b"
    ]

    assert runs["guided_recycle"]["end_a_checkpoint_fingerprint"] == runs["static"][
        "end_a_checkpoint_fingerprint"
    ]
    assert static_b_rows
    assert all(not row["transfers"] for row in static_b_rows)
    assert all(row["active_rank"] == row["active_rank_budget"] for row in static_b_rows)


def test_random_control_replays_every_guided_opportunity_from_the_same_checkpoint(
    smoke_report,
):
    report = smoke_report
    runs = {row["condition"]: row for row in report["runs"]}

    def transfer_trace(condition):
        return [
            (row["global_step"], row["transfers"][0])
            for row in report["trajectory"]
                if row["condition"] == condition
                and row["phase"] in {"learn_b", "return_a"}
                and row["transfers"]
        ]

    assert runs["guided_recycle"]["end_a_checkpoint_fingerprint"] == runs["random"][
        "end_a_checkpoint_fingerprint"
    ]
    guided = transfer_trace("guided_recycle")
    random = transfer_trace("random")
    assert [step for step, _ in guided] == [step for step, _ in random]
    for (_, guided_event), (_, random_event) in zip(guided, random, strict=True):
        assert random_event["evidence_source"] == "same_timing_random_legal_transfer"
        assert random_event["replay_guided_donor"] == [
            guided_event["donor"],
            guided_event["donor_component_index"],
        ]
        assert random_event["replay_guided_recipient"] == [
            guided_event["recipient"],
            guided_event["recipient_component_index"],
        ]


def test_future_aware_fixed_split_reserves_rank_without_migration(smoke_report):
    report = smoke_report
    rows = [
        row for row in report["trajectory"] if row["condition"] == "fixed_split"
    ]

    assert rows
    assert all(not row["transfers"] for row in rows)
    assert all(sorted(row["rank_map"].values()) == [1, 1, 2, 2] for row in rows)
    assert all(row["active_rank"] == row["active_rank_budget"] == 6 for row in rows)


def test_joint_sufficient_capacity_retains_both_learned_tasks(smoke_report):
    report = smoke_report
    rows = [
        row for row in report["trajectory"] if row["condition"] == "extra_capacity"
    ]
    end_b = [row for row in rows if row["phase"] == "learn_b"][-1]

    assert all(row["active_rank"] == row["active_rank_budget"] == 8 for row in rows)
    assert end_b["a_score"] > 0.8
    assert end_b["b_score"] > 0.8


def test_recycle_resets_both_factor_sides_and_preserves_unrelated_masters(
    smoke_report,
):
    transfers = [
        event
        for row in smoke_report["trajectory"]
        if row["condition"] == "guided_recycle"
        for event in row["transfers"]
    ]

    assert transfers
    assert all(event["non_donor_master_parameters_exact"] for event in transfers)
    assert all(event["released_a_column_zero"] for event in transfers)
    assert all(
        event["released_b_row_matches_deterministic_replacement"]
        for event in transfers
    )
    assert all(event["recycled_slot_reset_verified"] for event in transfers)
    assert all(event["historical_erasure_claimed"] is False for event in transfers)
    assert all("dormant_state_retained" not in event for event in transfers)
    assert smoke_report["integrity"]["strict_recycle_slot_reset_verified"] is True
    assert smoke_report["integrity"]["historical_erasure_claimed"] is False


def test_nonfinite_probe_fails_fast_and_restores_all_gates():
    protocol = _Protocol()
    _, manager = _build_trial(0, protocol)
    params = manager.trainable_parameters()
    gates_before = {
        name: adapter.active_component_indices
        for name, adapter in manager.iter_adapters()
    }

    def nan_loss(_params):
        return mx.array(float("nan"), dtype=mx.float32)

    with pytest.raises(FloatingPointError, match="probe baseline"):
        _probe_loss_guided_transfer(manager, nan_loss, params, protocol)

    assert {
        name: adapter.active_component_indices
        for name, adapter in manager.iter_adapters()
    } == gates_before


def test_checkpoint_fingerprint_includes_float32_master_state():
    protocol = _Protocol()
    _, manager = _build_trial(0, protocol)
    params = manager.trainable_parameters()
    before = _checkpoint_fingerprint(manager, params)
    perturbed = list(params)
    perturbed[0] = perturbed[0] + mx.ones_like(perturbed[0]) * 1e-8
    manager.set_trainable_parameters(perturbed)

    assert mx.array_equal(manager.trainable_parameters()[0], params[0]).item()
    assert _checkpoint_fingerprint(manager, perturbed) != before


def test_smoke_reports_control_metrics_without_promoting_the_thesis(smoke_report):
    aggregate = {row["condition"]: row for row in smoke_report["aggregate"]}

    assert set(aggregate) == {
        "guided_recycle",
        "static",
        "random",
        "fixed_split",
        "extra_capacity",
    }
    assert aggregate["guided_recycle"]["b_score_auc_mean"] > aggregate["static"][
        "b_score_auc_mean"
    ]
    assert aggregate["guided_recycle"]["b_score_auc_mean"] > aggregate["random"][
        "b_score_auc_mean"
    ]
    assert aggregate["guided_recycle"]["b_score_auc_mean"] > aggregate["fixed_split"][
        "b_score_auc_mean"
    ]
    assert smoke_report["gates"]["passed"] is False
    assert {
        item["id"]: item["passed"] for item in smoke_report["gates"]["criteria"]
    }["at_least_ten_confirmatory_seeds"] is False
    assert smoke_report["evidence_status"] == "learned_mlx_smoke_only"


def test_report_declares_seed_and_return_measurement_boundaries(smoke_report):
    assert smoke_report["seed_split"] == {
        "frozen": True,
        "development_seeds": [0],
        "confirmatory_seeds": list(range(1, 11)),
        "selected_partition": "development",
        "selected_seeds": [0],
    }
    semantics = smoke_report["measurement_semantics"]
    assert semantics["canonical_a_return_metric"] == (
        "a_return_post_supervised_probe_pre_update_score"
    )
    assert semantics["deprecated_alias"] == "a_return_immediate_score"
    assert semantics["parameter_update_precedes_measurement"] is False
    assert semantics["unlabeled_cue_wake_measured"] is False
    assert all(
        run["a_return_post_supervised_probe_pre_update_score"]
        == run["a_return_immediate_score"]
        for run in smoke_report["runs"]
    )
    criteria = {item["id"]: item["passed"] for item in smoke_report["gates"]["criteria"]}
    assert criteria["unlabeled_cue_wake_measured"] is False
    assert "full_forgetting_control_matrix_present" not in criteria


def test_localization_gate_uses_complete_guided_random_alignment_pairs(smoke_report):
    comparison = smoke_report["gates"]["paired_comparisons"][
        "guided_recycle_vs_random_b_final_alignment"
    ]
    criteria = {item["id"]: item["passed"] for item in smoke_report["gates"]["criteria"]}

    assert comparison["pairs"] == 1
    assert criteria["guided_rank_localizes_b_more_than_random"] == (
        comparison["ci_lower"] > 0.0 and comparison["pairs"] == 1
    )

    incomplete_runs = [
        run for run in smoke_report["runs"] if run["condition"] != "random"
    ]
    incomplete_gates = _promotion_gates(
        incomplete_runs,
        _aggregate(incomplete_runs, CORE_CONDITIONS),
        configured_conditions=CORE_CONDITIONS,
        seeds=(0,),
        protocol=_Protocol(),
        integrity=smoke_report["integrity"],
        event_windows=smoke_report["event_windows"],
        failures=[],
    )
    incomplete_criteria = {
        item["id"]: item["passed"] for item in incomplete_gates["criteria"]
    }
    assert incomplete_gates["paired_comparisons"][
        "guided_recycle_vs_random_b_final_alignment"
    ]["pairs"] == 0
    assert incomplete_criteria["guided_rank_localizes_b_more_than_random"] is False


def test_rank_ledger_separates_active_capacity_from_resident_storage(smoke_report):
    rows = [
        row
        for row in smoke_report["trajectory"]
        if row["condition"] == "guided_recycle"
    ]

    assert rows
    assert all(row["active_rank"] == 6 for row in rows)
    assert all(row["physical_rank"] == 16 for row in rows)
    assert all(row["physical_fp16_parameter_bytes"] == 384 for row in rows)
    assert all(
        row["physical_float32_master_parameter_bytes"] == 768 for row in rows
    )
    assert all(row["optimizer_state_bytes"] == 0 for row in rows)
    assert all(
        row["learned_inactive_rank_a_column_lower_bound"] == 0 for row in rows
    )
    assert all(row["active_fp16_factor_bytes"] == 144 for row in rows)
    assert all(row["active_master_factor_bytes"] == 288 for row in rows)
    assert all(
        row["learned_inactive_fp16_factor_bytes_lower_bound"] == 0
        for row in rows
    )
    assert all(
        row["learned_inactive_master_factor_bytes_lower_bound"] == 0
        for row in rows
    )
    recycled = [row["recycled_rank_cumulative"] for row in rows]
    assert recycled == sorted(recycled)
    assert recycled[-1] > 0
    assert smoke_report["capacity_accounting"]["conserves_physical_bytes"] is False
    assert smoke_report["capacity_accounting"]["conserves_effective_active_rank"] is True
    assert smoke_report["capacity_accounting"]["dormant_ledger"] == {
        "status": "provisional_a_column_lower_bound",
        "detection": "inactive_A_columns_with_magnitude_above_1e-8",
        "detects_b_only_learned_state": False,
    }


def test_rank_move_precedes_a_matched_b_performance_advantage(smoke_report):
    windows = smoke_report["event_windows"]

    assert windows
    assert all(
        window["event_semantics"] == "first_b_phase_loss_guided_transfer"
        for window in windows
    )
    assert all(window["rank_move_step"] < window["measurement_step"] for window in windows)
    assert all(abs(window["pre_transfer_score_gap"]) < 1e-6 for window in windows)
    assert all(window["post_window_score_advantage"] > 0.0 for window in windows)
    assert smoke_report["event_window_semantics"]["directionality_claimed"] is False


def test_development_matrix_exercises_vault_recycle_oracle_and_never_a(
    development_control_report,
):
    report = development_control_report
    conditions = {row["condition"] for row in report["runs"]}
    assert {"guided_vault", "oracle", "never_a"} <= conditions

    def rows(condition, phase):
        return [
            row
            for row in report["trajectory"]
            if row["condition"] == condition and row["phase"] == phase
        ]

    vault_return = rows("guided_vault", "return_a")
    recycle_return = rows("guided_recycle", "return_a")
    never_return = rows("never_a", "return_a")
    assert vault_return[0]["transfers"]
    assert vault_return[0]["a_score_post_supervised_probe_pre_update"] > 0.5
    assert (
        vault_return[0]["a_score_post_supervised_probe_pre_update"]
        > recycle_return[0]["a_score_post_supervised_probe_pre_update"] + 0.4
    )
    assert recycle_return[0]["a_score_post_supervised_probe_pre_update"] < 0.2
    assert never_return[0]["a_score_post_supervised_probe_pre_update"] < 0.2
    assert max(
        row["learned_inactive_rank_a_column_lower_bound"]
        for row in rows("guided_vault", "learn_b")
    ) > 0
    vault_transfers = [
        event
        for row in rows("guided_vault", "learn_b")
        for event in row["transfers"]
    ]
    assert vault_transfers
    assert all("inactive_a_column_state_retained" in event for event in vault_transfers)
    assert all("dormant_state_retained" not in event for event in vault_transfers)

    summaries = {row["condition"]: row for row in report["runs"]}
    assert summaries["oracle"]["b_final_alignment"] >= summaries["guided_recycle"][
        "b_final_alignment"
    ]
    assert report["evidence_status"] == "learned_mlx_development_only"


def test_learned_artifacts_match_the_standard_four_file_contract(
    smoke_report,
    tmp_path,
):
    paths = write_artifacts(smoke_report, tmp_path)

    assert set(paths) == {
        "trajectory_jsonl",
        "summary_json",
        "summary_csv",
        "summary_markdown",
    }
    assert len(paths["trajectory_jsonl"].read_text().splitlines()) == len(
        smoke_report["trajectory"]
    )
    summary = json.loads(paths["summary_json"].read_text())
    assert "trajectory" not in summary
    with paths["summary_csv"].open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(smoke_report["conditions"])
    markdown = paths["summary_markdown"].read_text()
    assert "real MLX loss" in markdown
    assert "cannot promote" in markdown
    assert "| condition | n |" in markdown
    assert "A return post-probe / pre-update" in markdown
    assert "inactive A-col R lower bound" in markdown
    assert "pairs=1" in markdown
    assert "Unlabeled cue-triggered wake is not measured" in markdown
    assert "no transfer directionality is claimed" in markdown


def test_same_seed_repeats_report_and_artifacts_byte_for_byte(
    smoke_report,
    tmp_path,
):
    repeated = run_learned_capacity_migration(
        LearnedMigrationConfig(mode="smoke", seeds=(0,))
    )
    assert repeated == smoke_report

    first_paths = write_artifacts(smoke_report, tmp_path / "first")
    repeated_paths = write_artifacts(repeated, tmp_path / "repeated")
    assert {
        name: path.read_bytes() for name, path in first_paths.items()
    } == {
        name: path.read_bytes() for name, path in repeated_paths.items()
    }


def test_learned_cli_uses_mode_protocol_default_and_enforces_smoke_gate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    assert main(
        [
            "--mode",
            "smoke",
            "--seeds",
            "0",
            "--require-pass",
        ]
    ) == 2
    output_dir = (
        tmp_path / "out" / "capacity_migration" / "tiny_mlx_dense_v1" / "smoke"
    )
    assert (output_dir / "summary.json").exists()


def test_run_records_injected_numerical_failure_without_relaxing_seed_split(
    monkeypatch,
):
    original_run_condition = migration._run_condition

    def injected_failure(seed, condition, protocol, *, replay=None):
        if condition == "random":
            raise FloatingPointError("injected random-control failure")
        return original_run_condition(seed, condition, protocol, replay=replay)

    monkeypatch.setattr(migration, "_run_condition", injected_failure)
    report = run_learned_capacity_migration(
        LearnedMigrationConfig(mode="smoke", seeds=(0,))
    )

    assert report["failures"]
    assert all(failure["seed"] == 0 for failure in report["failures"])
    assert any(failure["failure_type"] == "FloatingPointError" for failure in report["failures"])
    criteria = {item["id"]: item["passed"] for item in report["gates"]["criteria"]}
    assert criteria["complete_finite_seed_condition_matrix"] is False
    assert report["seed_split"]["selected_seeds"] == [0]
    assert report["evidence_status"] == "learned_mlx_smoke_only"
