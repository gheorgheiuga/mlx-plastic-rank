import csv
import json

import pytest

from mlx_plastic_rank.packs.multibatch_controller import (
    CONDITIONS,
    EVIDENCE_SEEDS,
    MultiBatchControllerConfig,
    run_multibatch_controller,
    write_artifacts,
)


@pytest.fixture(scope="module")
def smoke_report():
    return run_multibatch_controller(
        MultiBatchControllerConfig(mode="smoke", seeds=(0,))
    )


def test_multibatch_seed_split_is_frozen():
    assert MultiBatchControllerConfig(mode="smoke").resolved_seeds() == (0,)
    assert MultiBatchControllerConfig(mode="evidence").resolved_seeds() == tuple(
        range(21, 31)
    )
    assert EVIDENCE_SEEDS == tuple(range(21, 31))
    with pytest.raises(ValueError, match="development seed 0"):
        MultiBatchControllerConfig(mode="smoke", seeds=(21,)).resolved_seeds()
    with pytest.raises(ValueError, match="frozen seeds 21 through 30"):
        MultiBatchControllerConfig(mode="evidence", seeds=(21,)).resolved_seeds()


def test_smoke_runs_complete_control_matrix_from_one_checkpoint(smoke_report):
    assert {row["condition"] for row in smoke_report["runs"]} == set(CONDITIONS)
    fingerprints = {
        row["start_checkpoint_fingerprint"] for row in smoke_report["runs"]
    }
    assert len(fingerprints) == 1
    assert smoke_report["failures"] == []
    assert all(smoke_report["invariants"].values())


def test_directional_controls_commit_two_strict_recycle_transfers(smoke_report):
    by_condition = {row["condition"]: row for row in smoke_report["runs"]}
    for condition in (
        "b_horizon3",
        "fixed_random",
        "b_exact_one_step",
        "a_horizon3_wrong_task",
    ):
        assert by_condition[condition]["transfer_count"] == 2
        assert by_condition[condition]["budget_invariant"] is True
        assert by_condition[condition]["strict_recycle_invariant"] is True
        assert by_condition[condition]["shadow_restore_invariant"] is True
        assert by_condition[condition]["task_site_metadata_observed"] is False


def test_selection_compute_and_task_controls_are_explicit(smoke_report):
    by_condition = {row["condition"]: row for row in smoke_report["runs"]}
    assert (
        by_condition["b_horizon3"]["virtual_gradient_evaluations"]
        == by_condition["a_horizon3_wrong_task"]["virtual_gradient_evaluations"]
    )
    assert (
        by_condition["b_horizon3"]["virtual_gradient_evaluations"]
        > by_condition["b_exact_one_step"]["virtual_gradient_evaluations"]
        > 0
    )
    assert by_condition["fixed_random"]["virtual_gradient_evaluations"] == 0
    assert by_condition["site_oracle"]["b_final_alignment"] == 1.0
    assert by_condition["site_oracle"]["task_site_metadata_observed"] is True
    assert smoke_report["measurement_semantics"]["compute_efficiency_claimed"] is False
    assert (
        smoke_report["measurement_semantics"][
            "controller_selection_compute_matched"
        ]
        is False
    )


def test_smoke_remains_non_promotional(smoke_report):
    assert smoke_report["evidence_status"] == "multibatch_controller_development_only"
    assert smoke_report["gates"]["passed"] is False
    assert smoke_report["seed_split"]["selected_seeds"] == [0]
    assert smoke_report["measurement_semantics"]["experimental_unit"] == "fixture_seed"


def test_multibatch_artifact_contract(smoke_report, tmp_path):
    paths = write_artifacts(smoke_report, tmp_path, provenance={"test": True})
    assert set(paths) == {
        "protocol_json",
        "provenance_json",
        "trajectory_jsonl",
        "summary_json",
        "summary_csv",
        "diagnostics_csv",
        "interpretation_markdown",
    }
    assert len(paths["trajectory_jsonl"].read_text().splitlines()) == len(
        smoke_report["trajectory"]
    )
    assert "trajectory" not in json.loads(paths["summary_json"].read_text())
    with paths["summary_csv"].open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(smoke_report["aggregate"])
    assert "Frozen gate" in paths["interpretation_markdown"].read_text()
