import csv
import json

import pytest

from mlx_plastic_rank.packs.loss_lookahead_calibration import (
    EVIDENCE_SEEDS,
    LookaheadCalibrationConfig,
    _spearman,
    run_loss_lookahead_calibration,
    write_artifacts,
)


@pytest.fixture(scope="module")
def smoke_report():
    return run_loss_lookahead_calibration(
        LookaheadCalibrationConfig(mode="smoke", seeds=(0,))
    )


def test_calibration_seed_split_is_frozen():
    assert LookaheadCalibrationConfig(mode="smoke").resolved_seeds() == (0,)
    assert LookaheadCalibrationConfig(mode="evidence").resolved_seeds() == tuple(
        range(11, 21)
    )
    assert EVIDENCE_SEEDS == tuple(range(11, 21))
    with pytest.raises(ValueError, match="development seed 0"):
        LookaheadCalibrationConfig(mode="smoke", seeds=(1,)).resolved_seeds()
    with pytest.raises(ValueError, match="frozen seeds 11 through 20"):
        LookaheadCalibrationConfig(mode="evidence", seeds=(11,)).resolved_seeds()


def test_spearman_reports_ordering_and_ties():
    assert _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert -1.0 <= _spearman([1.0, 1.0, 2.0], [0.0, 1.0, 2.0]) <= 1.0


def test_smoke_branches_every_legal_transfer_from_one_checkpoint(smoke_report):
    seed = smoke_report["seeds"][0]
    transfer_rows = [
        row for row in smoke_report["branches"]
        if row["branch"] == "legal_transfer"
    ]

    assert seed["complete"] is True
    assert len(transfer_rows) == seed["candidate_count"]
    assert seed["candidate_count"] > 1
    assert seed["all_branch_checkpoints_match"] is True
    assert seed["budget_invariant"] is True
    assert seed["strict_recycle_invariant"] is True
    assert all(row["budget_after_transfer"] == 6 for row in transfer_rows)
    assert all(row["strict_recycle_verified"] for row in transfer_rows)
    assert {role for row in transfer_rows for role in row["selected_roles"]} == {
        "predicted_best",
        "predicted_worst",
        "wrong_task_best",
        "prediction_independent_random",
    }


def test_smoke_is_non_promotional_and_uses_explicit_metric_semantics(smoke_report):
    assert smoke_report["evidence_status"] == "lookahead_calibration_development_only"
    assert smoke_report["gates"]["passed"] is False
    assert smoke_report["seed_split"]["selected_seeds"] == [0]
    assert smoke_report["metric_semantics"]["experimental_unit"] == "fixture_seed"
    assert (
        smoke_report["metric_semantics"][
            "candidate_branches_are_independent_replicates"
        ]
        is False
    )


def test_calibration_artifact_contract(smoke_report, tmp_path):
    paths = write_artifacts(
        smoke_report,
        tmp_path,
        provenance={"test": True},
    )
    assert set(paths) == {
        "protocol_json",
        "provenance_json",
        "raw_results_jsonl",
        "summary_csv",
        "summary_json",
        "diagnostics_csv",
        "interpretation_markdown",
    }
    assert len(paths["raw_results_jsonl"].read_text().splitlines()) == len(
        smoke_report["branches"]
    )
    assert "branches" not in json.loads(paths["summary_json"].read_text())
    with paths["summary_csv"].open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(smoke_report["aggregate"])
    assert "Frozen gate" in paths["interpretation_markdown"].read_text()
