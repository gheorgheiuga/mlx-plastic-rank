import csv
import json

import pytest

from mlx_plastic_rank.packs.capacity_migration import (
    CONDITIONS,
    BenchmarkConfig,
    ConservedAllocator,
    Student,
    run_benchmark,
    write_artifacts,
)
from scripts.capacity_migration_benchmark import main


def _small_report():
    return run_benchmark(BenchmarkConfig(seeds=tuple(range(10))))


def test_reference_benchmark_is_deterministic_and_budget_conserved():
    config = BenchmarkConfig(seeds=(2, 7), phase_steps=6)
    left = run_benchmark(config)
    right = run_benchmark(config)

    assert left == right
    assert {row["condition"] for row in left["runs"]} == set(CONDITIONS)
    assert all(run["budget_invariant"] for run in left["runs"])
    assert all(
        task["task_a"]["site"] != task["task_b"]["site"]
        for task in left["tasks"]
    )
    assert all(
        row["active_rank"] == row["active_rank_budget"]
        for row in left["trajectory"]
    )

    with pytest.raises(ValueError, match="seeds must be unique"):
        BenchmarkConfig(seeds=(2, 2))


def test_non_restoring_activation_discards_stale_vault_state():
    student = Student(site_count=2, task_rank=1)
    student.activate_site(0)
    student.weights[0][0] = 1.0
    student.vault[(1, 0)] = 9.0

    ConservedAllocator().move_toward(
        student,
        target_site=1,
        max_transfers=1,
        park_released=False,
        restore_target=False,
        reason="test",
    )

    assert student.weights[1][0] == 0.0
    assert (1, 0) not in student.vault


def test_fixed_split_conserves_nondefault_task_rank_budget():
    report = run_benchmark(BenchmarkConfig(seeds=(0,), task_rank=4, phase_steps=4))
    fixed_split = [
        row for row in report["trajectory"] if row["condition"] == "fixed_split"
    ]

    assert fixed_split
    assert all(row["active_rank"] == row["active_rank_budget"] == 4 for row in fixed_split)


def test_vault_wakes_while_recycle_relearns_and_controls_bound_claim():
    report = _small_report()
    aggregate = {row["condition"]: row for row in report["aggregate"]}

    assert report["gates"]["passed"] is True
    assert report["evidence_status"] == "counterfactual_reference_mechanics_passed"
    assert aggregate["vault"]["a_score_end_b_mean"] < 0.1
    assert aggregate["vault"]["a_latent_score_end_b_mean"] > 0.9
    assert aggregate["vault"]["a_return_immediate_score_mean"] > 0.9
    assert aggregate["recycle"]["a_return_immediate_score_mean"] == 0.0
    assert aggregate["recycle"]["relearning_advantage_over_scratch_mean"] == 0.0
    assert aggregate["vault"]["b_score_auc_mean"] > aggregate["static"]["b_score_auc_mean"]
    assert aggregate["recycle"]["b_score_auc_mean"] > aggregate["random"]["b_score_auc_mean"]
    assert aggregate["extra_capacity"]["a_score_end_b_mean"] > 0.9
    assert all(
        comparison["ci_lower"] > 0.0
        for comparison in report["gates"]["paired_comparisons"].values()
    )
    assert all(
        transfer["reason"].startswith("counterfactual_gradient")
        for row in report["trajectory"]
        if row["condition"] in {"vault", "recycle"}
        for transfer in row["transfers"]
    )
    assert "does not show" in report["claim_boundary"]


def test_artifacts_include_trajectory_and_three_summary_formats(tmp_path):
    report = run_benchmark(BenchmarkConfig(seeds=(3,), phase_steps=4))
    paths = write_artifacts(report, tmp_path)

    assert set(paths) == {
        "trajectory_jsonl",
        "summary_json",
        "summary_csv",
        "summary_markdown",
    }
    rows = [json.loads(line) for line in paths["trajectory_jsonl"].read_text().splitlines()]
    assert len(rows) == len(CONDITIONS) * 3 * 4
    assert all("rank_map" in row and "transfers" in row for row in rows)
    summary = json.loads(paths["summary_json"].read_text())
    assert "trajectory" not in summary
    with paths["summary_csv"].open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(CONDITIONS)
    assert "Declared gates" in paths["summary_markdown"].read_text()


def test_cli_writes_runnable_artifacts(tmp_path):
    assert main(["--output-dir", str(tmp_path), "--seeds", "0,1", "--require-pass"]) == 0
    assert (tmp_path / "summary.json").exists()
