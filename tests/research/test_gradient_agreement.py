import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_plastic_rank.packs.gradient_admission import (
    CONDITIONS,
    load_spec,
    resolved_seeds,
    run_development,
    selection,
    summarize,
)
from mlx_plastic_rank.packs.gradient_agreement import (
    AuditedTrial,
    SelectionBatch,
    agreement_score,
    clipped_sgd,
    prospective_gradient,
    select_gradient,
    select_one_step,
)


def batch():
    rng = np.random.default_rng(71123)
    return SelectionBatch(mx.array(rng.normal(size=(32, 6)).astype("float32")),
                          mx.array(rng.normal(size=(32, 6)).astype("float32")),
                          mx.array([0., 0., 0., 1., 1., 1.]))


def test_reserved_seeds_and_eval_observations_are_inaccessible():
    assert resolved_seeds("development") == (31, 32, 33, 34, 35)
    with pytest.raises(ValueError, match="disabled"):
        resolved_seeds("evidence")
    with pytest.raises(ValueError, match="development seeds"):
        AuditedTrial(101)
    with pytest.raises(ValueError, match="evaluation data"):
        selection(None, "eval")
    assert load_spec()["evidence_enabled"] is False


def test_global_clip_scales_all_tensors_together_and_rejects_nan():
    params = [mx.zeros((1,)), mx.zeros((1,))]
    result, norm = clipped_sgd(params, [mx.array([3.]), mx.array([4.])])
    assert norm == pytest.approx(5.)
    np.testing.assert_allclose(np.asarray(result).reshape(-1), [-.9, -1.2], rtol=1e-6)
    result, norm = clipped_sgd(params, [mx.array([3e30]), mx.array([4e30])])
    assert np.isfinite(norm)
    np.testing.assert_allclose(np.asarray(result).reshape(-1), [-.9, -1.2], rtol=1e-6)
    with pytest.raises(FloatingPointError):
        clipped_sgd(params, [mx.array([float("nan")]), mx.zeros((1,))])


def test_prospective_gradient_matches_ungated_autodiff():
    b = batch().slice(0, 8)
    trial = AuditedTrial(0)
    route = trial.model.routes(b.features)[:, :1]
    row = trial.entries[0][1].B[3].astype(mx.float32)
    residual = trial.model(b.features) - b.targets

    def loss(a):
        prediction = residual + 2 * route * (b.features @ row)[:, None] * a[None, :]
        return mx.sum(prediction ** 2 * b.mask) / (8 * mx.sum(b.mask))

    expected = mx.grad(loss)(mx.zeros((6,)))
    actual = prospective_gradient(b, residual, route, row, 2.)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_agreement_distinguishes_opposing_gradients_from_energy():
    vectors = mx.array([[1., 2.], [-1., -2.], [0., 0.]])
    assert agreement_score(vectors) == pytest.approx(-5 / 3)
    assert agreement_score(vectors, energy=True) == pytest.approx(10 / 3)
    assert agreement_score(mx.array([[1., 2.]] * 3)) == pytest.approx(5)


def test_input_only_dormant_learning_is_detected():
    trial = AuditedTrial(0)
    # A remains exactly zero: the historical output-column-only audit missed this.
    trial.params[1] = trial.params[1].at[3, 0].add(.25)
    trial.manager.set_trainable_parameters(trial.params)
    with pytest.raises(RuntimeError, match="inactive input factor"):
        trial.audit()


def test_commit_restore_includes_reset_bank_and_both_parameter_representations():
    trial = AuditedTrial(0)
    before = trial.fingerprint()
    snapshot = trial.snapshot()
    donor, recipient = trial.candidates()[0]
    event = trial.commit(donor, recipient, 17)
    assert event["full_factor_audit"]
    assert trial.resets[donor] == 17
    assert trial.fingerprint() != before
    trial.restore(snapshot)
    assert trial.fingerprint() == before


def test_gradient_selector_is_read_only_and_energy_keeps_donor_rule():
    trial = AuditedTrial(0)
    before = trial.fingerprint()
    first = select_gradient(trial, batch())
    energy = select_gradient(trial, batch(), energy=True)
    assert first[0] == energy[0]
    assert before == trial.fingerprint()
    assert trial.work["actual_updates"] == trial.work["virtual_updates"] == 0
    assert first[2]["state_restored"]


def test_failed_shadow_update_restores_everything(monkeypatch):
    trial = AuditedTrial(0)
    before = trial.fingerprint()
    original = trial.update
    calls = 0

    def fail_on_candidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            trial.params[0] = mx.full(trial.params[0].shape, float("nan"))
            trial.manager.set_trainable_parameters(trial.params)
            raise FloatingPointError("broken candidate")
        return original(*args, **kwargs)

    monkeypatch.setattr(trial, "update", fail_on_candidate)
    with pytest.raises(FloatingPointError, match="broken candidate"):
        select_one_step(trial, batch().slice(0, 8), batch().slice(8, 24), 17)
    assert trial.fingerprint() == before


def fake_runs():
    return [{"seed": 0, "condition": c, "b_auc": .9, "b_final": .9,
             "a_readiness": .9, "a_final": .9, "b_coverage": 1.,
             "input_identity": "inputs", "initial_bank_identity": "initial",
             "start_checkpoint": "common" if i < 7 else c,
             "full_factor_audit": True,
             "transfer_count": 0 if c in {"static", "future_fixed_split", "joint_capacity"} else 2,
             "actual_b_updates": 24}
            for i, c in enumerate(CONDITIONS)]


def test_missing_duplicate_nonfinite_and_unpaired_results_cannot_pass():
    rows = fake_runs()
    assert summarize(rows, [], (0,), load_spec())["development_valid"]
    for invalid in (rows[:-1], rows + rows[:1], rows[:-1] + [rows[-1] | {"b_auc": float("nan")}],
                    rows[:-1] + [rows[-1] | {"input_identity": "different"}]):
        assert not summarize(invalid, [], (0,), load_spec())["development_valid"]
    report = summarize(rows, [], (0,), load_spec())
    assert not report["admission_passed"]
    assert report["paired_intervals"] is None
    json.dumps(report, allow_nan=False)


def test_sufficient_capacity_failure_blocks_an_otherwise_complete_matrix():
    rows = fake_runs()
    rows[-1]["b_final"] = .79
    report = summarize(rows, [], (0,), load_spec())
    assert report["checks"]["complete_finite_matrix"]
    assert not report["checks"]["joint_solvability"]
    assert not report["development_valid"]


def test_donor_cost_matches_real_gate_removal_after_learning():
    trial = AuditedTrial(0)
    observations = batch()
    for _ in range(5):
        trial.update(observations)
    before = trial.fingerprint()
    checkpoint = trial.snapshot()
    _, _, diagnostics = select_gradient(trial, observations)
    selected = observations.slice(0, 24)
    base_loss = trial.value(selected)
    for candidate in diagnostics["donor_costs"]:
        name, component = candidate["slot"]
        adapter = dict(trial.entries)[name]
        adapter.set_active_components(c for c in adapter.active_component_indices if c != component)
        measured = trial.value(selected) - base_loss
        assert candidate["cost"] == pytest.approx(measured, abs=2e-6)
        trial.restore(checkpoint)
    assert trial.fingerprint() == before


def test_actual_and_virtual_updates_apply_identical_clipping():
    trial = AuditedTrial(0)
    observations = batch()
    observations = SelectionBatch(observations.features, observations.targets * 100, observations.mask)
    checkpoint = trial.snapshot()
    actual = trial.update(observations)
    after = trial.fingerprint()
    trial.restore(checkpoint)
    virtual = trial.update(observations, virtual=True)
    assert actual["clipped"] and virtual["clipped"]
    assert actual == virtual
    assert trial.fingerprint() == after


def test_timeout_is_retained_as_failure_not_a_partial_pass():
    events = []
    report = run_development("smoke", time_limit_seconds=1e-12,
                             emit=lambda kind, row: events.append((kind, row)))
    assert not report["development_valid"]
    assert report["runs"] == []
    assert report["failures"][0]["error_type"] == "TimeoutError"
    assert events[0][0] == "failures"


def test_unexpected_static_transfer_invalidates_the_matrix():
    rows = fake_runs()
    rows[1]["transfer_count"] = 1
    report = summarize(rows, [], (0,), load_spec())
    assert report["checks"]["complete_finite_matrix"]
    assert not report["checks"]["mechanics"]
