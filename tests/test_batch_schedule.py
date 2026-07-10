from pathlib import Path

import mlx.core as mx
import pytest

from mlx_plastic_rank.packs import train as train_module
from mlx_plastic_rank.packs.batch_schedule import (
    MinibatchSchedule,
    generate_minibatch_schedule,
    load_minibatch_schedule,
    load_or_create_minibatch_schedule,
    save_minibatch_schedule,
)
from mlx_plastic_rank.packs.train import TrainingConfig


def test_generated_schedule_is_independent_of_mlx_rng_state():
    mx.random.seed(3)
    mx.eval(mx.random.normal((32,)))
    first = generate_minibatch_schedule(
        dataset_size=19,
        batch_size=3,
        steps=7,
        seed=41,
    )

    mx.random.seed(999)
    mx.eval(mx.random.normal((257,)))
    second = generate_minibatch_schedule(
        dataset_size=19,
        batch_size=3,
        steps=7,
        seed=41,
    )

    assert first == second
    assert first.steps == 7
    assert all(len(batch) == 3 for batch in first.indices)


def test_minibatch_schedule_round_trips_with_stable_digest(tmp_path: Path):
    path = tmp_path / "schedule.json"
    schedule = generate_minibatch_schedule(
        dataset_size=11,
        batch_size=2,
        steps=5,
        seed=17,
        dataset_fingerprint="dataset-v1",
    )

    save_minibatch_schedule(schedule, path)
    loaded = load_minibatch_schedule(path)

    assert loaded == schedule
    assert loaded.digest == schedule.digest
    assert loaded.batch(0) == schedule.indices[0]


def test_load_or_create_reuses_persisted_schedule_and_validates_shape(tmp_path: Path):
    path = tmp_path / "paired.json"
    created = load_or_create_minibatch_schedule(
        path,
        dataset_size=13,
        batch_size=2,
        steps=6,
        seed=7,
        dataset_fingerprint="dataset-v1",
    )

    reused = load_or_create_minibatch_schedule(
        path,
        dataset_size=13,
        batch_size=2,
        steps=4,
        seed=999,
        dataset_fingerprint="dataset-v1",
    )

    assert reused == created
    with pytest.raises(ValueError, match="dataset size"):
        load_or_create_minibatch_schedule(
            path,
            dataset_size=12,
            batch_size=2,
            steps=4,
            seed=7,
            dataset_fingerprint="dataset-v1",
        )
    with pytest.raises(ValueError, match="only 6 steps"):
        load_or_create_minibatch_schedule(
            path,
            dataset_size=13,
            batch_size=2,
            steps=7,
            seed=7,
            dataset_fingerprint="dataset-v1",
        )
    with pytest.raises(ValueError, match="dataset fingerprint"):
        load_or_create_minibatch_schedule(
            path,
            dataset_size=13,
            batch_size=2,
            steps=4,
            seed=7,
            dataset_fingerprint="dataset-v2",
        )


def test_schedule_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="outside dataset bounds"):
        MinibatchSchedule(
            dataset_size=3,
            batch_size=2,
            seed=0,
            indices=((0, 3),),
        )


def test_training_config_resolves_precomputed_schedule_without_regenerating():
    schedule = generate_minibatch_schedule(
        dataset_size=9,
        batch_size=2,
        steps=4,
        seed=5,
    )
    config = TrainingConfig(
        steps=3,
        batch_size=2,
        batch_seed=123,
        batch_schedule=schedule,
    )

    assert config.resolve_batch_schedule(dataset_size=9) is schedule
    assert config.resolved_batch_schedule_digest == schedule.digest


def test_training_config_can_persist_and_reuse_a_schedule(tmp_path: Path):
    path = tmp_path / "shared.json"
    first = TrainingConfig(
        steps=5,
        batch_size=2,
        batch_seed=31,
        batch_schedule_path=path,
        dataset_fingerprint="dataset-v1",
    ).resolve_batch_schedule(dataset_size=8)
    second = TrainingConfig(
        steps=3,
        batch_size=2,
        batch_seed=999,
        batch_schedule_path=path,
        dataset_fingerprint="dataset-v1",
    ).resolve_batch_schedule(dataset_size=8)

    assert path.exists()
    assert second is not first
    assert second == first


def test_supervised_training_resets_rng_and_consumes_supplied_schedule(monkeypatch):
    schedule = generate_minibatch_schedule(
        dataset_size=3,
        batch_size=1,
        steps=2,
        seed=11,
    )
    requested_steps: list[int] = []
    original_batch = schedule.batch

    class RecordingSchedule:
        digest = schedule.digest

        def validate_for(self, **kwargs):
            schedule.validate_for(**kwargs)

        def batch(self, step: int):
            requested_steps.append(step)
            return original_batch(step)

    class FakeManager:
        def __init__(self):
            self.params = [mx.array([1.0])]

        def trainable_parameters(self):
            return self.params

        def set_trainable_parameters(self, params):
            self.params = params

        def set_dropout(self, _rate):
            pass

    class FakeModel:
        def eval(self):
            pass

    seeded: list[int] = []

    def fake_value_and_grad(_loss_fn):
        def calculate(params):
            return mx.array(0.0), [mx.zeros_like(param) for param in params]

        return calculate

    monkeypatch.setattr(train_module.mx.random, "seed", seeded.append)
    monkeypatch.setattr(train_module.mx, "value_and_grad", fake_value_and_grad)
    config = TrainingConfig(
        steps=2,
        batch_size=1,
        log_interval=10,
        training_seed=73,
        batch_schedule=RecordingSchedule(),
    )
    tokens = mx.array([[1, 2], [3, 4], [5, 6]], dtype=mx.int32)
    masks = mx.ones(tokens.shape, dtype=mx.float32)

    train_module.train_lora_supervised(FakeManager(), FakeModel(), tokens, masks, config)

    assert seeded == [73]
    assert requested_steps == [0, 1]
