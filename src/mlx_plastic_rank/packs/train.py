"""LoRA training utilities for tiny adapter fine-tunes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .batch_schedule import (
    MinibatchSchedule,
    generate_minibatch_schedule,
    load_or_create_minibatch_schedule,
)
from .manager import LoRAManager


@dataclass
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-3
    sequence_length: int = 128
    log_interval: int = 100
    lora_dropout: float = 0.0
    dynamic_rank: bool = False
    dynamic_rank_interval: int = 50
    dynamic_rank_warmup: int = 50
    dynamic_rank_min: int = 2
    dynamic_rank_grow_threshold: float = 0.25
    dynamic_rank_prune_threshold: float = 0.03
    dynamic_rank_allowed_ranks: tuple[int, ...] = ()
    batch_seed: int = 42
    training_seed: int = 42
    batch_schedule: MinibatchSchedule | None = None
    batch_schedule_path: Path | str | None = None
    dataset_fingerprint: str | None = None
    resolved_batch_schedule_digest: str | None = field(init=False, default=None)

    def resolve_batch_schedule(self, *, dataset_size: int) -> MinibatchSchedule:
        """Return one schedule isolated from adapter and dropout RNG state."""

        if self.batch_schedule is not None and self.batch_schedule_path is not None:
            raise ValueError("Set batch_schedule or batch_schedule_path, not both")
        if self.batch_schedule is not None:
            self.batch_schedule.validate_for(
                dataset_size=dataset_size,
                batch_size=self.batch_size,
                steps=self.steps,
                dataset_fingerprint=self.dataset_fingerprint,
            )
            schedule = self.batch_schedule
        elif self.batch_schedule_path is not None:
            schedule = load_or_create_minibatch_schedule(
                self.batch_schedule_path,
                dataset_size=dataset_size,
                batch_size=self.batch_size,
                steps=self.steps,
                seed=self.batch_seed,
                dataset_fingerprint=self.dataset_fingerprint,
            )
        else:
            schedule = generate_minibatch_schedule(
                dataset_size=dataset_size,
                batch_size=self.batch_size,
                steps=self.steps,
                seed=self.batch_seed,
                dataset_fingerprint=self.dataset_fingerprint,
            )
        self.resolved_batch_schedule_digest = schedule.digest
        return schedule


def extract_logits(output):
    """Return logits from raw MLX outputs or model output containers."""

    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    return output


def model_logits(model, inputs: mx.array) -> mx.array:
    """Run a text forward pass across mlx-lm and mlx-vlm style models."""

    try:
        output = model(inputs)
    except TypeError:
        output = model(input_ids=inputs)
    return extract_logits(output)


def train_lora(
    manager: LoRAManager,
    model,
    dataset: mx.array,
    config: TrainingConfig,
) -> float:
    model.eval()
    params = manager.trainable_parameters()
    if not params:
        raise ValueError("No LoRA parameters initialised for training")

    manager.set_dropout(config.lora_dropout)
    schedule = config.resolve_batch_schedule(dataset_size=int(dataset.shape[0]))
    current_batch: mx.array

    def loss_fn(param_arrays: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(param_arrays)
        inputs = current_batch[:, :-1]
        targets = current_batch[:, 1:]
        logits = model_logits(model, inputs)
        loss = nn.losses.cross_entropy(logits, targets).mean()
        return loss

    param_arrays = params
    start = time.time()
    mx.random.seed(config.training_seed)
    for step in range(1, config.steps + 1):
        batch_indices = mx.array(schedule.batch(step - 1), dtype=mx.int32)
        current_batch = dataset[batch_indices]
        loss, grads = mx.value_and_grad(loss_fn)(param_arrays)
        param_arrays = [p - config.learning_rate * g for p, g in zip(param_arrays, grads)]
        manager.set_trainable_parameters(param_arrays)
        mx.eval(loss, *param_arrays)
        _maybe_adjust_dynamic_ranks(manager, config, step)
        if step % config.log_interval == 0 or step == config.steps:
            elapsed = time.time() - start
            print(f"step {step}/{config.steps} loss={float(loss):.4f} elapsed={elapsed:.1f}s")
    manager.set_dropout(0.0)
    return float(loss)


def train_lora_supervised(
    manager: LoRAManager,
    model,
    tokens: mx.array,
    masks: mx.array,
    config: TrainingConfig,
) -> float:
    model.eval()
    params = manager.trainable_parameters()
    if not params:
        raise ValueError("No LoRA parameters initialised for training")

    manager.set_dropout(config.lora_dropout)
    schedule = config.resolve_batch_schedule(dataset_size=int(tokens.shape[0]))
    current_tokens: mx.array
    current_masks: mx.array

    def loss_fn(param_arrays: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(param_arrays)
        inputs = current_tokens[:, :-1]
        targets = current_tokens[:, 1:]
        target_mask = current_masks[:, 1:]
        logits = model_logits(model, inputs)
        token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        denom = mx.sum(target_mask) + 1e-8
        return mx.sum(token_losses * target_mask) / denom

    param_arrays = params
    start = time.time()
    mx.random.seed(config.training_seed)
    for step in range(1, config.steps + 1):
        batch_indices = mx.array(schedule.batch(step - 1), dtype=mx.int32)
        current_tokens = tokens[batch_indices]
        current_masks = masks[batch_indices]
        loss, grads = mx.value_and_grad(loss_fn)(param_arrays)
        param_arrays = [p - config.learning_rate * g for p, g in zip(param_arrays, grads)]
        manager.set_trainable_parameters(param_arrays)
        mx.eval(loss, *param_arrays)
        _maybe_adjust_dynamic_ranks(manager, config, step)
        if step % config.log_interval == 0 or step == config.steps:
            elapsed = time.time() - start
            print(f"step {step}/{config.steps} supervised_loss={float(loss):.4f} elapsed={elapsed:.1f}s")
    manager.set_dropout(0.0)
    return float(loss)


def _maybe_adjust_dynamic_ranks(
    manager: LoRAManager,
    config: TrainingConfig,
    step: int,
) -> None:
    if not config.dynamic_rank:
        return
    if step < config.dynamic_rank_warmup:
        return
    if config.dynamic_rank_interval <= 0 or step % config.dynamic_rank_interval != 0:
        return
    events = manager.adjust_dynamic_ranks(
        allowed_ranks=config.dynamic_rank_allowed_ranks,
        min_rank=config.dynamic_rank_min,
        grow_threshold=config.dynamic_rank_grow_threshold,
        prune_threshold=config.dynamic_rank_prune_threshold,
    )
    for event in events:
        print(
            "dynamic-rank "
            f"step={step} adapter={event['adapter']} action={event['action']} "
            f"rank={event['from_rank']}->{event['to_rank']} "
            f"signal={event['signal']:.4g} global={event['global_signal']:.4g}"
        )
