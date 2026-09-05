"""LoRA training utilities for tiny adapter fine-tunes."""

from __future__ import annotations

import math
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
    batch_seed: int = 42
    training_seed: int = 42
    batch_schedule: MinibatchSchedule | None = None
    batch_schedule_path: Path | str | None = None
    dataset_fingerprint: str | None = None
    resolved_batch_schedule_digest: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "sequence_length", "log_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be finite and in [0, 1)")

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
    manager: LoRAManager, model, dataset: mx.array, config: TrainingConfig,
) -> float:
    """Train on every target token; return the last pre-update batch loss."""
    return _train(manager, model, dataset, None, config)


def train_lora_supervised(
    manager: LoRAManager, model, tokens: mx.array, masks: mx.array, config: TrainingConfig,
) -> float:
    """Train on masked target tokens; return the last pre-update batch loss."""
    return _train(manager, model, tokens, masks, config)


def _require_finite(values: list[mx.array], label: str) -> None:
    if not all(bool(mx.all(mx.isfinite(value)).item()) for value in values):
        raise ValueError(f"Non-finite {label}; training step was not committed")


def _train(
    manager: LoRAManager, model, tokens: mx.array, masks: mx.array | None,
    config: TrainingConfig,
) -> float:
    config.__post_init__()  # Also validate configurations edited after construction.
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError("Training tokens must be a nonempty matrix with at least two columns")
    if masks is not None:
        if masks.shape != tokens.shape:
            raise ValueError("Training masks must match the tokens")
        _require_finite([masks], "training masks")
        if bool(mx.any((masks < 0) | (masks > 1)).item()) or bool(mx.any(mx.sum(masks[:, 1:], axis=1) <= 0).item()):
            raise ValueError("Every training row needs positive target weight and masks in [0, 1]")
    param_arrays = manager.trainable_parameters()
    if not param_arrays:
        raise ValueError("No LoRA parameters initialised for training")
    _require_finite([p.astype(mx.float16) for p in param_arrays], "initial parameters")
    schedule = config.resolve_batch_schedule(dataset_size=int(tokens.shape[0]))
    model.eval()
    current_tokens: mx.array
    current_masks: mx.array | None

    def loss_fn(values: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(values)
        logits = model_logits(model, current_tokens[:, :-1])
        losses = nn.losses.cross_entropy(logits, current_tokens[:, 1:], reduction="none")
        if current_masks is None:
            return losses.mean()
        target_mask = current_masks[:, 1:]
        return mx.sum(losses * target_mask) / mx.sum(target_mask)

    start = time.time()
    mx.random.seed(config.training_seed)
    try:
        manager.set_dropout(config.lora_dropout)
        for step in range(1, config.steps + 1):
            indices = mx.array(schedule.batch(step - 1), dtype=mx.int32)
            current_tokens = tokens[indices]
            current_masks = masks[indices] if masks is not None else None
            loss, grads = mx.value_and_grad(loss_fn)(param_arrays)
            _require_finite([loss, *grads], "loss or gradients")
            next_params = [p - config.learning_rate * g for p, g in zip(param_arrays, grads)]
            # The live adapters store fp16. A finite fp32 master can still
            # overflow on conversion, so validate the actual storage format.
            _require_finite([p.astype(mx.float16) for p in next_params], "updated parameters")
            manager.set_trainable_parameters(next_params)
            param_arrays = next_params
            if step % config.log_interval == 0 or step == config.steps:
                elapsed = time.time() - start
                print(f"step {step}/{config.steps} pre_update_loss={float(loss):.4f} elapsed={elapsed:.1f}s")
    except Exception:
        manager.set_trainable_parameters(param_arrays)
        raise
    finally:
        manager.set_dropout(0.0)
    return float(loss)
