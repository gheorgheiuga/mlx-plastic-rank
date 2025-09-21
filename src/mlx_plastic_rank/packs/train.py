"""LoRA training utilities for tiny adapter fine-tunes."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .dataset import sample_minibatch
from .manager import LoRAManager


@dataclass
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-3
    sequence_length: int = 128
    log_interval: int = 100
    lora_dropout: float = 0.0


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

    def loss_fn(param_arrays: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(param_arrays)
        batch = sample_minibatch(dataset, config.batch_size)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = nn.losses.cross_entropy(logits, targets).mean()
        return loss

    param_arrays = params
    start = time.time()
    for step in range(1, config.steps + 1):
        loss, grads = mx.value_and_grad(loss_fn)(param_arrays)
        param_arrays = [p - config.learning_rate * g for p, g in zip(param_arrays, grads)]
        manager.set_trainable_parameters(param_arrays)
        if step % config.log_interval == 0 or step == config.steps:
            elapsed = time.time() - start
            print(f"step {step}/{config.steps} loss={float(loss):.4f} elapsed={elapsed:.1f}s")
    manager.set_dropout(0.0)
    return float(loss)
