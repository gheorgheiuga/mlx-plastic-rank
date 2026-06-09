"""LoRA training utilities for tiny adapter fine-tunes."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .dataset import sample_minibatch, sample_supervised_minibatch
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

    def loss_fn(param_arrays: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(param_arrays)
        batch = sample_minibatch(dataset, config.batch_size)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model_logits(model, inputs)
        loss = nn.losses.cross_entropy(logits, targets).mean()
        return loss

    param_arrays = params
    start = time.time()
    for step in range(1, config.steps + 1):
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

    def loss_fn(param_arrays: list[mx.array]) -> mx.array:
        manager.set_trainable_parameters(param_arrays)
        batch_tokens, batch_masks = sample_supervised_minibatch(tokens, masks, config.batch_size)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        target_mask = batch_masks[:, 1:]
        logits = model_logits(model, inputs)
        token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        denom = mx.sum(target_mask) + 1e-8
        return mx.sum(token_losses * target_mask) / denom

    param_arrays = params
    start = time.time()
    for step in range(1, config.steps + 1):
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
