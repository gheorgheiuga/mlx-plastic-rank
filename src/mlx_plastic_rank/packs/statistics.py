"""Paired uncertainty estimates for answer-mode perplexity evaluations."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class PairedPerplexityComparison:
    """Result of resampling aligned evaluation examples with replacement.

    ``ppl_difference`` is ``left - right``, so negative values favor the left
    candidate. ``relative_advantage`` is ``(right - left) / right``, so positive
    values favor the left candidate. Confidence intervals use bootstrap
    percentiles over examples, preserving each example's loss sum and token
    count as one cluster.
    """

    examples: int
    total_tokens: int
    left_perplexity: float
    right_perplexity: float
    ppl_difference: float
    ppl_difference_ci: tuple[float, float]
    relative_advantage: float
    relative_advantage_ci: tuple[float, float]
    probability_left_better: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit CI bounds."""

        return {
            "method": "paired_example_bootstrap",
            "examples": self.examples,
            "total_tokens": self.total_tokens,
            "left_perplexity": self.left_perplexity,
            "right_perplexity": self.right_perplexity,
            "ppl_difference": self.ppl_difference,
            "ppl_difference_ci": {
                "confidence_level": self.confidence_level,
                "lower": self.ppl_difference_ci[0],
                "upper": self.ppl_difference_ci[1],
            },
            "relative_advantage": self.relative_advantage,
            "relative_advantage_ci": {
                "confidence_level": self.confidence_level,
                "lower": self.relative_advantage_ci[0],
                "upper": self.relative_advantage_ci[1],
            },
            "probability_left_better": self.probability_left_better,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
        }


def _loss_vector(values: Sequence[Real], *, name: str) -> tuple[float, ...]:
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a real number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name}[{index}] must be finite")
        if converted < 0.0:
            raise ValueError(f"{name}[{index}] must be non-negative")
        result.append(converted)
    return tuple(result)


def _token_vector(values: Sequence[Integral], *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name}[{index}] must be an integer")
        converted = int(value)
        if converted <= 0:
            raise ValueError(f"{name}[{index}] must be positive")
        result.append(converted)
    return tuple(result)


def _perplexity(loss_sum: float, token_count: int) -> float:
    try:
        value = math.exp(loss_sum / token_count)
    except OverflowError as exc:
        raise ValueError("Perplexity overflows finite float precision") from exc
    if not math.isfinite(value):
        raise ValueError("Perplexity must be finite")
    return value


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _confidence_interval(
    values: list[float], confidence_level: float
) -> tuple[float, float]:
    values.sort()
    tail = (1.0 - confidence_level) / 2.0
    return _percentile(values, tail), _percentile(values, 1.0 - tail)


def compare_paired_perplexity(
    left_loss_sums: Sequence[Real],
    left_token_counts: Sequence[Integral],
    right_loss_sums: Sequence[Real],
    right_token_counts: Sequence[Integral],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> PairedPerplexityComparison:
    """Compare two candidates by paired, example-clustered bootstrapping.

    The vectors must describe the same examples in the same order. Token counts
    must match pairwise; a mismatch normally means the candidate artifacts were
    built from different evaluation rows or preprocessing.
    """

    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be between 0 and 1")
    checked_confidence = float(confidence_level)

    left_losses = _loss_vector(left_loss_sums, name="left_loss_sums")
    right_losses = _loss_vector(right_loss_sums, name="right_loss_sums")
    left_tokens = _token_vector(left_token_counts, name="left_token_counts")
    right_tokens = _token_vector(right_token_counts, name="right_token_counts")

    lengths = {
        len(left_losses),
        len(right_losses),
        len(left_tokens),
        len(right_tokens),
    }
    if len(lengths) != 1:
        raise ValueError("Loss-sum and token-count vectors must have equal length")
    examples = len(left_losses)
    if examples == 0:
        raise ValueError("At least one aligned evaluation example is required")
    if left_tokens != right_tokens:
        mismatch = next(
            index
            for index, (left_count, right_count) in enumerate(
                zip(left_tokens, right_tokens, strict=True)
            )
            if left_count != right_count
        )
        raise ValueError(
            "Aligned candidates must have identical token counts; "
            f"example {mismatch} has {left_tokens[mismatch]} and {right_tokens[mismatch]}"
        )

    total_tokens = sum(left_tokens)
    left_perplexity = _perplexity(sum(left_losses), total_tokens)
    right_perplexity = _perplexity(sum(right_losses), total_tokens)
    ppl_difference = left_perplexity - right_perplexity
    relative_advantage = (right_perplexity - left_perplexity) / right_perplexity

    rng = random.Random(seed)
    bootstrap_differences: list[float] = []
    bootstrap_advantages: list[float] = []
    left_wins = 0
    for _ in range(resamples):
        sampled_left_loss = 0.0
        sampled_right_loss = 0.0
        sampled_tokens = 0
        for _ in range(examples):
            index = rng.randrange(examples)
            sampled_left_loss += left_losses[index]
            sampled_right_loss += right_losses[index]
            sampled_tokens += left_tokens[index]

        sampled_left_ppl = _perplexity(sampled_left_loss, sampled_tokens)
        sampled_right_ppl = _perplexity(sampled_right_loss, sampled_tokens)
        difference = sampled_left_ppl - sampled_right_ppl
        advantage = (sampled_right_ppl - sampled_left_ppl) / sampled_right_ppl
        bootstrap_differences.append(difference)
        bootstrap_advantages.append(advantage)
        if difference < 0.0:
            left_wins += 1

    return PairedPerplexityComparison(
        examples=examples,
        total_tokens=total_tokens,
        left_perplexity=left_perplexity,
        right_perplexity=right_perplexity,
        ppl_difference=ppl_difference,
        ppl_difference_ci=_confidence_interval(
            bootstrap_differences, checked_confidence
        ),
        relative_advantage=relative_advantage,
        relative_advantage_ci=_confidence_interval(
            bootstrap_advantages, checked_confidence
        ),
        probability_left_better=left_wins / resamples,
        confidence_level=checked_confidence,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
    )


def _metric_sequence(metrics: Mapping[str, Any], key: str) -> Sequence[Any]:
    if key not in metrics:
        raise ValueError(f"Answer-mode metrics are missing {key!r}")
    value = metrics[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Answer-mode metric {key!r} must be an array")
    return value


def compare_answer_mode_metrics(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> PairedPerplexityComparison:
    """Compare two answer-mode metric rows emitted by pack evaluation."""

    if "provenance" in left or "provenance" in right:
        left_provenance, right_provenance = left.get("provenance"), right.get("provenance")
        if not isinstance(left_provenance, Mapping) or not isinstance(right_provenance, Mapping):
            raise ValueError("Both paired reports must contain provenance")
        if {k: v for k, v in left_provenance.items() if k != "pack"} != {
            k: v for k, v in right_provenance.items() if k != "pack"
        }:
            raise ValueError("Paired report provenance differs: model, examples, or preprocessing do not match")

    return compare_paired_perplexity(
        _metric_sequence(left, "example_loss_sums"),
        _metric_sequence(left, "example_token_counts"),
        _metric_sequence(right, "example_loss_sums"),
        _metric_sequence(right, "example_token_counts"),
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


__all__ = [
    "PairedPerplexityComparison",
    "compare_answer_mode_metrics",
    "compare_paired_perplexity",
]
