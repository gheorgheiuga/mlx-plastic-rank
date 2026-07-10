"""Deterministic, reusable minibatch index schedules."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_FORMAT_VERSION = 1


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class MinibatchSchedule:
    """A fixed sequence of zero-based dataset row indices."""

    dataset_size: int
    batch_size: int
    seed: int | None
    indices: tuple[tuple[int, ...], ...]
    dataset_fingerprint: str | None = None
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        dataset_size = _positive_int(self.dataset_size, name="dataset_size")
        batch_size = _positive_int(self.batch_size, name="batch_size")
        if batch_size > dataset_size:
            raise ValueError("Batch size exceeds dataset samples")
        if self.format_version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported minibatch schedule version {self.format_version}; "
                f"expected {_FORMAT_VERSION}"
            )
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or null")
        if self.dataset_fingerprint is not None and (
            not isinstance(self.dataset_fingerprint, str) or not self.dataset_fingerprint.strip()
        ):
            raise ValueError("dataset_fingerprint must be a non-empty string or null")

        normalized: list[tuple[int, ...]] = []
        for step, raw_batch in enumerate(self.indices):
            batch = tuple(raw_batch)
            if len(batch) != batch_size:
                raise ValueError(
                    f"Schedule step {step} has batch size {len(batch)}; expected {batch_size}"
                )
            for index in batch:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError(f"Schedule step {step} contains a non-integer index")
                if index < 0 or index >= dataset_size:
                    raise ValueError(
                        f"Schedule step {step} index {index} is outside dataset bounds "
                        f"[0, {dataset_size})"
                    )
            normalized.append(batch)
        if not normalized:
            raise ValueError("Minibatch schedule must contain at least one step")
        object.__setattr__(self, "indices", tuple(normalized))

    @property
    def steps(self) -> int:
        return len(self.indices)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def batch(self, step: int) -> tuple[int, ...]:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0 or step >= self.steps:
            raise IndexError(f"Schedule step {step} is outside [0, {self.steps})")
        return self.indices[step]

    def validate_for(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        steps: int,
        dataset_fingerprint: str | None = None,
    ) -> None:
        if dataset_size != self.dataset_size:
            raise ValueError(
                f"Schedule dataset size {self.dataset_size} does not match "
                f"requested dataset size {dataset_size}"
            )
        if batch_size != self.batch_size:
            raise ValueError(
                f"Schedule batch size {self.batch_size} does not match "
                f"requested batch size {batch_size}"
            )
        requested_steps = _positive_int(steps, name="steps")
        if requested_steps > self.steps:
            raise ValueError(
                f"Schedule has only {self.steps} steps; {requested_steps} required"
            )
        if dataset_fingerprint is not None:
            if self.dataset_fingerprint is None:
                raise ValueError("Schedule does not contain a dataset fingerprint")
            if dataset_fingerprint != self.dataset_fingerprint:
                raise ValueError(
                    "Schedule dataset fingerprint does not match the requested dataset preprocessing"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "indices": [list(batch) for batch in self.indices],
        }


def generate_minibatch_schedule(
    *,
    dataset_size: int,
    batch_size: int,
    steps: int,
    seed: int,
    dataset_fingerprint: str | None = None,
) -> MinibatchSchedule:
    """Generate with replacement using an RNG isolated from MLX model state."""

    checked_dataset_size = _positive_int(dataset_size, name="dataset_size")
    checked_batch_size = _positive_int(batch_size, name="batch_size")
    checked_steps = _positive_int(steps, name="steps")
    if checked_batch_size > checked_dataset_size:
        raise ValueError("Batch size exceeds dataset samples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    indices = tuple(
        tuple(rng.randrange(checked_dataset_size) for _ in range(checked_batch_size))
        for _ in range(checked_steps)
    )
    return MinibatchSchedule(
        dataset_size=checked_dataset_size,
        batch_size=checked_batch_size,
        seed=seed,
        indices=indices,
        dataset_fingerprint=dataset_fingerprint,
    )


def save_minibatch_schedule(schedule: MinibatchSchedule, path: Path | str) -> None:
    """Persist a schedule as human-inspectable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(schedule.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_minibatch_schedule(path: Path | str) -> MinibatchSchedule:
    """Load and validate a persisted schedule."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Minibatch schedule {source} must contain a JSON object")

    required = {"format_version", "dataset_size", "batch_size", "steps", "seed", "indices"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Minibatch schedule {source} is missing fields: {missing}")
    raw_indices = raw["indices"]
    if not isinstance(raw_indices, list):
        raise ValueError(f"Minibatch schedule {source} indices must be a JSON array")
    declared_steps = raw["steps"]
    if isinstance(declared_steps, bool) or not isinstance(declared_steps, int):
        raise ValueError(f"Minibatch schedule {source} steps must be an integer")
    if declared_steps != len(raw_indices):
        raise ValueError(
            f"Minibatch schedule {source} declares {declared_steps} steps "
            f"but contains {len(raw_indices)}"
        )

    try:
        indices = tuple(tuple(batch) for batch in raw_indices)
    except TypeError as exc:
        raise ValueError(f"Minibatch schedule {source} contains a non-array batch") from exc
    return MinibatchSchedule(
        format_version=raw["format_version"],
        dataset_size=raw["dataset_size"],
        batch_size=raw["batch_size"],
        seed=raw["seed"],
        indices=indices,
        dataset_fingerprint=raw.get("dataset_fingerprint"),
    )


def load_or_create_minibatch_schedule(
    path: Path | str,
    *,
    dataset_size: int,
    batch_size: int,
    steps: int,
    seed: int,
    dataset_fingerprint: str | None = None,
) -> MinibatchSchedule:
    """Reuse a valid schedule at ``path`` or create it deterministically."""

    schedule_path = Path(path)
    if schedule_path.exists():
        schedule = load_minibatch_schedule(schedule_path)
        schedule.validate_for(
            dataset_size=dataset_size,
            batch_size=batch_size,
            steps=steps,
            dataset_fingerprint=dataset_fingerprint,
        )
        return schedule

    schedule = generate_minibatch_schedule(
        dataset_size=dataset_size,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
    )
    save_minibatch_schedule(schedule, schedule_path)
    return schedule


__all__ = [
    "MinibatchSchedule",
    "generate_minibatch_schedule",
    "load_minibatch_schedule",
    "load_or_create_minibatch_schedule",
    "save_minibatch_schedule",
]
