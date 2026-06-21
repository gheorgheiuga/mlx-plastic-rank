"""Local-device memory profiles and memory-ledger helpers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .inspection import summarize_pack

BYTES_PER_GB = 1024**3


class DeviceProfileError(ValueError):
    """Raised when a device profile or memory ledger input is invalid."""


def gb_to_bytes(value: float) -> int:
    return int(math.ceil(float(value) * BYTES_PER_GB))


def bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(BYTES_PER_GB)


@dataclass(frozen=True)
class LocalDeviceProfile:
    """Memory and run-configuration defaults for one local machine class."""

    name: str
    total_memory_gb: int
    os_reserved_gb: float
    soft_budget_gb: float
    hard_budget_gb: float
    loader: str
    batch_size: int
    sequence_length: int
    eval_sample_cap: int
    rank_ceiling: int
    candidate_families: tuple[str, ...]
    notes: str

    @property
    def total_memory_bytes(self) -> int:
        return gb_to_bytes(self.total_memory_gb)

    @property
    def os_reserved_bytes(self) -> int:
        return gb_to_bytes(self.os_reserved_gb)

    @property
    def soft_budget_bytes(self) -> int:
        return gb_to_bytes(self.soft_budget_gb)

    @property
    def hard_budget_bytes(self) -> int:
        return gb_to_bytes(self.hard_budget_gb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_memory_gb": self.total_memory_gb,
            "total_memory_bytes": self.total_memory_bytes,
            "os_reserved_gb": self.os_reserved_gb,
            "os_reserved_bytes": self.os_reserved_bytes,
            "soft_budget_gb": self.soft_budget_gb,
            "soft_budget_bytes": self.soft_budget_bytes,
            "hard_budget_gb": self.hard_budget_gb,
            "hard_budget_bytes": self.hard_budget_bytes,
            "loader": self.loader,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "eval_sample_cap": self.eval_sample_cap,
            "rank_ceiling": self.rank_ceiling,
            "candidate_families": list(self.candidate_families),
            "notes": self.notes,
        }


LOCAL_DEVICE_PROFILES: dict[str, LocalDeviceProfile] = {
    "8gb": LocalDeviceProfile(
        name="8gb",
        total_memory_gb=8,
        os_reserved_gb=2.0,
        soft_budget_gb=5.4,
        hard_budget_gb=6.0,
        loader="auto",
        batch_size=1,
        sequence_length=128,
        eval_sample_cap=32,
        rank_ceiling=8,
        candidate_families=(
            "fixed_r4",
            "fixed_r8",
            "rank_map_json",
            "random_same_budget",
            "shuffled_discovered",
        ),
        notes="Tiny local profile; intended for smoke tests and small quantized bases.",
    ),
    "16gb": LocalDeviceProfile(
        name="16gb",
        total_memory_gb=16,
        os_reserved_gb=3.0,
        soft_budget_gb=11.7,
        hard_budget_gb=13.0,
        loader="auto",
        batch_size=1,
        sequence_length=256,
        eval_sample_cap=100,
        rank_ceiling=16,
        candidate_families=(
            "fixed_r8",
            "fixed_r16",
            "rank_map_json",
            "random_same_budget",
            "shuffled_discovered",
            "spectral_rank_map",
        ),
        notes="Default single-batch profile for modest local quantized-model experiments.",
    ),
    "32gb": LocalDeviceProfile(
        name="32gb",
        total_memory_gb=32,
        os_reserved_gb=4.0,
        soft_budget_gb=25.2,
        hard_budget_gb=28.0,
        loader="auto",
        batch_size=1,
        sequence_length=256,
        eval_sample_cap=300,
        rank_ceiling=32,
        candidate_families=(
            "fixed_r16",
            "fixed_r32",
            "dynamic_rank",
            "rank_map_json",
            "random_same_budget",
            "shuffled_discovered",
            "spectral_rank_map",
        ),
        notes="Main local PopRank profile for full held-out fault-code style evaluation.",
    ),
    "48gb": LocalDeviceProfile(
        name="48gb",
        total_memory_gb=48,
        os_reserved_gb=5.0,
        soft_budget_gb=38.7,
        hard_budget_gb=43.0,
        loader="auto",
        batch_size=2,
        sequence_length=512,
        eval_sample_cap=1000,
        rank_ceiling=64,
        candidate_families=(
            "fixed_r16",
            "fixed_r32",
            "fixed_r64",
            "dynamic_rank",
            "rank_map_json",
            "random_same_budget",
            "shuffled_discovered",
            "spectral_rank_map",
        ),
        notes="Larger local profile for broader sweeps and longer-context evaluation.",
    ),
}

PROFILE_ORDER = ("8gb", "16gb", "32gb", "48gb")

PROFILE_ALIASES = {
    "8": "8gb",
    "8g": "8gb",
    "8gb": "8gb",
    "16": "16gb",
    "16g": "16gb",
    "16gb": "16gb",
    "32": "32gb",
    "32g": "32gb",
    "32gb": "32gb",
    "48": "48gb",
    "48g": "48gb",
    "48gb": "48gb",
}


def normalize_device_profile_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", "").replace("-", "")
    if key not in PROFILE_ALIASES:
        raise DeviceProfileError(
            f"Unsupported local device profile {name!r}; expected one of {sorted(LOCAL_DEVICE_PROFILES)}."
        )
    return PROFILE_ALIASES[key]


def get_device_profile(name: str) -> LocalDeviceProfile:
    return LOCAL_DEVICE_PROFILES[normalize_device_profile_name(name)]


def iter_device_profiles(names: Iterable[str] | None = None) -> list[LocalDeviceProfile]:
    if names is None:
        return [LOCAL_DEVICE_PROFILES[name] for name in PROFILE_ORDER]
    return [get_device_profile(name) for name in names]


def parse_profile_names(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(PROFILE_ORDER)
    return [normalize_device_profile_name(part) for part in raw.split(",") if part.strip()]


def device_profiles_report(names: Iterable[str] | None = None) -> dict[str, Any]:
    profiles = iter_device_profiles(names)
    return {
        "kind": "local_device_memory_profiles",
        "profiles": [profile.to_dict() for profile in profiles],
    }


def device_profiles_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Local Device Memory Profiles",
        "",
        "| Profile | Total GB | Soft GB | Hard GB | Batch | Seq Len | Eval Cap | Rank Ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in report.get("profiles", []):
        lines.append(
            f"| `{profile['name']}` | {profile['total_memory_gb']} | "
            f"{profile['soft_budget_gb']} | {profile['hard_budget_gb']} | "
            f"{profile['batch_size']} | {profile['sequence_length']} | "
            f"{profile['eval_sample_cap']} | {profile['rank_ceiling']} |"
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeviceProfileError(f"Memory artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DeviceProfileError(f"Memory artifact is invalid JSON: {path}: {exc}") from exc


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted) or converted < 0.0:
        return None
    return converted


def _observation(
    *,
    source: str,
    kind: str,
    metric: str,
    peak_gb: float,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    peak_bytes = gb_to_bytes(peak_gb)
    return {
        "source": source,
        "kind": kind,
        "metric": metric,
        "peak_memory_gb": float(peak_gb),
        "peak_memory_bytes": peak_bytes,
        "details": dict(details or {}),
    }


def _iter_mapping_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
        summary = payload.get("summary")
        if isinstance(summary, Mapping):
            return [summary]
        return [payload]
    return []


def memory_observations_from_eval_report(
    payload: Any,
    *,
    source: str,
    pack: str | None = None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_mapping_rows(payload)):
        row_pack = row.get("pack")
        if pack is not None and row_pack not in {pack, None}:
            continue
        peak = _as_float(row.get("peak_memory_gb"))
        if peak is None:
            continue
        observations.append(
            _observation(
                source=source,
                kind="eval",
                metric="peak_memory_gb",
                peak_gb=peak,
                details={
                    "row": index,
                    "pack": row_pack,
                    "model": row.get("model"),
                    "loss_mode": row.get("loss_mode"),
                },
            )
        )
    return observations


def memory_observations_from_eval_batch_report(
    payload: Any,
    *,
    source: str,
    pack: str | None = None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, row in enumerate(_iter_mapping_rows(payload)):
        row_pack = row.get("pack")
        if pack is not None and row_pack not in {pack, None}:
            continue
        fields = ["vram_peak_base"]
        if pack is None or row_pack in {pack, None}:
            fields.append("vram_peak_pack")
        for field in fields:
            peak = _as_float(row.get(field))
            if peak is None:
                continue
            observations.append(
                _observation(
                    source=source,
                    kind="eval_batch",
                    metric=field,
                    peak_gb=peak,
                    details={
                        "row": index,
                        "pack": row_pack,
                        "domain": row.get("domain"),
                        "batch_size": row.get("batch_size"),
                        "sequence_length": row.get("sequence_length"),
                    },
                )
            )
    return observations


def memory_observations_from_generation_report(
    payload: Any,
    *,
    source: str,
    pack: str | None = None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    rows: list[Mapping[str, Any]]
    if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
        rows = [payload["summary"]]
    else:
        rows = _iter_mapping_rows(payload)
    for index, row in enumerate(rows):
        row_pack = row.get("pack")
        if pack is not None and row_pack not in {pack, None}:
            continue
        peak = _as_float(row.get("peak_memory_gb"))
        if peak is None:
            continue
        observations.append(
            _observation(
                source=source,
                kind="generation",
                metric="peak_memory_gb",
                peak_gb=peak,
                details={
                    "row": index,
                    "pack": row_pack,
                    "base": row.get("base"),
                    "examples": row.get("examples"),
                    "max_tokens": row.get("max_tokens"),
                },
            )
        )
    return observations


def memory_observations_from_report(
    path: Path,
    *,
    report_kind: str,
    pack: str | None = None,
) -> list[dict[str, Any]]:
    payload = _load_json(path)
    source = str(path)
    if report_kind == "eval":
        return memory_observations_from_eval_report(payload, source=source, pack=pack)
    if report_kind == "eval_batch":
        return memory_observations_from_eval_batch_report(payload, source=source, pack=pack)
    if report_kind == "generation":
        return memory_observations_from_generation_report(payload, source=source, pack=pack)
    raise DeviceProfileError(f"Unsupported memory report kind {report_kind!r}.")


def adapter_bytes_from_rank_budget_report(payload: Any) -> tuple[int | None, dict[str, Any] | None]:
    if not isinstance(payload, Mapping):
        return None, None
    summary = payload.get("normalized_summary") or payload.get("summary")
    if not isinstance(summary, Mapping):
        return None, None
    total = summary.get("tensor_bytes", summary.get("total_bytes"))
    try:
        return int(total), dict(summary)
    except (TypeError, ValueError):
        return None, None


def load_rank_budget_bytes(path: Path) -> tuple[int | None, dict[str, Any] | None]:
    return adapter_bytes_from_rank_budget_report(_load_json(path))


def pack_memory_inputs(pack_dir: Path) -> dict[str, Any]:
    metadata, _, total_params, tensor_bytes, non_lora = summarize_pack(pack_dir)
    pack_file = pack_dir / "pack.safetensors"
    file_bytes = os.path.getsize(pack_file) if pack_file.exists() else None
    return {
        "pack_dir": str(pack_dir),
        "metadata": metadata.to_dict(),
        "adapter_params": total_params,
        "adapter_tensor_bytes": tensor_bytes,
        "adapter_file_bytes": file_bytes,
        "non_lora_tensors": non_lora,
    }


def _peak_bytes(observations: Sequence[Mapping[str, Any]], kind: str | None = None) -> int | None:
    peaks = []
    for row in observations:
        if kind is not None and row.get("kind") != kind:
            continue
        value = row.get("peak_memory_bytes")
        if value is None:
            continue
        try:
            peaks.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(peaks) if peaks else None


def _status_for_profile(
    profile: LocalDeviceProfile,
    assessed_peak_bytes: int | None,
) -> tuple[str, str]:
    if assessed_peak_bytes is None:
        return "unknown", "No observed runtime peak, host RSS peak, or base-model memory estimate was provided."
    if assessed_peak_bytes <= profile.soft_budget_bytes:
        return "pass", "Assessed peak is within the soft profile budget."
    if assessed_peak_bytes <= profile.hard_budget_bytes:
        return "warn", "Assessed peak exceeds the soft budget but remains within the hard budget."
    return "fail", "Assessed peak exceeds the hard profile budget."


def build_memory_ledger(
    *,
    profiles: Iterable[str] | None = None,
    name: str | None = None,
    pack_info: Mapping[str, Any] | None = None,
    adapter_tensor_bytes: int | None = None,
    adapter_file_bytes: int | None = None,
    base_model_bytes: int | None = None,
    extra_overhead_bytes: int = 0,
    host_rss_peak_bytes: int | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
    rank_budget_sources: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a memory ledger against local-device profiles."""

    profile_rows = iter_device_profiles(profiles)
    observation_rows = [dict(row) for row in observations or []]
    adapter_bytes = adapter_tensor_bytes
    file_bytes = adapter_file_bytes
    if pack_info is not None:
        adapter_bytes = int(pack_info.get("adapter_tensor_bytes", adapter_bytes or 0))
        maybe_file = pack_info.get("adapter_file_bytes", file_bytes)
        file_bytes = int(maybe_file) if maybe_file is not None else None
    if adapter_bytes is None:
        adapter_bytes = 0
    if adapter_bytes < 0:
        raise DeviceProfileError("adapter_tensor_bytes must be >= 0.")
    if extra_overhead_bytes < 0:
        raise DeviceProfileError("extra_overhead_bytes must be >= 0.")
    if base_model_bytes is not None and base_model_bytes < 0:
        raise DeviceProfileError("base_model_bytes must be >= 0.")
    if host_rss_peak_bytes is not None and host_rss_peak_bytes < 0:
        raise DeviceProfileError("host_rss_peak_bytes must be >= 0.")

    observed_mlx_peak_bytes = _peak_bytes(observation_rows)
    observed_with_overhead = (
        observed_mlx_peak_bytes + extra_overhead_bytes
        if observed_mlx_peak_bytes is not None
        else None
    )
    static_estimate_bytes = None
    if base_model_bytes is not None:
        static_estimate_bytes = base_model_bytes + adapter_bytes + extra_overhead_bytes

    assessed_candidates = [
        value
        for value in (observed_with_overhead, host_rss_peak_bytes, static_estimate_bytes)
        if value is not None
    ]
    assessed_peak_bytes = max(assessed_candidates) if assessed_candidates else None

    profile_results = []
    for profile in profile_rows:
        status, reason = _status_for_profile(profile, assessed_peak_bytes)
        hard_slack = (
            profile.hard_budget_bytes - assessed_peak_bytes
            if assessed_peak_bytes is not None
            else None
        )
        soft_slack = (
            profile.soft_budget_bytes - assessed_peak_bytes
            if assessed_peak_bytes is not None
            else None
        )
        hard_utilization = (
            assessed_peak_bytes / float(profile.hard_budget_bytes)
            if assessed_peak_bytes is not None and profile.hard_budget_bytes > 0
            else None
        )
        row = {
            **profile.to_dict(),
            "fit_status": status,
            "fit_reason": reason,
            "assessed_peak_bytes": assessed_peak_bytes,
            "assessed_peak_gb": bytes_to_gb(assessed_peak_bytes),
            "soft_slack_bytes": soft_slack,
            "soft_slack_gb": bytes_to_gb(soft_slack),
            "hard_slack_bytes": hard_slack,
            "hard_slack_gb": bytes_to_gb(hard_slack),
            "hard_budget_utilization": hard_utilization,
        }
        profile_results.append(row)

    passing = [row for row in profile_results if row["fit_status"] == "pass"]
    hard_fit = [row for row in profile_results if row["fit_status"] in {"pass", "warn"}]
    if passing:
        status = "passed"
    elif hard_fit:
        status = "warn"
    elif assessed_peak_bytes is None:
        status = "unknown"
    else:
        status = "failed"

    return {
        "kind": "local_device_memory_ledger",
        "status": status,
        "name": name,
        "pack": dict(pack_info or {}),
        "rank_budget_sources": [dict(row) for row in rank_budget_sources or []],
        "memory_inputs": {
            "adapter_tensor_bytes": adapter_bytes,
            "adapter_tensor_gb": bytes_to_gb(adapter_bytes),
            "adapter_file_bytes": file_bytes,
            "adapter_file_gb": bytes_to_gb(file_bytes),
            "base_model_bytes": base_model_bytes,
            "base_model_gb": bytes_to_gb(base_model_bytes),
            "extra_overhead_bytes": extra_overhead_bytes,
            "extra_overhead_gb": bytes_to_gb(extra_overhead_bytes),
            "host_rss_peak_bytes": host_rss_peak_bytes,
            "host_rss_peak_gb": bytes_to_gb(host_rss_peak_bytes),
            "observed_mlx_peak_bytes": observed_mlx_peak_bytes,
            "observed_mlx_peak_gb": bytes_to_gb(observed_mlx_peak_bytes),
            "observed_mlx_peak_plus_overhead_bytes": observed_with_overhead,
            "observed_mlx_peak_plus_overhead_gb": bytes_to_gb(observed_with_overhead),
            "static_estimate_bytes": static_estimate_bytes,
            "static_estimate_gb": bytes_to_gb(static_estimate_bytes),
            "assessed_peak_bytes": assessed_peak_bytes,
            "assessed_peak_gb": bytes_to_gb(assessed_peak_bytes),
            "assessed_peak_basis": (
                "max(observed_mlx_peak+extra_overhead, host_rss_peak, static_estimate)"
                if assessed_peak_bytes is not None
                else "unknown"
            ),
        },
        "profiles": profile_results,
        "observations": observation_rows,
        "smallest_soft_fit_profile": passing[0]["name"] if passing else None,
        "smallest_hard_fit_profile": hard_fit[0]["name"] if hard_fit else None,
    }


def memory_ledger_rows_for_csv(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs = report.get("memory_inputs", {})
    if not isinstance(inputs, Mapping):
        inputs = {}
    for profile in report.get("profiles", []):
        rows.append(
            {
                "ledger": report.get("name"),
                "profile": profile.get("name"),
                "fit_status": profile.get("fit_status"),
                "assessed_peak_gb": profile.get("assessed_peak_gb"),
                "soft_budget_gb": profile.get("soft_budget_gb"),
                "hard_budget_gb": profile.get("hard_budget_gb"),
                "soft_slack_gb": profile.get("soft_slack_gb"),
                "hard_slack_gb": profile.get("hard_slack_gb"),
                "hard_budget_utilization": profile.get("hard_budget_utilization"),
                "observed_mlx_peak_gb": inputs.get("observed_mlx_peak_gb"),
                "host_rss_peak_gb": inputs.get("host_rss_peak_gb"),
                "base_model_gb": inputs.get("base_model_gb"),
                "adapter_tensor_gb": inputs.get("adapter_tensor_gb"),
            }
        )
    return rows


def memory_ledger_markdown(report: Mapping[str, Any]) -> str:
    inputs = report["memory_inputs"]
    lines = [
        f"# {report.get('name') or 'Local Device Memory Ledger'}",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Assessed peak GB: {inputs.get('assessed_peak_gb')}",
        f"- Observed MLX peak GB: {inputs.get('observed_mlx_peak_gb')}",
        f"- Host RSS peak GB: {inputs.get('host_rss_peak_gb')}",
        f"- Static estimate GB: {inputs.get('static_estimate_gb')}",
        f"- Adapter tensor GB: {inputs.get('adapter_tensor_gb')}",
        f"- Smallest soft-fit profile: `{report.get('smallest_soft_fit_profile')}`",
        f"- Smallest hard-fit profile: `{report.get('smallest_hard_fit_profile')}`",
        "",
        "## Profiles",
        "",
        "| Profile | Status | Assessed GB | Soft GB | Hard GB | Hard Slack GB | Rank Ceiling |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("profiles", []):
        lines.append(
            f"| `{row['name']}` | `{row['fit_status']}` | {row['assessed_peak_gb']} | "
            f"{row['soft_budget_gb']} | {row['hard_budget_gb']} | "
            f"{row['hard_slack_gb']} | {row['rank_ceiling']} |"
        )
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "| Source | Kind | Metric | Peak GB |",
            "| --- | --- | --- | ---: |",
        ]
    )
    observations = report.get("observations", [])
    if not observations:
        lines.append("| _none_ |  |  |  |")
    for row in observations:
        lines.append(
            f"| `{row['source']}` | {row['kind']} | {row['metric']} | {row['peak_memory_gb']} |"
        )
    return "\n".join(lines) + "\n"
