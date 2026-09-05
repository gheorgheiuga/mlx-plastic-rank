"""One-command pack bakeoff orchestration.

The bakeoff runner is deliberately a thin orchestration layer around existing
CLI commands. It records the exact train/eval/ledger/proof commands, keeps
phase logs, and builds a compact summary from completed artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .provenance import content_sha256, digest_json
from .statistics import compare_answer_mode_metrics

VALID_CANDIDATE_MODES = {
    "fixed_rank",
    "dynamic_rank",
    "random_same_budget",
    "rank_map_from_candidate",
    "rank_map_from_pack",
    "rank_map_json",
    "resume_pack",
    "shuffled_discovered",
}

CONTROL_CANDIDATE_MODES = {"random_same_budget", "shuffled_discovered"}


class BakeoffError(ValueError):
    """Raised when a bakeoff spec or phase cannot be executed."""


@dataclass(frozen=True)
class BakeoffCandidate:
    candidate_id: str
    pack: str
    mode: str
    raw: Mapping[str, Any]
    quality_reference: bool = False
    small_reference: bool = False
    tradeoff_candidate: bool = False


@dataclass(frozen=True)
class BakeoffSpec:
    name: str
    domain: str
    base: str
    loader: str
    train_data: Path
    eval_data: Path
    output_dir: Path
    layers: str
    profile: str
    root: Path = field(default_factory=Path.cwd)
    train: Mapping[str, Any] = field(default_factory=dict)
    eval: Mapping[str, Any] = field(default_factory=dict)
    proof: Mapping[str, Any] = field(default_factory=dict)
    promotion_gates: Mapping[str, Any] = field(default_factory=dict)
    candidates: tuple[BakeoffCandidate, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidates_by_id(self) -> dict[str, BakeoffCandidate]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}


@dataclass(frozen=True)
class BakeoffPhase:
    candidate_id: str
    phase: str
    command: tuple[str, ...]
    log_path: Path
    skip_path: Path | None = None
    additional_skip_paths: tuple[Path, ...] = ()
    output_path: Path | None = None
    input_paths: tuple[Path, ...] = ()
    base_reference: str | None = None

    @property
    def receipt_path(self) -> Path:
        return self.log_path.with_suffix(".receipt.json")

    def _outputs(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.skip_path, *self.additional_skip_paths) if path is not None)

    def _input_fingerprint(self) -> str:
        source_root = Path(__file__).resolve().parents[1]
        source_identity = {
            str(path.relative_to(source_root)): content_sha256(path)
            for path in sorted(source_root.rglob("*.py"))
        }
        project_root = source_root.parent.parent
        for name in ("pyproject.toml", "uv.lock"):
            path = project_root / name
            if path.is_file():
                source_identity[name] = content_sha256(path)
        model_path = Path(self.base_reference).expanduser() if self.base_reference else None
        if self.base_reference and model_path is not None and not model_path.exists():
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(self.base_reference, "config.json")
            model_path = Path(cached).parent if isinstance(cached, str) else None
        if self.base_reference and model_path is None:
            raise BakeoffError(f"Cannot identify checkpoint for {self.base_reference}")
        return digest_json({
            "command": [part for part in self.command if part != "--force"],
            "implementation": source_identity,
            "inputs": {str(path): content_sha256(path) if path.exists() else None for path in self.input_paths},
            "model_reference": self.base_reference,
            "model_sha256": content_sha256(model_path) if model_path is not None else None,
        })

    def record_completion(self) -> None:
        """Record a receipt only after every declared output has been produced."""
        outputs = self._outputs()
        if not outputs or not all(path.is_file() for path in outputs):
            raise BakeoffError(f"Incomplete outputs for {self.candidate_id} {self.phase}")
        if not all(path.is_file() for path in self.input_paths):
            raise BakeoffError(f"Missing inputs for {self.candidate_id} {self.phase}")
        receipt = {
            "version": 1, "inputs": self._input_fingerprint(),
            "outputs": {str(path): content_sha256(path) for path in outputs},
        }
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.receipt_path)

    def should_skip(self, *, force: bool) -> bool:
        if force:
            return False
        try:
            receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
            outputs = self._outputs()
            return (
                receipt.get("version") == 1
                and bool(outputs) and all(path.is_file() for path in outputs)
                and receipt.get("inputs") == self._input_fingerprint()
                and receipt.get("outputs") == {str(path): content_sha256(path) for path in outputs}
            )
        except (OSError, ValueError, AttributeError):
            return False


def load_bakeoff_spec(path: Path, *, root: Path | None = None) -> BakeoffSpec:
    """Load and validate a bakeoff JSON spec."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BakeoffError(f"Bakeoff spec not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BakeoffError(f"Bakeoff spec is invalid JSON: {path}: {exc}") from exc
    return validate_bakeoff_spec(payload, root=root)


def validate_bakeoff_spec(payload: Any, *, root: Path | None = None) -> BakeoffSpec:
    """Validate a bakeoff spec payload and return normalized fields."""

    if not isinstance(payload, Mapping):
        raise BakeoffError("Bakeoff spec must be a JSON object.")
    root_path = (root or Path.cwd()).resolve()

    def required_text(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BakeoffError(f"Bakeoff spec requires a non-empty '{key}' field.")
        return value.strip()

    def optional_mapping(key: str) -> Mapping[str, Any]:
        value = payload.get(key, {})
        if not isinstance(value, Mapping):
            raise BakeoffError(f"Bakeoff spec field '{key}' must be an object.")
        return value

    train_data = _spec_path(required_text("train_data"), root_path)
    eval_data = _spec_path(required_text("eval_data"), root_path)
    if not train_data.exists():
        raise BakeoffError(f"Bakeoff train_data does not exist: {train_data}")
    if not eval_data.exists():
        raise BakeoffError(f"Bakeoff eval_data does not exist: {eval_data}")

    candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list) or not candidates_payload:
        raise BakeoffError("Bakeoff spec requires a non-empty 'candidates' list.")

    seen: dict[str, BakeoffCandidate] = {}
    candidates: list[BakeoffCandidate] = []
    for raw_candidate in candidates_payload:
        if not isinstance(raw_candidate, Mapping):
            raise BakeoffError("Each bakeoff candidate must be an object.")
        candidate_id = _required_candidate_text(raw_candidate, "id")
        if candidate_id in seen:
            raise BakeoffError(f"Duplicate bakeoff candidate id: {candidate_id}")
        pack = _required_candidate_text(raw_candidate, "pack")
        mode = _normalize_mode(_required_candidate_text(raw_candidate, "mode"))
        _validate_candidate_mode(raw_candidate, mode, seen)
        candidate = BakeoffCandidate(
            candidate_id=candidate_id,
            pack=pack,
            mode=mode,
            raw=raw_candidate,
            quality_reference=bool(raw_candidate.get("quality_reference")),
            small_reference=bool(raw_candidate.get("small_reference")),
            tradeoff_candidate=bool(raw_candidate.get("tradeoff_candidate")),
        )
        seen[candidate_id] = candidate
        candidates.append(candidate)

    metadata = optional_mapping("metadata")
    base = required_text("base")
    local_base = _spec_path(base, root_path)
    if local_base.exists():
        base = str(local_base.resolve())
    return BakeoffSpec(
        name=required_text("name"),
        domain=required_text("domain"),
        base=base,
        loader=str(payload.get("loader", "auto")),
        train_data=train_data,
        eval_data=eval_data,
        output_dir=_spec_path(required_text("output_dir"), root_path),
        layers=str(payload.get("layers", "attn.q_proj,attn.k_proj,attn.v_proj")),
        profile=str(payload.get("profile", "lite")),
        root=root_path.resolve(),
        train=optional_mapping("train"),
        eval=optional_mapping("eval"),
        proof=optional_mapping("proof"),
        promotion_gates=optional_mapping("promotion_gates"),
        candidates=tuple(candidates),
        metadata=metadata,
    )


def build_bakeoff_plan(spec: BakeoffSpec, *, force: bool = False) -> list[BakeoffPhase]:
    """Build deterministic create/eval/ledger/proof phases for a spec."""

    phases: list[BakeoffPhase] = []
    for candidate in spec.candidates:
        phase_paths = _candidate_paths(spec, candidate)
        if candidate.mode in CONTROL_CANDIDATE_MODES:
            phases.append(
                BakeoffPhase(
                    candidate_id=candidate.candidate_id,
                    phase="rank-map",
                    command=tuple(_control_rank_map_command(spec, candidate, phase_paths)),
                    log_path=phase_paths["rank_map_log"],
                    skip_path=phase_paths["rank_map_json"],
                    additional_skip_paths=(phase_paths["rank_map_report_json"],),
                    output_path=phase_paths["rank_map_json"],
                )
            )
        phases.append(
            BakeoffPhase(
                candidate_id=candidate.candidate_id,
                phase="create",
                command=tuple(_create_command(spec, candidate, force=force)),
                log_path=phase_paths["create_log"],
                skip_path=spec.root / "packs" / candidate.pack / "meta.json",
                additional_skip_paths=(spec.root / "packs" / candidate.pack / "pack.safetensors",),
            )
        )
        phases.append(
            BakeoffPhase(
                candidate_id=candidate.candidate_id,
                phase="eval",
                command=tuple(_eval_command(spec, candidate, phase_paths["eval_json"], phase_paths["eval_csv"])),
                log_path=phase_paths["eval_log"],
                skip_path=phase_paths["eval_json"],
                output_path=phase_paths["eval_json"],
            )
        )
        phases.append(
            BakeoffPhase(
                candidate_id=candidate.candidate_id,
                phase="rank-ledger",
                command=tuple(_ledger_command(candidate, phase_paths["ledger_json"], phase_paths["ledger_csv"])),
                log_path=phase_paths["ledger_log"],
                skip_path=phase_paths["ledger_json"],
                output_path=phase_paths["ledger_json"],
            )
        )
        phases.append(
            BakeoffPhase(
                candidate_id=candidate.candidate_id,
                phase="proof",
                command=tuple(_proof_command(spec, candidate, phase_paths["eval_json"], phase_paths["ledger_json"], phase_paths["proof_json"])),
                log_path=phase_paths["proof_log"],
                skip_path=phase_paths["proof_json"],
                output_path=phase_paths["proof_json"],
            )
        )
    return [_bind_phase_inputs(spec, phase) for phase in phases]


def _bind_phase_inputs(spec: BakeoffSpec, phase: BakeoffPhase) -> BakeoffPhase:
    candidate = spec.candidates_by_id[phase.candidate_id]
    inputs: list[Path] = []

    def add_pack(reference: str) -> None:
        location = _spec_path(reference, spec.root)
        if not location.is_dir():
            location = spec.root / "packs" / reference
        inputs.extend([location / "meta.json", location / "pack.safetensors"])

    paths = _candidate_paths(spec, candidate)
    if phase.phase != "create" and phase.phase != "rank-map":
        add_pack(candidate.pack)
    if phase.phase == "create":
        inputs.append(spec.train_data)
        for flag in ("--rank-map-json", "--batch-schedule"):
            if flag in phase.command:
                inputs.append(_spec_path(phase.command[phase.command.index(flag) + 1], spec.root))
        for flag in ("--rank-map-from-pack", "--resume-pack"):
            if flag in phase.command:
                add_pack(phase.command[phase.command.index(flag) + 1])
    elif phase.phase == "rank-map":
        add_pack(_control_source_pack(spec, candidate))
    elif phase.phase == "eval":
        inputs.append(spec.eval_data)
    elif phase.phase == "proof":
        inputs.extend([spec.train_data, spec.eval_data, paths["eval_json"], paths["ledger_json"]])
        if "generation_report" in candidate.raw:
            inputs.append(_spec_path(str(candidate.raw["generation_report"]), spec.root))
    return replace(
        phase, input_paths=tuple(inputs),
        base_reference=spec.base if phase.phase in {"create", "eval"} else None,
    )


def bakeoff_plan_payload(spec: BakeoffSpec, *, force: bool = False) -> dict[str, Any]:
    """Return a JSON-serializable dry-run payload."""

    phases = build_bakeoff_plan(spec, force=force)
    return {
        "kind": "pack_bakeoff_plan",
        "name": spec.name,
        "domain": spec.domain,
        "base_model": spec.base,
        "output_dir": str(spec.output_dir),
        "summary_json": str(_summary_json_path(spec)),
        "summary_csv": str(_summary_csv_path(spec)),
        "phases": [
            {
                "candidate": phase.candidate_id,
                "phase": phase.phase,
                "command": list(phase.command),
                "display": shlex.join(phase.command),
                "log": str(phase.log_path),
                "output": str(phase.output_path) if phase.output_path else None,
                "skip_path": str(phase.skip_path) if phase.skip_path else None,
                "additional_skip_paths": [str(path) for path in phase.additional_skip_paths],
                "would_skip": phase.should_skip(force=force),
            }
            for phase in phases
        ],
    }


def run_bakeoff(spec: BakeoffSpec, *, force: bool = False, cwd: Path | None = None) -> dict[str, Any]:
    """Execute a bakeoff spec and write compact summary artifacts."""

    working_dir = cwd or spec.root
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    for phase in build_bakeoff_plan(spec, force=force):
        phase.log_path.parent.mkdir(parents=True, exist_ok=True)
        if phase.should_skip(force=force):
            print(f"Skipping {phase.candidate_id} {phase.phase}; found {phase.skip_path}")
            continue
        if not force and any(path.exists() for path in phase._outputs()):
            raise BakeoffError(
                f"Stale or unverified artifacts for {phase.candidate_id} {phase.phase}; "
                "use new pack/output names to preserve them, or --force to regenerate."
            )
        print(f"Running {phase.candidate_id} {phase.phase}: {shlex.join(phase.command)}")
        with phase.log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"$ {shlex.join(phase.command)}\n\n")
            result = subprocess.run(
                phase.command,
                cwd=working_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise BakeoffError(
                f"Bakeoff phase failed: candidate={phase.candidate_id} "
                f"phase={phase.phase} exit={result.returncode}; see {phase.log_path}"
            )
        phase.record_completion()

    summary = build_bakeoff_summary(spec)
    write_bakeoff_summary(spec, summary)
    return summary


def build_bakeoff_summary(spec: BakeoffSpec) -> dict[str, Any]:
    """Build a compact bakeoff summary from completed candidate artifacts."""

    base_metrics: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    paired_eval_rows: dict[str, Mapping[str, Any]] = {}
    for candidate in spec.candidates:
        paths = _candidate_paths(spec, candidate)
        eval_rows = _load_eval_rows(paths["eval_json"])
        base_row, pack_row = _find_eval_pair(eval_rows, candidate.pack)
        paired_eval_rows[candidate.candidate_id] = pack_row
        if base_metrics is None:
            base_metrics = _base_metrics(base_row)
        assert base_metrics is not None
        ledger_report = _load_json(paths["ledger_json"])
        ledger = _report_summary(ledger_report)
        proof = _load_json(paths["proof_json"])
        control_report = (
            _validated_control_report(spec, candidate, paths, ledger_report)
            if candidate.mode in CONTROL_CANDIDATE_MODES
            else {}
        )
        rows.append(
            _candidate_summary_row(
                spec,
                candidate,
                base_metrics,
                pack_row,
                ledger,
                proof,
                control_report,
            )
        )

    if base_metrics is None:
        raise BakeoffError("Cannot build bakeoff summary without candidate eval artifacts.")

    verified = all(phase.should_skip(force=False) for phase in build_bakeoff_plan(spec))
    promotion = _promotion_gate_summary(spec, base_metrics, rows, paired_eval_rows=paired_eval_rows)
    if promotion is not None:
        promotion["artifact_provenance_verified"] = verified
        promotion["passed"] = bool(promotion["passed"] and verified)

    return {
        "kind": "pack_bakeoff_summary",
        "artifact_provenance_verified": verified,
        "name": spec.name,
        "domain": spec.domain,
        "base_model": spec.base,
        "loader": spec.loader,
        "train_data": str(spec.train_data),
        "eval_data": str(spec.eval_data),
        "output_dir": str(spec.output_dir),
        "metadata": dict(spec.metadata),
        "base_metrics": base_metrics,
        "rows": rows,
        "winner_quality": _winner_quality(rows),
        "winner_tradeoff": _winner_tradeoff(rows),
        "promotion_gates": promotion,
    }


def write_bakeoff_summary(spec: BakeoffSpec, summary: Mapping[str, Any]) -> None:
    """Write summary JSON and CSV artifacts."""

    json_path = _summary_json_path(spec)
    csv_path = _summary_csv_path(spec)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = summary.get("rows", [])
    if isinstance(rows, list) and rows:
        fieldnames = [
            "candidate",
            "pack",
            "mode",
            "control_type",
            "control_source",
            "control_seed",
            "control_rank_map",
            "control_report",
            "control_reference_bytes",
            "control_candidate_bytes",
            "control_budget_slack_bytes",
            "size_mb",
            "pack_size_bytes",
            "declared_rank",
            "effective_rank",
            "rank_slack",
            "perplexity",
            "ppl_delta_pct",
            "ppl_improvement_pct",
            "token_accuracy",
            "accuracy_gain_vs_base",
            "max_logit_diff",
            "proof_status",
            "improvement_per_mb",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})


def _cli_prefix() -> list[str]:
    return [sys.executable, "-m", "mlx_plastic_rank.packs.cli"]


def _create_command(spec: BakeoffSpec, candidate: BakeoffCandidate, *, force: bool) -> list[str]:
    train = spec.train
    raw = candidate.raw
    command = [
        *_cli_prefix(),
        "create",
        "--name",
        candidate.pack,
        "--base",
        spec.base,
        "--loader",
        str(raw.get("loader", spec.loader)),
        "--layers",
        str(raw.get("layers", spec.layers)),
        "--data",
        str(spec.train_data),
        "--steps",
        str(_setting(raw, train, "steps", 1500)),
        "--batch-size",
        str(_setting(raw, train, "batch_size", 4)),
        "--learning-rate",
        str(_setting(raw, train, "learning_rate", 1e-4)),
        "--sequence-length",
        str(_setting(raw, train, "sequence_length", 128)),
        "--seed",
        str(_setting(raw, train, "seed", 42)),
        "--loss-mode",
        str(_setting(raw, train, "loss_mode", "full")),
        "--profile",
        str(raw.get("profile", spec.profile)),
        "--lora-dropout",
        str(_setting(raw, train, "lora_dropout", 0.0)),
    ]
    if bool(_setting(raw, train, "chat_template", False)):
        command.append("--chat-template")
    _append_option(command, "--batch-seed", _setting(raw, train, "batch_seed", None))
    _append_option(command, "--initialization", _setting(raw, train, "initialization", "legacy"))
    _append_option(command, "--training-seed", _setting(raw, train, "training_seed", None))
    _append_option(command, "--batch-schedule", _setting(raw, train, "batch_schedule", None))
    if bool(_setting(raw, train, "train_fp16_fallback", False)):
        command.append("--train-fp16-fallback")
    if force:
        command.append("--force")

    if candidate.mode in {"fixed_rank", "dynamic_rank"}:
        command.extend(["--rank", str(_required_candidate_int(raw, "rank"))])
    if candidate.mode == "dynamic_rank":
        command.append("--dynamic-rank")
        _append_option(command, "--dynamic-initial-rank", raw.get("dynamic_initial_rank"))
        _append_option(command, "--dynamic-min-rank", raw.get("dynamic_min_rank"))
        _append_option(command, "--dynamic-rank-warmup", raw.get("dynamic_rank_warmup"))
        _append_option(command, "--dynamic-rank-interval", raw.get("dynamic_rank_interval"))
        _append_option(command, "--dynamic-grow-threshold", raw.get("dynamic_grow_threshold"))
        _append_option(command, "--dynamic-prune-threshold", raw.get("dynamic_prune_threshold"))
    elif candidate.mode == "rank_map_from_candidate":
        source_id = _required_candidate_text(raw, "rank_map_from_candidate")
        source = spec.candidates_by_id[source_id]
        command.extend(["--rank-map-from-pack", source.pack])
    elif candidate.mode == "rank_map_from_pack":
        command.extend(["--rank-map-from-pack", _required_candidate_text(raw, "rank_map_from_pack")])
    elif candidate.mode == "rank_map_json":
        command.extend(["--rank-map-json", _required_candidate_text(raw, "rank_map_json")])
    elif candidate.mode == "resume_pack":
        command.extend(["--resume-pack", _required_candidate_text(raw, "resume_pack")])
    elif candidate.mode in CONTROL_CANDIDATE_MODES:
        command.extend(["--rank-map-json", str(_candidate_paths(spec, candidate)["rank_map_json"])])

    if "notes" in raw:
        command.extend(["--notes", str(raw["notes"])])
    if "min_rank" in raw:
        command.extend(["--min-rank", str(raw["min_rank"])])
    return command


def _control_rank_map_command(
    spec: BakeoffSpec,
    candidate: BakeoffCandidate,
    paths: Mapping[str, Path],
) -> list[str]:
    """Build the deterministic rank-map preflight for a control candidate."""

    if candidate.mode not in CONTROL_CANDIDATE_MODES:
        raise BakeoffError(f"Candidate {candidate.candidate_id!r} is not a control candidate.")
    subcommand = candidate.mode.replace("_", "-")
    command = [
        *_cli_prefix(),
        "rank-map",
        subcommand,
        "--source-pack",
        _control_source_pack(spec, candidate),
        "--profile",
        str(candidate.raw.get("profile", spec.profile)),
        "--seed",
        str(_control_seed(spec, candidate)),
        "--out",
        str(paths["rank_map_report_json"]),
        "--markdown",
        str(paths["rank_map_report_markdown"]),
        "--rank-map-out",
        str(paths["rank_map_json"]),
    ]
    if bool(candidate.raw.get("allow_over_budget", False)):
        command.append("--allow-over-budget")
    _append_option(command, "--tensor-dtype", candidate.raw.get("tensor_dtype"))
    _append_option(command, "--alpha-dtype", candidate.raw.get("alpha_dtype"))
    _append_option(command, "--file-overhead-bytes", candidate.raw.get("file_overhead_bytes"))
    return command


def _control_source_pack(spec: BakeoffSpec, candidate: BakeoffCandidate) -> str:
    source_candidate = candidate.raw.get("control_source_candidate")
    if isinstance(source_candidate, str) and source_candidate.strip():
        return spec.candidates_by_id[source_candidate.strip()].pack
    return _required_candidate_text(candidate.raw, "control_source_pack")


def _control_seed(spec: BakeoffSpec, candidate: BakeoffCandidate) -> int:
    value = candidate.raw.get("control_seed", _setting(candidate.raw, spec.train, "seed", 42))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BakeoffError(
            f"Control candidate {candidate.candidate_id!r} requires a non-negative integer seed."
        )
    return value


def _validated_control_report(
    spec: BakeoffSpec,
    candidate: BakeoffCandidate,
    paths: Mapping[str, Path],
    ledger_report: Any,
) -> Mapping[str, Any]:
    report = _load_json(paths["rank_map_report_json"])
    rank_map_payload = _load_json(paths["rank_map_json"])
    if not isinstance(report, Mapping) or not isinstance(rank_map_payload, Mapping):
        raise BakeoffError(
            f"Control artifacts for {candidate.candidate_id!r} must be JSON objects."
        )
    if report.get("control") != candidate.mode:
        raise BakeoffError(
            f"Control report for {candidate.candidate_id!r} has mode "
            f"{report.get('control')!r}; expected {candidate.mode!r}."
        )
    if _int_or_none(report.get("seed")) != _control_seed(spec, candidate):
        raise BakeoffError(
            f"Control report for {candidate.candidate_id!r} has a stale or mismatched seed."
        )
    if rank_map_payload.get("rank_map") != report.get("rank_map"):
        raise BakeoffError(
            f"Control rank map for {candidate.candidate_id!r} does not match its report."
        )
    if rank_map_payload.get("alpha_map") != report.get("alpha_map"):
        raise BakeoffError(
            f"Control alpha map for {candidate.candidate_id!r} does not match its report."
        )
    ledger_rank_map, ledger_alpha_map = _rank_alpha_from_ledger(
        ledger_report,
        label=f"Control ledger for {candidate.candidate_id!r}",
    )
    if ledger_rank_map != report.get("rank_map"):
        raise BakeoffError(
            f"Trained control pack {candidate.candidate_id!r} does not match its generated rank map."
        )
    report_alpha_map = report.get("alpha_map")
    if not _alpha_maps_match(ledger_alpha_map, report_alpha_map):
        raise BakeoffError(
            f"Trained control pack {candidate.candidate_id!r} does not match its generated alpha map."
        )
    source_rank_map, source_alpha_map = _control_source_rank_alpha(spec, candidate)
    if source_rank_map != report.get("reference_rank_map"):
        raise BakeoffError(
            f"Control {candidate.candidate_id!r} was not generated from its declared source rank map."
        )
    if not _alpha_maps_match(source_alpha_map, report.get("reference_alpha_map")):
        raise BakeoffError(
            f"Control {candidate.candidate_id!r} was not generated from its declared source alpha map."
        )
    reference = report.get("reference_summary")
    normalized = report.get("normalized_summary")
    if not isinstance(reference, Mapping) or not isinstance(normalized, Mapping):
        raise BakeoffError(
            f"Control report for {candidate.candidate_id!r} is missing budget summaries."
        )
    reference_bytes = _int_or_none(reference.get("total_bytes"))
    candidate_bytes = _int_or_none(normalized.get("total_bytes"))
    if reference_bytes is None or candidate_bytes is None:
        raise BakeoffError(
            f"Control report for {candidate.candidate_id!r} has invalid byte totals."
        )
    if candidate_bytes > reference_bytes and not bool(candidate.raw.get("allow_over_budget", False)):
        raise BakeoffError(
            f"Control {candidate.candidate_id!r} exceeds its reference budget: "
            f"{candidate_bytes} > {reference_bytes} bytes."
        )
    return report


def _rank_alpha_from_ledger(
    ledger_report: Any,
    *,
    label: str,
) -> tuple[dict[str, int], dict[str, float]]:
    if not isinstance(ledger_report, Mapping) or not isinstance(
        ledger_report.get("adapters"), list
    ):
        raise BakeoffError(f"{label} is missing adapter provenance.")
    rank_map: dict[str, int] = {}
    alpha_map: dict[str, float] = {}
    for row in ledger_report["adapters"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("adapter"), str):
            raise BakeoffError(f"{label} has an invalid adapter row.")
        adapter = str(row["adapter"])
        rank = _int_or_none(row.get("declared_rank"))
        alpha = _float_or_none(row.get("alpha"))
        if rank is None or alpha is None:
            raise BakeoffError(f"{label} has invalid rank/alpha provenance.")
        rank_map[adapter] = rank
        alpha_map[adapter] = alpha
    return rank_map, alpha_map


def _alpha_maps_match(actual: Mapping[str, float], expected: Any) -> bool:
    if not isinstance(expected, Mapping) or set(actual) != set(expected):
        return False
    for adapter, alpha in actual.items():
        expected_alpha = _float_or_none(expected.get(adapter))
        if expected_alpha is None or not math.isclose(
            alpha,
            expected_alpha,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return False
    return True


def _control_source_rank_alpha(
    spec: BakeoffSpec,
    candidate: BakeoffCandidate,
) -> tuple[dict[str, int], dict[str, float]]:
    source_candidate_id = candidate.raw.get("control_source_candidate")
    if isinstance(source_candidate_id, str) and source_candidate_id.strip():
        source = spec.candidates_by_id[source_candidate_id.strip()]
        source_ledger = _load_json(_candidate_paths(spec, source)["ledger_json"])
        return _rank_alpha_from_ledger(
            source_ledger,
            label=f"Source ledger for control {candidate.candidate_id!r}",
        )

    source_reference = _required_candidate_text(candidate.raw, "control_source_pack")
    source_path = Path(source_reference).expanduser()
    if not source_path.exists():
        source_path = Path("packs") / source_reference
    metadata = _load_json(source_path / "meta.json")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("rank_map"), Mapping):
        raise BakeoffError(
            f"Source metadata for control {candidate.candidate_id!r} has no rank map."
        )
    try:
        rank_map = {str(key): int(value) for key, value in metadata["rank_map"].items()}
    except (TypeError, ValueError) as exc:
        raise BakeoffError(
            f"Source metadata for control {candidate.candidate_id!r} has an invalid rank map."
        ) from exc
    raw_alpha_map = metadata.get("alpha_map")
    if not isinstance(raw_alpha_map, Mapping):
        raw_alpha_map = {}
    try:
        alpha_map = {
            key: float(raw_alpha_map.get(key, 2.0 * rank))
            for key, rank in rank_map.items()
        }
    except (TypeError, ValueError) as exc:
        raise BakeoffError(
            f"Source metadata for control {candidate.candidate_id!r} has an invalid alpha map."
        ) from exc
    return rank_map, alpha_map


def _eval_command(spec: BakeoffSpec, candidate: BakeoffCandidate, out_json: Path, out_csv: Path) -> list[str]:
    eval_settings = spec.eval
    command = [
        *_cli_prefix(),
        "eval",
        "--base",
        spec.base,
        "--loader",
        spec.loader,
        "--pack",
        candidate.pack,
        "--data-path",
        str(spec.eval_data),
        "--sequence-length",
        str(eval_settings.get("sequence_length", spec.train.get("sequence_length", 128))),
        "--num-samples",
        str(eval_settings.get("num_samples", 100)),
        "--batch-size",
        str(eval_settings.get("batch_size", 8)),
        "--loss-mode",
        str(eval_settings.get("loss_mode", spec.train.get("loss_mode", "full"))),
        "--out",
        str(out_json),
        "--csv",
        str(out_csv),
    ]
    if bool(eval_settings.get("chat_template", spec.train.get("chat_template", False))):
        command.append("--chat-template")
    return command


def _ledger_command(candidate: BakeoffCandidate, out_json: Path, out_csv: Path) -> list[str]:
    return [
        *_cli_prefix(),
        "rank-ledger",
        "--name",
        candidate.pack,
        "--out",
        str(out_json),
        "--csv",
        str(out_csv),
    ]


def _proof_command(
    spec: BakeoffSpec,
    candidate: BakeoffCandidate,
    eval_json: Path,
    ledger_json: Path,
    out_json: Path,
) -> list[str]:
    proof = spec.proof
    command = [
        *_cli_prefix(),
        "proof",
        "--base",
        spec.base,
        "--pack",
        candidate.pack,
        "--domain",
        spec.domain,
        "--train-data",
        str(spec.train_data),
        "--eval-data",
        str(spec.eval_data),
        "--eval-report",
        str(eval_json),
        "--ledger-report",
        str(ledger_json),
        "--out",
        str(out_json),
    ]
    if bool(proof.get("require_ledger", True)):
        command.append("--require-ledger")
    if bool(proof.get("fail_on_regression", True)):
        command.append("--fail-on-regression")
    if bool(proof.get("require_generation", False)):
        command.append("--require-generation")
    if "generation_report" in candidate.raw:
        command.extend(["--generation-report", str(candidate.raw["generation_report"])])
    _append_option(command, "--min-ppl-improvement-pct", proof.get("min_ppl_improvement_pct"))
    _append_option(command, "--min-token-accuracy-gain", proof.get("min_token_accuracy_gain"))
    _append_option(command, "--min-generation-overlap-gain", proof.get("min_generation_overlap_gain"))
    _append_option(command, "--min-logit-diff", proof.get("min_logit_diff"))
    return command


def _candidate_paths(spec: BakeoffSpec, candidate: BakeoffCandidate) -> dict[str, Path]:
    base = spec.output_dir / candidate.candidate_id
    return {
        "rank_map_log": spec.output_dir / f"{candidate.candidate_id}_rank_map.log",
        "rank_map_report_json": base.with_name(f"{base.name}_rank_map_report.json"),
        "rank_map_report_markdown": base.with_name(f"{base.name}_rank_map_report.md"),
        "rank_map_json": base.with_name(f"{base.name}_rank_map.json"),
        "create_log": spec.output_dir / f"{candidate.candidate_id}_create.log",
        "eval_log": spec.output_dir / f"{candidate.candidate_id}_eval.log",
        "ledger_log": spec.output_dir / f"{candidate.candidate_id}_rank_ledger.log",
        "proof_log": spec.output_dir / f"{candidate.candidate_id}_proof.log",
        "eval_json": base.with_name(f"{base.name}_eval.json"),
        "eval_csv": base.with_name(f"{base.name}_eval.csv"),
        "ledger_json": base.with_name(f"{base.name}_rank_ledger.json"),
        "ledger_csv": base.with_name(f"{base.name}_rank_ledger.csv"),
        "proof_json": base.with_name(f"{base.name}_proof.json"),
    }


def _summary_json_path(spec: BakeoffSpec) -> Path:
    return spec.output_dir / f"{spec.name}_summary.json"


def _summary_csv_path(spec: BakeoffSpec) -> Path:
    return spec.output_dir / f"{spec.name}_summary.csv"


def _required_candidate_text(candidate: Mapping[str, Any], key: str) -> str:
    value = candidate.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BakeoffError(f"Bakeoff candidate requires a non-empty '{key}' field.")
    return value.strip()


def _required_candidate_int(candidate: Mapping[str, Any], key: str) -> int:
    value = candidate.get(key)
    if not isinstance(value, int) or value <= 0:
        raise BakeoffError(f"Bakeoff candidate requires a positive integer '{key}' field.")
    return value


def _normalize_mode(mode: str) -> str:
    normalized = mode.replace("-", "_")
    if normalized not in VALID_CANDIDATE_MODES:
        raise BakeoffError(
            f"Unsupported bakeoff candidate mode {mode!r}; "
            f"expected one of {sorted(VALID_CANDIDATE_MODES)}"
        )
    return normalized


def _validate_candidate_mode(
    candidate: Mapping[str, Any],
    mode: str,
    seen: Mapping[str, BakeoffCandidate],
) -> None:
    if mode in {"fixed_rank", "dynamic_rank"}:
        _required_candidate_int(candidate, "rank")
    elif mode == "rank_map_from_candidate":
        source_id = _required_candidate_text(candidate, "rank_map_from_candidate")
        if source_id not in seen:
            raise BakeoffError(
                f"Candidate {candidate.get('id')!r} references rank_map_from_candidate "
                f"{source_id!r}, which must appear earlier in the candidates list."
            )
    elif mode == "rank_map_from_pack":
        _required_candidate_text(candidate, "rank_map_from_pack")
    elif mode == "rank_map_json":
        _required_candidate_text(candidate, "rank_map_json")
    elif mode == "resume_pack":
        _required_candidate_text(candidate, "resume_pack")
    elif mode in CONTROL_CANDIDATE_MODES:
        _validate_control_candidate(candidate, seen)


def _validate_control_candidate(
    candidate: Mapping[str, Any],
    seen: Mapping[str, BakeoffCandidate],
) -> None:
    source_candidate = candidate.get("control_source_candidate")
    source_pack = candidate.get("control_source_pack")
    has_candidate = isinstance(source_candidate, str) and bool(source_candidate.strip())
    has_pack = isinstance(source_pack, str) and bool(source_pack.strip())
    if has_candidate == has_pack:
        raise BakeoffError(
            f"Control candidate {candidate.get('id')!r} requires exactly one of "
            "'control_source_candidate' or 'control_source_pack'."
        )
    if has_candidate:
        source_id = str(source_candidate).strip()
        if source_id not in seen:
            raise BakeoffError(
                f"Control candidate {candidate.get('id')!r} references "
                f"control_source_candidate {source_id!r}, which must appear earlier "
                "in the candidates list."
            )
    if "control_seed" in candidate:
        value = candidate["control_seed"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BakeoffError(
                f"Control candidate {candidate.get('id')!r} requires a non-negative integer seed."
            )


def _spec_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _setting(raw: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return raw.get(key, defaults.get(key, fallback))


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BakeoffError(f"Bakeoff artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BakeoffError(f"Bakeoff artifact is invalid JSON: {path}: {exc}") from exc


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    raise BakeoffError(f"Eval artifact must be a row list or contain rows: {path}")


def _find_eval_pair(rows: list[dict[str, Any]], pack_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base_row: dict[str, Any] | None = None
    pack_row: dict[str, Any] | None = None
    for row in rows:
        pack = row.get("pack")
        label = str(row.get("label") or "").lower()
        if pack in (None, "", "base") or label == "base":
            base_row = row
        if pack == pack_name:
            pack_row = row
    if base_row is None:
        raise BakeoffError("Eval artifact does not contain a base row.")
    if pack_row is None:
        raise BakeoffError(f"Eval artifact does not contain a row for pack {pack_name!r}.")
    return base_row, pack_row


def _report_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
        return dict(payload["summary"])
    return {}


def _base_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "perplexity": _float_or_none(row.get("perplexity", row.get("ppl"))),
        "token_accuracy": _float_or_none(row.get("token_accuracy", row.get("domain_metric"))),
        "eval_time_s": _float_or_none(row.get("eval_time_s")),
        "tokens_per_sec": _float_or_none(row.get("tokens_per_sec")),
        "peak_memory_gb": _float_or_none(row.get("peak_memory_gb")),
    }


def _candidate_summary_row(
    spec: BakeoffSpec,
    candidate: BakeoffCandidate,
    base: Mapping[str, Any],
    eval_row: Mapping[str, Any],
    ledger: Mapping[str, Any],
    proof: Any,
    control_report: Any,
) -> dict[str, Any]:
    base_ppl = _float_or_none(base.get("perplexity"))
    pack_ppl = _float_or_none(eval_row.get("perplexity", eval_row.get("ppl")))
    ppl_improvement_pct = None
    ppl_delta_pct = _float_or_none(eval_row.get("ppl_delta_pct"))
    if base_ppl is not None and pack_ppl is not None and base_ppl > 0:
        ppl_improvement_pct = ((base_ppl - pack_ppl) / base_ppl) * 100.0
        if ppl_delta_pct is None:
            ppl_delta_pct = -ppl_improvement_pct

    base_acc = _float_or_none(base.get("token_accuracy"))
    pack_acc = _float_or_none(eval_row.get("token_accuracy", eval_row.get("domain_metric")))
    accuracy_gain = None
    if base_acc is not None and pack_acc is not None:
        accuracy_gain = pack_acc - base_acc

    size_mb = _float_or_none(eval_row.get("size_mb"))
    improvement_per_mb = None
    if size_mb is not None and size_mb > 0 and ppl_improvement_pct is not None:
        improvement_per_mb = ppl_improvement_pct / size_mb

    proof_status = None
    if isinstance(proof, Mapping):
        proof_status = proof.get("status")

    declared_control_type = candidate.raw.get("control_type")
    control_type = (
        candidate.mode
        if candidate.mode in CONTROL_CANDIDATE_MODES
        else str(declared_control_type).strip()
        if isinstance(declared_control_type, str) and declared_control_type.strip()
        else None
    )
    control_source = None
    control_seed = None
    control_rank_map = None
    control_report_path = None
    control_reference_bytes = None
    control_candidate_bytes = None
    control_budget_slack_bytes = None
    if candidate.mode in CONTROL_CANDIDATE_MODES:
        paths = _candidate_paths(spec, candidate)
        control_source = _control_source_pack(spec, candidate)
        control_seed = _control_seed(spec, candidate)
        control_rank_map = str(paths["rank_map_json"])
        control_report_path = str(paths["rank_map_report_json"])
        if isinstance(control_report, Mapping):
            reference = control_report.get("reference_summary")
            normalized = control_report.get("normalized_summary")
            if isinstance(reference, Mapping):
                control_reference_bytes = _int_or_none(reference.get("total_bytes"))
            if isinstance(normalized, Mapping):
                control_candidate_bytes = _int_or_none(normalized.get("total_bytes"))
                control_budget_slack_bytes = _int_or_none(normalized.get("budget_slack_bytes"))
    elif control_type is not None:
        control_source = candidate.raw.get("control_source")
        control_seed = _int_or_none(candidate.raw.get("control_seed"))
        control_rank_map = candidate.raw.get("rank_map_json")
        control_report_path = candidate.raw.get("control_report")
        control_reference_bytes = _int_or_none(candidate.raw.get("control_reference_bytes"))
        control_candidate_bytes = _int_or_none(candidate.raw.get("control_candidate_bytes"))
        if control_reference_bytes is not None and control_candidate_bytes is not None:
            control_budget_slack_bytes = control_reference_bytes - control_candidate_bytes

    return {
        "candidate": candidate.candidate_id,
        "pack": candidate.pack,
        "mode": candidate.mode,
        "control_type": control_type,
        "control_source": control_source,
        "control_seed": control_seed,
        "control_rank_map": control_rank_map,
        "control_report": control_report_path,
        "control_reference_bytes": control_reference_bytes,
        "control_candidate_bytes": control_candidate_bytes,
        "control_budget_slack_bytes": control_budget_slack_bytes,
        "size_mb": size_mb,
        "pack_size_bytes": _int_or_none(eval_row.get("pack_size_bytes")),
        "declared_rank": _int_or_none(ledger.get("declared_rank")),
        "effective_rank": _int_or_none(ledger.get("effective_rank")),
        "rank_slack": _int_or_none(ledger.get("rank_slack")),
        "perplexity": pack_ppl,
        "ppl_delta_pct": ppl_delta_pct,
        "ppl_improvement_pct": ppl_improvement_pct,
        "token_accuracy": pack_acc,
        "accuracy_gain_vs_base": accuracy_gain,
        "max_logit_diff": _float_or_none(eval_row.get("max_logit_diff")),
        "proof_status": proof_status,
        "improvement_per_mb": improvement_per_mb,
    }


def _winner_quality(rows: list[dict[str, Any]]) -> str | None:
    eligible = [row for row in rows if row.get("perplexity") is not None]
    if not eligible:
        return None
    return str(min(eligible, key=lambda row: float(row["perplexity"]))["candidate"])


def _winner_tradeoff(rows: list[dict[str, Any]]) -> str | None:
    eligible = [row for row in rows if row.get("improvement_per_mb") is not None]
    if not eligible:
        return None
    return str(max(eligible, key=lambda row: float(row["improvement_per_mb"]))["candidate"])


def _promotion_gate_summary(
    spec: BakeoffSpec,
    base: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    paired_eval_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    by_id = {row["candidate"]: row for row in rows}
    tradeoff = _first_flagged(spec, "tradeoff_candidate")
    quality = _first_flagged(spec, "quality_reference")
    small = _first_flagged(spec, "small_reference")
    if tradeoff is None or quality is None:
        return None
    tradeoff_row = by_id.get(tradeoff.candidate_id)
    quality_row = by_id.get(quality.candidate_id)
    small_row = by_id.get(small.candidate_id) if small else None
    if tradeoff_row is None or quality_row is None:
        return None

    base_ppl = _float_or_none(base.get("perplexity"))
    quality_ppl = _float_or_none(quality_row.get("perplexity"))
    tradeoff_ppl = _float_or_none(tradeoff_row.get("perplexity"))
    quality_size = _float_or_none(quality_row.get("size_mb"))
    tradeoff_size = _float_or_none(tradeoff_row.get("size_mb"))
    if None in (base_ppl, quality_ppl, tradeoff_ppl, quality_size, tradeoff_size):
        return None
    assert base_ppl is not None
    assert quality_ppl is not None
    assert tradeoff_ppl is not None
    assert quality_size is not None
    assert tradeoff_size is not None

    quality_gain = base_ppl - quality_ppl
    tradeoff_gain = base_ppl - tradeoff_ppl
    retention = tradeoff_gain / quality_gain if quality_gain > 0 else 0.0
    size_ratio = tradeoff_size / quality_size if quality_size > 0 else 0.0

    min_retention = float(spec.promotion_gates.get("retain_fixed_r32_improvement", 0.9))
    max_size_ratio = float(spec.promotion_gates.get("max_fixed_r32_size_ratio", 0.6))
    beats_small = True
    if small_row is not None:
        small_ppl = _float_or_none(small_row.get("perplexity"))
        small_acc = _float_or_none(small_row.get("token_accuracy"))
        tradeoff_acc = _float_or_none(tradeoff_row.get("token_accuracy"))
        beats_small = (
            (small_ppl is not None and tradeoff_ppl < small_ppl)
            or (small_acc is not None and tradeoff_acc is not None and tradeoff_acc > small_acc)
        )

    proof_passed = tradeoff_row.get("proof_status") == "passed"
    control_rows = [row for row in rows if row.get("control_type")]
    require_beats_controls = bool(
        spec.promotion_gates.get("require_beats_controls", bool(control_rows))
    )
    min_control_advantage = float(
        spec.promotion_gates.get("min_control_ppl_advantage_pct", 0.0)
    )
    require_paired_ci = bool(spec.promotion_gates.get("require_paired_ci", False))
    bootstrap_resamples = int(spec.promotion_gates.get("paired_bootstrap_resamples", 10_000))
    bootstrap_seed = int(spec.promotion_gates.get("paired_bootstrap_seed", 0))
    control_comparisons = []
    for control_row in control_rows:
        control_ppl = _float_or_none(control_row.get("perplexity"))
        advantage_pct = None
        control_passed = False
        if control_ppl is not None and control_ppl > 0:
            advantage_pct = ((control_ppl - tradeoff_ppl) / control_ppl) * 100.0
            control_passed = advantage_pct > min_control_advantage
        paired = None
        paired_error = None
        paired_ci_passed = None
        if paired_eval_rows is not None:
            tradeoff_eval = paired_eval_rows.get(tradeoff.candidate_id)
            control_id = control_row.get("candidate")
            control_eval = (
                paired_eval_rows.get(str(control_id)) if control_id is not None else None
            )
            if tradeoff_eval is not None and control_eval is not None:
                try:
                    paired = compare_answer_mode_metrics(
                        tradeoff_eval,
                        control_eval,
                        resamples=bootstrap_resamples,
                        seed=bootstrap_seed,
                    ).to_dict()
                    ci = paired["ppl_difference_ci"]
                    paired_ci_passed = float(ci["upper"]) < 0.0
                except ValueError as exc:
                    paired_error = str(exc)
        if require_paired_ci:
            control_passed = control_passed and paired_ci_passed is True
        control_comparisons.append(
            {
                "candidate": control_row.get("candidate"),
                "mode": control_row.get("mode"),
                "control_type": control_row.get("control_type"),
                "perplexity": control_ppl,
                "tradeoff_ppl_advantage_pct": advantage_pct,
                "min_tradeoff_ppl_advantage_pct": min_control_advantage,
                "paired": paired,
                "paired_error": paired_error,
                "paired_ci_passed": paired_ci_passed,
                "passed": control_passed,
            }
        )
    random_comparisons = [
        comparison
        for comparison in control_comparisons
        if comparison.get("control_type") == "random_same_budget"
    ]
    structured_comparisons = [
        comparison
        for comparison in control_comparisons
        if comparison.get("control_type") != "random_same_budget"
    ]
    default_random_required = len(random_comparisons)
    min_random_controls_beaten = int(
        spec.promotion_gates.get("min_random_controls_beaten", default_random_required)
    )
    random_controls_beaten = sum(
        1 for comparison in random_comparisons if comparison["passed"]
    )
    random_controls_passed = random_controls_beaten >= min_random_controls_beaten
    require_all_structured = bool(
        spec.promotion_gates.get("require_all_structured_controls", True)
    )
    structured_controls_passed = (
        all(comparison["passed"] for comparison in structured_comparisons)
        if require_all_structured
        else True
    )
    controls_passed = (
        bool(control_comparisons)
        and random_controls_passed
        and structured_controls_passed
    )
    if not require_beats_controls:
        controls_passed = True

    passed = (
        proof_passed
        and beats_small
        and retention >= min_retention
        and size_ratio <= max_size_ratio
        and controls_passed
    )
    return {
        "candidate": tradeoff.candidate_id,
        "quality_reference": quality.candidate_id,
        "small_reference": small.candidate_id if small else None,
        "passed": passed,
        "proof_passed": proof_passed,
        "beats_small_reference": beats_small,
        "retained_quality_gain_ratio": retention,
        "min_retained_quality_gain_ratio": min_retention,
        "size_ratio_vs_quality_reference": size_ratio,
        "max_size_ratio_vs_quality_reference": max_size_ratio,
        "require_beats_controls": require_beats_controls,
        "require_paired_ci": require_paired_ci,
        "paired_bootstrap_resamples": bootstrap_resamples,
        "paired_bootstrap_seed": bootstrap_seed,
        "random_controls_beaten": random_controls_beaten,
        "random_controls_total": len(random_comparisons),
        "min_random_controls_beaten": min_random_controls_beaten,
        "random_controls_passed": random_controls_passed,
        "require_all_structured_controls": require_all_structured,
        "structured_controls_passed": structured_controls_passed,
        "controls_passed": controls_passed,
        "control_comparisons": control_comparisons,
    }


def _first_flagged(spec: BakeoffSpec, field_name: str) -> BakeoffCandidate | None:
    for candidate in spec.candidates:
        if bool(getattr(candidate, field_name)):
            return candidate
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
