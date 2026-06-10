"""One-command pack bakeoff orchestration.

The bakeoff runner is deliberately a thin orchestration layer around existing
CLI commands. It records the exact train/eval/ledger/proof commands, keeps
phase logs, and builds a compact summary from completed artifacts.
"""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

VALID_CANDIDATE_MODES = {
    "fixed_rank",
    "dynamic_rank",
    "rank_map_from_candidate",
    "rank_map_from_pack",
    "rank_map_json",
    "resume_pack",
}


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
    output_path: Path | None = None

    def should_skip(self, *, force: bool) -> bool:
        return not force and self.skip_path is not None and self.skip_path.exists()


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
    root_path = root or Path.cwd()

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
    return BakeoffSpec(
        name=required_text("name"),
        domain=required_text("domain"),
        base=required_text("base"),
        loader=str(payload.get("loader", "auto")),
        train_data=train_data,
        eval_data=eval_data,
        output_dir=_spec_path(required_text("output_dir"), root_path),
        layers=str(payload.get("layers", "attn.q_proj,attn.k_proj,attn.v_proj")),
        profile=str(payload.get("profile", "lite")),
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
        phases.append(
            BakeoffPhase(
                candidate_id=candidate.candidate_id,
                phase="create",
                command=tuple(_create_command(spec, candidate, force=force)),
                log_path=phase_paths["create_log"],
                skip_path=Path("packs") / candidate.pack / "meta.json",
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
    return phases


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
                "would_skip": phase.should_skip(force=force),
            }
            for phase in phases
        ],
    }


def run_bakeoff(spec: BakeoffSpec, *, force: bool = False, cwd: Path | None = None) -> dict[str, Any]:
    """Execute a bakeoff spec and write compact summary artifacts."""

    working_dir = cwd or Path.cwd()
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    for phase in build_bakeoff_plan(spec, force=force):
        phase.log_path.parent.mkdir(parents=True, exist_ok=True)
        if phase.should_skip(force=force):
            print(f"Skipping {phase.candidate_id} {phase.phase}; found {phase.skip_path}")
            continue
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

    summary = build_bakeoff_summary(spec)
    write_bakeoff_summary(spec, summary)
    return summary


def build_bakeoff_summary(spec: BakeoffSpec) -> dict[str, Any]:
    """Build a compact bakeoff summary from completed candidate artifacts."""

    base_metrics: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for candidate in spec.candidates:
        paths = _candidate_paths(spec, candidate)
        eval_rows = _load_eval_rows(paths["eval_json"])
        base_row, pack_row = _find_eval_pair(eval_rows, candidate.pack)
        if base_metrics is None:
            base_metrics = _base_metrics(base_row)
        assert base_metrics is not None
        ledger = _load_optional_summary(paths["ledger_json"])
        proof = _load_json(paths["proof_json"])
        rows.append(_candidate_summary_row(candidate, base_metrics, pack_row, ledger, proof))

    if base_metrics is None:
        raise BakeoffError("Cannot build bakeoff summary without candidate eval artifacts.")

    return {
        "kind": "pack_bakeoff_summary",
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
        "promotion_gates": _promotion_gate_summary(spec, base_metrics, rows),
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

    if "notes" in raw:
        command.extend(["--notes", str(raw["notes"])])
    if "min_rank" in raw:
        command.extend(["--min-rank", str(raw["min_rank"])])
    return command


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


def _load_optional_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
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
    candidate: BakeoffCandidate,
    base: Mapping[str, Any],
    eval_row: Mapping[str, Any],
    ledger: Mapping[str, Any],
    proof: Any,
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

    return {
        "candidate": candidate.candidate_id,
        "pack": candidate.pack,
        "mode": candidate.mode,
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
    passed = proof_passed and beats_small and retention >= min_retention and size_ratio <= max_size_ratio
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
