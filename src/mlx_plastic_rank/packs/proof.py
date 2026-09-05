"""Evidence reports for domain pack DLC-style improvement claims."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inspection import summarize_pack
from .io import load_pack_metadata
from .provenance import content_sha256, dataset_example_keys, pack_identity


@dataclass(frozen=True)
class DomainPackProofConfig:
    base_model: str
    pack: str
    domain: str
    train_data: Path
    eval_data: Path | None
    pack_dir: Path
    eval_report: Path
    generation_report: Path | None = None
    ledger_report: Path | None = None
    min_ppl_improvement_pct: float = 1.0
    min_token_accuracy_gain: float = 0.0
    min_generation_overlap_gain: float = 0.0
    min_logit_diff: float = 0.0
    require_generation: bool = False
    require_ledger: bool = False


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Proof input not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Proof input is invalid JSON: {path}: {exc}") from exc


def _as_float(value: Any, default: float | None = None) -> float:
    return _metric(default if value is None and default is not None else value, "report metric")


def _metric(value: Any, name: str, minimum: float = 0.0, maximum: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric metric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Eval report must be a list of metric rows or contain a 'rows' list: {path}")


def _find_eval_pair(rows: list[dict[str, Any]], pack_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base_row: dict[str, Any] | None = None
    pack_row: dict[str, Any] | None = None
    for row in rows:
        pack = row.get("pack")
        label = str(row.get("label") or "").lower()
        if pack in (None, "", "base") or label == "base":
            if base_row is not None:
                raise ValueError("Eval report contains multiple base rows.")
            base_row = row
        if pack == pack_name:
            if pack_row is not None:
                raise ValueError(f"Eval report contains multiple rows for pack {pack_name!r}.")
            pack_row = row
    if base_row is None:
        raise ValueError("Eval report does not contain a base row.")
    if pack_row is None:
        raise ValueError(f"Eval report does not contain a row for pack {pack_name!r}.")
    return base_row, pack_row


def _generation_metrics(path: Path, pack_name: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), dict):
        raise ValueError(f"Generation report must contain a summary object: {path}")
    summary = payload["summary"]
    if summary.get("pack") != pack_name:
        raise ValueError(
            f"Generation report pack {summary.get('pack')!r} does not match requested pack {pack_name!r}."
        )
    def rate(key: str) -> float:
        return _metric(summary.get(key), key, 0.0, 1.0)

    base_overlap = rate("base_solution_keyword_overlap")
    pack_overlap = rate("pack_solution_keyword_overlap")
    return {
        "examples": int(_as_float(summary.get("examples"))),
        "base_solution_keyword_overlap": base_overlap,
        "pack_solution_keyword_overlap": pack_overlap,
        "solution_keyword_overlap_gain": pack_overlap - base_overlap,
        "base_contains_brand_rate": rate("base_contains_brand_rate"),
        "pack_contains_brand_rate": rate("pack_contains_brand_rate"),
        "base_contains_code_rate": rate("base_contains_code_rate"),
        "pack_contains_code_rate": rate("pack_contains_code_rate"),
    }


def _ledger_metrics(path: Path, pack_name: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Rank ledger report must be a JSON object: {path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("pack_name") != pack_name:
        raise ValueError(f"Rank ledger report metadata does not match pack {pack_name!r}.")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Rank ledger report must contain a summary object: {path}")
    return {
        "adapter_count": int(_as_float(summary.get("adapter_count"))),
        "declared_rank": int(_as_float(summary.get("declared_rank"))),
        "effective_rank": int(_as_float(summary.get("effective_rank"))),
        "rank_slack": int(_as_float(summary.get("rank_slack"))),
        "rank_efficiency": _as_float(summary.get("rank_efficiency")),
        "bytes": int(_as_float(summary.get("bytes"))),
        "bytes_per_effective_rank": _as_float(summary.get("bytes_per_effective_rank")),
    }


def _requirement(
    requirement_id: str,
    passed: bool,
    summary: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "status": "passed" if passed else "failed",
        "summary": summary,
        "evidence": evidence,
    }


def build_domain_pack_proof(config: DomainPackProofConfig) -> dict[str, Any]:
    """Build a pass/fail proof report for a base+pack domain improvement claim."""

    requirements: list[dict[str, Any]] = []
    metadata = load_pack_metadata(config.pack_dir / "meta.json")
    training_provenance = _object(metadata.training_config.get("provenance"))
    recorded_keys = training_provenance.get("training_example_keys")
    lineage_complete = (
        training_provenance.get("lineage_complete") is True
        and isinstance(recorded_keys, list) and bool(recorded_keys)
        and all(_sha256(key) for key in recorded_keys)
    )
    inherited_keys = set(recorded_keys) if lineage_complete and isinstance(recorded_keys, list) else set()
    overlap = None
    training_keys: set[str] = set()
    if config.eval_data is not None and config.train_data.is_file() and config.eval_data.is_file():
        training_keys = dataset_example_keys(config.train_data)
        seen_keys = training_keys | inherited_keys
        overlap = len(seen_keys & dataset_example_keys(config.eval_data))
    requirements.append(_requirement(
        "held_out_dataset", overlap == 0 and lineage_complete,
        "Evaluation data is supplied and has no exact examples from the recorded training lineage.",
        {"overlapping_examples": overlap, "lineage_complete": lineage_complete,
         "scope": "exact content only; not semantic or pretraining overlap"},
    ))
    _, _, _, total_bytes, non_lora = summarize_pack(config.pack_dir)
    pack_file = config.pack_dir / "pack.safetensors"
    train_exists = config.train_data.exists()
    eval_exists = config.eval_data.exists() if config.eval_data else None
    metadata_training_data = metadata.training_data
    metadata_train_matches = (
        metadata_training_data is not None
        and Path(metadata_training_data).expanduser() == config.train_data.expanduser()
    )

    requirements.append(
        _requirement(
            "pack_artifact_exists",
            metadata.pack_name == config.pack and pack_file.exists() and not non_lora and total_bytes > 0,
            "Pack is a loadable SafeTensors LoRA artifact with matching metadata.",
            {
                "pack_dir": str(config.pack_dir),
                "pack_file": str(pack_file),
                "metadata_pack_name": metadata.pack_name,
                "pack_size_bytes": total_bytes,
                "non_lora_tensors": non_lora,
            },
        )
    )
    requirements.append(
        _requirement(
            "training_dataset_evidence",
            train_exists,
            "Training dataset exists; new packs also persist this in metadata.",
            {
                "train_data": str(config.train_data),
                "train_data_exists": train_exists,
                "metadata_training_data": metadata_training_data,
                "metadata_training_data_matches": metadata_train_matches,
                "metadata_training_config": metadata.training_config,
            },
        )
    )

    rows = _load_eval_rows(config.eval_report)
    base_row, pack_row = _find_eval_pair(rows, config.pack)
    base_provenance = _object(base_row.get("provenance"))
    pack_provenance = _object(pack_row.get("provenance"))
    current_pack = pack_identity(config.pack_dir)
    train_sha = content_sha256(config.train_data) if config.train_data.is_file() else None
    eval_sha = content_sha256(config.eval_data) if config.eval_data and config.eval_data.is_file() else None
    model_sha = base_provenance.get("model_sha256")
    tokenizer_sha = base_provenance.get("tokenizer_sha256")
    settings = _object(base_provenance.get("settings"))
    same_evaluation = {
        key: value for key, value in base_provenance.items() if key != "pack"
    } == {key: value for key, value in pack_provenance.items() if key != "pack"}
    provenance_ok = (
        base_provenance.get("version") == 1
        and same_evaluation
        and _sha256(model_sha) and _sha256(tokenizer_sha)
        and _sha256(base_provenance.get("tokenized_sha256"))
        and {"loader", "loss_mode", "chat_template", "sequence_length", "num_samples"} <= settings.keys()
        and settings.get("loss_mode") in {"answer", "full"}
        and base_provenance.get("pack") is None
        and pack_provenance.get("pack") == current_pack
        and eval_sha is not None and base_provenance.get("dataset_sha256") == eval_sha
        and train_sha is not None and training_provenance.get("dataset_sha256") == train_sha
        and lineage_complete and training_keys <= inherited_keys
        and training_provenance.get("model_sha256") == model_sha
        and training_provenance.get("tokenizer_sha256") == tokenizer_sha
    )
    requirements.append(_requirement(
        "artifact_provenance", provenance_ok,
        "Training and paired evaluation are bound to the current data, pack, checkpoint, tokenizer, and preprocessing.",
        {"pack": current_pack, "train_sha256": train_sha, "eval_sha256": eval_sha,
         "model_sha256": model_sha, "same_evaluation": same_evaluation},
    ))
    base_model_matches = base_row.get("model") == config.base_model and pack_row.get("model") == config.base_model
    if config.eval_data is not None:
        eval_data_known = bool(eval_exists)
    else:
        eval_data_known = True
    requirements.append(
        _requirement(
            "base_and_pack_eval_present",
            base_model_matches and eval_data_known,
            "Eval report contains matching base and base+pack rows for the requested base model.",
            {
                "eval_report": str(config.eval_report),
                "eval_data": str(config.eval_data) if config.eval_data else None,
                "eval_data_exists": eval_exists,
                "base_model": config.base_model,
                "base_row_model": base_row.get("model"),
                "pack_row_model": pack_row.get("model"),
            },
        )
    )

    base_ppl = _metric(base_row.get("perplexity", base_row.get("ppl")), "base perplexity", 1.0)
    pack_ppl = _metric(pack_row.get("perplexity", pack_row.get("ppl")), "pack perplexity", 1.0)
    ppl_improvement_pct = ((base_ppl - pack_ppl) / base_ppl) * 100.0 if base_ppl > 0 else 0.0
    base_acc = _metric(base_row.get("token_accuracy", base_row.get("domain_metric")), "base token accuracy", 0.0, 1.0)
    pack_acc = _metric(pack_row.get("token_accuracy", pack_row.get("domain_metric")), "pack token accuracy", 0.0, 1.0)
    token_accuracy_gain = pack_acc - base_acc
    max_logit_diff = _metric(pack_row.get("max_logit_diff"), "max logit difference")
    attach_passed = (
        pack_row.get("pack") == config.pack
        and _as_float(pack_row.get("pack_size_bytes", total_bytes)) > 0
        and max_logit_diff > config.min_logit_diff
    )
    requirements.append(
        _requirement(
            "pack_attachment_changes_model",
            attach_passed,
            "Base+pack eval row shows the pack was attached and changed logits.",
            {
                "pack": pack_row.get("pack"),
                "pack_size_bytes": pack_row.get("pack_size_bytes"),
                "max_logit_diff": max_logit_diff,
                "min_logit_diff": config.min_logit_diff,
            },
        )
    )
    requirements.append(
        _requirement(
            "domain_eval_improves",
            ppl_improvement_pct >= config.min_ppl_improvement_pct
            and token_accuracy_gain >= config.min_token_accuracy_gain,
            "Base+pack improves held-out domain answer metrics over the base model.",
            {
                "base_perplexity": base_ppl,
                "pack_perplexity": pack_ppl,
                "ppl_improvement_pct": ppl_improvement_pct,
                "min_ppl_improvement_pct": config.min_ppl_improvement_pct,
                "base_token_accuracy": base_acc,
                "pack_token_accuracy": pack_acc,
                "token_accuracy_gain": token_accuracy_gain,
                "min_token_accuracy_gain": config.min_token_accuracy_gain,
            },
        )
    )

    generation: dict[str, Any] | None = None
    if config.generation_report is not None:
        generation = _generation_metrics(config.generation_report, config.pack)
        generation_provenance = _object(_load_json(config.generation_report).get("provenance"))
        requirements.append(
            _requirement(
                "generation_response_improves",
                generation["examples"] > 0
                and generation["solution_keyword_overlap_gain"] >= config.min_generation_overlap_gain
                and generation_provenance.get("pack") == current_pack
                and generation_provenance.get("model_sha256") == model_sha
                and generation_provenance.get("dataset_sha256") == eval_sha,
                "Base+pack improves generated answer keyword overlap on sampled domain prompts.",
                {
                    "generation_report": str(config.generation_report),
                    **generation,
                    "min_generation_overlap_gain": config.min_generation_overlap_gain,
                },
            )
        )
    elif config.require_generation:
        requirements.append(
            _requirement(
                "generation_response_improves",
                False,
                "Generation proof is required but no generation report was provided.",
                {"generation_report": None},
            )
        )

    ledger: dict[str, Any] | None = None
    if config.ledger_report is not None:
        ledger = _ledger_metrics(config.ledger_report, config.pack)
        ledger_provenance = _object(_load_json(config.ledger_report).get("provenance"))
        requirements.append(
            _requirement(
                "rank_ledger_valid",
                ledger["adapter_count"] > 0 and ledger["effective_rank"] > 0
                and ledger_provenance.get("pack") == current_pack,
                "Rank ledger verifies the attached pack has active low-rank adapter capacity.",
                {
                    "ledger_report": str(config.ledger_report),
                    **ledger,
                },
            )
        )
    elif config.require_ledger:
        requirements.append(
            _requirement(
                "rank_ledger_valid",
                False,
                "Rank-ledger proof is required but no ledger report was provided.",
                {"ledger_report": None},
            )
        )

    status = "passed" if all(row["status"] == "passed" for row in requirements) else "failed"
    return {
        "kind": "domain_pack_proof",
        "status": status,
        "claim": (
            "A base model can load a LoRA skill pack as a DLC-style artifact, and "
            "base+pack improves held-out response quality on the stated domain."
        ),
        "domain": config.domain,
        "base_model": config.base_model,
        "pack": config.pack,
        "pack_dir": str(config.pack_dir),
        "train_data": str(config.train_data),
        "eval_data": str(config.eval_data) if config.eval_data else None,
        "metrics": {
            "base_perplexity": base_ppl,
            "pack_perplexity": pack_ppl,
            "ppl_improvement_pct": ppl_improvement_pct,
            "base_token_accuracy": base_acc,
            "pack_token_accuracy": pack_acc,
            "token_accuracy_gain": token_accuracy_gain,
            "max_logit_diff": max_logit_diff,
            "pack_size_bytes": total_bytes,
            "generation": generation,
            "rank_ledger": ledger,
        },
        "metadata": metadata.to_dict(),
        "requirements": requirements,
    }
