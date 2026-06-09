"""Command line interface for LoRA skill packs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.quantized import QuantizedLinear

from .capabilities import capability_report, missing_capabilities
from .dataset import build_supervised_token_dataset, build_token_dataset, load_jsonl_texts
from .eval_utils import load_domain_prompts, parse_batch_sizes, parse_thinking_option
from .inspection import TensorInfo, allowed_ranks_for, size_limit_for, summarize_pack
from .io import compute_sha256, save_pack, save_pack_metadata
from .manager import LoRAManager, PackApplicationError
from .rank_ledger import (
    compare_pack_rank_ledgers,
    comparison_rows_for_csv,
    ledger_rows_for_csv,
    pack_rank_ledger,
)
from .router import DomainPackRouter, load_domain_map
from .train import TrainingConfig, model_logits, train_lora, train_lora_supervised

LORA_ALIAS_MAP = {
    "q": "attn.q_proj",
    "k": "attn.k_proj",
    "v": "attn.v_proj",
    "o": "attn.o_proj",
}

_eval_ppl: Any
_load_data: Any
_load_model: Any
_load_vlm_model: Any

try:
    from mlx_lm.perplexity import eval_ppl as _eval_ppl_import
    from mlx_lm.perplexity import load_data as _load_data_import
    from mlx_lm.utils import load as _load_model_import

    _eval_ppl = _eval_ppl_import
    _load_data = _load_data_import
    _load_model = _load_model_import
except ModuleNotFoundError:
    _eval_ppl = None
    _load_data = None
    _load_model = None

try:
    from mlx_vlm.utils import load as _load_vlm_model_import

    _load_vlm_model = _load_vlm_model_import
except ModuleNotFoundError:
    _load_vlm_model = None


def _resolve_target(name: str) -> str:
    canonical = name.strip()
    return LORA_ALIAS_MAP.get(canonical, canonical)


def _require_mlx_lm():
    if _load_model is None or _eval_ppl is None or _load_data is None:
        raise SystemExit(
            "The packs CLI requires `mlx-lm`. Install it with: uv pip install -e '.[packs]'"
        )
    return _load_model, _eval_ppl, _load_data


def _require_load_model():
    if _load_model is None:
        raise SystemExit(
            "The packs CLI requires `mlx-lm`. Install it with: uv pip install -e '.[packs]'"
        )
    return _load_model


def _require_mlx_vlm():
    if _load_vlm_model is None:
        raise SystemExit(
            "Gemma 4 any-to-any bases require `mlx-vlm`. Install it with: uv pip install -e '.[packs]'"
        )
    return _load_vlm_model


def _looks_like_vlm_ref(base_ref: str) -> bool:
    lowered = base_ref.lower()
    if "optiq" in lowered:
        return False
    return "gemma-4" in lowered or "gemma4" in lowered


def _tokenizer_from_processor(processor: Any):
    tokenizer = getattr(processor, "tokenizer", processor)
    if not hasattr(tokenizer, "encode"):
        raise SystemExit("Loaded processor does not expose tokenizer.encode for text pack training/eval.")
    return tokenizer


def _load_base_model(base_ref: str, loader: str = "auto"):
    if loader == "mlx-vlm" or (loader == "auto" and _looks_like_vlm_ref(base_ref)):
        load_vlm = _require_mlx_vlm()
        loaded = load_vlm(base_ref)
        model, processor = loaded[0], loaded[1]
        return model, _tokenizer_from_processor(processor)
    load_model = _require_load_model()
    loaded = load_model(base_ref)
    return loaded[0], loaded[1]


def _resolve_base_checkpoint(base_ref: str) -> Path | None:
    candidate = Path(base_ref).expanduser()
    if candidate.exists():
        return candidate
    return None


def _parse_lora_dropout(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid dropout value '{raw}'") from exc
    if value < 0.0 or value >= 1.0:
        raise argparse.ArgumentTypeError("LoRA dropout must be in the range [0.0, 1.0).")
    return value


def _parse_min_rank(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid min rank '{raw}'") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("Minimum rank must be >= 0")
    return value


def _parse_rank(raw: str) -> int:
    value = _parse_min_rank(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Rank must be > 0")
    return value


def _tokenise_prompt(tokenizer, prompt: str, max_tokens: int) -> mx.array:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [0]
    clipped = token_ids[: max(1, max_tokens)]
    return mx.array([clipped], dtype=mx.int32)


def _evaluate_perplexity(model, dataset: mx.array, batch_size: int) -> Dict[str, float]:
    if dataset.shape[0] == 0:
        raise ValueError("Dataset is empty")
    mx.reset_peak_memory()
    all_losses: List[mx.array] = []
    first_token_ms: float | None = None
    start_time = time.time()
    for start in range(0, dataset.shape[0], batch_size):
        batch = dataset[start : start + batch_size]
        step_start = time.time()
        logits = model_logits(model, batch[:, :-1]).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(losses)
        if first_token_ms is None:
            first_token_ms = (time.time() - step_start) * 1000.0
        all_losses.append(losses.flatten())
    total_time = time.time() - start_time
    tokens = dataset.shape[0] * (dataset.shape[1] - 1)
    tps = tokens / max(total_time, 1e-6)
    losses_concat = mx.concatenate(all_losses)
    mean_loss = losses_concat.mean().item()
    ppl = math.exp(mean_loss)
    if losses_concat.size > 1:
        std_dev = float(mx.sqrt(mx.var(losses_concat, ddof=1)).item())
        se_loss = std_dev / math.sqrt(losses_concat.size)
        se_ppl = ppl * se_loss
    else:
        se_ppl = 0.0
    vram_peak = mx.get_peak_memory() / (1024**3)
    return {
        "ppl": float(ppl),
        "ppl_se": float(se_ppl),
        "tps": float(tps),
        "first_token_ms": float(first_token_ms or 0.0),
        "vram_peak": float(vram_peak),
        "eval_time_s": float(total_time),
        "tokens": int(tokens),
    }


def _evaluate_supervised_perplexity(
    model,
    tokens: mx.array,
    masks: mx.array,
    batch_size: int,
) -> Dict[str, float]:
    if tokens.shape[0] == 0:
        raise ValueError("Dataset is empty")
    mx.reset_peak_memory()
    all_losses: List[mx.array] = []
    first_token_ms: float | None = None
    start_time = time.time()
    total_tokens = 0.0
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size]
        mask = masks[start : start + batch_size]
        step_start = time.time()
        logits = model_logits(model, batch[:, :-1]).astype(mx.float32)
        target_mask = mask[:, 1:]
        losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        masked_losses = losses * target_mask
        mx.eval(masked_losses)
        if first_token_ms is None:
            first_token_ms = (time.time() - step_start) * 1000.0
        all_losses.append(masked_losses.flatten())
        total_tokens += float(mx.sum(target_mask).item())
    total_time = time.time() - start_time
    losses_concat = mx.concatenate(all_losses)
    denom = max(total_tokens, 1.0)
    mean_loss = float(mx.sum(losses_concat).item()) / denom
    ppl = math.exp(mean_loss)
    vram_peak = mx.get_peak_memory() / (1024**3)
    tps = total_tokens / max(total_time, 1e-6)
    return {
        "ppl": float(ppl),
        "ppl_se": 0.0,
        "tps": float(tps),
        "first_token_ms": float(first_token_ms or 0.0),
        "vram_peak": float(vram_peak),
        "eval_time_s": float(total_time),
        "tokens": int(total_tokens),
    }

PACK_ROOT = Path("packs")


def _default_pack_dir(name: str) -> Path:
    return PACK_ROOT / name


def cmd_create(args: argparse.Namespace) -> None:
    base_ref = str(args.base)
    base_checkpoint = _resolve_base_checkpoint(base_ref)
    pack_dir = _default_pack_dir(args.name)
    if pack_dir.exists() and not args.force:
        raise SystemExit(f"Pack '{args.name}' already exists. Use --force to overwrite.")

    print(f"Loading base model from {base_ref}...")
    model, tokenizer = _load_base_model(base_ref, args.loader)
    if getattr(args, "train_fp16_fallback", False):
        converted = _dequantize_linear_inplace(model)
        if converted:
            print(f"Dequantized {converted} quantized linear layers for training fallback")

    manager = LoRAManager(model, base_checkpoint=base_checkpoint, base_model=base_ref)
    if base_checkpoint is None:
        print("Warning: --base is not a local path; base hash verification will be skipped.")

    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    canonical_layers = [_resolve_target(layer) for layer in layers]
    allowed_targets = {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"}
    disallowed = [layer for layer in canonical_layers if layer not in allowed_targets]
    if disallowed:
        raise SystemExit(f"Unsupported LoRA targets: {disallowed}. Only q,k,v,o projections are allowed.")

    try:
        allowed_ranks = allowed_ranks_for(args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc))

    rank_map: Dict[str, int] = {}
    alpha_map: Dict[str, float] = {}

    def _snap_to_allowed(candidate: int) -> int:
        for allowed in sorted(allowed_ranks):
            if allowed >= int(candidate):
                return allowed
        return sorted(allowed_ranks)[-1]

    if args.rank is not None:
        if args.rank not in allowed_ranks:
            raise SystemExit(
                f"Rank {args.rank} is not allowed for profile={args.profile}; "
                f"allowed ranks are {list(allowed_ranks)}"
            )
        print(f"Using explicit ranks (profile={args.profile}):")
        for target in canonical_layers:
            rank_map[target] = args.rank
            alpha_map[target] = 2.0 * args.rank
            print(f"  {target}: rank={rank_map[target]}")
    else:
        auto_rank_map, _, residuals = manager.compute_auto_ranks(
            canonical_layers,
            strategy=args.rank_strategy,
            target_compression=args.target_compression,
            eps=args.rank_eps,
            allowed_ranks=allowed_ranks,
        )
        print(f"Auto-selected ranks ({args.rank_strategy}, profile={args.profile}):")
        for target in canonical_layers:
            selected_rank = auto_rank_map[target]
            if args.min_rank > 0:
                selected_rank = _snap_to_allowed(max(selected_rank, args.min_rank))
            rank_map[target] = selected_rank
            alpha_map[target] = 2.0 * selected_rank
            res = residuals[target]
            print(f"  {target}: rank={rank_map[target]} residual={res:.4g}")

    base_rank = rank_map.get("attn.q_proj", next(iter(rank_map.values())))
    base_alpha = alpha_map.get("attn.q_proj", 2.0 * base_rank)

    print(f"Initialising adapters on layers: {canonical_layers}")
    if args.dynamic_rank:
        print(
            "Dynamic rank enabled: "
            f"max ranks={rank_map} initial_active_rank={args.dynamic_initial_rank}"
        )
    adapters = manager.initialize_adapters(
        canonical_layers,
        rank=base_rank,
        alpha=base_alpha,
        seed=args.seed,
        rank_map=rank_map,
        alpha_map=alpha_map,
        dropout=args.lora_dropout,
        allowed_ranks=allowed_ranks,
        initial_active_rank=args.dynamic_initial_rank if args.dynamic_rank else None,
    )
    if args.zero_init:
        for adapter in adapters.values():
            adapter.A = mx.zeros_like(adapter.A)
            adapter.B = mx.zeros_like(adapter.B)

    config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        log_interval=max(1, args.steps // 10),
        lora_dropout=args.lora_dropout,
        dynamic_rank=args.dynamic_rank,
        dynamic_rank_interval=args.dynamic_rank_interval,
        dynamic_rank_warmup=args.dynamic_rank_warmup,
        dynamic_rank_min=args.dynamic_min_rank,
        dynamic_rank_grow_threshold=args.dynamic_grow_threshold,
        dynamic_rank_prune_threshold=args.dynamic_prune_threshold,
        dynamic_rank_allowed_ranks=allowed_ranks,
    )
    print(
        f"Training LoRA adapters: steps={config.steps} batch_size={config.batch_size} "
        f"lr={config.learning_rate} loss_mode={args.loss_mode}"
    )
    if args.loss_mode == "answer":
        tokens, masks = build_supervised_token_dataset(
            Path(args.data),
            tokenizer,
            args.sequence_length,
            chat_template=args.chat_template,
        )
        final_loss = train_lora_supervised(manager, model, tokens, masks, config)
    else:
        texts = load_jsonl_texts(Path(args.data), tokenizer, args.chat_template)
        dataset = build_token_dataset(texts, tokenizer, args.sequence_length)
        final_loss = train_lora(manager, model, dataset, config)
    print(f"Training complete. Final loss={final_loss:.4f}")

    tensors, metadata = manager.export_active_pack(
        args.name,
        PACK_ROOT,
        notes=args.notes,
        profile=args.profile,
    )
    tensor_path = pack_dir / "pack.safetensors"
    meta_path = pack_dir / "meta.json"
    save_pack(tensors, tensor_path)
    save_pack_metadata(metadata, meta_path)
    print(f"Saved pack tensors to {tensor_path}")
    print(f"Saved metadata to {meta_path}")

    metadata, infos, _, total_bytes, non_lora = summarize_pack(pack_dir)
    if non_lora:
        raise SystemExit(f"Pack contains non-LoRA tensors: {non_lora}")
    limit = size_limit_for(metadata)
    if total_bytes > limit:
        raise SystemExit(
            f"Pack size {total_bytes / (1024**2):.2f} MB exceeds limit {(limit / (1024**2)):.1f} MB"
        )
    print(f"Pack size: {total_bytes / (1024**2):.2f} MB (limit {(limit / (1024**2)):.1f} MB)")


def cmd_apply(args: argparse.Namespace) -> None:
    base_ref = str(args.base)
    base_checkpoint = _resolve_base_checkpoint(base_ref)
    pack_dir = _default_pack_dir(args.name)
    if not pack_dir.exists():
        raise SystemExit(f"Pack '{args.name}' not found at {pack_dir}")

    print(f"Inspecting pack '{args.name}'...")
    metadata, infos, _, total_bytes, non_lora = summarize_pack(pack_dir)
    if base_checkpoint is not None:
        base_hash = compute_sha256(base_checkpoint)
        if metadata.base_hash and metadata.base_hash != base_hash:
            raise SystemExit(
                f"Base hash mismatch: pack built for {metadata.base_hash[:8]}, base is {base_hash[:8]}"
            )
    elif metadata.base_hash:
        print("Warning: --base is not a local path; base hash verification skipped.")
    if non_lora:
        raise SystemExit(f"Pack contains non-LoRA tensors: {non_lora}")
    limit = size_limit_for(metadata)
    if total_bytes > limit:
        raise SystemExit(
            f"Pack size {total_bytes / (1024**2):.2f} MB exceeds limit {(limit / (1024**2)):.1f} MB"
        )

    if args.dry_run:
        print("Dry run summary:")
        for info in infos:
            print(
                f"  {info.key:<32} shape={info.shape} dtype={info.dtype} params={info.params} bytes={info.bytes}"
            )
        print(f"Total bytes: {total_bytes} ({total_bytes / (1024**2):.2f} MB)")
        print(f"Size limit: {(limit / (1024**2)):.1f} MB")
        print(f"Target layers: {metadata.target_layers}")
        print(f"Ranks: {metadata.rank_map}")
        print(f"Estimated VRAM footprint: {total_bytes / (1024**2):.2f} MB")
        return

    model, _ = _load_base_model(base_ref, args.loader)
    manager = LoRAManager(model, base_checkpoint=base_checkpoint, base_model=base_ref)
    try:
        metadata = manager.apply_pack(pack_dir)
    except PackApplicationError as exc:
        raise SystemExit(str(exc))
    print(f"Pack '{metadata.pack_name}' applied successfully.")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        session = {
            "pack": metadata.pack_name,
            "pack_dir": str(pack_dir),
            "base": base_ref,
            "base_hash": metadata.base_hash,
            "timestamp": time.time(),
        }
        out_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        print(f"Session state written to {out_path}")


def cmd_remove(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if not out.exists():
        print(f"Session file {out} does not exist.")
        return
    out.unlink()
    print(f"Removed session file {out}.")


def cmd_inspect(args: argparse.Namespace) -> None:
    pack_dir = _default_pack_dir(args.name)
    try:
        metadata, infos, total_params, total_bytes, non_lora = summarize_pack(pack_dir)
    except FileNotFoundError:
        raise SystemExit(f"Pack '{args.name}' not found at {pack_dir}")
    if non_lora:
        raise SystemExit(f"Pack contains non-LoRA tensors: {non_lora}")
    limit = size_limit_for(metadata)
    if total_bytes > limit:
        raise SystemExit(
            f"Pack size {total_bytes / (1024**2):.2f} MB exceeds limit {(limit / (1024**2)):.1f} MB"
        )

    print(f"Pack: {metadata.pack_name}")
    print(f"Profile: {metadata.profile}")
    print(f"Base hash: {metadata.base_hash}")
    print(f"Target layers: {metadata.target_layers}")
    print(f"Ranks: {metadata.rank_map}")
    print("Tensors:")
    for info in infos:
        print(
            f"  {info.key:<32} shape={info.shape} dtype={info.dtype} params={info.params} bytes={info.bytes}"
        )
    print(f"Total params: {total_params}")
    print(f"Total size: {total_bytes} bytes ({total_bytes / (1024**2):.2f} MB)")
    print(f"Size limit: {(limit / (1024**2)):.1f} MB")

    expected_params = 0
    expected_bytes = 0
    grouped: Dict[str, Dict[str, TensorInfo]] = {}
    for info in infos:
        if ".lora." not in info.key:
            continue
        prefix, suffix = info.key.split(".lora.")
        grouped.setdefault(prefix, {})[suffix] = info
    for prefix, tensors_by_name in grouped.items():
        a = tensors_by_name.get("A")
        b = tensors_by_name.get("B")
        if not a or not b:
            continue
        out_dim, rank = a.shape
        rank2, in_dim = b.shape
        rank = min(rank, rank2)
        params = out_dim * rank + rank * in_dim
        expected_params += params
        expected_bytes += params * 2 + 4  # fp16 params + fp32 alpha
    print(f"Expected params (LoRA fp16): {expected_params}")
    print(f"Expected size estimate: {expected_bytes} bytes ({expected_bytes / (1024**2):.2f} MB)")


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_rank_ledger(args: argparse.Namespace) -> None:
    pack_dir = _default_pack_dir(args.name)
    if not pack_dir.exists():
        raise SystemExit(f"Pack '{args.name}' not found at {pack_dir}")

    if args.compare:
        compare_dir = _default_pack_dir(args.compare)
        if not compare_dir.exists():
            raise SystemExit(f"Pack '{args.compare}' not found at {compare_dir}")
        report = compare_pack_rank_ledgers(pack_dir, compare_dir, rank_tol=args.rank_tol)
        csv_rows = comparison_rows_for_csv(report)
    else:
        report = pack_rank_ledger(pack_dir, rank_tol=args.rank_tol)
        csv_rows = ledger_rows_for_csv(report)

    print(json.dumps(report, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Rank ledger written to {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        _write_csv_rows(csv_path, csv_rows)
        print(f"Rank ledger CSV written to {csv_path}")


def cmd_list(args: argparse.Namespace) -> None:
    if not PACK_ROOT.exists():
        print("No packs found.")
        return
    rows: List[str] = []
    for pack_dir in sorted(PACK_ROOT.iterdir()):
        if not pack_dir.is_dir():
            continue
        try:
            metadata, _, _, total_bytes, _ = summarize_pack(pack_dir)
        except FileNotFoundError:
            continue
        size_mb = total_bytes / (1024 * 1024)
        ranks = sorted(set(metadata.rank_map.values()))
        created = metadata.created_at or ""
        limit_bytes = size_limit_for(metadata)
        limit_mb = limit_bytes / (1024 * 1024)
        over = "!" if total_bytes > limit_bytes else ""
        rows.append(
            f"{metadata.pack_name:<16} profile={metadata.profile:<5} size={size_mb:5.2f}MB/{limit_mb:4.1f}MB{over} ranks={ranks} layers={len(metadata.target_layers)} created={created[:19]} base={metadata.base_hash[:8]}"
        )
    if not rows:
        print("No packs found.")
    else:
        print("Installed packs:")
        for row in rows:
            print("  " + row)


def cmd_capabilities(args: argparse.Namespace) -> None:
    rows = capability_report()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("MLX modality capabilities:")
        for row in rows:
            status = "installed" if row["installed"] else "missing"
            version = f" {row['version']}" if row["version"] else ""
            print(f"  {row['name']:<10} {status}{version}")
            print(f"    {row['summary']}")
            print(f"    Features: {', '.join(row['features'])}")
            print(f"    Commands: {', '.join(row['commands'])}")

    missing = missing_capabilities(rows)
    if args.check and missing:
        raise SystemExit(f"Missing MLX modality capabilities: {', '.join(missing)}")


def cmd_eval(args: argparse.Namespace) -> None:
    data_path = args.data_path

    dataset_cached: mx.array | tuple[mx.array, mx.array] | None = None
    dataset_type: str | None = None
    base_logits_sample: mx.array | None = None

    def evaluate(model_ref: str, pack_dir: Path | None = None) -> dict:
        nonlocal dataset_cached, dataset_type, base_logits_sample
        start_load = time.time()
        base_checkpoint = _resolve_base_checkpoint(model_ref)
        model, tokenizer = _load_base_model(model_ref, args.loader)
        manager = LoRAManager(model, base_checkpoint=base_checkpoint, base_model=model_ref)
        pack_name = None
        pack_size = 0
        if pack_dir is not None:
            metadata = manager.apply_pack(pack_dir)
            pack_name = metadata.pack_name
            pack_file = pack_dir / "pack.safetensors"
            if pack_file.exists():
                pack_size = os.path.getsize(pack_file)
        load_time_ms = (time.time() - start_load) * 1000.0

        if dataset_cached is None:
            path_obj = Path(data_path)
            if path_obj.exists() and path_obj.is_file():
                if args.loss_mode == "answer":
                    tokens, masks = build_supervised_token_dataset(
                        path_obj,
                        tokenizer,
                        args.sequence_length,
                        chat_template=args.chat_template,
                    )
                    if args.num_samples > 0:
                        tokens = tokens[: args.num_samples]
                        masks = masks[: args.num_samples]
                    dataset_cached = (tokens, masks)
                    dataset_type = "jsonl-answer"
                else:
                    texts = load_jsonl_texts(path_obj, tokenizer, args.chat_template)
                    token_dataset = build_token_dataset(texts, tokenizer, args.sequence_length)
                    if args.num_samples > 0:
                        token_dataset = token_dataset[: args.num_samples]
                    dataset_cached = token_dataset
                    dataset_type = "jsonl"
            else:
                if args.loss_mode == "answer":
                    raise SystemExit("--loss-mode answer requires a local JSONL data path")
                _, _, load_data = _require_mlx_lm()
                dataset_cached = load_data(
                    tokenizer,
                    data_path,
                    num_samples=args.num_samples,
                    sequence_length=args.sequence_length,
                )
                dataset_type = "hf"

        cached_dataset = dataset_cached
        if dataset_type == "jsonl-answer":
            assert isinstance(cached_dataset, tuple)
            tokens, masks = cached_dataset
            metrics = _evaluate_supervised_perplexity(model, tokens, masks, args.batch_size)
            diff_batch = tokens[:1]
        else:
            assert cached_dataset is not None and not isinstance(cached_dataset, tuple)
            metrics = _evaluate_perplexity(model, cached_dataset, args.batch_size)
            diff_batch = cached_dataset[:1]
        eval_time = metrics["eval_time_s"]
        ppl = metrics["ppl"]
        se = metrics["ppl_se"]
        peak_mem = metrics["vram_peak"]
        tps = metrics["tps"]

        logits = model_logits(model, diff_batch[:, :-1]).astype(mx.float32)
        mx.eval(logits)
        max_diff = 0.0
        if base_logits_sample is None:
            base_logits_sample = logits
        else:
            diff = mx.abs(logits - base_logits_sample)
            max_diff = float(mx.max(diff).item())

        token_accuracy = None
        if dataset_type == "jsonl":
            assert cached_dataset is not None and not isinstance(cached_dataset, tuple)
            correct = 0.0
            total = 0
            for start_idx in range(0, cached_dataset.shape[0], args.batch_size):
                batch_tokens = cached_dataset[start_idx : start_idx + args.batch_size]
                if batch_tokens.shape[0] == 0:
                    continue
                inputs = batch_tokens[:, :-1]
                targets = batch_tokens[:, 1:]
                preds = mx.argmax(model_logits(model, inputs), axis=-1)
                matches = mx.equal(preds, targets).astype(mx.float32)
                correct += float(mx.sum(matches).item())
                total += int(targets.size)
            if total > 0:
                token_accuracy = correct / float(total)
        elif dataset_type == "jsonl-answer":
            assert isinstance(cached_dataset, tuple)
            tokens, masks = cached_dataset
            correct = 0.0
            total_answer = 0.0
            for start_idx in range(0, tokens.shape[0], args.batch_size):
                batch_tokens = tokens[start_idx : start_idx + args.batch_size]
                batch_masks = masks[start_idx : start_idx + args.batch_size]
                if batch_tokens.shape[0] == 0:
                    continue
                inputs = batch_tokens[:, :-1]
                targets = batch_tokens[:, 1:]
                target_mask = batch_masks[:, 1:]
                preds = mx.argmax(model_logits(model, inputs), axis=-1)
                matches = mx.equal(preds, targets).astype(mx.float32) * target_mask
                correct += float(mx.sum(matches).item())
                total_answer += float(mx.sum(target_mask).item())
            if total_answer > 0:
                token_accuracy = correct / total_answer

        return {
            "model": model_ref,
            "pack": pack_name,
            "pack_size_bytes": pack_size,
            "size_mb": pack_size / (1024 * 1024),
            "load_time_ms": load_time_ms,
            "perplexity": ppl,
            "perplexity_se": se,
            "eval_time_s": eval_time,
            "tokens_per_sec": tps,
            "peak_memory_gb": peak_mem,
            "max_logit_diff": max_diff,
            "token_accuracy": token_accuracy,
            "loss_mode": args.loss_mode,
        }

    base_metrics = evaluate(str(args.base), None)
    results = [base_metrics]
    if args.pack:
        results.append(evaluate(str(args.base), _default_pack_dir(args.pack)))

    base_ppl = results[0]["perplexity"]
    for row in results:
        row["ppl_delta_pct"] = (
            0.0
            if row is results[0]
            else ((row["perplexity"] - base_ppl) / base_ppl) * 100.0
        )
        row.setdefault("token_accuracy", None)
        row.setdefault("domain_metric", row["token_accuracy"])

    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Evaluation metrics saved to {args.out}")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Evaluation metrics CSV written to {csv_path}")


def cmd_eval_batch(args: argparse.Namespace) -> None:
    base_ref = str(args.base)
    base_checkpoint = _resolve_base_checkpoint(base_ref)
    batch_sizes = parse_batch_sizes(args.batch_size)
    thinking_mode, cap_tokens = parse_thinking_option(args.thinking)
    prompts_by_domain = load_domain_prompts(Path(args.input), thinking_mode, cap_tokens)

    model, tokenizer = _load_base_model(base_ref, args.loader)
    manager = LoRAManager(model, base_checkpoint=base_checkpoint, base_model=base_ref)

    datasets: Dict[str, mx.array] = {}
    for domain, texts in prompts_by_domain.items():
        dataset = build_token_dataset(texts, tokenizer, args.sequence_length)
        datasets[domain] = dataset

    results: List[dict] = []
    base_metrics: Dict[tuple[str, int], Dict[str, float]] = {}
    for domain, dataset in datasets.items():
        for bs in batch_sizes:
            metrics = _evaluate_perplexity(model, dataset, bs)
            base_metrics[(domain, bs)] = metrics

    pack_metrics: Dict[tuple[str, int], Dict[str, float]] = {}
    pack_size_mb = 0.0
    if args.pack:
        pack_dir = _default_pack_dir(args.pack)
        metadata, _, _, total_bytes, non_lora = summarize_pack(pack_dir)
        if non_lora:
            raise SystemExit(f"Pack contains non-LoRA tensors: {non_lora}")
        limit = size_limit_for(metadata)
        if total_bytes > limit:
            raise SystemExit(
                f"Pack size {total_bytes / (1024**2):.2f} MB exceeds limit {(limit / (1024**2)):.1f} MB"
            )
        if base_checkpoint is not None:
            base_hash = compute_sha256(base_checkpoint)
            if metadata.base_hash and metadata.base_hash != base_hash:
                raise SystemExit(
                    f"Base hash mismatch: pack built for {metadata.base_hash[:8]}, base is {base_hash[:8]}"
                )
        elif metadata.base_hash:
            print("Warning: --base is not a local path; base hash verification skipped.")
        pack_size_mb = total_bytes / (1024**2)
        try:
            manager.apply_pack(pack_dir)
        except PackApplicationError as exc:
            raise SystemExit(str(exc))
        for domain, dataset in datasets.items():
            for bs in batch_sizes:
                metrics = _evaluate_perplexity(model, dataset, bs)
                pack_metrics[(domain, bs)] = metrics
        manager.detach_pack()

    for domain in sorted(datasets.keys()):
        dataset = datasets[domain]
        for bs in batch_sizes:
            base = base_metrics[(domain, bs)]
            pack = pack_metrics.get((domain, bs))
            ppl_delta_pct = None
            tps_loss_pct = None
            if pack:
                ppl_delta_pct = ((pack["ppl"] - base["ppl"]) / base["ppl"]) * 100.0 if base["ppl"] else None
                tps_loss_pct = ((base["tps"] - pack["tps"]) / base["tps"]) * 100.0 if base["tps"] else None
            row = {
                "domain": domain,
                "batch_size": bs,
                "num_sequences": int(dataset.shape[0]),
                "sequence_length": args.sequence_length,
                "thinking": args.thinking,
                "ppl_base": base["ppl"],
                "ppl_pack": pack["ppl"] if pack else None,
                "ppl_delta_pct": ppl_delta_pct,
                "tps_base": base["tps"],
                "tps_pack": pack["tps"] if pack else None,
                "tps_loss_pct": tps_loss_pct,
                "first_token_ms_base": base["first_token_ms"],
                "first_token_ms_pack": pack["first_token_ms"] if pack else None,
                "vram_peak_base": base["vram_peak"],
                "vram_peak_pack": pack["vram_peak"] if pack else None,
                "pack_size_mb": pack_size_mb if pack else 0.0,
                "tokens_evaluated": base["tokens"],
                "eval_time_s_base": base["eval_time_s"],
                "eval_time_s_pack": pack["eval_time_s"] if pack else None,
                "pack": args.pack,
            }
            results.append(row)

    print(json.dumps(results, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Evaluation metrics saved to {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        fieldnames = list(results[0].keys()) if results else []
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Evaluation metrics CSV written to {csv_path}")


def cmd_route(args: argparse.Namespace) -> None:
    base_ref = str(args.base)
    base_checkpoint = _resolve_base_checkpoint(base_ref)
    domain_map_path = Path(args.domain_map)
    input_path = Path(args.input)

    if not domain_map_path.exists():
        raise SystemExit(f"Domain map file not found: {domain_map_path}")
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        domain_map = load_domain_map(domain_map_path, pack_root=PACK_ROOT)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))

    print(f"Loaded {len(domain_map)} domain entries from {domain_map_path}")
    print(f"Loading base model from {base_ref}...")
    model, tokenizer = _load_base_model(base_ref, args.loader)
    manager = LoRAManager(model, base_checkpoint=base_checkpoint, base_model=base_ref)
    router = DomainPackRouter(
        manager,
        domain_map,
        default_domain=args.default_domain,
        ttl_seconds=args.ttl_seconds,
        max_recent_domains=args.max_recent_domains,
    )

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.append_out:
            out_path.unlink()

    total = 0
    forward_total_ms = 0.0
    with input_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {lineno}: {exc}") from exc

            domain = payload.get("domain")
            prompt = payload.get("prompt") or payload.get("text") or ""
            route_start = time.time()
            event = router.route(domain)
            route_ms = (time.time() - route_start) * 1000.0

            forward_ms = None
            if args.probe_forward:
                tokens = _tokenise_prompt(tokenizer, str(prompt), args.max_tokens)
                start = time.time()
                logits = model_logits(model, tokens)
                mx.eval(logits)
                forward_ms = (time.time() - start) * 1000.0
                forward_total_ms += forward_ms

            row = {
                "line": lineno,
                "requested_domain": event.requested_domain,
                "resolved_domain": event.resolved_domain,
                "action": event.action,
                "reason": event.reason,
                "active_domain": event.active_domain,
                "pack": event.pack,
                "route_ms": route_ms,
                "forward_ms": forward_ms,
            }
            print(json.dumps(row))
            if out_path:
                with out_path.open("a", encoding="utf-8") as out_file:
                    out_file.write(json.dumps(row) + "\n")

            total += 1
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    detached = router.force_detach()
    summary = {
        "requests": total,
        "ttl_expirations": router.expirations,
        "recent_domains": router.recent_domains(),
        "probe_forward_total_ms": forward_total_ms,
        "detached_on_exit": detached,
    }
    print(json.dumps({"summary": summary}))
    if out_path:
        with out_path.open("a", encoding="utf-8") as out_file:
            out_file.write(json.dumps({"summary": summary}) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="packs", description="Manage MLX LoRA skill packs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Train and export a LoRA pack")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--base",
        required=True,
        help="Base model path or ID (e.g. mlx-community/gemma-4-12B-mxfp8, qwen3-4b-2507-mlx-4bit)",
    )
    create.add_argument("--layers", default="attn.q_proj,attn.k_proj,attn.v_proj")
    create.add_argument(
        "--loader",
        choices=["auto", "mlx-lm", "mlx-vlm"],
        default="auto",
        help="Model loader; auto uses mlx-vlm for Gemma 4 unified any-to-any bases.",
    )
    create.add_argument(
        "--rank-strategy",
        choices=["stable", "theorem"],
        default="theorem",
        help="Rank selection strategy for automatic LoRA rank selection",
    )
    create.add_argument(
        "--target-compression",
        type=float,
        default=0.9,
        help="Energy fraction for automatic rank selection",
    )
    create.add_argument(
        "--rank-eps",
        type=float,
        default=1e-6,
        help="Tolerance for theorem-guided rank selection",
    )
    create.add_argument("--data", required=True)
    create.add_argument(
        "--chat-template",
        action="store_true",
        help="Render JSONL messages/prompt+answer through the tokenizer chat template before tokenizing.",
    )
    create.add_argument(
        "--loss-mode",
        choices=["full", "answer"],
        default="full",
        help="Training loss target: full text or answer-only supervised loss for JSONL prompt/answer data.",
    )
    create.add_argument("--steps", type=int, default=1500)
    create.add_argument("--batch-size", type=int, default=4)
    create.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    create.add_argument("--sequence-length", type=int, default=128)
    create.add_argument("--seed", type=int, default=42)
    create.add_argument(
        "--rank",
        type=_parse_rank,
        help="Use one explicit rank for every selected target instead of automatic rank selection.",
    )
    create.add_argument(
        "--min-rank",
        type=_parse_min_rank,
        default=0,
        help="Optional minimum rank floor after auto-selection (snapped to allowed ranks)",
    )
    create.add_argument("--zero-init", action="store_true")
    create.add_argument("--lora-dropout", type=_parse_lora_dropout, default=0.0)
    create.add_argument(
        "--dynamic-rank",
        action="store_true",
        help="Train with gated active rank and export only active rank columns.",
    )
    create.add_argument(
        "--dynamic-initial-rank",
        type=_parse_rank,
        default=4,
        help="Initial active rank when --dynamic-rank is enabled.",
    )
    create.add_argument(
        "--dynamic-min-rank",
        type=_parse_rank,
        default=2,
        help="Minimum active rank per adapter when dynamic rank can shrink.",
    )
    create.add_argument(
        "--dynamic-rank-interval",
        type=int,
        default=50,
        help="Training steps between dynamic rank adjustments.",
    )
    create.add_argument(
        "--dynamic-rank-warmup",
        type=int,
        default=50,
        help="Training steps before dynamic rank adjustments start.",
    )
    create.add_argument(
        "--dynamic-grow-threshold",
        type=float,
        default=0.25,
        help="Grow adapters whose rank signal is at least this fraction of the strongest adapter.",
    )
    create.add_argument(
        "--dynamic-prune-threshold",
        type=float,
        default=0.03,
        help="Shrink adapters whose rank signal is at most this fraction of the strongest adapter.",
    )
    create.add_argument("--notes", default="")
    create.add_argument(
        "--profile",
        choices=["lite", "heavy"],
        default="lite",
        help="Pack profile: lite keeps strict rank/size guardrails, heavy allows larger ranks and pack sizes",
    )
    create.add_argument("--force", action="store_true")
    create.add_argument(
        "--train-fp16-fallback",
        action="store_true",
        help="Dequantize quantized linears to fp16 before training if required",
    )
    create.set_defaults(func=cmd_create)

    apply = subparsers.add_parser("apply", help="Validate and activate a pack against a base model")
    apply.add_argument("--name", required=True)
    apply.add_argument(
        "--base",
        required=True,
        help="Base model path or ID (e.g. mlx-community/gemma-4-12B-mxfp8, qwen3-4b-2507-mlx-4bit)",
    )
    apply.add_argument(
        "--loader",
        choices=["auto", "mlx-lm", "mlx-vlm"],
        default="auto",
        help="Model loader; auto uses mlx-vlm for Gemma 4 unified any-to-any bases.",
    )
    apply.add_argument("--out", default="run/session_state.json")
    apply.add_argument("--dry-run", action="store_true")
    apply.set_defaults(func=cmd_apply)

    remove = subparsers.add_parser("remove", help="Remove session state created by apply")
    remove.add_argument("--out", default="run/session_state.json")
    remove.set_defaults(func=cmd_remove)

    inspect = subparsers.add_parser("inspect", help="Inspect pack tensor shapes and sizes")
    inspect.add_argument("--name", required=True)
    inspect.set_defaults(func=cmd_inspect)

    rank_ledger = subparsers.add_parser(
        "rank-ledger",
        help="Measure effective rank, slack, and overlap for LoRA pack operators",
    )
    rank_ledger.add_argument("--name", required=True, help="Pack name under packs/")
    rank_ledger.add_argument(
        "--compare",
        help="Optional second pack name to compare against --name",
    )
    rank_ledger.add_argument(
        "--rank-tol",
        type=float,
        default=1e-5,
        help="Relative singular-value tolerance for numerical rank",
    )
    rank_ledger.add_argument("--out", help="Write JSON ledger to this path")
    rank_ledger.add_argument("--csv", help="Write adapter/pair rows to this CSV path")
    rank_ledger.set_defaults(func=cmd_rank_ledger)

    list_cmd = subparsers.add_parser("list", help="List available packs")
    list_cmd.set_defaults(func=cmd_list)

    capabilities = subparsers.add_parser(
        "capabilities",
        help="Report optional MLX modality packages available in this environment",
    )
    capabilities.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable capability status",
    )
    capabilities.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any MLX modality package is missing",
    )
    capabilities.set_defaults(func=cmd_capabilities)

    eval_cmd = subparsers.add_parser("eval", help="Compare base vs pack on perplexity and performance")
    eval_cmd.add_argument(
        "--base",
        required=True,
        help="Base model path or ID (e.g. mlx-community/gemma-4-12B-mxfp8, qwen3-4b-2507-mlx-4bit)",
    )
    eval_cmd.add_argument(
        "--loader",
        choices=["auto", "mlx-lm", "mlx-vlm"],
        default="auto",
        help="Model loader; auto uses mlx-vlm for Gemma 4 unified any-to-any bases.",
    )
    eval_cmd.add_argument("--pack", help="Pack name to evaluate alongside base")
    eval_cmd.add_argument("--data-path", default="roneneldan/TinyStories")
    eval_cmd.add_argument(
        "--chat-template",
        action="store_true",
        help="Render local JSONL messages/prompt+answer through the tokenizer chat template before tokenizing.",
    )
    eval_cmd.add_argument(
        "--loss-mode",
        choices=["full", "answer"],
        default="full",
        help="Evaluation loss target: full text or answer-only supervised loss for local JSONL prompt/answer data.",
    )
    eval_cmd.add_argument("--sequence-length", type=int, default=128)
    eval_cmd.add_argument("--num-samples", type=int, default=100)
    eval_cmd.add_argument("--batch-size", type=int, default=8)
    eval_cmd.add_argument("--out")
    eval_cmd.add_argument("--csv")
    eval_cmd.set_defaults(func=cmd_eval)

    eval_batch = subparsers.add_parser(
        "eval-batch",
        help="Batch evaluate prompts by domain with base vs pack metrics",
    )
    eval_batch.add_argument(
        "--base",
        required=True,
        help="Base model path or ID (e.g. mlx-community/gemma-4-12B-mxfp8, qwen3-4b-2507-mlx-4bit)",
    )
    eval_batch.add_argument(
        "--loader",
        choices=["auto", "mlx-lm", "mlx-vlm"],
        default="auto",
        help="Model loader; auto uses mlx-vlm for Gemma 4 unified any-to-any bases.",
    )
    eval_batch.add_argument("--pack", help="Pack name to evaluate (under packs/<name>)")
    eval_batch.add_argument("--input", required=True, help="Path to prompts JSONL grouped by domain")
    eval_batch.add_argument(
        "--batch-size",
        default="8,16,32",
        help="Comma-separated batch sizes to evaluate (e.g. 8,16,32)",
    )
    eval_batch.add_argument("--sequence-length", type=int, default=512)
    eval_batch.add_argument(
        "--thinking",
        default="strip",
        help="Thinking control: keep, strip, or cap=N",
    )
    eval_batch.add_argument("--out")
    eval_batch.add_argument("--csv")
    eval_batch.set_defaults(func=cmd_eval_batch)

    route = subparsers.add_parser(
        "route",
        help="Route request domains to packs with on-demand attach/detach",
    )
    route.add_argument(
        "--base",
        required=True,
        help="Base model path or ID",
    )
    route.add_argument(
        "--loader",
        choices=["auto", "mlx-lm", "mlx-vlm"],
        default="auto",
        help="Model loader; auto uses mlx-vlm for Gemma 4 unified any-to-any bases.",
    )
    route.add_argument(
        "--domain-map",
        required=True,
        type=Path,
        help="JSON mapping domain -> pack reference (or null for core)",
    )
    route.add_argument(
        "--input",
        required=True,
        type=Path,
        help="JSONL requests with fields {domain,prompt} or {domain,text}",
    )
    route.add_argument("--ttl-seconds", type=float, default=120.0)
    route.add_argument("--max-recent-domains", type=int, default=8)
    route.add_argument("--default-domain", default="core")
    route.add_argument("--probe-forward", action="store_true")
    route.add_argument("--max-tokens", type=int, default=128)
    route.add_argument("--sleep-between", type=float, default=0.0)
    route.add_argument("--out", type=Path)
    route.add_argument(
        "--append-out",
        action="store_true",
        help="Append to --out if it exists (default overwrites)",
    )
    route.set_defaults(func=cmd_route)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
def _dequantize_linear_inplace(module: nn.Module) -> int:
    """Convert quantized linears under module to fp16 linears."""

    converted = 0
    items = list(module.items()) if isinstance(module, nn.Module) else []
    for key, child in items:
        if isinstance(child, QuantizedLinear):
            bits = int(getattr(child, "bits", 4))
            factor = 32 // bits if bits else 1
            out_dim = int(child.weight.shape[0])
            in_dim = int(child.weight.shape[1]) * factor
            weight = mx.dequantize(
                child.weight,
                child.scales,
                child.get("biases"),
                group_size=child.group_size,
                bits=bits,
                mode=child.mode,
            )
            linear = nn.Linear(in_dim, out_dim, bias="bias" in child)
            linear.weight = weight.astype(mx.float16)
            if "bias" in child:
                linear.bias = child["bias"].astype(mx.float16)
            module[key] = linear
            converted += 1
        elif isinstance(child, nn.Module):
            converted += _dequantize_linear_inplace(child)
    return converted
