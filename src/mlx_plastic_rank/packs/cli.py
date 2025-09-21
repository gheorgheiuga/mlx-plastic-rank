"""Command line interface for LoRA skill packs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.perplexity import eval_ppl, load_data
from mlx_lm.utils import load as load_model
from mlx.nn.layers.quantized import QuantizedLinear

from .dataset import build_token_dataset, load_jsonl_texts
from .inspection import ALLOWED_RANKS, TensorInfo, size_limit_for, summarize_pack
from .io import compute_sha256, save_pack, save_pack_metadata
from .eval_utils import load_domain_prompts, parse_batch_sizes, parse_thinking_option
from .manager import LoRAManager, PackApplicationError
from .train import TrainingConfig, train_lora

LORA_ALIAS_MAP = {
    "q": "attn.q_proj",
    "k": "attn.k_proj",
    "v": "attn.v_proj",
}


def _resolve_target(name: str) -> str:
    canonical = name.strip()
    return LORA_ALIAS_MAP.get(canonical, canonical)


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
        logits = model(batch[:, :-1]).astype(mx.float32)
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

PACK_ROOT = Path("packs")


def _default_pack_dir(name: str) -> Path:
    return PACK_ROOT / name


def cmd_create(args: argparse.Namespace) -> None:
    base_path = Path(args.base)
    pack_dir = _default_pack_dir(args.name)
    if pack_dir.exists() and not args.force:
        raise SystemExit(f"Pack '{args.name}' already exists. Use --force to overwrite.")

    print(f"Loading base model from {base_path}...")
    model, tokenizer = load_model(str(base_path))
    if getattr(args, "train_fp16_fallback", False):
        converted = _dequantize_linear_inplace(model)
        if converted:
            print(f"Dequantized {converted} quantized linear layers for training fallback")

    manager = LoRAManager(model, base_checkpoint=base_path)

    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    canonical_layers = [_resolve_target(layer) for layer in layers]
    allowed_targets = {"attn.q_proj", "attn.k_proj", "attn.v_proj"}
    disallowed = [layer for layer in canonical_layers if layer not in allowed_targets]
    if disallowed:
        raise SystemExit(f"Unsupported LoRA targets: {disallowed}. Only q,k,v projections are allowed.")

    auto_rank_map, auto_alpha_map, residuals = manager.compute_auto_ranks(
        canonical_layers,
        strategy="theorem",
        target_compression=args.target_compression,
        eps=args.rank_eps,
    )
    print("Auto-selected ranks (Pop theorem):")
    rank_map: Dict[str, int] = {}
    alpha_map: Dict[str, float] = {}
    for target in canonical_layers:
        rank_map[target] = auto_rank_map[target]
        alpha_map[target] = auto_alpha_map[target]
        res = residuals[target]
        print(f"  {target}: rank={rank_map[target]} residual={res:.4g}")

    base_rank = rank_map.get("attn.q_proj", next(iter(rank_map.values())))
    base_alpha = alpha_map.get("attn.q_proj", 2.0 * base_rank)

    print(f"Initialising adapters on layers: {canonical_layers}")
    adapters = manager.initialize_adapters(
        canonical_layers,
        rank=base_rank,
        alpha=base_alpha,
        seed=args.seed,
        rank_map=rank_map,
        alpha_map=alpha_map,
        dropout=args.lora_dropout,
    )
    if args.zero_init:
        for adapter in adapters.values():
            adapter.A = mx.zeros_like(adapter.A)
            adapter.B = mx.zeros_like(adapter.B)

    texts = load_jsonl_texts(Path(args.data))
    dataset = build_token_dataset(texts, tokenizer, args.sequence_length)

    config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        log_interval=max(1, args.steps // 10),
        lora_dropout=args.lora_dropout,
    )
    print(
        f"Training LoRA adapters: steps={config.steps} batch_size={config.batch_size} lr={config.learning_rate}"
    )
    final_loss = train_lora(manager, model, dataset, config)
    print(f"Training complete. Final loss={final_loss:.4f}")

    tensors, metadata = manager.export_active_pack(
        args.name,
        PACK_ROOT,
        notes=args.notes,
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
    base_path = Path(args.base)
    pack_dir = _default_pack_dir(args.name)
    if not pack_dir.exists():
        raise SystemExit(f"Pack '{args.name}' not found at {pack_dir}")

    print(f"Inspecting pack '{args.name}'...")
    metadata, infos, _, total_bytes, non_lora = summarize_pack(pack_dir)
    base_hash = compute_sha256(base_path)
    if metadata.base_hash and metadata.base_hash != base_hash:
        raise SystemExit(
            f"Base hash mismatch: pack built for {metadata.base_hash[:8]}, base is {base_hash[:8]}"
        )
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

    model, _ = load_model(str(base_path))
    manager = LoRAManager(model, base_checkpoint=base_path)
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
            "base": str(base_path),
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
            f"{metadata.pack_name:<16} size={size_mb:5.2f}MB/{limit_mb:4.1f}MB{over} ranks={ranks} layers={len(metadata.target_layers)} created={created[:19]} base={metadata.base_hash[:8]}"
        )
    if not rows:
        print("No packs found.")
    else:
        print("Installed packs:")
        for row in rows:
            print("  " + row)


def cmd_eval(args: argparse.Namespace) -> None:
    data_path = args.data_path

    dataset_cached = None
    dataset_type = None
    base_logits_sample = None

    def evaluate(model_path: Path, pack_dir: Path | None = None) -> dict:
        nonlocal dataset_cached, dataset_type, base_logits_sample
        start_load = time.time()
        model, tokenizer = load_model(str(model_path))
        manager = LoRAManager(model, base_checkpoint=model_path)
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
                texts = load_jsonl_texts(path_obj)
                dataset = build_token_dataset(texts, tokenizer, args.sequence_length)
                if args.num_samples > 0:
                    dataset = dataset[: args.num_samples]
                dataset_cached = dataset
                dataset_type = "jsonl"
            else:
                dataset_cached = load_data(
                    tokenizer,
                    data_path,
                    num_samples=args.num_samples,
                    sequence_length=args.sequence_length,
                )
                dataset_type = "hf"

        dataset = dataset_cached
        mx.reset_peak_memory()
        eval_start = time.time()
        ppl, se = eval_ppl(model, dataset, batch_size=args.batch_size)
        eval_time = time.time() - eval_start
        peak_mem = mx.get_peak_memory() / (1024**3)
        tokens = dataset.shape[0] * (dataset.shape[1] - 1)
        tps = tokens / max(eval_time, 1e-6)

        batch = dataset[:1]
        logits = model(batch[:, :-1]).astype(mx.float32)
        mx.eval(logits)
        max_diff = 0.0
        if base_logits_sample is None:
            base_logits_sample = logits
        else:
            diff = mx.abs(logits - base_logits_sample)
            max_diff = float(mx.max(diff).item())

        token_accuracy = None
        if dataset_type == "jsonl":
            correct = 0.0
            total = 0
            for start_idx in range(0, dataset.shape[0], args.batch_size):
                batch_tokens = dataset[start_idx : start_idx + args.batch_size]
                if batch_tokens.shape[0] == 0:
                    continue
                inputs = batch_tokens[:, :-1]
                targets = batch_tokens[:, 1:]
                preds = mx.argmax(model(inputs), axis=-1)
                matches = mx.equal(preds, targets).astype(mx.float32)
                correct += float(mx.sum(matches).item())
                total += int(targets.size)
            if total > 0:
                token_accuracy = correct / float(total)

        return {
            "model": str(model_path),
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
        }

    base_metrics = evaluate(Path(args.base), None)
    results = [base_metrics]
    if args.pack:
        results.append(evaluate(Path(args.base), _default_pack_dir(args.pack)))

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
    base_path = Path(args.base)
    batch_sizes = parse_batch_sizes(args.batch_size)
    thinking_mode, cap_tokens = parse_thinking_option(args.thinking)
    prompts_by_domain = load_domain_prompts(Path(args.input), thinking_mode, cap_tokens)

    model, tokenizer = load_model(str(base_path))
    manager = LoRAManager(model, base_checkpoint=base_path)

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
        base_hash = compute_sha256(base_path)
        if metadata.base_hash and metadata.base_hash != base_hash:
            raise SystemExit(
                f"Base hash mismatch: pack built for {metadata.base_hash[:8]}, base is {base_hash[:8]}"
            )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="packs", description="Manage GPT-2 LoRA skill packs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Train and export a LoRA pack")
    create.add_argument("--name", required=True)
    create.add_argument(
        "--base",
        required=True,
        help="Base model path or ID (e.g. qwen3-4b-2507-mlx-4bit, qwen3-4b-thinking-2507-mlx-4bit, llama-3-8b-instruct-mlx-4bit)",
    )
    create.add_argument("--layers", default="attn.q_proj,attn.k_proj,attn.v_proj")
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
    create.add_argument("--steps", type=int, default=1500)
    create.add_argument("--batch-size", type=int, default=4)
    create.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    create.add_argument("--sequence-length", type=int, default=128)
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--zero-init", action="store_true")
    create.add_argument("--lora-dropout", type=float, default=0.0)
    create.add_argument("--notes", default="")
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
        help="Base model path or ID (e.g. qwen3-4b-2507-mlx-4bit, qwen3-4b-thinking-2507-mlx-4bit, llama-3-8b-instruct-mlx-4bit)",
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

    list_cmd = subparsers.add_parser("list", help="List available packs")
    list_cmd.set_defaults(func=cmd_list)

    eval_cmd = subparsers.add_parser("eval", help="Compare base vs pack on perplexity and performance")
    eval_cmd.add_argument(
        "--base",
        required=True,
        help="Base model path or ID (e.g. qwen3-4b-2507-mlx-4bit, qwen3-4b-thinking-2507-mlx-4bit, llama-3-8b-instruct-mlx-4bit)",
    )
    eval_cmd.add_argument("--pack", help="Pack name to evaluate alongside base")
    eval_cmd.add_argument("--data-path", default="roneneldan/TinyStories")
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
        help="Base model path or ID (e.g. qwen3-4b-2507-mlx-4bit, qwen3-4b-thinking-2507-mlx-4bit, llama-3-8b-instruct-mlx-4bit)",
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
