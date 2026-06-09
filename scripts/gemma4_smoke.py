"""Smoke-test Gemma 4 MLX bases and optional no-op pack mechanics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import model_info, snapshot_download

from mlx_plastic_rank.packs.io import save_pack, save_pack_metadata
from mlx_plastic_rank.packs.manager import LoRAManager, PackApplicationError

DEFAULT_MODELS = (
    "mlx-community/gemma-4-12B-mxfp8",
    "mlx-community/gemma-4-12B-it-qat-mxfp8",
)
BF16_MODEL = "mlx-community/gemma-4-12B-bf16"
DEFAULT_PROMPTS = (
    "In one sentence, explain what a LoRA skill pack changes without changing the base model.",
    "Classify this request as core, code, audio, or vision: summarize this meeting recording.",
)


@dataclass
class RepoSummary:
    model: str
    task: str | None
    library_name: str | None
    downloads: int | None
    likes: int | None
    last_modified: str | None
    files: int
    bytes: int
    size_gb: float


@dataclass
class GenerationMetric:
    model: str
    prompt_index: int
    prompt: str
    load_time_s: float
    generation_time_s: float
    first_token_ms: float | None
    prompt_tokens: int | None
    generation_tokens: int | None
    total_tokens: int | None
    prompt_tps: float | None
    generation_tps: float | None
    peak_memory_gb: float | None
    finish_reason: str | None
    text: str


@dataclass
class NoopProbeMetric:
    model: str
    pack_dir: str
    targets: list[str]
    rank: int
    pack_size_bytes: int
    baseline_text: str
    packed_text: str
    text_match: bool


def parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def repo_summary(model: str) -> RepoSummary:
    info = model_info(model, files_metadata=True)
    siblings = getattr(info, "siblings", []) or []
    total_bytes = sum((getattr(sibling, "size", None) or 0) for sibling in siblings)
    card_data = getattr(info, "card_data", None)
    task = getattr(card_data, "pipeline_tag", None) or getattr(info, "pipeline_tag", None)
    return RepoSummary(
        model=model,
        task=task,
        library_name=getattr(info, "library_name", None),
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        last_modified=str(getattr(info, "last_modified", None)),
        files=len(siblings),
        bytes=total_bytes,
        size_gb=round(total_bytes / (1024**3), 3),
    )


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                payload = json.loads(line)
                text = payload.get("prompt") or payload.get("text")
                if text:
                    prompts.append(str(text))
            else:
                prompts.append(line)
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


def load_vlm_model(model_ref: str, *, local_files_only: bool = False):
    from mlx_vlm import load

    snapshot_path = snapshot_download(model_ref, local_files_only=local_files_only)
    loaded = load(snapshot_path)
    return loaded[0], loaded[1]


def resolve_snapshot(model_ref: str, *, local_files_only: bool = False) -> str:
    return snapshot_download(model_ref, local_files_only=local_files_only)


def run_generation(
    model,
    processor,
    *,
    model_ref: str,
    prompt: str,
    prompt_index: int,
    load_time_s: float,
    max_tokens: int,
    temperature: float,
    image: list[str],
    audio: list[str],
    video: list[str],
    chat_template: bool,
) -> GenerationMetric:
    from mlx_vlm.generate import stream_generate

    rendered_prompt = render_chat_prompt(processor, prompt) if chat_template else prompt
    mx.reset_peak_memory()
    start = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    last_response: Any | None = None

    for response in stream_generate(
        model,
        processor,
        rendered_prompt,
        image=image or None,
        audio=audio or None,
        video=video or None,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    ):
        piece = getattr(response, "text", "")
        if piece and first_token_at is None:
            first_token_at = time.perf_counter()
        text_parts.append(piece)
        last_response = response

    elapsed = time.perf_counter() - start
    return GenerationMetric(
        model=model_ref,
        prompt_index=prompt_index,
        prompt=prompt,
        load_time_s=load_time_s,
        generation_time_s=elapsed,
        first_token_ms=((first_token_at - start) * 1000.0) if first_token_at else None,
        prompt_tokens=getattr(last_response, "prompt_tokens", None),
        generation_tokens=getattr(last_response, "generation_tokens", None),
        total_tokens=getattr(last_response, "total_tokens", None),
        prompt_tps=getattr(last_response, "prompt_tps", None),
        generation_tps=getattr(last_response, "generation_tps", None),
        peak_memory_gb=getattr(last_response, "peak_memory", None),
        finish_reason=getattr(last_response, "finish_reason", None),
        text="".join(text_parts).strip(),
    )


def export_noop_pack(
    model,
    *,
    model_ref: str,
    name: str,
    targets: list[str],
    rank: int,
    pack_root: Path,
) -> Path:
    manager = LoRAManager(model, base_model=model_ref)
    adapters = manager.initialize_adapters(
        targets=targets,
        rank=rank,
        alpha=0.0,
        seed=0,
        alpha_map={target: 0.0 for target in targets},
        allowed_ranks=(2, 4, 8),
    )
    if not adapters:
        raise PackApplicationError("No adapters were initialised for no-op pack export")
    tensors, metadata = manager.export_active_pack(
        name,
        pack_root,
        notes=f"alpha-zero Gemma 4 smoke pack for {model_ref}",
    )
    pack_dir = pack_root / name
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(metadata, pack_dir / "meta.json")
    manager.clear()
    return pack_dir


def run_noop_probe(
    model,
    processor,
    *,
    model_ref: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    chat_template: bool,
    pack_name: str,
    pack_root: Path,
    targets: list[str],
    rank: int,
) -> NoopProbeMetric:
    baseline = run_generation(
        model,
        processor,
        model_ref=model_ref,
        prompt=prompt,
        prompt_index=0,
        load_time_s=0.0,
        max_tokens=max_tokens,
        temperature=temperature,
        image=[],
        audio=[],
        video=[],
        chat_template=chat_template,
    )
    pack_dir = export_noop_pack(
        model,
        model_ref=model_ref,
        name=pack_name,
        targets=targets,
        rank=rank,
        pack_root=pack_root,
    )
    manager = LoRAManager(model, base_model=model_ref)
    manager.apply_pack(pack_dir)
    packed = run_generation(
        model,
        processor,
        model_ref=model_ref,
        prompt=prompt,
        prompt_index=0,
        load_time_s=0.0,
        max_tokens=max_tokens,
        temperature=temperature,
        image=[],
        audio=[],
        video=[],
        chat_template=chat_template,
    )
    manager.clear()
    pack_file = pack_dir / "pack.safetensors"
    return NoopProbeMetric(
        model=model_ref,
        pack_dir=str(pack_dir),
        targets=targets,
        rank=rank,
        pack_size_bytes=pack_file.stat().st_size if pack_file.exists() else 0,
        baseline_text=baseline.text,
        packed_text=packed.text,
        text_match=baseline.text == packed.text,
    )


def write_outputs(payload: dict[str, Any], out_path: Path | None, csv_path: Path | None) -> None:
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON metrics to {out_path}")
    if csv_path is not None:
        rows = payload.get("generations", [])
        if not rows:
            return
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote generation CSV to {csv_path}")


def render_chat_prompt(processor, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise SystemExit("Selected processor/tokenizer does not expose apply_chat_template")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated HF model IDs to compare",
    )
    parser.add_argument(
        "--include-bf16",
        action="store_true",
        help=f"Append {BF16_MODEL}; this is a much larger download/runtime target.",
    )
    parser.add_argument("--prompt-file", type=Path, help="Plain-text or JSONL prompts")
    parser.add_argument("--image", action="append", default=[], help="Image path/URL to attach")
    parser.add_argument("--audio", action="append", default=[], help="Audio path/URL to attach")
    parser.add_argument("--video", action="append", default=[], help="Video path/URL to attach")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render text prompts through processor/tokenizer chat template before generation.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch repo metadata and skip model downloads/generation",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached model files when loading models",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Resolve/download model snapshots and skip loading/generation",
    )
    parser.add_argument("--out", type=Path, default=Path("out/gemma4_smoke.json"))
    parser.add_argument("--csv", type=Path, default=Path("out/gemma4_smoke.csv"))
    parser.add_argument(
        "--noop-pack",
        action="store_true",
        help="Export/reapply an alpha-zero pack and compare deterministic output on the first prompt.",
    )
    parser.add_argument("--noop-pack-name", default="gemma4-noop")
    parser.add_argument("--noop-targets", default="attn.q_proj,attn.k_proj,attn.v_proj")
    parser.add_argument("--noop-rank", type=int, default=2)
    parser.add_argument("--pack-root", type=Path, default=Path("packs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    models = parse_csv(args.models)
    if args.include_bf16 and BF16_MODEL not in models:
        models.append(BF16_MODEL)
    if not models:
        raise SystemExit("No models selected")

    prompts = load_prompts(args.prompt_file)
    summaries = [asdict(repo_summary(model)) for model in models]
    print("Model metadata:")
    for row in summaries:
        print(
            f"  {row['model']}: task={row['task']} files={row['files']} size={row['size_gb']:.2f}GB"
        )

    payload: dict[str, Any] = {
        "models": models,
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / (1024**3), 3),
        "metadata": summaries,
        "generations": [],
        "noop_probe": [],
    }

    if args.metadata_only:
        write_outputs(payload, args.out, args.csv)
        return

    if args.download_only:
        snapshots: dict[str, str] = {}
        for model_ref in models:
            print(f"Resolving snapshot for {model_ref}...")
            snapshots[model_ref] = resolve_snapshot(
                model_ref,
                local_files_only=args.local_files_only,
            )
            print(f"  {snapshots[model_ref]}")
        payload["snapshots"] = snapshots
        write_outputs(payload, args.out, args.csv)
        return

    for model_ref in models:
        print(f"Loading {model_ref}...")
        load_start = time.perf_counter()
        model, processor = load_vlm_model(model_ref, local_files_only=args.local_files_only)
        load_time_s = time.perf_counter() - load_start
        print(f"Loaded {model_ref} in {load_time_s:.1f}s")

        for index, prompt in enumerate(prompts):
            metric = run_generation(
                model,
                processor,
                model_ref=model_ref,
                prompt=prompt,
                prompt_index=index,
                load_time_s=load_time_s,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                image=args.image,
                audio=args.audio,
                video=args.video,
                chat_template=args.chat_template,
            )
            payload["generations"].append(asdict(metric))
            print(
                f"  prompt {index}: first_token={metric.first_token_ms}ms "
                f"gen_tps={metric.generation_tps} peak={metric.peak_memory_gb}GB"
            )

        if args.noop_pack:
            targets = parse_csv(args.noop_targets)
            if not targets:
                raise SystemExit("--noop-targets must not be empty")
            probe = run_noop_probe(
                model,
                processor,
                model_ref=model_ref,
                prompt=prompts[0],
                max_tokens=args.max_tokens,
                temperature=0.0,
                chat_template=args.chat_template,
                pack_name=args.noop_pack_name,
                pack_root=args.pack_root,
                targets=targets,
                rank=args.noop_rank,
            )
            payload["noop_probe"].append(asdict(probe))
            print(
                f"  noop pack: match={probe.text_match} size={probe.pack_size_bytes} path={probe.pack_dir}"
            )

    write_outputs(payload, args.out, args.csv)


if __name__ == "__main__":
    main()
