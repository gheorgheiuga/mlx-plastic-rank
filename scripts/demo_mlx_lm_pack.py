"""Demonstrate applying an mlx-plastic-rank LoRA pack on an mlx-lm GPT-2 model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load as load_model

from mlx_plastic_rank.packs.manager import LoRAManager


def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    """Greedy/temperature sampling loop using MLX arrays."""
    mx.random.seed(seed)
    np.random.seed(seed)
    input_ids = tokenizer.encode(prompt)
    tokens = mx.array([input_ids], dtype=mx.int32)
    mx.eval(tokens)

    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_token = int(mx.argmax(next_logits, axis=-1).item())
        else:
            scaled = next_logits / temperature
            probs = mx.softmax(scaled, axis=-1)
            mx.eval(probs)
            probs_np = np.array(probs)[0]
            next_token = int(np.random.choice(len(probs_np), p=probs_np))
        next_id = mx.array([[next_token]], dtype=mx.int32)
        tokens = mx.concatenate([tokens, next_id], axis=1)

    output_ids = [int(t) for t in tokens[0].tolist()]
    return tokenizer.decode(output_ids)


def run_demo(
    base_path: Path,
    pack_dir: Optional[Path],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> None:
    print(f"Loading model from {base_path}...")
    model, tokenizer = load_model(str(base_path))
    manager = LoRAManager(model, base_checkpoint=base_path)

    print("\n=== Base model ===")
    base_text = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    print(base_text)

    if pack_dir:
        print(f"\nApplying pack from {pack_dir}...")
        metadata = manager.apply_pack(pack_dir)
        pack_bytes = 0
        tensor_path = pack_dir / "pack.safetensors"
        if tensor_path.exists():
            pack_bytes = tensor_path.stat().st_size
        size_mb = pack_bytes / (1024**2)
        print(f"Applied pack '{metadata.pack_name}' (~{size_mb:.2f} MB)")
        print("\n=== Pack enabled ===")
        pack_text = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        print(pack_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("out/gpt2_mlx_base"))
    parser.add_argument("--pack", type=Path, help="Pack directory to apply before generation")
    parser.add_argument("--prompt", default="Request: JFK to Midtown East. Plan the trip and estimate fare.")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_demo(
        base_path=args.base,
        pack_dir=args.pack,
        prompt=args.prompt,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
