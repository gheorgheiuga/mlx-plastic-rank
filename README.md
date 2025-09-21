# mlx-plastic-rank

Adaptive low-rank compression for MLX with neuroplastic “wake and sleep” triggers. The toolkit re-expresses pruned weights as reversible low-rank factors so capacity can shrink, regrow, or park on demand while preserving the base checkpoint.

## Why Plastic Rank?
Traditional pruning and distillation discard parameters permanently. Plastic rank treats compression as a reversible transformation: slices are factored into low-rank adapters that can be re-activated when conditions warrant. The result is a tunable system that keeps accuracy, exposes rank controls, and surfaces clear telemetry for every change.

## Key Capabilities
- Dynamic low-rank factors with deterministic MLX kernels
- Plasticity manager for pruning/waking flows and rank heuristics
- LoRA “skill pack” CLI for training, exporting, and evaluating adapters
- SafeTensors export utilities plus inspection and logging helpers
- Benchmarks and demos for profiling rank/latency trade-offs

## Project Layout
- `src/mlx_plastic_rank/` – core modules (`lowrank`, `plasticity_manager`, `packs`, utilities)
- `scripts/` – compression demos, memory benchmarks, CLI helpers
- `packs/` – generated skill packs (git-ignored by default)
- `data/` – small finetuning/evaluation samples
- `tests/` – pytest suite covering rank logic, LoRA manager, CLI workflows
- `codex/` – runbook, ADRs, DSNs, and research notes for maintainers

## Quick Start
1. Create an environment (optional):
   - `uv venv`
   - `source .venv/bin/activate`
2. Install the project in editable mode: `uv pip install -e .`
3. Add extras when exploring compression flows: `uv pip install -e '.[compress]'`
4. Run the sanity check: `uv run python main.py`
5. Execute the plasticity demo: `uv run python plastic_rank.py --steps 10`

## LoRA Skill Packs
- Train a pack: `uv run packs create --name domain-demo --base qwen3-4b-2507-mlx-4bit --layers attn.q_proj,attn.k_proj,attn.v_proj --rank 8 --alpha 16 --data data/domain_prompts.jsonl --steps 1000 --lr 1e-4 --lora-dropout 0.05`
  
  Qwen3 4-bit checkpoints stay quantized end-to-end; per-slice ranks adjust automatically (`q` uses the requested rank, `k/v` default to the grouped head width). Add `--train-fp16-fallback` if a projection fails geometry checks.
- Inspect metadata: `uv run packs inspect --name domain-demo`
- Apply safely: `uv run packs apply --name domain-demo --base qwen3-4b-2507-mlx-4bit --dry-run`
- Evaluate: `uv run packs eval --base qwen3-4b-2507-mlx-4bit --pack domain-demo --data-path data/domain_prompts.jsonl --csv results.csv`
- Batch evaluation with VRAM/latency guardrails lives in `scripts/demo_plasticity_blocks.py`.
- Compare base vs pack across domains: `uv run packs eval-batch --base qwen3-4b-2507-mlx-4bit --pack domain-demo --input data/domain_prompts.jsonl --batch-size 8,16,32 --sequence-length 256 --thinking strip` (outputs PPL, TPS, first-token ms, VRAM, pack size).

## Benchmarks & Utilities
- Compression baseline: `uv run python scripts/compress_llm_mlx.py --hf mlx-community/qwen3-4b-2507-mlx-4bit --out out/qwen3_mlx_compressed --svd randomized --batch-size 20`
- Memory profiler: `uv run python scripts/bench_memory.py --m 2048 --n 512`
- Export factors directly: `uv run python -m mlx_plastic_rank.export_safetensors --from-weight weight.npy --rank 64 --bits 8 --out out/weight_lr.safetensors`

## Testing & Quality Gates
- Unit tests: `uv run pytest -q`
- Focused suites: `uv run pytest -q -k rank_layer`
- Static analysis: `uv run ruff check`; types: `uv run mypy`
- Ensure tests and demos pass before sending a PR; include representative logs or CSV snippets when adding new experiments.

## Requirements & Notes
- Apple Silicon with MLX installed is required for GPU-backed ops; SVD falls back to CPU streams.
- Packs enforce `.lora.{A,B,alpha}` tensor schema, fp16 matrices, fp32 alpha, ≤10 MB total.
- RNG seeds are fixed in tests to keep MLX operations deterministic.

## License
Licensed under the [MIT License](LICENSE).
