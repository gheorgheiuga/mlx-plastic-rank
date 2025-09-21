# Codex Runbook — mlx-plastic-rank

This runbook captures the working state of the project, the preferred workflows, and the current research tracks. Update it as experiments land.

## Mission
Deliver an MLX toolkit for adaptive, reversible low-rank compression. Keep the base checkpoint immutable while LoRA-style skill packs introduce domain skills, report rank/latency trade-offs, and remain easy to audit or roll back.

## Repository Snapshot (2025-09-20)
- `src/mlx_plastic_rank/`: low-rank primitives, plasticity manager, rank selection, pack I/O
- `packs/`: generated skill packs (SafeTensors `.lora.{A,B,alpha}` schema, ≤10 MB)
- `scripts/`: compression demos (`compress_llm_mlx.py`), pack helpers (`demo_mlx_lm_pack.py`), benchmarks
- `tests/`: pytest coverage for rank heuristics, LoRA wrappers, CLI smoke tests
- `codex/`: ADRs, DSNs, research inbox, and this runbook

## Core Commands
- Bootstrap: `uv venv && uv pip install -e .`
- Demos: `uv run python main.py`, `uv run python plastic_rank.py --steps 10`
- Packs workflow:
  - Train: `uv run packs create --name domain-demo --base qwen3-4b-2507-mlx-4bit --layers attn.q_proj,attn.k_proj,attn.v_proj --rank 8 --alpha 16 --data data/domain_prompts.jsonl --steps 1000 --lr 1e-4 --lora-dropout 0.05`
    - Quantized Qwen3 bases stay on 4-bit weights; k/v ranks downshift automatically. Use `--train-fp16-fallback` to dequantize stubborn layers.
  - Inspect: `uv run packs inspect --name domain-demo`
  - Apply safely: `uv run packs apply --name domain-demo --base qwen3-4b-2507-mlx-4bit --dry-run`
  - Evaluate: `uv run packs eval --base qwen3-4b-2507-mlx-4bit --pack domain-demo --data-path data/domain_prompts.jsonl --csv results.csv`
- Compression baseline: `uv run python scripts/compress_llm_mlx.py --hf mlx-community/qwen3-4b-2507-mlx-4bit --out out/qwen3_mlx_compressed --svd randomized --batch-size 20`
- QA: `uv run pytest -q`, `uv run ruff check`, `uv run mypy`

## Implementation Principles
- Python 3.13 + MLX arrays for heavy math; seed RNG for reproducible tests.
- Base checkpoints stay immutable; only LoRA packs mutate runtime state.
- Enforce guardrails: attention ranks ≤8 by default, alpha = 2r unless overridden, packs store only `.lora.{A,B,alpha}` tensors, size cap 10 MB, base hash must match on apply.
- Log rank/alpha/dropout values whenever packs initialize or attach.
- Prefer deterministic kernels in demos so metrics compare cleanly.

## Active Tracks
1. **First production pack** – curate `data/domain_prompts.jsonl`, sweep `r_q ∈ {4,8}` (k/v down-rank automatically), `steps ∈ {500,1000,1500}`, evaluate on the base validation set plus the domain corpus, log CSV (`pack_name,layers,rank_map,alpha,size_MB,load_ms,vram_GB,tps,ppl_base,ppl_pack,ppl_delta_pct,domain_metric`). Target ≤10 % general perplexity delta.
2. **Ablations** – compare layer sets (`qv`, `qkv`, `qkv+out_proj`), rank maps (mixed q/k/v), dropout sweeps (`0.0,0.05,0.1`). Capture outcomes in the same CSV schema.
3. **Path-B export** – implement `packs from-delta` to turn finetuned checkpoints into LoRA packs via ΔW SVD; evaluate with Track 1 metrics.
4. **Reporting** – once experiments stabilize, surface results in README (tables/plots) and link datasets/packs in `codex/dsn/`.

## Testing & Benchmarks
- Unit coverage: `uv run pytest -q`
- Focused: `uv run pytest -q -k manager_adapters`, `uv run pytest -q -k rank_layer`
- Perf sanity: `uv run python scripts/bench_memory.py --m 2048 --n 512`
- Pack smoke test (manual for now):
  - `uv run packs inspect --name noop`
  - `uv run packs apply --name noop --base qwen3-4b-2507-mlx-4bit --dry-run`
  - `uv run packs eval --base qwen3-4b-2507-mlx-4bit --pack noop --data-path data/domain_prompts.jsonl --csv eval_noop.csv`
  - `uv run packs eval-batch --base qwen3-4b-2507-mlx-4bit --pack noop --input data/domain_prompts.jsonl --batch-size 8,16,32 --sequence-length 256 --thinking strip`

## Research Inbox Guidance
Log new papers or experiments under “Research Inbox” below using the template. When an entry drives a decision, add an ADR in `codex/decisions.md` or a DSN and reference it from the PR.

### Research Inbox
- Template:
  - Citation/DOI:
  - Link:
  - Key idea (2–4 bullets):
  - Impacted files/modules:
  - Open questions / risks:

Keep this section lightweight—no PDFs or large datasets in the repo.
