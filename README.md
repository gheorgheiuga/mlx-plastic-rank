# mlx-plastic-rank

Local low-rank adaptation experiments for MLX. The practical goal is a general-capable base model that can load small domain "skill packs"; the Pop Rank research question is whether LoRA rank can describe useful added capability, not just adapter size.

## Current Thesis
Static LoRA rank is normally treated as a fixed hyperparameter. Pop Rank explores a different path: let training discover where adapter capacity is useful, measure that rank allocation, and then test whether the discovered heterogeneous map can deliver similar quality with fewer exported adapter bytes.

This is still research. The repo now has working mechanics for dynamic active-rank gates, frozen heterogeneous continuation, fresh training from a discovered rank map, and a rank ledger for measuring effective rank, slack, and pack overlap. The current evidence is quality-positive on one local industrial-domain experiment; it is not proof of a general theorem.

## Current Best Signal
Fault-code maintenance pack experiment on `mlx-community/gemma-4-12B-it-qat-mxfp8`, evaluated with 300 answer-only held-out samples and 8 generation-check examples:

| Model/pack | Size | Effective rank | Answer PPL | Token Acc. | Generation solution-overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0 MB | - | 15.4316 | 0.6155 | 0.2723 |
| fixed r16 / 600 steps | 27.10 MB | 2176 | 8.6158 | 0.6507 | 0.2924 |
| dynamic two-phase 150+450 | 27.39 MB | 2288 | 6.2071 | 0.6734 | 0.3911 |
| discovered map from scratch / 600 steps | 27.39 MB | 2288 | 5.7811 | 0.6773 | 0.3911 |
| fixed r32 / 600 steps | 54.16 MB | 4352 | 5.6365 | 0.6802 | 0.4025 |

The strongest signal is the discovered-map-from-scratch run: dynamic rank was used to discover a per-layer heterogeneous rank map, then fresh adapters were trained from that map. It nearly matches fixed `r32/600` while exporting roughly half the adapter bytes, and it strongly beats the same-size fixed `r16/600` baseline. Use `out/fault_codes_all4_plus_base_eval_300.{json,csv}` as the compact artifact for this comparison.

## Why Plastic Rank?
Traditional pruning and distillation discard parameters permanently. Plastic rank started as a reversible compression idea: slices are factored into low-rank adapters that can be re-activated when conditions warrant. The pack tooling extends that rank-control surface from compression into local domain adaptation.

## Key Capabilities
- Dynamic low-rank factors with deterministic MLX kernels
- Plasticity manager for pruning/waking flows and rank heuristics
- LoRA “skill pack” CLI for training, exporting, and evaluating adapters
- Dynamic Pop Rank gates, frozen heterogeneous continuations, and fresh rank-map ablations
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
4. Install pack tooling extras before using `packs`: `uv pip install -e '.[packs]'`
5. Run the sanity check: `uv run python main.py`
6. Execute the plasticity demo: `uv run python plastic_rank.py --steps 10`

## LoRA Skill Packs
- Default Gemma 4 base: `mlx-community/gemma-4-12B-mxfp8`. It keeps the unified any-to-any MLX architecture while roughly halving the local footprint versus bf16. Use `mlx-community/gemma-4-12B-bf16` as a reference/high-fidelity checkpoint for rank probes or regression comparisons when memory allows.
- Train a pack: `uv run packs create --name domain-demo --base mlx-community/gemma-4-12B-mxfp8 --layers attn.q_proj,attn.k_proj,attn.v_proj --loader auto --rank-strategy theorem --target-compression 0.9 --steps 1000 --batch-size 2 --learning-rate 5e-5 --data data/domain_prompts.jsonl --lora-dropout 0.05`
  
  Gemma 4 unified checkpoints load through `mlx-vlm` in `--loader auto`; Qwen/Llama text-only checkpoints continue through `mlx-lm`. Per-slice ranks adjust automatically (`q` keeps the requested rank, `k/v` default to the grouped key/value head width). `attn.o_proj` is supported when you want a higher-capacity pack, but budget it explicitly against the size cap. Add `--train-fp16-fallback` if a quantized projection fails geometry checks.
- Check local macOS modality support: `uv run --extra packs packs capabilities --check`. The pack extra includes `mlx-lm`, `mlx-vlm`, and `mlx-audio`; `mlx-vlm` covers Gemma 4 unified image/audio/video prompting, while `mlx-audio` is the dedicated speech IO layer for TTS, STT, and STS workflows around packs.
- Inspect metadata: `uv run packs inspect --name domain-demo`
- Apply safely: `uv run packs apply --name domain-demo --base mlx-community/gemma-4-12B-mxfp8 --dry-run`
- Evaluate: `uv run packs eval --base mlx-community/gemma-4-12B-mxfp8 --pack domain-demo --data-path data/domain_prompts.jsonl --csv results.csv`
- For prompt/answer JSONL, add `--loss-mode answer` to train/evaluate only assistant answer tokens. This is the preferred mode for diagnostic or maintenance packs where the prompt is context, not a target to imitate.
- Batch evaluation with VRAM/latency guardrails lives in `scripts/demo_plasticity_blocks.py`.
- Compare base vs pack across domains: `uv run packs eval-batch --base mlx-community/gemma-4-12B-mxfp8 --pack domain-demo --input data/domain_prompts.jsonl --batch-size 8,16,32 --sequence-length 256 --thinking strip` (outputs PPL, TPS, first-token ms, VRAM, pack size).
- Gemma 4 bake-off metadata: `uv run --extra packs python scripts/gemma4_smoke.py --metadata-only --out out/gemma4_smoke.json`. Real mxfp8 generation/no-op pack smoke: `uv run --extra packs python scripts/gemma4_smoke.py --models mlx-community/gemma-4-12B-mxfp8 --max-tokens 32 --noop-pack`.
- Instruction-tuned UX smoke: `uv run --extra packs python scripts/gemma4_smoke.py --models mlx-community/gemma-4-12B-it-qat-mxfp8 --chat-template --max-tokens 32`.
- Heavy packs (bigger ranks + larger size cap): add `--profile heavy` to `packs create` when you want higher-capacity domain packs loaded on demand from SSD.
- To force heavier adapters even when auto-rank would stay small, add `--min-rank 16` (or higher; values snap to allowed heavy ranks). To bypass auto-rank entirely for controlled sweeps, use `--rank N`.
- To continue a trained heterogeneous pack with fixed ranks, use `--resume-pack SOURCE_PACK`. To train fresh weights from a discovered heterogeneous rank map, use `--rank-map-from-pack SOURCE_PACK`.

### IndustryBench Pilot
Use Alibaba's IndustryBench as a small industrial QA pack probe. The extractor keeps source metadata in JSON fields while letting the training text stay clean.
- Extract a small English split: `uv run python scripts/industrybench_extract.py --language en --source-limit 512 --train-size 128 --eval-size 32 --metadata-mode none --train-out data/industrybench_en_train.jsonl --eval-out data/industrybench_en_eval.jsonl`
- Baseline IT QAT eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --data-path data/industrybench_en_eval.jsonl --chat-template --sequence-length 256 --num-samples 32 --batch-size 1 --out out/industrybench_gemma4_it_baseline.json --csv out/industrybench_gemma4_it_baseline.csv`
- No-shrink heavy pilot pack: `uv run --extra packs packs create --name industrybench-en-gemma4-it-heavy-smoke --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/industrybench_en_train.jsonl --chat-template --steps 20 --batch-size 1 --sequence-length 256 --learning-rate 1e-5 --target-compression 0.7 --profile heavy`
- First result: the pack exported and applied successfully, with q/v rank 64, k rank 32, and size 91.75 MB. On the 32-row held-out split it changed logits (`max_logit_diff=3.0`) but worsened PPL by 1.45%, so this is an end-to-end mechanics proof, not a quality win.

### Industrial Fault-Code Pilot
Use `avneetsingla/industrial-fault-codes-sample` for the first practical industrial maintenance pack. It has 3,000 English fault-code rows with `brand`, `code`, `description`, and `solution` fields. License is `cc-by-nc-4.0`, so treat this as research/prototyping data unless licensing is resolved.
- Extract train/eval JSONL: `uv run python scripts/fault_codes_extract.py --train-size 2400 --eval-size 300 --train-out data/fault_codes_train.jsonl --eval-out data/fault_codes_eval.jsonl`
- Baseline answer-only eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --data-path data/fault_codes_eval.jsonl --chat-template --loss-mode answer --sequence-length 256 --num-samples 300 --batch-size 4 --out out/fault_codes_gemma4_it_answer_baseline_300.json --csv out/fault_codes_gemma4_it_answer_baseline_300.csv`
- Rank-16 pilot pack: `uv run --extra packs packs create --name fault-codes-gemma4-it-answer-r16-100 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 100 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 16 --profile heavy --lora-dropout 0.05`
- Best sweep pack: `uv run --extra packs packs create --name fault-codes-gemma4-it-answer-r32-300 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 300 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 32 --profile heavy --lora-dropout 0.05`
- Pack eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --pack fault-codes-gemma4-it-answer-r32-300 --data-path data/fault_codes_eval.jsonl --chat-template --loss-mode answer --sequence-length 256 --num-samples 300 --batch-size 4 --out out/fault_codes_gemma4_it_answer_r32_300_eval_300.json --csv out/fault_codes_gemma4_it_answer_r32_300_eval_300.csv`
- Generation check: `uv run --extra packs python scripts/fault_codes_generate_check.py --base mlx-community/gemma-4-12B-it-qat-mxfp8 --pack fault-codes-gemma4-it-answer-r32-300 --eval-data data/fault_codes_eval.jsonl --limit 8 --max-tokens 96 --temperature 0 --chat-template --out out/fault_codes_generation_r32_300_8.json --csv out/fault_codes_generation_r32_300_8.csv`
- Full 300-row sweep:

| Pack | Size | Answer PPL | Delta | Token Acc. | Generation solution-overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0 MB | 15.4316 | 0.00% | 0.6155 | 0.2723 |
| r16 / 100 steps | 27.10 MB | 14.4647 | -6.27% | 0.6197 | not run |
| r16 / 300 steps | 27.10 MB | 11.6834 | -24.29% | 0.6311 | 0.2584 |
| r32 / 100 steps | 54.16 MB | 12.8972 | -16.42% | 0.6259 | not run |
| r32 / 300 steps | 54.16 MB | 8.5175 | -44.80% | 0.6513 | 0.3619 |

Best 300-step fixed-rank sweep result: `fault-codes-gemma4-it-answer-r32-300`. Rank 16 / 300 is the best smaller pack by PPL-per-MB in that sweep, but rank 32 / 300 is the first pack that improves both full-token eval and generated solution-keyword overlap. The newer 600-step Pop Rank bakeoff is summarized near the top of this README.

### Pop Rank Ledger
Use the rank ledger to measure the algebraic footprint of a pack before claiming that rank allocation improved. It reconstructs each LoRA update in compressed form and reports effective rank, slack, stable rank, per-target rank budget, and pairwise pack overlap/composition.
- Inspect one pack: `uv run packs rank-ledger --name fault-codes-gemma4-it-answer-r32-300 --out out/fault_codes_rank_ledger_r32_300.json --csv out/fault_codes_rank_ledger_r32_300.csv`
- Compare two packs: `uv run packs rank-ledger --name fault-codes-gemma4-it-answer-r16-300 --compare fault-codes-gemma4-it-answer-r32-300 --out out/fault_codes_rank_compare_r16_300_vs_r32_300.json --csv out/fault_codes_rank_compare_r16_300_vs_r32_300.csv`

First ledger readout: `r32/300` has 136 adapters, declared rank 4352, effective rank 4352, and zero rank slack. Compared with `r16/300`, the shared adapters compose additively: left effective rank 2176, right effective rank 4352, composition rank 6528, rank savings 0, row/column overlap 0, mean absolute Frobenius cosine about 0.0097. The stronger pack is adding mostly new rank directions rather than duplicating the smaller pack.

### Dynamic Pop Rank
Use dynamic rank when you want the pack to start small and earn capacity during training. The requested `--rank` becomes the maximum rank; `--dynamic-initial-rank` sets the active rank prefix; train-time rank signals grow high-utility adapters and leave low-utility adapters small. Export writes only active rank columns, so the final pack can be smaller than its training ceiling.

Example fault-code run:
`uv run --extra packs packs create --name fault-codes-gemma4-it-answer-dynamic-r32-init4-300 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 300 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 32 --profile heavy --lora-dropout 0.05 --dynamic-rank --dynamic-initial-rank 4 --dynamic-rank-warmup 50 --dynamic-rank-interval 25 --dynamic-grow-threshold 0.25 --dynamic-prune-threshold 0.03`

Two follow-up paths are useful after a dynamic discovery run:
- Continue the discovered map with its learned weights frozen at exported ranks: `uv run --extra packs packs create --name phase-two --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --resume-pack fault-codes-gemma4-it-answer-dynamic-r32-init8-min4-150 --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 450 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --profile heavy --lora-dropout 0.05`
- Train fresh weights from only the discovered heterogeneous rank map: `uv run --extra packs packs create --name hetero-scratch --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --rank-map-from-pack fault-codes-gemma4-it-answer-dynamic-r32-init8-min4-150 --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 600 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --profile heavy --lora-dropout 0.05`

The current 600-step fault-code bakeoff is summarized near the top of this README. Treat it as a quality-positive local result for one industrial domain, not proof of the Pop Rank theorem.

### On-Demand Domain Routing (TTL + LRU)
Run a core model and attach/detach packs on demand using domain labels:
- Domain map JSON (example): `{"core": null, "taxi": "bench-r4"}`
- Requests JSONL (example): `{"domain":"taxi","prompt":"JFK to Midtown fare estimate"}`
- Runtime (CLI): `uv run packs route --base mlx-community/Qwen2.5-1.5B-Instruct-4bit --domain-map run/domain_map.json --input run/requests.jsonl --ttl-seconds 120 --max-recent-domains 8 --probe-forward --out run/route_log.jsonl`
- Runtime (script): `uv run python scripts/domain_router_runtime.py --base mlx-community/Qwen2.5-1.5B-Instruct-4bit --domain-map run/domain_map.json --input run/requests.jsonl --ttl-seconds 120 --max-recent-domains 8 --probe-forward --out run/route_log.jsonl`

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
- `packs` commands require `mlx-lm`; Gemma 4 any-to-any support also requires `mlx-vlm`, and speech IO support uses `mlx-audio` (all install with `uv pip install -e '.[packs]'`).
- Packs enforce `.lora.{A,B,alpha}` tensor schema, fp16 matrices, and fp32 alpha. Lite packs default to a 10 MB cap; heavy packs allow larger SSD-loaded adapters.
- RNG seeds are fixed in tests to keep MLX operations deterministic.

## License
Licensed under the [MIT License](LICENSE).
