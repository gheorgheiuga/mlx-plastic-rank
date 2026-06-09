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
- Bootstrap: `uv venv && uv pip install -e .` (add `uv pip install -e '.[packs]'` for packs CLI)
- Demos: `uv run python main.py`, `uv run python plastic_rank.py --steps 10`
- Modality stack check: `uv run --extra packs packs capabilities --check` verifies `mlx-lm`, `mlx-vlm`, and `mlx-audio`.
- Gemma bake-off:
  - Metadata/no-download: `uv run --extra packs python scripts/gemma4_smoke.py --metadata-only --out out/gemma4_smoke.json`
  - Real mxfp8 smoke + alpha-zero pack probe: `uv run --extra packs python scripts/gemma4_smoke.py --models mlx-community/gemma-4-12B-mxfp8 --max-tokens 32 --noop-pack`
  - IT QAT assistant smoke: `uv run --extra packs python scripts/gemma4_smoke.py --models mlx-community/gemma-4-12B-it-qat-mxfp8 --chat-template --max-tokens 32`
  - Current mxfp8 smoke: local cache 12 GB, load 2.6-3.5s, generation about 31 tok/s at 8 tokens, peak about 12.73 GB, no-op pack 3.4 MB with matching deterministic output.
  - Current IT QAT mxfp8 chat smoke: local cache 12 GB, warm load 2.8s, peak about 12.75 GB, coherent assistant response when `--chat-template` is used.
- Packs workflow:
  - Train: `uv run packs create --name domain-demo --base mlx-community/gemma-4-12B-mxfp8 --layers attn.q_proj,attn.k_proj,attn.v_proj --loader auto --rank-strategy theorem --target-compression 0.9 --steps 1000 --batch-size 2 --learning-rate 5e-5 --data data/domain_prompts.jsonl --lora-dropout 0.05`
    - Gemma 4 unified any-to-any bases load through `mlx-vlm`; q ranks keep the requested capacity while grouped k/v ranks scale to the key/value head width. Add `attn.o_proj` only after checking pack size.
  - Inspect: `uv run packs inspect --name domain-demo`
  - Apply safely: `uv run packs apply --name domain-demo --base mlx-community/gemma-4-12B-mxfp8 --dry-run`
  - Evaluate: `uv run packs eval --base mlx-community/gemma-4-12B-mxfp8 --pack domain-demo --data-path data/domain_prompts.jsonl --csv results.csv`
  - Prompt/answer data: add `--loss-mode answer` to train/evaluate only assistant answer tokens. Use this for maintenance, diagnostic, and support-style packs where prompts are conditioning context.
- IndustryBench pilot:
  - Extract clean English Q/A text: `uv run python scripts/industrybench_extract.py --language en --source-limit 512 --train-size 128 --eval-size 32 --metadata-mode none --train-out data/industrybench_en_train.jsonl --eval-out data/industrybench_en_eval.jsonl`
  - Baseline eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --data-path data/industrybench_en_eval.jsonl --chat-template --sequence-length 256 --num-samples 32 --batch-size 1 --out out/industrybench_gemma4_it_baseline.json --csv out/industrybench_gemma4_it_baseline.csv`
  - Heavy no-shrink pack smoke: `uv run --extra packs packs create --name industrybench-en-gemma4-it-heavy-smoke --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/industrybench_en_train.jsonl --chat-template --steps 20 --batch-size 1 --sequence-length 256 --learning-rate 1e-5 --target-compression 0.7 --profile heavy`
  - Result: exported/applied successfully, q/v rank 64, k rank 32, 91.75 MB. Held-out eval worsened PPL from 1125.20 to 1141.52 (+1.45%) and token accuracy from 0.28378 to 0.28342, so the run validates pack mechanics but not quality.
- Industrial fault-code pilot:
  - Dataset: `avneetsingla/industrial-fault-codes-sample` (3,000 English rows; `brand`, `code`, `description`, `solution`; `cc-by-nc-4.0`, research/prototyping only unless licensing is resolved).
  - Extract: `uv run python scripts/fault_codes_extract.py --train-size 2400 --eval-size 300 --train-out data/fault_codes_train.jsonl --eval-out data/fault_codes_eval.jsonl`
  - Baseline answer-only eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --data-path data/fault_codes_eval.jsonl --chat-template --loss-mode answer --sequence-length 256 --num-samples 300 --batch-size 4 --out out/fault_codes_gemma4_it_answer_baseline_300.json --csv out/fault_codes_gemma4_it_answer_baseline_300.csv`
  - Rank-16 100-step pack: `uv run --extra packs packs create --name fault-codes-gemma4-it-answer-r16-100 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 100 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 16 --profile heavy --lora-dropout 0.05`
  - Best sweep pack: `uv run --extra packs packs create --name fault-codes-gemma4-it-answer-r32-300 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 300 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 32 --profile heavy --lora-dropout 0.05`
  - Generation check: `uv run --extra packs python scripts/fault_codes_generate_check.py --base mlx-community/gemma-4-12B-it-qat-mxfp8 --pack fault-codes-gemma4-it-answer-r32-300 --eval-data data/fault_codes_eval.jsonl --limit 8 --max-tokens 96 --temperature 0 --chat-template --out out/fault_codes_generation_r32_300_8.json --csv out/fault_codes_generation_r32_300_8.csv`
  - Full 300-row sweep:

    | Pack | Size | Answer PPL | Delta | Token Acc. | Generation solution-overlap |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | base | 0 MB | 15.4316 | 0.00% | 0.6155 | 0.2723 |
    | r16 / 100 steps | 27.10 MB | 14.4647 | -6.27% | 0.6197 | not run |
    | r16 / 300 steps | 27.10 MB | 11.6834 | -24.29% | 0.6311 | 0.2584 |
    | r32 / 100 steps | 54.16 MB | 12.8972 | -16.42% | 0.6259 | not run |
    | r32 / 300 steps | 54.16 MB | 8.5175 | -44.80% | 0.6513 | 0.3619 |

  - Result: `fault-codes-gemma4-it-answer-r32-300` is the best current quality pack. `r16/300` is the best smaller pack by PPL-per-MB, but `r32/300` is the first pack that improves both full-token eval and generated solution-keyword overlap.
- Compression baseline: `uv run python scripts/compress_llm_mlx.py --hf mlx-community/gemma-4-12B-mxfp8 --out out/gemma4_mxfp8_compressed --svd randomized --batch-size 20`
- QA: `uv run pytest -q`, `uv run ruff check`, `uv run mypy`

## Implementation Principles
- Python 3.13 + MLX arrays for heavy math; seed RNG for reproducible tests.
- Base checkpoints stay immutable; only LoRA packs mutate runtime state.
- Default any-to-any base is `mlx-community/gemma-4-12B-mxfp8`; keep `mlx-community/gemma-4-12B-bf16` for reference comparisons when memory allows.
- Treat `mlx-vlm` as the Gemma unified image/audio/video runtime, and `mlx-audio` as the dedicated TTS/STT/STS speech IO layer around the pack system.
- Initialize LoRA train adapters with `B ~ N(0, 1/sqrt(input_dim))` and zero `A`; the earlier `1/input_dim` scale made short answer-only Gemma pilots quantize to no-op deltas.
- Enforce guardrails: attention ranks ≤8 by default, alpha = 2r unless overridden, packs store only `.lora.{A,B,alpha}` tensors, lite size cap 10 MB, heavy profile for larger SSD-loaded packs, base hash must match on apply.
- Log rank/alpha/dropout values whenever packs initialize or attach.
- Prefer deterministic kernels in demos so metrics compare cleanly.

## Active Tracks
1. **First production pack** – curate `data/domain_prompts.jsonl`, sweep `r_q ∈ {4,8}` (k/v down-rank automatically), `steps ∈ {500,1000,1500}`, evaluate on the base validation set plus the domain corpus, log CSV (`pack_name,layers,rank_map,alpha,size_MB,load_ms,vram_GB,tps,ppl_base,ppl_pack,ppl_delta_pct,domain_metric`). Target ≤10 % general perplexity delta.
2. **Ablations** – compare layer sets (`qv`, `qkv`, `qkv+out_proj`), rank maps (mixed q/k/v), dropout sweeps (`0.0,0.05,0.1`). Capture outcomes in the same CSV schema.
3. **Path-B export** – implement `packs from-delta` to turn finetuned checkpoints into LoRA packs via ΔW SVD; evaluate with Track 1 metrics.
4. **Reporting**
    - Job demo r4 (rank_q=4, kv=4): pack 3.83 MB; eval @100 prompts seq128 → PPL 11.58 (+0.09%), TPS 1.13k (-1.4%), token accuracy 0.527; batch eval (bs32 seq256) → PPL Δ +0.01%, TPS loss 7.2%, pack size <6 MB.
    - Job demo r8 (rank_q=8, kv=8): pack 7.62 MB; eval @100 prompts seq128 → PPL 11.58 (+0.22%), TPS 0.79k (-0.4%), token accuracy 0.527; batch eval (bs32 seq256) → PPL Δ +0.002%, TPS loss 7.1%, pack size <12 MB. – once experiments stabilize, surface results in README (tables/plots) and link datasets/packs in `codex/dsn/`.

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
