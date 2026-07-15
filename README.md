# mlx-plastic-rank

Local low-rank adaptation experiments for MLX. The practical goal is a general-capable base model that can load small domain "skill packs"; the Pop Rank research question is whether effective rank can expose and control where finite learning capacity moves over time.[^pop-theorem]

## Current Thesis
Pop Rank studies whether a capacity-limited model can move low-rank capacity as it learns: concentrate rank during specialization, park it when it is temporarily inactive, and recycle or revive it later. The central experiment holds the declared effective active-rank count constant, so growth at one site requires shrinkage at another. That invariant does not by itself conserve physical parameter bytes, optimizer state, resident learned factors, or information. Rank is the measurement and control surface for capacity migration, not merely a way to choose a smaller static adapter.

This is an experimental thesis, not an established result. Dynamic gates, sleep/wake paths, the rank ledger, heterogeneous rank maps, and same-budget controls provide enabling mechanics and measurement. A ten-seed orthogonal teacher/student fixture verifies the reference mechanics. The first frozen learned-MLX attempt then found a real partial signal: loss-guided rank movement beat static, had a positive conditional B-AUC difference across nine finite random pairs, and the first loss-guided transfer during the B phase preceded a matched performance advantage. The full promotion gate nevertheless failed because the random comparison was incomplete, another control also went non-finite, the learned controller did not beat a future-aware fixed split conclusively, and the joint-sufficient control narrowly missed threshold. Its A-return metric was measured after a supervised A-loss probe and any resulting gate transfer but before a parameter update; learned V1 did not test cue-triggered wake through an unlabeled retrieval path. The rank-one floor guaranteed at least one A-site component remained active, while some seeds retained more and dense routing could preserve A through other sites. Dormant-factor and strict-recycle counts are provisional A-column lower bounds because B-only learned state was not audited. See [the capacity-migration DSN](codex/dsn/dsn-20260713-capacity-migration.md), [reference evidence](codex/evidence/capacity_migration_reference_seed0_9.md), and the [failed learned-model gate](codex/evidence/capacity_migration_learned_dense_seed1_10.md).

A frozen follow-up then branched every legal first-B transfer from the same
A-trained checkpoint on untouched seeds 11–20. Predicted-best beat static,
prediction-independent random, and an A-loss-selected control, but the full
candidate ordering was not calibrated and predicted-best did not beat
predicted-worst conclusively. One-step loss lookahead is therefore demoted as a
directional controller; only a narrower exploratory move signal survives. See
[the selector calibration](codex/evidence/loss_lookahead_calibration_seed11_20.md).

A separately frozen two-transfer horizon-3 controller then beat static,
fixed-random, and wrong-task controls and moved more final rank toward B. Its
admission gate still failed: it did not separate from exact one-step, whose seed
26 path reproducibly went non-finite. Horizon-3 is demoted, further exhaustive
shadow-rollout tuning is parked, and the full return-A V2 remains blocked. See
[the multi-batch evidence](codex/evidence/multibatch_controller_v2_seed21_30.md).

## Best Supporting Rank-Placement Signal
Fault-code maintenance pack experiment on `mlx-community/gemma-4-12B-it-qat-mxfp8`, trained on 2,700 rows and evaluated on 300 answer-only held-out samples from `avneetsingla/industrial-fault-codes-sample`:[^fault-codes]

| Model/pack | Size | Effective rank | Answer PPL | Token Acc. |
| --- | ---: | ---: | ---: | ---: |
| base | 0 MB | - | 15.4316 | 0.6155 |
| fixed r16 context (unpaired) / 600 steps | 27.10 MB | 2176 | 8.5677 | 0.6515 |
| discovered map, paired seed 42 / 600 steps | 23.73 MB | 1984 | 5.8532 | 0.6757 |
| fixed r32 context (unpaired) / 600 steps | 54.16 MB | 4352 | 5.5622 | 0.6802 |

The strongest signal is now a paired falsification screen. Fresh adapters from the discovered map retained 97.05% of the contextual fixed `r32/600` held-out perplexity gain at 43.82% of its adapter size. With identical initialization, minibatch order, dropout seed, and evaluation examples, it beat five same-budget random maps, an exact-budget shuffle of its own rank histogram, and a target-constant `q16/k16/v8` rule; all paired 95% bootstrap intervals excluded zero.

The important caveat is cross-domain transfer. A Text-to-SQL map normalized to the fault-code budget reached PPL 5.9194, only 1.12% behind the native fault-code map (paired difference -0.0662, 95% CI [-0.0849, -0.0485]). In the reciprocal 300-example Text-to-SQL screen, the native map reached PPL 1.7172 versus 1.7429 for a byte-matched fault-code transplant, a 1.47% advantage (paired difference -0.0256, 95% CI [-0.0300, -0.0217]). Both native maps won, but only narrowly, so rank placement includes a small domain-native signal while much of the benefit may still be a transferable Gemma architecture/training prior. The committed screens are `codex/evidence/fault_codes_paired_control_screen_seed42.json` and `codex/evidence/text_to_sql_paired_transfer_screen_seed42.json`; the prior contextual snapshot remains in `codex/evidence/fault_codes_full2700_fullscale_summary.{json,csv}`. These are single-seed diagnostic results, not theorem validation or multi-seed generalization.

## Why Plastic Rank?
Traditional pruning and distillation discard parameters permanently. Plastic Rank began with a different question: can capacity grow, shrink, sleep, wake, and move as experience changes? Skill packs, rank maps, and quality-per-byte experiments remain useful supporting work, but the canonical goal is now the original one: observe and test capacity migration through learning, specialization, forgetting, and relearning.

## Key Capabilities
- Dynamic low-rank factors with deterministic MLX kernels
- Plasticity manager for pruning/waking flows and rank heuristics
- LoRA “skill pack” CLI for training, exporting, and evaluating adapters
- Dynamic Pop Rank gates, frozen heterogeneous continuations, and fresh rank-map ablations
- Bakeoff-native random and shuffled same-budget controls with promotion gates
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
7. Run the learned capacity-migration smoke matrix: `uv run python scripts/learned_capacity_migration_benchmark.py --mode smoke --output-dir out/capacity_migration/learned_mlx`

## Parked Research Track: Forgetting Machine

The controlled spectral Forgetting Machine is **parked, not discarded**. Its
executable prototype, tests, threat-boundary documentation, and experimental DSN
remain in the repository. It demonstrates deletion inside one trusted external
memory vault; it does not test the active capacity-migration thesis or establish
that model weights have forgotten information.

Return to this track after a learned MLX `A -> B -> A` model—not only the
orthogonal reference mechanism—passes the learned-model promotion gate and there is a
concrete need to compare dormant model capacity, overwritten capacity, and
governed external deletion. An explicit project-priority decision can also
unpark it. The durable parking record is
[`codex/parking-lot.md`](codex/parking-lot.md), and the full boundary is in
[`docs/forgetting_machine.md`](docs/forgetting_machine.md).

The retained demonstration remains runnable:

```bash
uv run python scripts/forgetting_machine_demo.py \
  --out-json out/forgetting_machine_demo.json \
  --out-html out/forgetting_machine_demo.html
```

Do not promote this track back into the Current Thesis without updating its DSN
status and recording new evidence in the canonical decision index.

## LoRA Skill Packs
- Default Gemma 4 base: `mlx-community/gemma-4-12B-mxfp8`. It keeps the unified any-to-any MLX architecture while roughly halving the local footprint versus bf16. Use `mlx-community/gemma-4-12B-bf16` as a reference/high-fidelity checkpoint for rank probes or regression comparisons when memory allows.
- Train a pack: `uv run packs create --name domain-demo --base mlx-community/gemma-4-12B-mxfp8 --layers attn.q_proj,attn.k_proj,attn.v_proj --loader auto --rank-strategy gram_energy --target-compression 0.9 --steps 1000 --batch-size 2 --learning-rate 5e-5 --data data/domain_prompts.jsonl --lora-dropout 0.05`
  
  Gemma 4 unified checkpoints load through `mlx-vlm` in `--loader auto`. Per-slice ranks adjust automatically (`q` keeps the requested rank, `k/v` default to the grouped key/value head width). `attn.o_proj` is supported when you want a higher-capacity pack, but budget it explicitly against the size cap. Add `--train-fp16-fallback` if a quantized projection fails geometry checks.
- Check local macOS modality support: `uv run --extra packs packs capabilities --check`. The pack extra includes `mlx-lm`, `mlx-vlm`, and `mlx-audio`; `mlx-vlm` covers Gemma 4 unified image/audio/video prompting, while `mlx-audio` is the dedicated speech IO layer for TTS, STT, and STS workflows around packs.
- Inspect metadata: `uv run packs inspect --name domain-demo`
- Apply safely: `uv run packs apply --name domain-demo --base mlx-community/gemma-4-12B-mxfp8 --dry-run`
- Evaluate: `uv run packs eval --base mlx-community/gemma-4-12B-mxfp8 --pack domain-demo --data-path data/domain_prompts.jsonl --csv results.csv`
- Audit the DLC-style improvement claim: `uv run packs proof --base mlx-community/gemma-4-12B-mxfp8 --pack domain-demo --domain my-domain --train-data data/domain_prompts.jsonl --eval-report results.json`
- After extracting the referenced train/eval JSONL, run a reproducible train/eval/rank-ledger/proof bakeoff from a JSON spec: `uv run --extra packs packs bakeoff --spec codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json --dry-run`
- Bakeoff specs can declare `random_same_budget` and `shuffled_discovered` candidates; each gets a resumable rank-map preflight, and promotion fails when the proposed tradeoff does not beat its included controls.
- `--rank-strategy gram_energy` names the actual Gram spectral-energy heuristic. The old `theorem` spelling remains a compatibility alias, not a theorem-backed selector.
- For prompt/answer JSONL, add `--loss-mode answer` to train/evaluate only assistant answer tokens. This is the preferred mode for diagnostic or maintenance packs where the prompt is context, not a target to imitate.
- Batch evaluation with VRAM/latency guardrails lives in `scripts/demo_plasticity_blocks.py`.
- Compare base vs pack across domains: `uv run packs eval-batch --base mlx-community/gemma-4-12B-mxfp8 --pack domain-demo --input data/domain_prompts.jsonl --batch-size 8,16,32 --sequence-length 256 --thinking strip` (outputs PPL, TPS, first-token ms, VRAM, pack size).
- Gemma 4 bake-off metadata: `uv run --extra packs python scripts/gemma4_smoke.py --metadata-only --out out/gemma4_smoke.json`. Real mxfp8 generation/no-op pack smoke: `uv run --extra packs python scripts/gemma4_smoke.py --models mlx-community/gemma-4-12B-mxfp8 --max-tokens 32 --noop-pack`.
- Instruction-tuned UX smoke: `uv run --extra packs python scripts/gemma4_smoke.py --models mlx-community/gemma-4-12B-it-qat-mxfp8 --chat-template --max-tokens 32`.
- Heavy packs (bigger ranks + larger size cap): add `--profile heavy` to `packs create` when you want higher-capacity domain packs loaded on demand from SSD.
- To force heavier adapters even when auto-rank would stay small, add `--min-rank 16` (or higher; values snap to allowed heavy ranks). To bypass auto-rank entirely for controlled sweeps, use `--rank N`.
- To continue a trained heterogeneous pack with fixed ranks, use `--resume-pack SOURCE_PACK`. To train fresh weights from a discovered heterogeneous rank map, use `--rank-map-from-pack SOURCE_PACK`; for standalone candidate maps, use `--rank-map-json path/to/rank_map.json`.

### IndustryBench Pilot
Use Alibaba's IndustryBench as a small industrial QA pack probe.[^industrybench] The extractor keeps source metadata in JSON fields while letting the training text stay clean.
- Extract a small English split: `uv run python scripts/industrybench_extract.py --language en --source-limit 512 --train-size 128 --eval-size 32 --metadata-mode none --train-out data/industrybench_en_train.jsonl --eval-out data/industrybench_en_eval.jsonl`
- Baseline IT QAT eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --data-path data/industrybench_en_eval.jsonl --chat-template --sequence-length 256 --num-samples 32 --batch-size 1 --out out/industrybench_gemma4_it_baseline.json --csv out/industrybench_gemma4_it_baseline.csv`
- No-shrink heavy pilot pack: `uv run --extra packs packs create --name industrybench-en-gemma4-it-heavy-smoke --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/industrybench_en_train.jsonl --chat-template --steps 20 --batch-size 1 --sequence-length 256 --learning-rate 1e-5 --target-compression 0.7 --profile heavy`
- First result: the pack exported and applied successfully, with q/v rank 64, k rank 32, and size 91.75 MB. On the 32-row held-out split it changed logits (`max_logit_diff=3.0`) but worsened PPL by 1.45%, so this is an end-to-end mechanics proof, not a quality win.

### Industrial Fault-Code Pilot
Use `avneetsingla/industrial-fault-codes-sample` for the first practical industrial maintenance pack.[^fault-codes] It has 3,000 English fault-code rows with `brand`, `code`, `description`, and `solution` fields. License is `cc-by-nc-4.0`, so treat this as research/prototyping data unless licensing is resolved.
- Extract train/eval JSONL: `uv run python scripts/fault_codes_extract.py --train-size 2400 --eval-size 300 --train-out data/fault_codes_train.jsonl --eval-out data/fault_codes_eval.jsonl`
- Baseline answer-only eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --data-path data/fault_codes_eval.jsonl --chat-template --loss-mode answer --sequence-length 256 --num-samples 300 --batch-size 4 --out out/fault_codes_gemma4_it_answer_baseline_300.json --csv out/fault_codes_gemma4_it_answer_baseline_300.csv`
- Rank-16 pilot pack: `uv run --extra packs packs create --name fault-codes-gemma4-it-answer-r16-100 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 100 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 16 --profile heavy --lora-dropout 0.05`
- Best sweep pack: `uv run --extra packs packs create --name fault-codes-gemma4-it-answer-r32-300 --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 300 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --rank 32 --profile heavy --lora-dropout 0.05`
- Pack eval: `uv run --extra packs packs eval --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --pack fault-codes-gemma4-it-answer-r32-300 --data-path data/fault_codes_eval.jsonl --chat-template --loss-mode answer --sequence-length 256 --num-samples 300 --batch-size 4 --out out/fault_codes_gemma4_it_answer_r32_300_eval_300.json --csv out/fault_codes_gemma4_it_answer_r32_300_eval_300.csv`
- Generation check: `uv run --extra packs python scripts/fault_codes_generate_check.py --base mlx-community/gemma-4-12B-it-qat-mxfp8 --pack fault-codes-gemma4-it-answer-r32-300 --eval-data data/fault_codes_eval.jsonl --limit 8 --max-tokens 96 --temperature 0 --chat-template --out out/fault_codes_generation_r32_300_8.json --csv out/fault_codes_generation_r32_300_8.csv`
- DLC proof report for the spectral-key candidate: `uv run --extra packs packs proof --base mlx-community/gemma-4-12B-it-qat-mxfp8 --pack spectral-key-candidate --domain industrial-fault-codes --train-data data/fault_codes_train.jsonl --eval-data data/fault_codes_eval.jsonl --eval-report out/fault_codes_gemma4_it_answer_spectral_key_candidate_eval_300.json --generation-report out/fault_codes_generation_spectral_key_candidate_8.json --ledger-report out/fault_codes_rank_ledger_spectral_key_candidate.json --require-generation --require-ledger --fail-on-regression --out out/fault_codes_domain_pack_proof_spectral_key_candidate.json`
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
- Generate the spectral-key-biased standalone rank map from q/k/v probe JSON: `uv run --extra packs packs rank-map spectral --source-pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-map-600 --q-spectral out/pop_poly_q_all_layers_low_mid_high_seed5.json --k-spectral out/pop_poly_k_all_layers_low_mid_high_seed5.json --v-spectral out/pop_poly_v_all_layers_low_mid_high_seed5.json --profile heavy --out out/pop_poly_rank_map_spectral_auto_balanced.json`
- Train a standalone proposed rank map: `uv run --extra packs packs create --name spectral-key-candidate --base mlx-community/gemma-4-12B-it-qat-mxfp8 --loader mlx-vlm --layers attn.q_proj,attn.k_proj,attn.v_proj --rank-map-json out/pop_poly_spectral_key_rank_map_candidate.json --data data/fault_codes_train.jsonl --chat-template --loss-mode answer --steps 600 --batch-size 1 --sequence-length 256 --learning-rate 5e-5 --profile heavy --lora-dropout 0.05`

Spectral-key candidate result: at 27.38 MB, `spectral-key-candidate` reached answer-token PPL 5.7641 and token accuracy 0.6786 on the 300-row fault-code eval, slightly ahead of the current hetero map at 27.39 MB, PPL 5.7811, accuracy 0.6773. Its 8-example generation solution-overlap matched the hetero/freeze result at 0.3911 and remains below fixed `r32/600` at 0.4025.

The product-style proof artifact is `out/fault_codes_domain_pack_proof_spectral_key_candidate.json`: base PPL 15.4316 to base+pack PPL 5.7641, token accuracy 0.6155 to 0.6786, generation solution-overlap 0.2723 to 0.3911, and attach evidence via `max_logit_diff=27.6875`.

The current 600-step fault-code bakeoff is summarized near the top of this README. Treat it as a quality-positive local result for one industrial domain, not proof of the Pop Rank theorem.

### On-Demand Domain Routing (TTL + LRU)
Run a core model and attach/detach packs on demand using domain labels:
- Domain map JSON (example): `{"core": null, "taxi": "bench-r4"}`
- Requests JSONL (example): `{"domain":"taxi","prompt":"JFK to Midtown fare estimate"}`
- Runtime (CLI): `uv run packs route --base mlx-community/gemma-4-12B-mxfp8 --domain-map run/domain_map.json --input run/requests.jsonl --ttl-seconds 120 --max-recent-domains 8 --probe-forward --out run/route_log.jsonl`
- Runtime (script): `uv run python scripts/domain_router_runtime.py --base mlx-community/gemma-4-12B-mxfp8 --domain-map run/domain_map.json --input run/requests.jsonl --ttl-seconds 120 --max-recent-domains 8 --probe-forward --out run/route_log.jsonl`

## Benchmarks & Utilities
- Compression baseline: `uv run python scripts/compress_llm_mlx.py --hf mlx-community/gemma-4-12B-mxfp8 --out out/gemma4_mxfp8_compressed --svd randomized --batch-size 20`
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

## Research Footnotes
[^pop-theorem]: Vasile Pop, "Relations between ranks of matrix polynomials", arXiv:2010.00634 [math.RA], submitted October 1, 2020. See also Vasile Pop and Alexandru Negrescu, "Three New Proofs of the Theorem rank f(M) + rank g(M) = rank (f,g)(M) + rank [f,g](M)", *Mathematics* 2024, 12(3), 360, https://doi.org/10.3390/math12030360. The theorem gives a rank identity for matrix polynomials; in this repo it is rank-accounting intuition, not validation that Pop Rank improves LoRA quality.

[^industrybench]: Alibaba Multimodal Industrial AI, `alibaba-multimodal-industrial-ai/IndustryBench`, MIT-licensed Hugging Face dataset. Its dataset card requests citation of Bai et al. (2026), "IndustryBench: Probing the Industrial Knowledge Boundaries of LLMs", arXiv:2605.10267.

[^fault-codes]: Avneet Singla, `avneetsingla/industrial-fault-codes-sample`, Hugging Face dataset tagged `cc-by-nc-4.0`. Treat experiments or packs trained on this source as research/prototype artifacts unless separate commercial rights are resolved.

## License
Licensed under the [MIT License](LICENSE). See [NOTICE.md](NOTICE.md) for
third-party dataset/model attribution and generated-data license notes.
