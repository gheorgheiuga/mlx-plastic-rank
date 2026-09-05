# PopRank / Plastic Rank Repository Audit

Date: 2026-06-21

**Historical snapshot.** This audit predates the September correctness repairs,
dependency cleanup and retired demos. Its paths and backlog are not current
instructions. Use [the consolidation DSN](../codex/dsn/dsn-20260905-prototype-consolidation.md)
and [current runbook](../codex/runbook.md) for active scope and commands.

Scope: current repository state only. This is an audit and implementation plan; it does not change training, evaluation, rank-map, or experiment code.

## 1. Current training entrypoints

The primary LoRA training entrypoint is the `packs` console script declared in `pyproject.toml`, which resolves to `mlx_plastic_rank.packs.cli:main`. Its `create` subcommand is implemented by `cmd_create` in `src/mlx_plastic_rank/packs/cli.py`.

`packs create` supports:

- model loading through `--loader auto|mlx-lm|mlx-vlm`;
- target selection through `--layers`, defaulting to `attn.q_proj,attn.k_proj,attn.v_proj`;
- fixed-rank training through `--rank`;
- automatic rank selection through `--rank-strategy stable|gram_energy`, `--target-compression`, `--rank-eps`, and `--min-rank` (`theorem` is retained only as a legacy alias);
- frozen continuation from an existing pack through `--resume-pack`;
- fresh training from an existing pack's exported heterogeneous map through `--rank-map-from-pack`;
- fresh training from a standalone rank-map JSON through `--rank-map-json`;
- dynamic-rank discovery through `--dynamic-rank` plus warmup, interval, min-rank, grow-threshold, and prune-threshold controls;
- training provenance persistence in `meta.json` under `training_data` and `training_config`.

The repeatable experiment entrypoint is `packs bakeoff --spec ...`, implemented in `src/mlx_plastic_rank/packs/bakeoff.py`. It plans and runs `create`, `eval`, `rank-ledger`, and `proof` phases for each candidate in a JSON spec. The current candidate modes are `fixed_rank`, `dynamic_rank`, `rank_map_from_candidate`, `rank_map_from_pack`, `rank_map_json`, and `resume_pack`.

There are also legacy/toy Plastic Rank entrypoints:

- `plastic_rank.py`, which trains a tiny `PlasticBlock` demo with `PlasticityManager`;
- `scripts/demo_plasticity_blocks.py`, which emits JSONL plasticity events and a compact table;
- `scripts/bench_memory.py`, which estimates dense vs quantized low-rank storage for synthetic matrices rather than training LoRA packs.

## 2. Current evaluation entrypoints

The main held-out evaluation command is `packs eval`, implemented by `cmd_eval` in `src/mlx_plastic_rank/packs/cli.py`. It evaluates the base row and, when `--pack` is provided, a base+pack row. Outputs are printed as JSON and optionally written to JSON/CSV.

`packs eval` reports:

- `model`;
- `pack`;
- `pack_size_bytes`;
- `size_mb`;
- `load_time_ms`;
- `perplexity`;
- `perplexity_se`;
- `eval_time_s`;
- `tokens_per_sec`;
- `peak_memory_gb`;
- `max_logit_diff`;
- `token_accuracy`;
- `loss_mode`;
- `ppl_delta_pct`;
- `domain_metric`.

Other evaluation/reporting entrypoints:

- `packs eval-batch` evaluates grouped prompt domains across batch sizes and reports PPL, TPS, first-token latency, peak VRAM, and pack size.
- `packs rank-ledger` builds algebraic rank ledgers for one pack or pairwise rank-composition comparisons between two packs.
- `packs proof` builds a pass/fail domain-pack proof from eval, generation, and ledger artifacts.
- `packs bakeoff` orchestrates all of the above and writes compact bakeoff summaries.
- `scripts/fault_codes_generate_check.py` runs base and optional pack generation checks for fault-code prompts, scoring brand/code presence and solution-keyword overlap.
- `scripts/gemma4_smoke.py` is a Gemma 4 runtime smoke tool, useful for load/generation sanity checks rather than PopRank-controlled evaluation.

## 3. Current LoRA rank-map format

The exported pack format is:

- `packs/<pack>/meta.json`;
- `packs/<pack>/pack.safetensors`.

`meta.json` is serialized from `PackMetadata` in `src/mlx_plastic_rank/packs/io.py`. The relevant rank fields are:

```json
{
  "profile": "heavy",
  "rank_map": {
    "blocks.0.attn.q_proj": 4,
    "blocks.0.attn.k_proj": 4
  },
  "alpha_map": {
    "blocks.0.attn.q_proj": 8.0,
    "blocks.0.attn.k_proj": 8.0
  },
  "target_layers": [
    "blocks.0.attn.q_proj",
    "blocks.0.attn.k_proj"
  ]
}
```

Adapter keys use the form `blocks.<layer_index>.attn.<q|k|v|o>_proj`. Exported ranks are integer values on the active allowed rank ladder for the pack profile. `alpha_map` uses the same keys and must be either `0.0` or `2 * rank`.

`pack.safetensors` uses tensor keys:

- `<adapter>.lora.A`, shape `[out_dim, rank]`, fp16;
- `<adapter>.lora.B`, shape `[rank, in_dim]`, fp16;
- `<adapter>.lora.alpha`, scalar fp32.

Standalone `--rank-map-json` accepts either a bare object of adapter/target keys to ranks or an object with `rank_map` and optional `alpha_map`. The loader defaults missing alpha values to `2 * rank`, rejects unsupported ranks for the selected profile, and rejects alpha values other than `0.0` or `2 * rank`.

The spectral rank-map generator emits an extended report with `kind`, `name`, `source_pack`, `policy`, budget fields, promotions/reductions, plus final `rank_map` and `alpha_map`. `packs create --rank-map-json` consumes only the rank and alpha maps.

## 4. Current dynamic rank discovery mechanism

Dynamic discovery is enabled by `packs create --dynamic-rank`. In this mode, `--rank` is the maximum allocated LoRA rank, while `--dynamic-initial-rank` controls the active prefix width at initialization.

The mechanism is:

1. `cmd_create` builds a max-rank `rank_map` and passes `initial_active_rank` to `LoRAManager.initialize_adapters`.
2. Each `SliceLoRA` stores a `gates` vector. The active prefix is set to 1.0; the inactive suffix is set to 0.0.
3. `SliceLoRA.delta` multiplies the projected rank channels by `gates`, so inactive channels do not contribute to the forward pass.
4. `train_lora` and `train_lora_supervised` call `_maybe_adjust_dynamic_ranks` after each optimizer step.
5. After warmup and on configured intervals, `LoRAManager.adjust_dynamic_ranks` computes a per-adapter utility signal as `sum(norm(A_col) * norm(B_row))` over active channels.
6. Signals are compared against the global max signal. Adapters grow one allowed-rank step when `signal >= grow_threshold * global_signal`; adapters shrink one allowed-rank step when `signal <= prune_threshold * global_signal`.
7. `export_active_pack` calls `SliceLoRA.export_arrays`, which writes only the active prefix columns/rows and records the exported active rank in metadata.

Current limitations:

- dynamic rank events are printed to stdout but are not persisted as a structured training artifact;
- metadata records the dynamic settings, not the full rank trajectory;
- grow/shrink is local greedy thresholding, not an equal-budget optimizer;
- exported rank maps record active ranks, but not why each adapter reached that rank.

## 5. Current ledger/report artifact format

`packs rank-ledger` single-pack JSON has this top-level shape:

```json
{
  "kind": "pack_rank_ledger",
  "rank_tol": 1e-5,
  "pack_dir": "packs/example",
  "metadata": {},
  "summary": {},
  "by_target": [],
  "adapters": []
}
```

`summary` contains:

- `adapter_count`;
- `declared_rank`;
- `effective_rank`;
- `rank_slack`;
- `rank_efficiency`;
- `bytes`;
- `fro_norm`;
- `bytes_per_effective_rank`.

Each adapter row contains:

- `adapter`;
- `target`;
- `declared_rank`;
- `effective_rank`;
- `rank_slack`;
- `rank_efficiency`;
- `alpha`;
- `scale`;
- `shape`;
- `params`;
- `bytes`;
- `fro_norm`;
- `spectral_norm`;
- `stable_rank`;
- up to eight `singular_values`.

The CSV form drops the singular-value list and flattens `shape`.

Pairwise `packs rank-ledger --compare` emits `kind: "pack_rank_comparison"` with left/right metadata, left-only/right-only adapter lists, a `summary`, and per-shared-adapter `pairs`. Pair rows include composition rank, rank savings, row/column overlap ranks, Frobenius cosine, and leading composition singular values. The comparison CSV drops the singular-value list.

`packs proof` emits `kind: "domain_pack_proof"` with `status`, `claim`, domain/base/pack identifiers, metrics, metadata, and requirement rows. It checks pack artifact presence, training data evidence, eval-pair presence, pack attachment via `max_logit_diff`, held-out metric improvement, optional generation improvement, and optional rank-ledger validity.

`packs bakeoff` writes a `pack_bakeoff_summary` JSON and CSV. The code-generated JSON contains `base_metrics`, candidate `rows`, `winner_quality`, `winner_tradeoff`, and `promotion_gates`. Candidate rows include size, rank, perplexity, accuracy, proof status, and improvement-per-MB fields. The committed `codex/evidence/fault_codes_full2700_fullscale_summary.json` is a compact evidence snapshot with the same core row fields but a custom `kind`.

## 6. Where adapter size is calculated

Exact tensor-byte counting is centralized in `src/mlx_plastic_rank/packs/inspection.py`:

- `summarize_pack(pack_dir)` loads `pack.safetensors`, sums `arr.nbytes`, returns tensor rows, total params, total bytes, and non-LoRA tensor names.
- `size_limit_for(metadata)` returns profile/base-size limits. `lite` defaults to 10 MiB, `heavy` defaults to 512 MiB, with a small legacy base-specific table for Llama 3 8B 4-bit.
- `allowed_ranks_for(profile)` defines `lite = (2, 4, 8)` and `heavy = (2, 4, 8, 16, 32, 64)`.

The exact tensor-byte path is used by `packs create`, `packs apply`, `packs inspect`, `packs list`, `packs eval-batch`, and `packs proof`.

Important secondary paths:

- `packs eval` reports `pack_size_bytes` using `os.path.getsize(pack.safetensors)`, which is file size rather than summed tensor `nbytes`.
- `packs rank-ledger` reports per-adapter `bytes` by summing grouped tensor `nbytes`, not full SafeTensors file size.
- `rank_map.py` estimates LoRA storage as `params * 2 + len(rank_map) * 4`, which covers fp16 A/B values plus fp32 alpha values but not SafeTensors file overhead or `meta.json`.
- `cmd_inspect` prints an expected-size estimate from tensor shapes, also using fp16 params plus fp32 alpha.

## 7. Where MLX memory or runtime memory is measured

Current MLX peak-memory measurement exists in evaluation and generation paths:

- `_evaluate_perplexity` calls `mx.reset_peak_memory()` before looping over eval batches and returns `mx.get_peak_memory() / 1024**3` as `vram_peak`.
- `_evaluate_supervised_perplexity` does the same for answer-only JSONL evaluation.
- `packs eval` exposes this as `peak_memory_gb`.
- `packs eval-batch` exposes this as `vram_peak_base` and `vram_peak_pack`.
- `scripts/fault_codes_generate_check.py` calls `mx.reset_peak_memory()` before generation and writes `peak_memory_gb` in the summary.
- `scripts/gemma4_smoke.py` has Gemma runtime peak-memory reporting for smoke runs.

Current gaps:

- no host RSS / process memory tracking;
- no training-time memory metric;
- eval memory is reset after model load, so it mostly measures eval-loop peak, not full load-plus-eval footprint;
- no device-profile preflight or pass/fail memory budget;
- `scripts/bench_memory.py` is a storage estimator for synthetic low-rank factors, not a runtime memory profiler despite the README label.

## 8. Where fixed r16/r32 experiments are configured

Fixed rank experiments are configured in three places:

- README and `codex/runbook.md` command examples for r16/r32 fault-code runs;
- `codex/bakeoffs/fault_codes_gemma4_it_fullscale.json`;
- `codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json`.

Both committed bakeoff specs define:

- `fixed_r16_600`, mode `fixed_rank`, rank `16`, marked as `small_reference`;
- `fixed_r32_600`, mode `fixed_rank`, rank `32`, marked as `quality_reference`;
- `dynamic_r32_init8_min4_150`, mode `dynamic_rank`, max rank `32`;
- `hetero_r32_init8_min4_map_600`, mode `rank_map_from_candidate`, marked as `tradeoff_candidate`.

The fixed runs are not a separate benchmark framework. They are bakeoff candidates that expand into `packs create --rank 16` or `packs create --rank 32`.

## 9. Next research-phase readiness

### Equal-budget heterogeneous rank maps

Implemented in `packs/rank_budget.py` and the `packs rank-map` CLI family. The shared layer now exposes shape-aware parameter/byte accounting, fixed-rank and rank-map targets, fixed-r32 percentages, profile/rank validation, deterministic normalization, and consumable `rank_map`/`alpha_map` artifacts. Bakeoff control candidates use the same interface before training. The conservative solver may leave budget slack; it is not a globally optimal allocator.

### Random same-budget rank maps

Implemented as `packs rank-map random-same-budget` and as the bakeoff candidate mode `random_same_budget`. The generator is seeded, shape-aware, budget-normalized, and writes both an audit report and a consumable rank map. Bakeoff records the control type, source pack, seed, and map artifact, then requires the tradeoff candidate to beat included controls before promotion by default.

Multi-seed expansion and aggregate confidence intervals remain open.

### Shuffled rank maps

Implemented as `packs rank-map shuffled-discovered` and as the bakeoff candidate mode `shuffled_discovered`. It seed-shuffles the discovered rank multiset, normalizes the shape-aware byte budget when adapter geometries differ, emits auditable artifacts, and runs through the same resumable bakeoff preflight as the random control.

Optional constrained shuffles by target, layer band, or shape class remain open.

### Causal rank-channel ablations

Implemented as `packs ablation-report` for channel, adapter, target, layer, and prefix interventions. It writes immutable ablated pack copies and can attach paired eval deltas while labeling proxy-only output separately from paired causal evidence.

Automatic evaluation orchestration, physical rank slicing controls, and confidence intervals remain open.

### 8GB/16GB/32GB/48GB device profiles

Implemented in `packs/device_profiles.py` and exposed through `packs memory-ledger`. The 8GB/16GB/32GB/48GB profiles are separate from adapter `lite`/`heavy` limits and produce feasibility reports from base, adapter, overhead, and optional observed-memory inputs.

Automatic training-time RSS capture and bakeoff promotion gates tied to observed profile budgets remain open.

### Statistical multi-seed reports

Missing. The CLI accepts a single `--seed`, and bakeoff specs can pass one seed into `packs create`, but there is no multi-seed orchestration or aggregate statistical report.

Needed:

- bakeoff schema support for `seeds`;
- deterministic pack/output naming per candidate and seed;
- aggregation over runs with mean, standard deviation, standard error, confidence intervals, and pairwise deltas;
- paired comparisons between discovered maps and random/shuffled controls;
- bootstrap or nonparametric summaries over eval examples, especially for answer-only loss where `perplexity_se` is currently `0.0`;
- pass/fail gates that require hetero maps to beat random/shuffled controls across seeds, not just one run.

## Implementation status and remaining file-level plan

1. Implemented: `src/mlx_plastic_rank/packs/rank_budget.py`.

   Centralizes adapter-shape loading, parameter/byte estimates, allowed-rank validation, and bounded budget matching.

2. Implemented: extend `src/mlx_plastic_rank/packs/rank_map.py`.

   Equal-budget, random same-budget, shuffled, and spectral policies now share the budget layer.

3. Implemented: extend `src/mlx_plastic_rank/packs/cli.py`.

   The `packs rank-map` commands write a common consumable `rank_map`/`alpha_map` format plus generator metadata.

4. Partially implemented: extend `src/mlx_plastic_rank/packs/bakeoff.py`.

   The resumable preflight map-generation phase and first-class random/shuffled control modes are implemented. Add `seeds` support that expands candidate IDs and pack names deterministically, then writes both per-run and aggregate summaries.

5. Implemented: `src/mlx_plastic_rank/packs/ablation.py`.

   Temporary channel, adapter, target, layer, and prefix interventions keep source packs immutable and write separate ablation artifacts.

6. Implemented: `src/mlx_plastic_rank/packs/device_profiles.py`.

   Defines 8GB/16GB/32GB/48GB runtime profiles and emits requested-versus-observed memory feasibility artifacts.

7. Add `src/mlx_plastic_rank/packs/stats.py`.

   Centralize aggregate statistics, paired deltas, bootstrap CIs, and control comparisons. Use this from bakeoff summaries and future proof gates.

8. Continue test expansion.

   Budget equality, seeded random reproducibility, shuffle preservation, ablation pack immutability, device profiles, and bakeoff control preflights are covered. Add multi-seed aggregation coverage with the future stats module.

9. Extend the experimental specs under `codex/bakeoffs/`.

   The fault-code and text-to-SQL specs now include fixed references, discovered hetero maps, and equal-budget random/shuffled controls. Add multi-seed reporting and optional ablation/profile gates while keeping decision status experimental until those runs complete.
