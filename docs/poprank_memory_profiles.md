# PopRank Local Memory Profiles

The local-device memory profiles define repeatable memory gates for PopRank experiments on Apple Silicon style local machines. They are separate from pack rank profiles such as `lite` and `heavy`.

## Profiles

The built-in profiles are:

| Profile | Total GB | Soft Budget GB | Hard Budget GB | Batch | Sequence Length | Eval Cap | Rank Ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8gb` | 8 | 5.4 | 6.0 | 1 | 128 | 32 | 8 |
| `16gb` | 16 | 11.7 | 13.0 | 1 | 256 | 100 | 16 |
| `32gb` | 32 | 25.2 | 28.0 | 1 | 256 | 300 | 32 |
| `48gb` | 48 | 38.7 | 43.0 | 2 | 512 | 1000 | 64 |

Soft budgets are intended for green-pass runs. Hard budgets allow a warning band for runs that may work but are close to local memory pressure. The reserved memory is deliberately simple and conservative:

```text
hard_budget_gb = total_memory_gb - os_reserved_gb
soft_budget_gb = 0.9 * hard_budget_gb
```

## CLI

List profiles:

```bash
uv run packs device-profiles \
  --out out/device_profiles.json \
  --markdown out/device_profiles.md
```

Build a memory ledger from a pack and an eval artifact:

```bash
uv run packs memory-ledger \
  --pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --eval-report out/fault_codes_gemma4_it_answer_hetero_r32_init8_min4_map_600_eval_300.json \
  --profiles 8gb,16gb,32gb,48gb \
  --out out/hetero_memory_ledger.json \
  --markdown out/hetero_memory_ledger.md \
  --csv out/hetero_memory_ledger.csv
```

Build a ledger for a generated rank-map artifact before training a pack:

```bash
uv run packs memory-ledger \
  --rank-budget-report out/random_same_budget_seed17.json \
  --base-model-gb 9.5 \
  --observed-peak-gb 10.8 \
  --extra-overhead-gb 0.5 \
  --out out/random_same_budget_memory_ledger.json
```

## Ledger Inputs

The ledger can combine:

- pack tensor bytes from `pack.safetensors`;
- rank-map budget reports from `packs rank-map budget-report`, `normalize-budget`, `random-same-budget`, or `shuffled-discovered`;
- `packs eval` reports through `peak_memory_gb`;
- `packs eval-batch` reports through `vram_peak_base` and `vram_peak_pack`;
- generation-check reports through top-level `summary.peak_memory_gb`;
- manual `--observed-peak-gb`;
- optional `--base-model-gb`;
- optional `--host-rss-peak-gb`;
- optional `--extra-overhead-gb`.

The assessed peak is:

```text
max(
  observed_mlx_peak + extra_overhead,
  host_rss_peak,
  base_model_estimate + adapter_tensor_bytes + extra_overhead
)
```

If no runtime peak, RSS peak, or base-model estimate is available, the ledger status is `unknown` even when adapter bytes are known. Adapter size alone is not enough to claim a local device fit.

## Status

For each profile:

- `pass`: assessed peak is within the soft budget;
- `warn`: assessed peak exceeds the soft budget but fits within the hard budget;
- `fail`: assessed peak exceeds the hard budget;
- `unknown`: not enough memory evidence was provided.

The report also records `smallest_soft_fit_profile` and `smallest_hard_fit_profile`.

## Limitations

- The ledger does not run a model or measure memory by itself; it consumes artifacts or manual values.
- MLX peak memory in current eval paths is reset after model load, so it may not capture full load-plus-eval pressure.
- Host RSS is optional and must be supplied externally.
- The profile defaults are local run defaults and gates, not proof that a specific base model exists or is appropriate for every profile.
