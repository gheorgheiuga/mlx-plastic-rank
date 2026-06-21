# PopRank Rank-Budget Core

The rank-budget module gives PopRank experiments a shared way to compare fixed, discovered, random, shuffled, and spectral rank maps under the same adapter-byte budget. It does not train adapters and does not change `packs create` behavior.

## Formulas

For one LoRA adapter with rank `r`, input dimension `in_dim`, and output dimension `out_dim`:

```text
params = r * (in_dim + out_dim)
A_bytes = out_dim * r * dtype_bytes
B_bytes = r * in_dim * dtype_bytes
alpha_bytes = alpha_dtype_bytes
adapter_tensor_bytes = A_bytes + B_bytes + alpha_bytes
```

The current repository convention is:

```text
alpha = 0.0 or 2 * rank
```

By default, A/B tensors are budgeted as fp16 and alpha is budgeted as fp32, matching the emitted pack tensor convention. The module can also account A/B and alpha as fp16, bf16, or fp32.

For a rank map:

```text
total_params = sum(adapter params)
total_tensor_bytes = sum(adapter_tensor_bytes)
total_bytes = total_tensor_bytes + optional_file_overhead_bytes
budget_slack_bytes = target_budget_bytes - total_bytes
```

## CLI Examples

Budget the fixed r16 map over the adapter set and shapes from an existing pack:

```bash
uv run packs rank-map budget-report \
  --source-pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --fixed-rank 16 \
  --profile heavy \
  --out out/r16_budget.json \
  --markdown out/r16_budget.md \
  --rank-map-out out/r16_rank_map.json
```

Validate a discovered heterogeneous map from a pack:

```bash
uv run packs rank-map validate \
  --source-pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --out out/hetero_validation.json \
  --markdown out/hetero_validation.md \
  --rank-map-out out/hetero_rank_map.json
```

Normalize an existing discovered map to 40 percent of the fixed r32 adapter-byte budget:

```bash
uv run packs rank-map normalize-budget \
  --source-pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --target fixed-r32-percent \
  --target-fixed-r32-pct 40 \
  --out out/hetero_40pct_r32_budget.json \
  --markdown out/hetero_40pct_r32_budget.md \
  --rank-map-out out/hetero_40pct_r32_rank_map.json
```

Generate a seeded random same-budget control. The source pack or `--rank-map-json` supplies the reference adapter set and byte budget:

```bash
uv run packs rank-map random-same-budget \
  --source-pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --seed 17 \
  --out out/random_same_budget_seed17.json \
  --markdown out/random_same_budget_seed17.md \
  --rank-map-out out/random_same_budget_seed17_rank_map.json
```

Generate a shuffled discovered control. The command permutes discovered ranks across adapters, then normalizes if the shape-aware byte budget changes:

```bash
uv run packs rank-map shuffled-discovered \
  --source-pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --seed 17 \
  --out out/shuffled_discovered_seed17.json \
  --markdown out/shuffled_discovered_seed17.md \
  --rank-map-out out/shuffled_discovered_seed17_rank_map.json
```

Normalize a source map to the same budget as another discovered map:

```bash
uv run packs rank-map normalize-budget \
  --source-pack source-pack \
  --rank-map-json out/source_candidate.json \
  --target rank-map \
  --target-pack discovered-reference-pack \
  --out out/source_same_budget.json \
  --markdown out/source_same_budget.md \
  --rank-map-out out/source_same_budget_rank_map.json
```

The `--rank-map-out` file is consumable by:

```bash
uv run packs create \
  --name same-budget-candidate \
  --base mlx-community/gemma-4-12B-it-qat-mxfp8 \
  --loader mlx-vlm \
  --layers attn.q_proj,attn.k_proj,attn.v_proj \
  --rank-map-json out/source_same_budget_rank_map.json \
  --data data/fault_codes_train_full2700.jsonl \
  --chat-template \
  --loss-mode answer \
  --steps 600 \
  --batch-size 1 \
  --sequence-length 256 \
  --learning-rate 5e-5 \
  --profile heavy
```

## Tensor Bytes Versus File Size

Rank-budget reports keep tensor bytes separate from file size.

Tensor bytes are the shape-derived LoRA payload:

- A matrix bytes;
- B matrix bytes;
- alpha scalar bytes.

SafeTensors files also include container metadata and alignment overhead. That overhead is useful for disk/storage reporting, but it should not be mixed with the mathematical adapter budget unless an experiment explicitly wants to include it. The CLI exposes `--file-overhead-bytes` for that case, while preserving `tensor_bytes` separately in the report.

## Why Equal-Budget Comparison Matters

PopRank claims are about whether a rank allocation is useful, not merely whether a larger adapter has more capacity. A discovered heterogeneous map can only be compared fairly against controls when the adapter-byte budget is controlled.

The minimum useful controls are:

- fixed r16 and fixed r32 maps budgeted over the same adapter set;
- the discovered heterogeneous map;
- same-budget random maps;
- same-budget shuffled maps;
- spectral or other proposed rank maps normalized to the same budget.

Without equal-budget controls, a quality win may just be a size win. Without random and shuffled same-budget controls, a discovered rank map may be indistinguishable from any arbitrary allocation at that size.

## Known Limitations

- The normalization solver is deterministic and conservative, not globally optimal. It never exceeds the target budget unless `--allow-over-budget` is used, but it may leave slack.
- Current normalization starts from an existing map, demotes if over budget, then promotes while changes still fit.
- Shape discovery from a pack only covers adapters present in that pack. Use a source pack with the intended adapter universe when comparing maps.
- Random and shuffled controls are generated as rank-map artifacts, but bakeoff specs do not yet have first-class candidate modes for them.
- A shuffled discovered control preserves the discovered rank multiset before normalization. If q/k/v/o adapter shapes differ, the final normalized map may need rank promotions or demotions to stay within the byte budget.
- Device-profile feasibility now lives in `packs memory-ledger`; the rank-budget module itself still only accounts adapter bytes.
- Training-time memory and host RSS are not measured automatically.
- Causal rank-channel ablation reports now live in `packs ablation-report`; the rank-budget module itself does not run ablations.
