# PopRank Causal Rank-Channel Ablation Reports

Rank-channel ablation reports inspect an exported LoRA pack and produce deterministic interventions for testing whether specific rank channels, adapters, targets, layers, or rank suffixes are causally useful.

The report is conservative:

- without paired eval artifacts, it is `mechanistic_proxy_only`;
- with baseline and ablated eval artifacts, it is `paired_eval`;
- source packs are never mutated.

## What Is Measured

For each adapter channel `i`, the proxy score is the Frobenius norm of that channel's scaled rank-one update:

```text
channel_update_i = (alpha / rank) * A[:, i] outer B[i, :]
proxy_update_fro_i = abs(alpha / rank) * ||A[:, i]|| * ||B[i, :]||
```

This is an intervention ranking signal. It is not a quality claim until the generated ablation pack is evaluated on the same examples as the original pack.

## CLI

Report the top channel ablations:

```bash
uv run packs ablation-report \
  --pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --unit channel \
  --top-k 20 \
  --out out/hetero_channel_ablation_report.json \
  --markdown out/hetero_channel_ablation_report.md \
  --csv out/hetero_channel_ablation_report.csv
```

Generate ablated pack copies for paired evaluation:

```bash
uv run packs ablation-report \
  --pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --unit channel \
  --top-k 8 \
  --ablation-pack-root out/ablations/hetero_channels \
  --out out/hetero_channel_ablation_manifest.json
```

Each generated pack has zeroed rank channels for one ablation unit and keeps the source pack untouched.

Report prefix/suffix ablations:

```bash
uv run packs ablation-report \
  --pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --unit prefix \
  --prefix-rank 8 \
  --targets attn.q_proj,attn.k_proj \
  --top-k 20 \
  --ablation-pack-root out/ablations/hetero_prefix8 \
  --out out/hetero_prefix8_ablation_manifest.json
```

Attach paired eval artifacts:

```bash
uv run packs ablation-report \
  --pack fault-codes-gemma4-it-answer-hetero-r32-init8-min4-full2700-map-600 \
  --unit channel \
  --top-k 8 \
  --baseline-eval out/hetero_eval.json \
  --ablation-eval channel-blocks_28_attn_q_proj-c0003=out/ablations/channel_0003_eval.json \
  --out out/hetero_channel_ablation_paired_report.json \
  --markdown out/hetero_channel_ablation_paired_report.md \
  --csv out/hetero_channel_ablation_paired_report.csv
```

## Units

Supported units:

- `channel`: zero one LoRA rank channel;
- `adapter`: zero all rank channels in one adapter;
- `target`: zero all channels for one target such as `attn.k_proj`;
- `layer`: zero all channels in one transformer layer;
- `prefix`: keep channels before `--prefix-rank` and zero the suffix.

Filters:

- `--targets attn.q_proj,attn.k_proj`;
- `--layers 0,1,2`;
- `--top-k 0` keeps all generated units.

## Report Schema

The JSON report contains:

- `kind: causal_rank_channel_ablation_report`;
- `evidence_status`;
- source pack metadata;
- policy fields;
- adapter/channel proxy summaries;
- `ablations`, each with an `ablation_id`, operation, affected adapters/channels, removed proxy norm, optional generated pack path, and optional paired-eval deltas.

Paired eval deltas include:

- `perplexity_delta`;
- `perplexity_delta_pct`;
- `token_accuracy_delta`;
- `domain_metric_delta`;
- `tokens_per_sec_delta`.

## Limitations

- The report builder does not run evaluation by itself.
- Proxy ranking is based on exported LoRA factor norms, not downstream quality.
- Causal claims require matched baseline and ablated evals on the same examples and settings.
- Generated ablation packs preserve tensor shapes and zero selected rank channels rather than shrinking ranks.
