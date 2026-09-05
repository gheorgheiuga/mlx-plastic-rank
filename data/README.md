# Data Directory

`data/` is ignored by Git except for this README. Use it for local generated
training and evaluation JSONL files, not for checked-in datasets.

Dataset downloads are optional. Extractors using Hugging Face `datasets` need
the `data` extra on each invocation, for example:

```sh
uv run --locked --extra data python scripts/text_to_sql_extract.py --help
uv run --locked --extra data python scripts/fault_codes_extract.py --help
```

The fault-code dataset-viewer path uses the standard library and does not need
that extra. Local JSONL pack loading and the default lifecycle demo do not
require `datasets`. Keep training and held-out files separate.

Before publishing or sharing generated data:

- Keep `source_dataset`, `source_dataset_url`, and license fields from the
  extraction scripts.
- Update `NOTICE.md` when a source dataset, model, license, or citation is
  referenced by repository docs/scripts or shipped with a published artifact.
- Treat `fault_codes_*.jsonl` as CC BY-NC 4.0 derived research/prototype data
  unless commercial rights are resolved.
- Treat `industrybench_en_*.jsonl` as MIT-licensed IndustryBench derived data
  and cite the upstream IndustryBench paper for published research.
- Treat `text_to_sql_*.jsonl` as Apache-2.0 derived data from
  `gretelai/synthetic_text_to_sql`; preserve the extractor's source and license
  fields if publishing an artifact.

Current local generators:

- `scripts/fault_codes_extract.py` -> `fault_codes_train.jsonl`,
  `fault_codes_eval.jsonl`; pass `--train-size 2700 --eval-size 300` for the
  full local `fault_codes_train_full2700.jsonl` /
  `fault_codes_eval_full300.jsonl` split used by the June 2026 bakeoff.
- `scripts/industrybench_extract.py` -> `industrybench_en_train.jsonl`,
  `industrybench_en_eval.jsonl`
- `scripts/text_to_sql_extract.py` -> `text_to_sql_train_10000.jsonl`,
  `text_to_sql_eval_1000.jsonl`
