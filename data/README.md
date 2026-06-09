# Data Directory

`data/` is ignored by Git except for this README. Use it for local generated
training and evaluation JSONL files, not for checked-in datasets.

Before publishing or sharing generated data:

- Keep `source_dataset`, `source_dataset_url`, and license fields from the
  extraction scripts.
- Update `NOTICE.md` if the source dataset, model, license, or citation changes.
- Treat `fault_codes_*.jsonl` as CC BY-NC 4.0 derived research/prototype data
  unless commercial rights are resolved.
- Treat `industrybench_en_*.jsonl` as MIT-licensed IndustryBench derived data
  and cite the upstream IndustryBench paper for published research.

Current local generators:

- `scripts/fault_codes_extract.py` -> `fault_codes_train.jsonl`,
  `fault_codes_eval.jsonl`
- `scripts/industrybench_extract.py` -> `industrybench_en_train.jsonl`,
  `industrybench_en_eval.jsonl`
