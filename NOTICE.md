# Third-Party Notices

Last checked: 2026-06-09.

`mlx-plastic-rank` is licensed under the MIT License in `LICENSE`. This notice
covers external data and model resources referenced by the repository or used by
local generated artifacts. Runtime Python dependencies are not vendored in this
repository; check upstream package metadata before redistributing an environment
or binary bundle.

## Data

- `data/fault_codes_*.jsonl`, when generated with
  `scripts/fault_codes_extract.py`, is derived from the Hugging Face dataset
  `avneetsingla/industrial-fault-codes-sample`
  (https://huggingface.co/datasets/avneetsingla/industrial-fault-codes-sample).
  The upstream license tag is `cc-by-nc-4.0` / CC BY-NC 4.0
  (https://creativecommons.org/licenses/by-nc/4.0/). Treat this data and packs
  trained from it as research or prototype artifacts unless separate commercial
  rights are resolved.

- `data/industrybench_en_*.jsonl`, when generated with
  `scripts/industrybench_extract.py`, is derived from the Hugging Face dataset
  `alibaba-multimodal-industrial-ai/IndustryBench`
  (https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench).
  The upstream license tag is `mit`. If you use this dataset in research or
  published evaluations, cite: Bai et al. (2026), "IndustryBench: Probing the
  Industrial Knowledge Boundaries of LLMs", arXiv:2605.10267
  (https://arxiv.org/abs/2605.10267).

- `data/domain_prompts.jsonl` is treated as local project sample data because no
  external source is recorded in this repository. If it is regenerated from a
  third-party source, add the source, license, and attribution here before
  publishing or sharing the generated file.

## Model Checkpoints

The repository references model checkpoints in docs, tests, and scripts, but it
does not vendor checkpoint weights. Verify the upstream card and license before
redistributing weights, generated packs, or derived artifacts.

- `mlx-community/gemma-4-12B-mxfp8` and
  `mlx-community/gemma-4-12B-bf16` are MLX conversions of `google/gemma-4-12B`.
  Their Hugging Face cards list `apache-2.0` and link to the Gemma 4 license
  terms (https://ai.google.dev/gemma/docs/gemma_4_license).

- `mlx-community/gemma-4-12B-it-qat-mxfp8` is used in local Gemma 4 IT QAT
  experiments. At the check date above, its Hugging Face API/card did not expose
  an explicit license tag. Treat it as subject to the upstream Gemma 4 / Google
  model terms until the card is made explicit, and do not redistribute weights or
  packs trained from it without confirming the applicable terms.

- `mlx-community/Qwen2.5-1.5B-Instruct-4bit` is an MLX checkpoint whose
  Hugging Face card lists `apache-2.0` and links to the upstream Qwen license:
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE.

- Historical Qwen3 pilot examples in contributor docs may reference
  `qwen3-4b-2507-mlx-4bit`. Confirm the current upstream repository and license
  before publishing weights, packs, or evaluation artifacts based on that model.
