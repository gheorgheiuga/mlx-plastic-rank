# Third-Party Notices

Last checked: 2026-06-10.

`mlx-plastic-rank` is licensed under the MIT License in `LICENSE`. This notice
covers external data and model resources referenced by repository docs,
examples, and local experiment scripts.

No third-party datasets, generated JSONL files, model checkpoints, or dependency
source trees are vendored in this repository. Before publishing generated data,
trained packs, checkpoint-derived artifacts, environments, or binary bundles,
carry the applicable upstream license, attribution, and citation into that
artifact.

## External Datasets

- The fault-code pilot references the Hugging Face dataset
  `avneetsingla/industrial-fault-codes-sample`
  (https://huggingface.co/datasets/avneetsingla/industrial-fault-codes-sample).
  The upstream license tag is `cc-by-nc-4.0` / CC BY-NC 4.0
  (https://creativecommons.org/licenses/by-nc/4.0/). Treat local data and packs
  trained from it as research or prototype artifacts unless separate commercial
  rights are resolved.

- The IndustryBench pilot references the Hugging Face dataset
  `alibaba-multimodal-industrial-ai/IndustryBench`
  (https://huggingface.co/datasets/alibaba-multimodal-industrial-ai/IndustryBench).
  The upstream license tag is `mit`. If you use this dataset in research or
  published evaluations, cite: Bai et al. (2026), "IndustryBench: Probing the
  Industrial Knowledge Boundaries of LLMs", arXiv:2605.10267
  (https://arxiv.org/abs/2605.10267).

- The Text-to-SQL replication spec references the Hugging Face dataset
  `gretelai/synthetic_text_to_sql`
  (https://huggingface.co/datasets/gretelai/synthetic_text_to_sql). At the
  check date above, the upstream license tag is `apache-2.0`. Generated local
  JSONL files from `scripts/text_to_sql_extract.py` should preserve the source
  dataset, attribution, and license fields.

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
