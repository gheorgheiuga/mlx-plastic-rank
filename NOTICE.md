# Third-Party Notices

Last checked: 2026-07-13.

`mlx-plastic-rank` is licensed under the MIT License in `LICENSE`. This notice
covers external data and model resources referenced by repository docs,
examples, and local experiment scripts.

No third-party datasets, generated JSONL files, model checkpoints, or dependency
source trees are vendored in this repository. Before publishing generated data,
trained packs, checkpoint-derived artifacts, environments, or binary bundles,
carry the applicable upstream license, attribution, and citation into that
artifact.

## Repository-Generated Synthetic Evidence

`codex/evidence/capacity_migration_reference_seed0_9.*` and
`codex/evidence/capacity_migration_learned_dense_seed1_10.*` are compact metrics
generated entirely from seeded local synthetic fixtures. They do not
contain third-party data, downloaded model weights, prompts, or checkpoints.
Their generator entry points and protocol boundaries are recorded in the
corresponding DSN and benchmark scripts. The learned V1 snapshot was produced
from a dirty worktree based on the revision recorded in its provenance section.
The retained outputs are hash-bound, but that base revision and the nearest
post-run source hashes cannot independently reconstruct or prove the exact
uncommitted generator state.

The September 2026 workflow regression fixture in `tests/packs/test_pack_workflow.py`
uses locally authored synthetic JSONL and a tiny in-memory model, with temporary
weights and reports confined to the test directory. It contains no third-party
training examples or downloaded checkpoints. The updated fault-code generation
helper records input and pack content identities in new output metadata; it does
not change dataset attribution or retroactively modify existing artifacts. This
implementation note does not update the upstream license check date above.

`codex/evidence/svd_workspace_20260905.json` contains 18 measurements of locally
generated Gaussian matrices, produced by `scripts/bench_svd_workspace.py` using
seeds 42–44. It includes no external dataset, model weights, prompts or generated
answers. Its metadata records matrix identities, source/generator hashes and
environment versions; the comparison baseline is an earlier MIT-licensed
revision of this repository. These are engineering measurements only.

`codex/research/gradient-agreement/seed-audit.json` is a local inventory of seed
fields and content hashes in retained synthetic experiment metadata. It contains
no training examples, external dataset or checkpoint. The adjacent proposed
protocol reuses the repository's synthetic task design and cites RigL and
AdaLoRA as related research; it includes no copied third-party implementation.

`codex/evidence/gradient_agreement_development_seed31_35.json` contains local
development measurements, source/output identities and repeatability checks from
`scripts/gradient_agreement_benchmark.py`. Its nine-condition matrix, synthetic
arrays and source snapshots remain under ignored `out/capacity_migration/` paths.
The generator writes the local synthetic origin, exact source bytes, environment,
commands and hashes into each package. The committed summary is derived from
those packages and records its audit method; it contains no external dataset or
weights and makes no confirmatory claim. These notes do not update the upstream
license check date above.

## External Datasets

Prototype consolidation (2026-09-05) moves the existing extractors' optional
`datasets` loader to the `data` extra. Their row construction, source/license
metadata and split generation are unchanged; no data was regenerated. The new
default lifecycle demo uses seeded synthetic residuals and prints only numerical
checks, with no external data or model weights. This does not refresh upstream
license verification dates.

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


## September prototype diagnostics

`research/baseline_diagnostic.py` derives dense reference coefficients and
measurements from the unchanged seed 31–35 archives of the local gradient-
agreement development package. No external data, teacher weights or new task
samples enter fitting. Outputs identify the original receipts, exact input and
source bytes, NumPy/runtime versions, command and protocol. Generated artifacts
remain under ignored `out/baseline_diagnostic/`; any committed summary contains
only derived measurements and identities. This does not refresh upstream license
checks or establish a learned allocation result.

`codex/evidence/baseline_diagnostic_seed31_35.json` records the completed twenty-fit
diagnostic and independent receipt/coefficient audit. The source snapshots in
the ignored result package preserve the attribution and declaration before the
first retained-data fit; the current result documentation was written afterwards.

`codex/evidence/forward_workspace_20260905.json` records twelve isolated synthetic
forward measurements from `scripts/bench_forward.py`, including timing follow-ups.
Seeded factors and inputs are locally generated. Baseline methods come from
MIT-licensed repository revision `1c308ec`; exact source and raw timing records
remain under ignored `out/forward_review_20260905/`. No third-party datasets or
model weights are included. These measurements describe temporary allocations,
not learned quality or whole-model memory savings.

The mathematical contract cites Pop (2020/2021), Pop–Negrescu (2024), and
Pop–Todea (2024). Implementations and small rational test fixtures are original
repository code; no third-party implementation or paper text is redistributed.
Moved research modules retain their original repository license and history.
