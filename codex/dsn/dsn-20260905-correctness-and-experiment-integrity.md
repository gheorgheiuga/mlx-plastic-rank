# Decision Support Note

**ID:** DSN-20260905-01  
**Title:** Correctness and content-bound experiment evidence  
**Date:** 2026-09-05  
**Status:** Experimental  
**Evidence Status:** Engineering behavior verified with local regression and workflow tests; confirmatory quality evidence pending  
**Canonical decision:** ADR-0012 in `codex/decisions.md`

## Findings and changes

The repository review reproduced batch broadcasting in `PlasticBlock`, sleeper
overwrites after non-LIFO wake, pruning of large negative factors, trainable base
weights documented as frozen, and ignored alpha-zero adapter initialization.
Regression tests now exercise and protect the corrected behavior.

Proof reports previously admitted missing metrics and did not bind results to
the evaluated data and artifacts. They now validate numeric domains, exact
training/evaluation disjointness including known continuation history, and
content identities for training data, checkpoint, tokenizer, tokenized evaluation,
adapter tensors and metadata. Bakeoff reuse requires completion receipts tied to
inputs, implementation and outputs; unverified artifacts cannot support promotion.

A shared RNG seed did not produce matching initial factor rows across different
rank maps. The explicit `component-v1` protocol addresses this for new studies.
The default remains `legacy`; no frozen study or raw evidence file was rewritten.
The older rank-placement and reciprocal-transfer comparisons remain diagnostic
measurements with an initialization confound, not isolated causal evidence.

## Validation

- Focused tests reproduced the defects before fixes and passed afterwards.
- A tiny local model completes train/export/attach/eval/ledger/proof, resumes
  without rerunning phases, and rejects changed data/checkpoint contents.
- Full pytest suite, Ruff and mypy run in the locked Python environment.
- The plasticity demo remains a smoke check, not a scientific result.
- No large-model training or new research promotion was performed.

The workflow test uses locally authored synthetic data and a tiny model. This
does not validate external model-loader compatibility for every architecture.
Operational boundaries and migration instructions are in
`docs/experiment_integrity.md`.

## Promotion or falsification

Keep this DSN Experimental. Promoting a research claim requires fresh untouched
seeds, `component-v1` matched initialization, shared schedules, complete finite
controls, held-out data with captured provenance, and the preregistered paired
quality gate. A rank-map win without those controls is insufficient. Capacity
migration still follows ADR-0011, including its failed gates and re-entry rules;
these engineering fixes do not reopen large-scale experiments automatically.
