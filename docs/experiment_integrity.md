# Experiment integrity

The September 2026 hardening changes make proof reports and bakeoff reuse depend
on content identities. They verify engineering behavior; they do not establish
Pop Rank quality benefit or repair the causal limitations of earlier studies.

## New experiments

Use the locked project environment (`uv sync --locked`, then `uv run --locked`;
include `--extra packs` for external model loaders). Use separate training and
evaluation JSONL files, new pack names, and a new output directory.

For comparisons across rank maps, set `train.initialization` to `component-v1`
in the bakeoff spec, or pass `--initialization component-v1` to `packs create`.
Each factor row is keyed by seed, adapter, input width, and component index.
Shared rows match across rank choices and target traversal order. Continue to
share minibatch schedules and training/dropout seeds. This controls starting
factors; it does not make subsequent gradients or dropout masks identical across
different model geometries. `legacy` remains the default for historical protocol
reproduction. Mixing initialization protocols is not an isolated rank comparison.

## Proof requirements

`packs create` records the training data hash, resolved checkpoint identity,
tokenizer identity, and exact-example identities. Continuations inherit the
parent's recorded example identities. A continuation from an older pack without
complete training provenance remains usable, but cannot establish held-out status.

`packs eval --out metrics.json` records the actual checkpoint, tokenizer,
preprocessing settings, data, ordered tokenized inputs/masks, and attached pack.
Hub model references are resolved to a snapshot before loading. Checkpoint
identities include all non-hidden files, including all weight shards and config.
Pack identities include both `pack.safetensors` and `meta.json`.

`packs proof` requires `--eval-data` to pass. It rejects missing, non-numeric,
non-finite, or impossible required metrics and ambiguous base/pack rows. It checks
that current training/evaluation files and pack contents match the reports, that
paired evaluation used the same model/tokenizer/settings/examples, and that no
exact examples overlap the recorded training lineage. Generation and ledger
reports, when supplied, must identify the same artifacts. The generation helper
`scripts/fault_codes_generate_check.py` and `packs rank-ledger` emit these identities.

The overlap check is conservative across supported text, prompt/answer and
message representations. It does not detect paraphrases, semantic leakage, or
base-model pretraining exposure. Hashes detect stale or mismatched inputs; they
are not signed attestations against deliberately fabricated reports. Unknown
tokenizer interfaces and non-local evaluation datasets cannot establish a proof.

## Resume and historical artifacts

A successful bakeoff phase writes a `.receipt.json` next to its log. Reuse requires
the same command, package source and lockfile, input content, checkpoint, and
every declared output. Missing or altered tensors invalidate a create phase even
when `meta.json` survives. Pack metadata edits also invalidate dependent reports.
A summary can display historical metrics, but promotion requires valid receipts.

Unverified or stale outputs stop execution before overwrite. Prefer new pack and
output names to preserve previous experiments. `--force` explicitly regenerates
outputs. Do not create receipts or add provenance retrospectively to make an old
result pass: identities must be captured by the producing run. Content hashing
large checkpoints adds startup and resume-check IO.

Historical evidence snapshots remain unchanged. The older paired rank-map
screens shared an RNG seed but did not match factor rows across ranks. Their
stored metric arithmetic can be reproduced; the stronger claim that rank
placement alone caused the difference requires new controlled runs. The failed
capacity-migration promotion gates remain failed.

## Verification and remaining work

The local test suite covers batch independence, non-LIFO sleep/wake cycles,
signed factor pruning, frozen base weights, nonzero alpha-zero adapters, matched
factor initialization, invalid/stale proof inputs, and receipt invalidation.
`tests/packs/test_pack_workflow.py` runs real tiny-model training, serialization,
attachment, evaluation, ledger and proof through the bakeoff phases, then checks
reuse and changed-input rejection. External model loading and the subprocess
boundary are substituted with a local synthetic fixture.

Large Gemma training was not repeated for this hardening pass. The subsequent
[compact SVD and focused structural cleanup](svd_factorization.md) is recorded
under ADR-0013 and DSN-20260905-02. A new confirmatory research protocol remains
separate work and must satisfy the existing re-entry rules. See ADR-0012 and
DSN-20260905-01 for the experiment-integrity policy.
