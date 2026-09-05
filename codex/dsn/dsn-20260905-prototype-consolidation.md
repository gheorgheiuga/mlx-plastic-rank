# DSN-20260905-04 — Consolidate the prototype around verified mechanics

- **Status:** Accepted as an engineering and scope decision
- **Evidence status:** Behavioral checks support the retained mechanics; learning quality and capacity-migration benefit remain Experimental
- **Decision index:** ADR-0015 in `codex/decisions.md`
- **Date:** 2026-09-05

## Problem and decision

The prototype accumulated overlapping demos, heavy default dependencies and
several research tracks whose failed gates did not reduce the active surface.
The default demo optimized an arbitrary output mean and invoked a heuristic
that rapidly expanded rank. This was a poor introduction to either conserved
capacity or useful adaptation.

Keep a small dependable substrate: low-rank factor lifecycle, compact matrix
factorization, static adapter packs, rank/byte accounting and content-bound
experiment reports. Give it one bounded local demo and one pack CLI. Preserve
research evidence while stopping implementation growth around failed controls.
This accepts a scope reduction, not a scientific result or production-readiness
claim. It supersedes older default-demo and large-model-priority guidance.

## Keep, retire and park

| Decision | Surface | Rationale and boundary |
| --- | --- | --- |
| Keep | `lowrank.py`, `factorization.py`, rank selection/accounting | Frozen bases, live factors, approximate sleep/wake, numerical SVD and budget diagnostics have behavioral tests; no learned allocation benefit follows |
| Keep | Pack create/inspect/apply/remove/eval, proof and bakeoff | Useful adapter lifecycle and artifact validation; matched initialization is required for new allocation comparisons |
| Replace | `plastic_rank.py` automatic-growth demo | Explicit four-component lifecycle; residual-relative error, finite outputs and frozen-base checks; extra steps hold state |
| Retire | `main.py` and `tests/test_main.py` | Banner-only entry point/test add no MLX validation; use the real lifecycle demo |
| Retire | `scripts/demo_plasticity_blocks.py` | Duplicate heuristic demo with no meaningful learning objective; use `plastic_rank.py` |
| Retire | `scripts/demo_mlx_lm_pack.py` | Separate model-loading/generation demo; use the tested pack CLI for adapter application/evaluation |
| Retire | `scripts/domain_router_runtime.py` | Duplicate routing runtime; use `packs route`, which shares canonical loader and checkpoint handling |
| Park | Automatic `PlasticityManager`, dynamic gates and rank-map discovery | Preserve explicit APIs/tests; utility/Gram signals and local rank-map observations have no established general allocation benefit |
| Park | Learned migration controllers and polynomial selector work | Prior ordering/admission gates failed or were incomplete; do not tune around them |
| Park | Large-model/modality expansion and forgetting vault | Outside the next discriminating test; retain source/evidence and documented return triggers |

The retired files are unchanged tracked files from revision
`1651a62c22c526427f0e890698dad912008f43be` and are recoverable from Git history.
Their deletion does not remove research journals, checkpoints or generated data.
The demo retains `--steps`, `--seed` and `--d-model`; `--lr` is removed because it
no longer trains. Historical class imports from `plastic_rank.py` still work.

The package's advertised `__all__` contains low-rank tools. Explicit legacy root
imports of manager, theorem-named and vault symbols still resolve; existing
callers need not migrate immediately. Prefer the dedicated modules for parked
APIs. No experiment package reshuffle or compatibility framework is introduced.

## Dependencies and maintained boundary

Core dependencies are MLX, NumPy and SafeTensors. Remove unused SymPy (and its
now-unneeded transitive mpmath). Move Hugging Face `datasets` into the optional
`data` extra. Extraction helpers explain how to enable it and preserve failures
from a broken transitive dependency. Existing `packs` and `compress` extras stay
available. Lockfile resolution changes only these dependency edges/removals;
other locked versions stay fixed.

Model-loader imports in the pack CLI and Hub imports in the retained Gemma
utility now occur only when requested. Compression helpers work locally without
Hub/progress packages; the compression extra retains download support and
progress bars. The cache-miss test supplies a fake Hub boundary instead of
requiring a real download library for an entirely local failure-path check.

The dependency guard now fails on missing named modules instead of silently
skipping them, parses imports, and checks that core/CLI startup does not import
model loaders, dataset tooling or parked experiments. NumPy remains necessary at
the SafeTensors boundary; compact factorization still uses MLX CPU operations.

## Evidence and limits

- DSN-20260905-01 fixes layer correctness and artifact integrity, while recording
  historical initialization/provenance confounds.
- DSN-20260905-02 measures compact SVD on synthetic matrices; it establishes no
  downstream quality or whole-model memory claim.
- DSN-20260905-03 completed 45 finite development runs but failed readiness and
  sufficient-capacity gates. That failure prevents controller admission.
- Consolidation validation: focused lifecycle, optional-loader and import-boundary
  checks pass in the minimal environment. Ruff and mypy pass (53 source files).
  Before suite organization, **332 tests passed in 29.53 seconds**, with no skips and
  no optional model/dataset packages installed. The ten-step demo keeps its base
  unchanged, conserves four total components and restores the residual with
  relative error 0.002172 (seed 42, width 32); extra steps hold that state.
  Local CLI help, documentation links and diff checks pass. Lockfile inspection
  confirms all retained package versions are unchanged. These checks do not
  include a new downloaded-model training run or scientific efficacy test.

Historical source snapshots, evidence JSON and failed thresholds remain intact.
A runnable experiment is not an endorsed active direction. The full ledger of
parked tracks and return triggers is [the parking lot](../parking-lot.md).

## Test organization follow-through

The 2026-09-05 cleanup gives the retained prototype a default suite of 259 cases:
45 under `tests/core`, 186 under `tests/packs`, and 28 under `tests/tools`.
The 71 parked experiment cases move unchanged to `tests/research`: migration
controllers (51), polynomial/spectral studies (9), and the external vault (11).
Default CI uses the three active directories; research is an explicit local
command or an option on a manual CI run. Shared-mechanic changes still require
the affected research checks. `pytest tests` remains the complete regression run.

Two weak standalone smoke checks are merged into existing tests: wrapper imports
now assert canonical function/class identity, and SVD reconstruction is checked
against the existing optimal-error result. The parser tests retain all 32
scenarios while shrinking from 711 to 269 lines through shared setup and explicit
case tables. This reduces repetition without dropping numerical/input coverage.
The README now leads with the runnable prototype; research details live behind
the decision/evidence links. [Commands and conventions](../../tests/README.md).

Validation: **259 default cases passed in 1.60 seconds** and **71 research cases
passed in 238.26 seconds**, in separate invocations with no skips. Ruff, mypy and
diff checks pass. Collection verifies that the default command excludes research
and `pytest tests` includes all 330 cases. An assertion audit confirms all 85
expected values in the 29 parser acceptance scenarios were retained, alongside
the two rejection cases and routing scenario. Current documentation links resolve.

## Next decision and stop conditions

[DSN-20260905-05](dsn-20260905-baseline-validity-diagnostic.md) proposes one dense
baseline diagnostic on stored development arrays. It is not implemented or run
by this consolidation. Its result must select one justified follow-up before
more adaptive comparisons. No reserved evidence seeds, controller retuning,
return-A runs, model downloads or broader product expansion are authorized by
this scope decision alone.

Promotion still requires the scientific gates in the relevant research DSN.
A failed diagnostic is retained and narrows the next question; it is not a reason
to lower historical thresholds or launch another sweep.
