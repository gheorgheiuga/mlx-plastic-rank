# DSN-20260905-03 — Gradient-agreement controller admission

- **Status:** Experimental; parked before evidence
- **Evidence status:** Implementation verified; complete development matrix failed readiness and solvability gates
- **Decision index:** ADR-0014 in `codex/decisions.md`
- **Protocol:** [Specification](../research/gradient-agreement/experiment-spec.md),
  [hypotheses](../research/gradient-agreement/hypothesis-ledger.md),
  [numeric declaration](../research/gradient-agreement/protocol.json)

## Decision requested by the research question

Can a controller without candidate training rollouts identify reusable capacity
better than exact one-step, random transfer and a future-aware fixed allocation?
Specify a cross-batch prospective-gradient controller, including a plain
gradient-energy ablation and a wrong-task control. Do not increase the demoted
horizon-3 controller's search horizon.

## Design and evidence boundary

The proposal separates identical-A-checkpoint selector comparisons from a
whole-strategy fixed-split comparison that starts at matched untrained factors.
It specifies component-stable initialization, uniformly clipped actual and
virtual SGD, two-factor recycle auditing, an extra-capacity solvability check,
complete paired seed-level inference and eight predeclared comparison gates.

Development seeds are 0 and 31–35; evidence seeds 101–120 are reserved subject
to a fresh audit at freeze. The bounded local JSON/JSONL inventory found no
intersection with the reserved set. They remain unused after the failed
development gate. The implementation is runnable for development only; source
snapshots are retained, but there is no evidence freeze.

## Development result, 2026-09-05

All 45 declared development runs were finite, checkpoint-paired and rank/state
valid. Common-A readiness was 0.6062 and future-fixed-split A readiness 0.4823;
joint-capacity A/B scores at B end were 0.6682/0.5811. Each required mean missed
0.8. Final seed-0 trajectory/event/input/preparation journals repeated exactly.
The full 325-test suite, Ruff and mypy passed.

Preserve this failed validity gate and park before confirmatory evidence. No
controller efficacy inference is warranted. Diagnose the reference baseline on
the stored development tasks under a separately declared test before changing
the schedule or data. [Results and artifacts](../research/gradient-agreement/development-results.md).

## Promotion and falsification

Before evidence, require finite development controls, invariant checks and a
source/content freeze. Admission requires all 20 evidence seeds and nine
conditions, minimum useful B acquisition gains, adjusted positive paired lower
bounds against all six B-AUC controls, improved B localization and the declared
A readiness and joint-capacity checks. No failed seed is excluded or replaced.

If fixed split or plain gradient energy matches treatment, demote the relevant
adaptive or agreement explanation. Numerical/solvability failures park inference
and require a new version. A full pass allows only the design of a separately
controlled return-A experiment. ADR-0011 remains Experimental; London/Gemma,
cue-triggered wake, erasure and Pop-theorem promotion remain unsupported.
