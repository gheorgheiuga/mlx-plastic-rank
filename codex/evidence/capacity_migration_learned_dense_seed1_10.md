# Learned Dense Capacity Migration — Confirmatory Seeds 1–10

**Verdict:** `learned_dense_capacity_migration_gate_failed`

Seed 0 was used to construct and debug the frozen `tiny_mlx_dense_v1`
protocol. Seeds 1–10 were then run as the untouched confirmatory matrix across
guided recycle, static, same-opportunity random, future-aware fixed split,
joint-sufficient extra capacity, guided vault, hidden-site oracle, and never-A
controls.

## What replicated

- The declared effective active-rank budget was conserved at every recorded
  checkpoint. Physical rank stayed preallocated at 16, physical fp16 adapter
  storage at 384 bytes, float32 master storage at 768 bytes, and stateless-SGD
  optimizer state at zero bytes. This is effective-rank conservation, not
  physical-memory conservation.
- Guided recycling reached mean B-acquisition AUC `0.5176`, versus `0.2792` for
  static allocation. The paired mean advantage was `+0.2384`, with a 95%
  fixture-seed interval `[+0.1822, +0.2974]`.
- Across the nine finite random-control pairs, the conditional mean advantage
  was `+0.1268`, 95% interval `[+0.0631, +0.1802]`. The tenth random run was
  non-finite, so the localization-versus-random gate was incomplete and failed.
- Across those same nine pairs, final B-site alignment had a conditional mean
  advantage of `+0.3333`, 95% interval `[+0.1852, +0.4815]`; it is positive
  partial evidence, not a substitute for the missing tenth confirmatory pair.
- The first loss-guided transfer during the B phase began from a score-matched
  static checkpoint on every seed and preceded a mean 12-step B-score advantage
  of `+0.1431`, 95% interval `[+0.0807, +0.2052]`.
- The hidden-site oracle reached full mean B alignment (`1.0`), bounding the
  learned controller's `0.8`. The never-A condition had zero
  post-supervised-loss-probe/pre-update A score, providing a scratch-access
  reference under that same measurement order.

## Why the promotion gate failed

- The complete finite matrix failed: random seed 2 and extra-capacity seed 3
  produced non-finite training gradients. The harness recorded these failures
  rather than converting them into no-transfer events or dropping the report.
- Guided recycling did not beat the future-aware fixed split conclusively:
  mean B-AUC difference `+0.0227`, 95% interval
  `[-0.0346, +0.0816]`.
- The joint-sufficient control missed the frozen `0.8` retention/acquisition
  gate, averaging A-after-B `0.7777` and B-final `0.7983` across nine finite
  runs.
- Strict recycle retained mean post-supervised-loss-probe/pre-update A score
  `0.2569`. `min_rank=1` guarantees at least one A-site component remains active,
  but some seeds retained more and dense routes may preserve A through other
  sites; the residual cannot be attributed to the floor alone.
- The vault's corresponding post-supervised-loss-probe/pre-update value was
  directionally stronger at `0.5270`. Learned V1 did **not** test cue-triggered
  wake through an unlabeled retrieval path: its A loss probe supplied supervised
  retrieval information before this metric was recorded.
- The strict-recycle cleanliness result is provisional. It verified released A
  columns and exact unrelated float32 master parameters, but did not audit
  learned state stored only in B rows. The vault's mean `4.3` dormant learned
  rank is likewise an A-column lower bound, not a complete dormant-state count.

## Interpretation

The experiment now shows a learned-loss signal that can move conserved
effective rank and whose movement precedes a matched performance advantage. It
does not yet show that dynamic migration is better than a strong future-aware
reservation, that the optimization is stable, or that recycling reproduces
human-like forgetting. The central thesis therefore remains Experimental, the
London audience experiment remains blocked, and the Forgetting Machine remains
parked.

V2 must be a separately named protocol with development-only stability
calibration, a predeclared A/B baseline ledger, a genuinely unlabeled cue path
measured before any supervised loss probe, and learned-state accounting across
both A columns and B rows.

## Provenance

- Exact command: `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/learned_capacity_migration_benchmark.py --mode evidence --output-dir out/capacity_migration/learned_mlx_dense_seed1_10 --require-pass` (expected exit `2` for the failed gate, after writing artifacts)
- Base Git revision: `55d94388848d0ce3be6ff22baa97d47a3cdeb9ab`
- Worktree state: dirty, with uncommitted benchmark changes. The base revision
  alone cannot reconstruct the exact frozen V1 generator.
- Retained output SHA-256 values: `summary.json` `9c93a8fe...1b49ff`,
  `summary.csv` `a6a5e3b0...0a0c1`, `summary.md` `f392589b...62ccf`, and
  `trajectory.jsonl` `9a2bc1b6...63832` (full values are in the compact JSON).
- The nearest retained generator hashes were captured after the run, before the
  reporting audit: script `35c181b2...44473`, learned module
  `f95399ee...70d0da`, manager `e2e63035...54cd37`, and artifact writer
  `03e288c7...df38`. They document the nearest retained source state but cannot
  independently prove the exact dirty-worktree code that generated the run.

The full local trajectory and summaries are under
`out/capacity_migration/learned_mlx_dense_seed1_10/`.
