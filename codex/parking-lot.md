# Research Parking Lot

This register preserves research tracks that are intentionally out of the active
critical path. Parking is neither rejection nor deletion. Unparking a track
requires an explicit status update in its DSN and in `codex/dsn/dsn-log.md`.

## Forgetting Machine

- **Parked on:** 2026-07-13
- **Status:** Parked Experimental; retained and runnable
- **Why parked:** The prototype tests governed deletion from a controlled
  external record vault. It does not test the restored Pop Rank thesis: whether
  finite model capacity migrates through learning, specialization, forgetting,
  and relearning.
- **Preserved work:**
  - `src/mlx_plastic_rank/forgetting_vault.py`
  - `scripts/forgetting_machine_demo.py`
  - `tests/test_forgetting_vault.py`
  - `tests/test_forgetting_machine_demo.py`
  - `docs/forgetting_machine.md`
  - `codex/dsn/dsn-20260713-forgetting-machine.md`
- **Return trigger:** Resume when a learned MLX `A -> B -> A` model—not only the
  orthogonal reference mechanism—in
  `codex/dsn/dsn-20260713-capacity-migration.md` passes its learned-model promotion
  gate and needs an external-deletion comparison, or when an explicit project-
  priority decision resumes storage governance or unlearning work.
- **Latest trigger check:** The frozen `tiny_mlx_dense_v1` run on confirmatory
  seeds 1–10 failed its learned-model promotion gate on 2026-07-13, so the
  return trigger was not met and this track remains parked. Its A-return metric
  followed a supervised A-loss probe; it did not test wake through an unlabeled
  cue path.
- **Unparking checklist:** Record the new question, update the threat boundary,
  name the evidence capable of falsifying the new claim, and update this file,
  the DSN, and the DSN log before changing implementation scope.

The runnable prototype remains evidence for controlled logical deletion only.
Its existence is not evidence of neural-network unlearning, physical erasure, or
publicly verifiable deletion.
