# Decision Support Note (DSN)

**ID:** DSN-20260713-01
**Title:** Build a controlled spectral forgetting vault
**Date:** 2026-07-13
**Status:** Parked (Experimental)
**Evidence Status:** Controlled-vault mechanics verified; model-weight forgetting untested; no active development
**Related Research Inbox Entry:** None

---

## Context

The project needs a concrete, audience-readable demonstration of a new Pop-theorem
application beyond rank-map selection. A right-to-forget workflow is compelling only
if the demonstration makes its trust boundary explicit: deleting controlled records
is testable, while claiming that a transformer has forgotten matching information
requires substantially different evidence.

## Options Considered

1. Claim selective unlearning from transformer weights.
   - Pros: The boldest product story.
   - Cons: The repository has no extraction, membership-inference, or retraining
     comparison evidence to support the claim.
2. Build a controlled external-memory vault with spectral attestations.
   - Pros: Produces a real, deterministic deletion operation; supports independent
     recalculation of disclosed algebra and exact Pop GCD/LCM rank accounting.
   - Cons: Certifies only the state controlled by the vault, not every copy of the
     data or information encoded in model weights.
3. Build a presentation-only simulation.
   - Pros: Fast to stage.
   - Cons: Would provide visuals without an executable or falsifiable mechanism.

## Decision

- Chosen option: 2.
- Each record receives a distinct integer eigenvalue in a controlled diagonal
  operator. A forget policy becomes an annihilating polynomial over selected modes;
  an exact rational Lagrange projector applies the deletion while preserving all
  unmatched modes. Attestations are canonical JSON, HMAC-authenticated, linked,
  and include Pop rank identities for overlapping policy targets.
- Canonical decision record: this remains an experimental DSN and does not create
  an Accepted entry in `codex/decisions.md`.

### Parking decision (2026-07-13)

- Preserve the implementation, tests, demonstration, and boundary documentation,
  but stop active development while the repository returns to its founding
  capacity-migration question.
- This track is adjacent rather than discarded: it distinguishes governed
  external-record deletion from model inaccessibility, dormant factors, and
  overwritten capacity, but it cannot answer where learning capacity moves.
- Return trigger: a learned MLX `A -> B -> A` capacity-migration model—not only
  the orthogonal reference mechanism—passes its learned-model promotion gate and
  needs an external-deletion comparison, or an explicit project-priority
  decision resumes storage governance/unlearning work.
- Durable parking record: `codex/parking-lot.md`.

## Consequences

- The core interface stays small: search active records, forget by deterministic
  policy, inspect a payload-free metadata snapshot, and verify a serialized attestation.
- The certificate authenticates a transition assertion from the trusted controlled
  vault emitter. It does not independently inspect historical state or prove
  physical media sanitization, erasure from caller-owned
  values, logs, caches, backups, process memory, or neural-network weights.
- Exact interpolation is intentionally optimized for clarity and capped at 128
  records, not high-volume storage.
- Promotion to a stronger claim requires a persistent backend with explicit key
  custody, external audit replay, and leakage tests appropriate to every claimed
  storage or model boundary.

## Follow-ups

- [ ] On unpark, update this status, `codex/dsn/dsn-log.md`, and
  `codex/parking-lot.md` before expanding the claim or implementation.
- [ ] Add a persistent storage adapter with transactional deletion and key rotation.
- [ ] Replace the flat spectrum commitment with a scalable authenticated index.
- [ ] Add extraction and membership-inference tests before discussing model unlearning.
- [ ] Evaluate whether a learned state-space memory can expose the same controlled
  operator without weakening the certificate boundary.

---

The completed prototype remains reproducible while parked. This DSN stays
Experimental until evidence supports a broader decision; parking does not count
as rejection or promotion.
