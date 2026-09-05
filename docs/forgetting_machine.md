# The Forgetting Machine: controlled spectral vault

> **Parked research track (2026-07-13).** This implementation, its tests, and
> its threat-boundary documentation are deliberately retained, but active
> development is paused while Pop Rank tests conserved model-capacity migration.
> Parking is not rejection. Resume after a learned MLX capacity-migration model,
> not only the orthogonal reference mechanism, passes its learned-model promotion gate and
> needs an external-deletion comparison, or by an explicit project-priority
> decision. See `codex/parking-lot.md` and
> `codex/dsn/dsn-20260713-forgetting-machine.md`.

## Status and defensible claim

`ForgettingVault` is a controlled, in-memory prototype for deleting selected
record payloads and emitting authenticated, mathematically checkable evidence
about that state transition. It gives each record its own mode in a deliberately
constructed diagonal operator, represents deletion policies by polynomial roots,
uses an exact Lagrange projector to preserve every unselected mode, and records
Pop's GCD/LCM rank identity when policies overlap.

The defensible claim is:

> A trusted vault emitter atomically applies and attests selection, payload
> tombstoning, and exact preservation inside its controlled record store. A
> retained verifier can authenticate the attestation, check its internal
> invariants and polynomial algebra, and validate history continuity.

It does **not** prove that a trained model has unlearned a fact. It does not
inspect or erase transformer weights, embeddings outside the vault, caller-owned
objects, logs, caches, backups, or residual copies in process memory.

The implementation is in
`src/mlx_plastic_rank/forgetting_vault.py`; focused behavioral and tamper tests
are in `tests/research/test_forgetting_vault.py`.

## Interface

The public interface is intentionally small:

| API | Purpose |
| --- | --- |
| `MemoryRecord(record_id, subject, category, payload)` | One payload governed by the vault. All fields must be non-empty strings and `record_id` must be unique. |
| `ForgetPolicy(policy_id, subject=None, category=None, record_ids=(), reason="")` | A deterministic selector. If several selectors are supplied, **all** must match. Use separate policies for union semantics. |
| `ForgettingVault(records, integrity_key=..., key_id=..., vault_id=...)` | Constructs the controlled operator and in-memory state. The HMAC secret must contain at least 32 bytes; key length is not an entropy guarantee. The current exact representation is capped at 128 records. |
| `vault.search(...)` | Returns active records matching every supplied selector. Tombstoned records are never returned. |
| `vault.forget(policy)` | Builds and verifies a prospective transition, commits it, and returns a `DeletionCertificate`. |
| `vault.snapshot()` | Returns a payload-free but metadata-bearing summary containing active/forgotten IDs, a keyed state commitment, certificate count, and latest certificate digest. |
| `certificate.to_dict()` / `to_json()` | Returns isolated or canonical-JSON evidence. JSON loading rejects duplicate object keys. |
| `DeletionCertificate.from_json(raw)` | Reconstructs bounded certificate JSON while rejecting duplicate keys, invalid UTF-8, and non-finite values. |
| `certificate.verify(integrity_key)` | Authenticates one emitter attestation and checks its cross-field invariants, polynomial data, projector values, active-count assertion, and Pop compositions. It does not inspect historical storage. |
| `verify_certificate_chain(certificates, integrity_key, expected_head_digest=...)` | Checks vault/operator identity, sequence, predecessor links, and state continuity for a retained history. An external head anchor is required to detect a truncated tail. |

An empty selector, an unknown `record_id`, a policy that matches no record, or
reuse of a `policy_id` is rejected. A new policy may select records that were
already tombstoned; this is idempotent and the certificate distinguishes
`newly_tombstoned_record_ids` from `already_tombstoned_record_ids`.

## Controlled spectral model

For `n` records, the vault sorts record IDs and assigns the distinct integer
eigenvalues `2, 3, ..., n + 1`. Conceptually, it constructs

\[
M = \operatorname{diag}(\lambda_1,\ldots,\lambda_n),
\]

with one record per eigenmode. This operator is deterministic and exact; it is
not learned from a model and does not claim that ordinary memories naturally
occupy isolated spectral modes.

The implementation retains record ID, subject, category, and eigenvalue after
forgetting, sets the selected payload to `None`, and closes its scalar access
gate. This is a logical tombstone in the controlled Python object, not a secure
overwrite of physical memory.

## Annihilator evidence versus the scrubber projector

The distinction between the policy polynomial and the state-changing projector
is important.

For the selected eigenvalue set `S`, the certificate first records the monic
annihilator

\[
f_S(t) = \prod_{\lambda \in S}(t-\lambda).
\]

On the diagonal operator, `f_S(M)` is zero on each selected mode and nonzero on
each unselected mode, so

\[
\operatorname{rank} f_S(M) = n-|S|.
\]

However, an annihilator is not generally a keep projector: on unselected modes,
`f_S(M)` can scale values by numbers other than one. Applying it directly would
not prove exact preservation.

The actual scrubber is therefore the exact Lagrange polynomial `q_S` satisfying

\[
q_S(\lambda)=
\begin{cases}
0,&\lambda\in S,\\
1,&\lambda\notin S.
\end{cases}
\]

The implementation constructs `q_S` with rational arithmetic (`Fraction`),
checks that every spectral value is exactly zero or one, applies it to the
prospective access gates, and nulls the selected payloads. The certificate
serializes every coefficient as `[numerator, denominator]` and records the
value at every eigenvalue. A domain-separated keyed commitment over all
unselected records is computed
before and after the prospective transition; the transition commits only after
the generated certificate verifies.

In short:

- `policy_filter` is the root-carrying annihilator used for rank and policy
  composition evidence;
- `scrubber_projector` is the exactly selective polynomial used to preserve or
  close the controlled modes;
- setting selected payloads to `None` is the vault's actual logical deletion
  action.

## Pop GCD/LCM composition certificate

Suppose policies select eigenvalue sets `S` and `T`, with annihilators `f_S`
and `f_T`. Because this prototype uses unique eigenvalues and simple roots:

\[
\operatorname{roots}(\gcd(f_S,f_T))=S\cap T,
\qquad
\operatorname{roots}(\operatorname{lcm}(f_S,f_T))=S\cup T.
\]

The composition entry therefore labels:

- GCD roots as `shared_target_record_ids`;
- LCM roots as `combined_target_record_ids`.

It records all four polynomials, their exact integer coefficients, and the rank
identity

\[
\operatorname{rank}f_S(M)+\operatorname{rank}f_T(M)
=
\operatorname{rank}\gcd(f_S,f_T)(M)
+
\operatorname{rank}\operatorname{lcm}(f_S,f_T)(M).
\]

The ranks count modes on which a polynomial is **nonzero**, not deleted modes.
Thus the GCD rank is `n - |S intersection T|` and the LCM rank is
`n - |S union T|`. The certificate verifier reconstructs the intersection,
union, coefficients, record mappings, ranks, and both sides of the identity.

Each new certificate includes a composition against every earlier policy in the
same vault history. In this controlled diagonal construction, the identity is
exact but reduces to finite-set inclusion/exclusion. This demonstrates correct
polynomial mechanics and auditable policy composition; it does not yet show an
advantage over ordinary set accounting in a production deletion system.

## Certificate authentication and retained-history chain

Certificate JSON is canonicalized with sorted keys and compact separators. The
vault then computes:

1. `certificate_digest = SHA256(canonical_payload)` before the digest and tag
   fields are added;
2. `authentication_tag = HMAC-SHA256(integrity_key, certificate_digest)`;
3. `previous_certificate_digest`, which is 64 zeroes for the first certificate
   and the immediately preceding certificate digest thereafter.

Each authenticated payload also carries `vault_id`, a monotone
`event_sequence`, the operator commitment, `key_id`, and keyed before/after
state commitments. `verify_certificate_chain()` checks those fields across an
ordered retained history. Passing an externally retained latest digest detects
a missing or replaced tail.

Because the previous digest is inside the authenticated payload, changing a
link without the key invalidates the current certificate. Verification with the
wrong key or after mutation reports failures instead of accepting the evidence.

This is a keyed, tamper-evident retained history, not a public proof or a
complete durable audit log. The current prototype has no persistent append-only
store, trusted timestamp, built-in external head anchor, or key rotation.
Without an independently retained head digest, a valid prefix cannot be
distinguished from a complete history. Anyone holding the shared HMAC key can
produce apparently valid certificates, so the tag does not provide public
verifiability or non-repudiation.

## Run the demo

From the repository root, run the focused test suite:

```bash
uv run --locked pytest -q tests/research/test_forgetting_vault.py
```

Run the stage narrative and produce self-contained evidence artifacts:

```bash
uv run python scripts/forgetting_machine_demo.py \
  --out-json out/forgetting_machine_demo.json \
  --out-html out/forgetting_machine_demo.html
```

The CLI uses a fresh 32-byte secret for each run and does not export it. Its
saved report records that integrity, algebra, and history were checked during
generation; it is intentionally not a publicly re-verifiable certificate. The
library example below retains a private key so the caller can re-verify later.

The following standalone demo deletes all Alice records, then all medical
records. Their overlap is `alice-medical`; their union also includes
`alice-travel` and `bob-medical`.

```bash
uv run python - <<'PY'
import json

from mlx_plastic_rank.forgetting_vault import (
    ForgetPolicy,
    ForgettingVault,
    MemoryRecord,
)

key = b"demo-only-key-with-at-least-32-bytes"
vault = ForgettingVault(
    [
        MemoryRecord("alice-medical", "alice", "medical", "Penicillin allergy"),
        MemoryRecord("alice-travel", "alice", "travel", "Flight to Bucharest"),
        MemoryRecord("bob-medical", "bob", "medical", "Blood type O positive"),
        MemoryRecord("pump-fault", "plant", "maintenance", "Bearing temperature high"),
    ],
    integrity_key=key,
)

alice = vault.forget(
    ForgetPolicy("delete-alice", subject="alice", reason="user request")
)
medical = vault.forget(
    ForgetPolicy("delete-medical", category="medical", reason="retention policy")
)

composition = medical.to_dict()["pop_compositions"][0]
print("Alice certificate valid:", alice.verify(key).valid)
print("Medical certificate valid:", medical.verify(key).valid)
print("Shared targets:", composition["shared_target_record_ids"])
print("Combined targets:", composition["combined_target_record_ids"])
print("Pop ranks:", json.dumps(composition["ranks"], sort_keys=True))
print("Remaining records:", [record.record_id for record in vault.search()])
print("Snapshot:", json.dumps(vault.snapshot(), indent=2, sort_keys=True))
PY
```

To demonstrate tamper detection, add this after creating a certificate:

```python
import json
from mlx_plastic_rank.forgetting_vault import DeletionCertificate

tampered_payload = json.loads(medical.to_json())
tampered_payload["active_records_after"] = 999
tampered = DeletionCertificate.from_json(json.dumps(tampered_payload))
print(tampered.verify(key))
```

The report should be invalid and include `certificate_digest_mismatch`.

## Threat and proof scope

Within one trusted `ForgettingVault` instance, certificate generation checks:

- deterministic record selection under intersection semantics;
- exact annihilator roots, coefficients, and ranks;
- exact zero/one Lagrange projector values on the controlled spectrum;
- logical erasure of every selected payload in the prospective state;
- unchanged keyed commitments for every active, unselected record;
- active-record counts and before/after keyed state commitments;
- GCD/LCM policy overlap, union, and Pop rank equality;
- per-certificate digest and HMAC authentication.

Later `DeletionCertificate.verify()` can recompute authentication, cross-field
consistency, and the disclosed mathematics from the serialized certificate. It
cannot reconstruct the vault's historical RAM state from the certificate alone.
In particular, the preservation and payload-erasure declarations are
authenticated evidence produced by the trusted vault code, not a remote
inspection of the original storage medium.

The prototype is useful as a laboratory for a stronger future system: put
retrievable memory behind a narrow controlled boundary, make deletion a checked
state transition, and expose composition evidence rather than a bare success
flag.

## Explicit limitations

- **No transformer-weight unlearning.** The vault neither edits nor tests model
  parameters, gradients, optimizer state, activations, or memorized behavior.
- **No guarantee against extraction elsewhere.** A model, caller, database,
  vector index, log, cache, backup, network peer, or previously returned
  `MemoryRecord` may retain the same information.
- **No secure physical erasure.** Assigning `None` does not overwrite Python
  heap pages, allocator copies, swap, crash dumps, or storage blocks.
- **Metadata remains.** Tombstoned record IDs, subjects, categories, spectral
  assignments, and history remain in vault state. Certificates omit raw policy
  selectors but still carry record IDs and spectral assignments, so neither
  surface is a confidentiality-preserving artifact.
- **One artificial mode per record.** The diagonal, distinct-integer spectrum is
  deliberately engineered. It does not establish that semantic facts in a
  learned representation are separable, stable, or polynomially addressable.
- **Logical checks depend on trusted code and key custody.** A compromised
  process or HMAC key can alter state and mint valid-looking evidence.
- **Verification is not storage attestation.** It proves authenticated internal
  consistency, not that all copies were deleted or that the producer reported
  every relevant store.
- **The audit chain is volatile.** History exists only in memory, and no external
  anchor prevents deletion of a tail or rollback to an earlier snapshot.
- **Only process-local concurrency control.** An in-process re-entrant lock
  serializes searches, snapshots, and complete forget transitions. There are no
  transactions across processes, crash recovery, persistent schema migration,
  authorization layer, or key-management service.
- **No legal or regulatory certification.** The evidence may support an audit
  design, but by itself does not establish compliance with a deletion request or
  retention regime.
- **No demonstrated theorem-derived product advantage yet.** In this prototype,
  GCD/LCM roots mirror set intersection/union. A stronger result would require a
  setting where several memories share structured modes and polynomial
  composition predicts behavior that simpler accounting does not.

Any public demonstration should therefore say **"observed payload removal with
an authenticated vault attestation"**, not **"the AI proved that it forgot"**.
