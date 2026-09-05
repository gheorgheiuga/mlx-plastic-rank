# DSN-20260905-06 — Pop's identity as a precise subspace contract

- **Status:** Experimental
- **Evidence status:** Small exact and numerical implementation checks; no learned allocation or quality result
- **Decision index:** ADR-0017 in `codex/decisions.md`
- **Scope:** Mathematical foundation for the narrowed prototype; does not reopen parked controllers

## Objects and invariant

For a field K, one square matrix M over K, and scalar polynomials f,g in K[x],
let d=gcd(f,g) and m=lcm(f,g). Pop's identity is

`rank f(M) + rank g(M) = rank d(M) + rank m(M)`.

All four operators use the **same M** and algebraic rank. Polynomial images give
the useful interpretation: the image of d(M) is the sum of the images of f(M)
and g(M); the image of m(M) is their intersection. This accounts for subspace
sharing. It does not specify a loss, update rule, useful task support, or a
conserved byte/compute budget across unrelated layers.

Pop's [original paper](https://arxiv.org/pdf/2010.00634) proves the identity by
invertible block transformations. [Pop–Negrescu (2024)](https://doi.org/10.3390/math12030360)
provides Jordan-block and kernel/image proofs; root multiplicities and repeated
eigenspaces matter. [Pop–Todea (2024)](https://www.scientificbulletin.upb.ro/static/pdfs/rez20c_605444.pdf)
extends the identity to rank functions satisfying product monotonicity and
block-diagonal additivity. Energy cutoffs and stable rank are not such a rank
function by default: stable ranks of diag(2,1) and diag(1/2,1) are both 1.25,
while their product has stable rank 2, violating product monotonicity.

## Numerical implementation boundary

`tests/research/test_pop_contract.py` independently checks each rank using small
rational row reduction on exactly representable cases: zero/full rank,
idempotents, repeated eigenvalues and Jordan blocks. Generic full-rank equality
is explicitly included as a nonselective case. Perturbation tests distinguish
algebraic rank from a declared relative singular-value threshold.

The retained `research/pop_polynomial_probe.py` is a floating-point diagnostic.
Its symmetric spectral path removes whole eigenspaces within an explicit
float32 clustering tolerance; near-coincident eigenvalues remain approximate.
Jordan cases use direct polynomial evaluation. A numerical `identity_holds`
flag is never sufficient evidence that all ranks are correct or useful.
The pack ledger obtains numerical ranks and bases from the same compact update
decomposition; equivalent reciprocal factorizations must produce equal overlap.
Neither numerical ledger totals nor gate counts are exact neural capacity.

## One candidate bridge to learning

An explicit restricted model could use a fixed, small symmetric operator H and
task updates `Delta_t = L p_t(H) R^T`, where L and R have orthonormal columns.
This preserves the latent operator's rank in rectangular projections. With
distinct eigenvalues, task support masks can be represented by interpolation
polynomials; runtime evaluation should use the shared eigenbasis and masks,
not unstable high-degree polynomial products.

**Hypothesis:** selecting reusable task subspaces can improve A→B→A retention
under a fixed total storage and update budget. The theorem supplies accounting;
task-dependent selection remains an additional, unvalidated mechanism.

**Strong alternative:** any improvement comes from the shared basis, ordinary
mask selection, or extra training. A same-basis mask is mathematically equivalent
to its interpolating polynomial, so polynomial notation alone cannot add value.

**Falsification:** after a valid learning fixture exists, compare the proposed
selection with fixed and random support choices in the same basis, with matched
initialization, data exposure, actual updates and total bytes. Include the cost
of L, R, H, inactive storage and selection. If advantage disappears under these
controls, retain the accounting interpretation and park a theorem-quality claim.
Specify thresholds and untouched evaluation data in a separate protocol first.

## Next action

DSN-20260905-05 passed on stored development data, including all broken-pairing
controls. Its [result](../research/baseline-diagnostic-results.md) narrows the next
question to a separately declared factorized baseline diagnosis with matched
checkpoints. It does not admit this learning hypothesis, a new controller matrix,
or a large-model run.
