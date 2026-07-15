# Evaluating Pop's Theorem for AI

## Scope

The theorem considered here is the matrix-polynomial rank identity

`rank f(M) + rank g(M) = rank (f,g)(M) + rank [f,g](M)`,

where `(f,g)` and `[f,g]` denote the polynomial gcd and lcm under the theorem's
hypotheses. Its direct object is the rank structure of polynomial functions of
a common matrix operator. LoRA adapters are not automatically such functions,
so applying the identity to adapter training requires an additional modeling
argument or a falsifiable proxy—not just observing that ranks differ.

## Decision

Pop's matrix-polynomial rank identity is useful here as an exact accounting
invariant and as a source of testable spectral hypotheses. The repository does
not yet show that the theorem itself selects better LoRA ranks. The strongest
validated result is narrower: heterogeneous rank placement can preserve most of
a fixed-`r32` adapter's quality gain with less than half of its bytes on one
paired fault-code screen.

Accordingly:

- keep Pop-inspired rank accounting and spectral diagnostics;
- treat discovered rank maps as an efficiency mechanism worth testing;
- do not call Gram energy, factor-norm gating, or a heterogeneous-map win a
  theorem-derived quality result;
- require causal controls and training-seed replication before promoting a
  selector.

## Evaluated AI uses

| Candidate use | What Pop contributes | Current evidence | Verdict |
| --- | --- | --- | --- |
| Adapter rank accounting | An exact language for image, kernel, composition, and overlap ranks | `packs rank-ledger` reports effective rank, slack, composition rank, overlap, and savings | Useful instrumentation; theorem causality is not required |
| Efficient LoRA rank placement | The hypothesis that useful capability can be represented by a non-uniform rank allocation | A fresh discovered fault-code map beat random, shuffled, target-constant, and cross-domain maps under a shared schedule | Strongest practical use, but evidence supports placement rather than the theorem |
| Spectral rank-map proposals | Polynomial roots/notches expose selected eigenspaces and motivate promotions or reductions | Generic polynomial pairs were non-discriminating; a later low-spectrum `k_proj` candidate slightly improved local fault-code PPL at the same size | Promising diagnostic; needs paired causal and multi-seed tests |
| Online rank allocation | Rank could become a training-time control variable instead of a fixed hyperparameter | Hard active-rank gates, grow/shrink mechanics, and compact export work | Mechanics only; the controller uses factor norms, has no global byte budget, and is not theorem-informed |
| Pack composition and interference | If two adapters are functions of a common operator, the identity may help account for shared image/kernel structure | The ledger can measure overlap and composition, but packs have not been shown to be polynomial functions of one operator | Plausible future use; ordinary low-rank algebra currently explains the measurements |
| Polynomial AI operators | Graph filters, diffusion steps, Krylov/recurrent transforms, and other modules built as `f(M)` are a direct mathematical fit | No such learning benchmark exists in this repository | Mathematically cleanest direct use; not yet evaluated |
| Compact domain skill packs | Rank maps make exportable local capability smaller and easier to route | Attach/proof/bakeoff workflows improve held-out fault-code and Text-to-SQL quality | Useful product mechanism; not evidence that Pop's theorem caused the gain |

## Evidence that narrows the claim

The [paired fault-code evidence](../codex/evidence/fault_codes_paired_control_screen_seed42.json)
is the current strongest artifact. The discovered map reached PPL `5.8532` at
`23.73 MB`, retained `97.05%` of the contextual fixed-`r32` gain at `43.82%`
of its size, and beat five same-budget random maps, an exact-budget shuffle, and
a target-constant rule with paired 95% bootstrap intervals excluding zero.

That screen also supplies the main counter-explanation. A Text-to-SQL map
normalized to the same byte budget was only `1.12%` worse on fault codes. Rank
placement matters locally, but much of the useful placement may be a reusable
Gemma architecture/training prior rather than domain semantics or the theorem.

The reciprocal [Text-to-SQL transfer screen](../codex/evidence/text_to_sql_paired_transfer_screen_seed42.json)
found the same direction on 300 examples: the native map reached PPL `1.7172`
versus `1.7429` for a byte-matched fault-code transplant, a `1.47%` relative
advantage with paired 95% CI `[1.29%, 1.66%]`. The native map retained `97.50%`
of the contextual fixed-`r32` gain at `44.16%` of its size. Native maps therefore
won in both directions, but the margins remain small and single-seed.

Other narrowing results matter:

- generic Pop-polynomial pairs verified the identity but did not discriminate
  useful adapter locations;
- fixed `r32` remains the absolute local quality ceiling, so the current value
  proposition is efficiency rather than higher maximum quality;
- Text-to-SQL reproduced the quality/size tradeoff, but its original result was
  not a paired same-budget control screen;
- the spectral-key candidate's gain over the prior heterogeneous map was small
  and did not improve the generation-overlap check;
- IndustryBench smoke training was mechanics-positive but quality-negative.

## Falsification ladder

### 1. Domain-by-map transfer

Train native and reciprocal transplanted maps from fresh weights under the same
initialization, minibatch schedule, dropout seed, steps, evaluation examples,
and tensor-byte budget. Repeat in both fault codes and Text-to-SQL.

This separates:

- placement usefulness: non-uniform maps beat random, shuffled, and simple
  structured controls;
- domain semantics: each native map beats the reciprocal transplant;
- architecture prior: both maps transfer with little or no native advantage.

Use at least three training seeds for a diagnostic result and five for a
confirmatory result. Per-example bootstrap intervals do not replace uncertainty
across training seeds.

Kill the domain-specific interpretation if reciprocal maps remain within `1%`
of native maps without a consistent native advantage. An architecture-prior
allocator may still be useful in that outcome.

### 2. Pop-specific selector test

After the transfer question is resolved, compare a Pop spectral-notch map with:

- the factor-norm discovered map used by the current dynamic controller;
- an anti-Pop map that reverses the same score ordering;
- an exact-rank-histogram shuffle;
- five random same-budget maps;
- a target-constant map;
- fixed `r16` and `r32` anchors.

Maps must be frozen before fresh candidate training, match tensor bytes within
`0.1%`, and share the stochastic schedule within each seed. The primary
estimand is paired answer-token NLL/PPL versus the strongest theorem-free
control. Secondary metrics are token accuracy, task-specific generation
quality, effective rank, pack bytes, and spectral-order stability.

Kill Pop spectral allocation as a quality selector if, after at least three
seeds, it fails to beat both factor-norm and anti-Pop/shuffled controls, its
seed-level 95% interval includes zero, its relative PPL advantage is below `1%`
in both domains, or its adapter ordering is unstable across discovery/projection
seeds. The existing practical gate also remains: retain at least `90%` of the
fixed-`r32` gain at no more than `60%` of its size.

### 3. Direct polynomial-operator benchmark

Only after the adapter tests should the project claim a direct theorem use.
Build a small graph-filter, diffusion, or Krylov-style benchmark where learned
modules are explicitly polynomials of the same operator. Test whether the rank
identity predicts reusable image/kernel structure, memory needs, or safe
composition better than ordinary numerical-rank baselines.

## Current conclusion

The most defensible AI value today is **rank-aware capacity allocation and
accounting**. Pop's theorem supplies mathematical discipline and useful
hypotheses, but the quality-positive evidence belongs to heterogeneous LoRA
placement. Reciprocal transfer now supports a small domain-native placement
effect, while also showing that most of the benefit is portable across domains.
A theorem-specific AI claim remains open until a Pop-derived selector beats
strong theorem-free controls across seeds and domains.
