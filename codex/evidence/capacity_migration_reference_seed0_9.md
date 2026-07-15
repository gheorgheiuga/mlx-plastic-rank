# Capacity Migration Reference Benchmark

**Verdict:** `counterfactual_reference_mechanics_passed`

Passing verifies effective active-rank accounting, paired transfers, and cue-triggered dormant-factor restoration in an orthogonal synthetic fixture with idealized component-level counterfactual gradients. That signal is a routing upper bound and does not show that a trained neural model can discover the same sites. Vault rows also report resident rank: parked factors are stored information, not erased parameters. The fixed student preallocates every possible slot, so this is not parameter-memory conservation.

| condition | B AUC | B alignment | A after B | A cue return pre-update | budget pass |
|---|---:|---:|---:|---:|---:|
| vault | 0.900 | 1.000 | 0.000 | 1.000 | 1.000 |
| recycle | 0.900 | 1.000 | 0.000 | 0.000 | 1.000 |
| static | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| fixed_split | 0.508 | 0.500 | 0.501 | 0.501 | 1.000 |
| random | 0.362 | 0.400 | 0.000 | 0.000 | 1.000 |
| oracle | 0.958 | 1.000 | 0.000 | 0.000 | 1.000 |
| extra_capacity | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 |

## Declared gates

- [x] `active_budget_invariant`
- [x] `rank_reaches_new_task_site`
- [x] `migration_beats_static`
- [x] `migration_beats_future_aware_fixed_split`
- [x] `migration_beats_same_timing_random`
- [x] `vault_is_inaccessible_then_cue_wakeable`
- [x] `recycle_has_no_immediate_memory`
- [x] `recycle_matches_never_a_schedule`
- [x] `extra_capacity_can_retain_both_tasks`

## Paired fixture-seed B-acquisition comparisons

- `vault_vs_static_b_score_auc`: mean=0.900, 95% CI [0.898, 0.902]
- `vault_vs_fixed_split_b_score_auc`: mean=0.392, 95% CI [0.376, 0.409]
- `vault_vs_random_b_score_auc`: mean=0.538, 95% CI [0.363, 0.714]
- `recycle_vs_static_b_score_auc`: mean=0.900, 95% CI [0.898, 0.902]
- `recycle_vs_fixed_split_b_score_auc`: mean=0.392, 95% CI [0.377, 0.408]
- `recycle_vs_random_b_score_auc`: mean=0.538, 95% CI [0.376, 0.713]

## Integration seam

LoRAManager currently supplies mask-based actuation and norm-product donor heuristics, but it does not accept the benchmark's inactive-component demand signal. The learned MLX bridge must add a recipient-demand seam derived from real gradients, loss ablations, or probe activations before claiming that it reproduces this fixture. Preserve the trajectory schema and active-rank invariant.
