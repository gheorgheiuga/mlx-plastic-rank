# Repository-only research

This package preserves experiments, controllers and their failed gates outside
the installed prototype. The maintained package provides fixed-allocation packs
and factor mechanics. See the [parking register](../codex/parking-lot.md) for
return conditions and the [mathematical contract](../codex/dsn/dsn-20260905-pop-mathematical-contract.md)
for what Pop's theorem does and does not establish.

Run research modules from the repository root with the locked environment:

```sh
uv run --locked pytest -q tests/research
uv run --locked python -m research.pop_polynomial_probe --help
uv run --locked python -m research.gradient_agreement_benchmark --help
uv run --locked python -m research.baseline_diagnostic --output-dir out/baseline_diagnostic/new-run
```

The last command requires the verified, locally retained seed 31–35 development
package. It does not download or regenerate missing inputs. The completed
[baseline result](../codex/research/baseline-diagnostic-results.md) supports a
future factorized-baseline diagnosis; it does not admit a controller.

Other benchmark entry points follow `python -m research.<name>`, including
`capacity_migration_benchmark`, `learned_capacity_migration_benchmark`,
`loss_lookahead_calibration_benchmark`, `multibatch_controller_benchmark` and
`forgetting_machine_demo`. `ResearchLoRAManager` in `rank_manager.py` retains
the old automatic/dynamic policies; its default initialization stays `legacy`
to avoid silently rewriting historical comparisons.

The former dynamic pack-training flags and spectral-map CLI were retired.
There is no replacement general research CLI. Exact historical replay uses the
source snapshots and lockfile in each retained artifact, or its recorded Git
revision. Current modules include subsequent correctness repairs, so rerunning
them produces a new result. Preserve old artifacts, thresholds and input hashes;
always choose a fresh output directory. A runnable module does not reopen its
research track or authorize reserved evidence seeds.
