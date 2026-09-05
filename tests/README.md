# Test suites

Ordinary development runs the retained prototype checks:

```sh
uv run --locked pytest -q
```

`pyproject.toml` selects these directories by default:

| Suite | Protects |
| --- | --- |
| `core/` | Factors, quantization, numerical decomposition, lifecycle and imports |
| `packs/` | Adapters, training/evaluation, persistence, CLI contracts and artifact integrity |
| `tools/` | Compression utilities and dataset preparation helpers |

Parked experiment regressions remain available explicitly:

```sh
uv run --locked pytest -q tests/research
uv run --locked pytest -q tests
```

The first command runs only research; the second runs every suite. Normal CI
runs the default suites. A manual CI run can also select the research suite.
Run research checks when changing its source or shared mechanics it exercises;
passing them does not reopen a parked track or prove a quality benefit.

Place new tests in the directory matching the behavior they protect. Test public
behavior, meaningful failure modes and numerical edge cases. Use parameterized
cases for repeated scenarios; keep individual failures identifiable. Avoid
standalone import/banner checks or copying an implementation formula as the
only assertion. The parser suite uses compact scenario tables with explicit
expected values.

All suites use local synthetic models/artifacts and require no downloaded model
or dataset. MLX imports/execution require Metal access on Apple Silicon.
