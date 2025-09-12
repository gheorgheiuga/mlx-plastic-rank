# Codex runbook — mlx-plastic-rank

Goal
- Build an MLX toolkit for adaptive low‑rank compression with neuroplastic triggers. Keep components reversible, support parking pruned factors for reactivation, export weights, and ship a demo plus tests.

Repo layout (current → target)
- Current: flat modules `main.py`, `plastic_rank.py`, tests in `tests/`.
- Target (Phase 2):
  - `src/mlx_plastic_rank/`
    - `lowrank.py` (RankLayer and utility ops)
    - `plasticity_manager.py`
    - `rank_select.py` (policies/heuristics)
    - `utils.py` (quantize/dequantize, stable_rank, seeds)
    - `export_safetensors.py` (optional)
  - Tests remain in `tests/`

Commands
- Setup environment: `uv venv && uv pip install -e .` (Apple Silicon + MLX required for MLX features)
- Run sanity check: `uv run python main.py`
- Run demo (Apple Silicon + MLX required): `uv run python plastic_rank.py`
- Run tests: `uv run pytest -q` (or `-k rank_layer` for targeted tests)
- Reset env: `rm -rf .venv && uv venv`

Implementation rules
- Use Python 3.13. Package with uv.
- Use MLX APIs for arrays and linear algebra.
- Prefer reversible operations; do not hard‑delete capacity—park pruned components in a dict and allow wake‑up.
- Ensure determinism: seed MLX RNG for tests when feasible.
- Keep dependencies minimal; provide clear logs for rank and pruning decisions per layer.

Status checklist
- [x] Rebrand to `mlx-plastic-rank` (pyproject + banner + tests)
- [x] Add `codex/` with runbook and ADRs
 - [x] Fix `quantise` shadowing of `mx` module
 - [x] Fix `bias` truthiness in `RankLayer.__call__`
 - [x] Fix `compress_dict_size` unpack bug
 - [x] Remove unused imports in `plastic_rank.py`
 - [~] Simplify demo loop (no param grads yet)
- [ ] Move modules under `src/mlx_plastic_rank/` and adjust imports
- [ ] Expand tests: rank add/prune/wake; shape/dtype checks

Research Inbox (add items here; do not commit large PDFs)
- Template:
  - Citation/DOI:
  - Link:
  - Key idea (2–4 bullets):
  - How it maps to our code (files/functions):
  - Open questions or conflicts with current design:

Notes
- If research guidance conflicts with existing code, document the decision in `codex/decisions.md` and reference the Research Inbox entry.
- Avoid committing large artifacts; prefer links and notes.

## Key modules to implement

1. rank_select.py
   - `stable_rank(A, eps=1e-6) -> float`
   - `theorem_guided_rank(A, target_compression: float) -> int`
   - Implement the polynomial check and the heuristic described in the Grok report. Use Horner evaluation and a tolerant equality check. Document that the theorem check is used as a heuristic for `r`. See the report notes on saving to SafeTensors and the example layer traversal for q, k, v, o and gate, up, down projections. [grok_report]

2. lowrank.py
   - `svd_lowrank(A, r) -> A_approx`
   - `factorized_lowrank(A, r) -> (U, S, Vh)` for reversible storage
   - Quantize to 8‑bit (optional): `quantize_factors(U, S, Vh, bits=8)`

3. plasticity_manager.py
   - Maintains moving validation loss and a delta threshold `delta`.
   - When loss change < `delta`, trigger growth or shrink on selected layers.
   - Selection policy: placeholder LRP score or simple gradient‑norm ranking.
   - Strategy options: `"stable"`, `"theorem"`, `"activation"`.
   - Growth: add LoRA‑style factors of rank `k`.
   - Shrink: soft‑threshold singular values, park tiny components in `sleep_dict`.
   - Reactivation path if validation worsens after shrink.

4. export_safetensors.py
   - Extract current model weights as dict, then `mx.save_safetensors(path, weights)`.
   - Print instructions to convert to GGUF with `llama.cpp/convert_hf_to_gguf.py` and flags to carry tokenizer and chat template metadata as the Grok report advises. [grok_report]

5. utils.py
   - Helpers for Jacobian or activation covariance rank estimation.
   - Logging utilities.

### Demo script
- `scripts/demo_plasticity_blocks.py`
  - Build a small stub model from `PlasticBlock` modules (no checkpoint I/O).
  - Run a short plasticity phase with `PlasticityManager` on random inputs.
  - Log rank changes to JSONL and print a compact table with bytes units.

### Compression script
- `scripts/compress_gemma2_mlx.py` (optional)
  - Downloads a Hugging Face MLX checkpoint and compresses 2‑D tensors with SVD,
    choosing rank via `rank_select.choose_rank` ("stable" or "theorem").
  - Writes compressed `.safetensors` and copies tokenizer/config alongside; emits a small meta JSON.
  - Requires extra deps (e.g., `huggingface_hub`).

### Tests
1. test_rank_select.py
   - Random matrix and near‑idempotent matrix cases.
   - Assert `theorem_guided_rank` returns a sensible `r` within `[1, min(A.shape)]`.

2. test_roundtrip.py
   - For a random weight `A`: compress with `svd_lowrank`, reconstruct, and check relative Frobenius error below a threshold expected for rank `r`.

3. test_plasticity_triggers.py
   - Simulate validation loss plateau; ensure `PlasticityManager` increases or decreases rank and logs the action.

### Dev environment
Create `pyproject.toml` with uv groups:

```ini
[project]
name = "mlx-plastic-rank"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["mlx>=0.20.0", "numpy", "sympy", "safetensors"]

[tool.uv]
dev-dependencies = ["pytest", "ruff", "mypy", "pytest-cov"]
```
