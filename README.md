mlx-plastic-rank
================

Why Plastic Rank?

Most approaches to model compression prune or distill, permanently cutting capacity and often losing subtle behaviors. Plastic rank takes inspiration from neuroplasticity: instead of discarding, weights are re-expressed in adaptive low-rank factors that can grow, shrink, or wake as needed. This makes compression reversible, transparent, and tunable — a closer analogy to how brains reorganize rather than amputate. The result is not just smaller models, but flexible ones that adjust rank dynamically while preserving knowledge.

[![CI](https://github.com/gheorgheiuga/mlx-plastic-rank/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gheorgheiuga/mlx-plastic-rank/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Adaptive low-rank compression with neuroplastic triggers for MLX. Includes reversible factors, quantized sleep store, SafeTensors export, and demos/tests.

Quick start
- Create env: `uv venv` (optional) and `source .venv/bin/activate` or use `uv run`
- Install: `uv pip install -e .`
- Extras for compression tools: `uv pip install -e '.[compress]'`
- Tests: `uv run pytest -q`
- Demo: `uv run python scripts/demo_plasticity_blocks.py --strategy stable --compress 0.3 --steps 5`
- Optional (requires installing the extras for huggingface_hub):
  
`uv run python scripts/compress_gemma2_mlx.py --hf openai-community/gpt2 --out out/gpt2_mlx_compressed --strategy stable --compress 0.30 --svd randomized --svd-oversamples 6 --svd-iters 0 --device gpu --gpu-max-bytes 800000000 --gpu-max-dim 2048 --gpu-chunk-k 8 --min-dim 64 --max-rank 128`   

Results in “compressed 50 of 160 tensors.” That shows it’s doing something — but I’ll admit: I don’t yet know how to verify or interpret the impact on the model’s guts (perplexity, activations, speed)

- Bench: `uv run python scripts/bench_memory.py --m 2048 --n 512`

Notes
- Apple Silicon + MLX required for most ops; SVD forced on CPU stream.
- Export factors: `python -m mlx_plastic_rank.export_safetensors --from-weight weight.npy --rank 64 --bits 8 --out out/weight_lr.safetensors` - not tested yet.

Research preview
- This is a research preview; APIs and behaviors may change between minor versions.

License
- MIT
