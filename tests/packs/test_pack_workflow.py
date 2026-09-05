"""Exercise real pack training and evidence production on a tiny local fixture."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_plastic_rank.packs import bakeoff, cli


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text):
        # Deliberately trivial prediction task; no external data or checkpoint.
        return [2] * len(text)

    def get_vocab(self):
        return {str(i): i for i in range(8)}


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.q_proj.weight = mx.zeros((8, 8))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [TinyBlock()]
        self.config = SimpleNamespace(hidden_size=8)
        self.model_type = "gemma"

    def __call__(self, inputs):
        return self.layers[0].self_attn.q_proj(mx.ones((*inputs.shape, 8)))


def test_bakeoff_trains_proves_resumes_and_rejects_changed_inputs(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "tiny-base"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"hidden_size": 8}')
    mx.save_safetensors(str(checkpoint / "model.safetensors"), {"weight": mx.zeros((8, 8))})
    train = tmp_path / "train.jsonl"
    train.write_text('{"prompt": "training", "answer": "yes"}\n')
    evaluate = tmp_path / "eval.jsonl"
    evaluate.write_text('{"prompt": "evaluation", "answer": "yes"}\n')
    monkeypatch.setattr(cli, "PACK_ROOT", tmp_path / "packs")
    monkeypatch.setattr(cli, "_require_load_model", lambda: lambda path: (TinyModel(), TinyTokenizer()))
    phases_run = []

    def run_cli(command, **kwargs):
        # Replace only the subprocess boundary and external model loader.
        # Training, serialization, attachment, eval, ledger and proof stay real.
        phases_run.append(command[3])
        args = cli.build_parser().parse_args(command[3:])
        args.func(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bakeoff.subprocess, "run", run_cli)
    spec = bakeoff.validate_bakeoff_spec({
        "name": "tiny-workflow", "domain": "synthetic", "base": str(checkpoint),
        "loader": "mlx-lm", "train_data": str(train), "eval_data": str(evaluate),
        "output_dir": "out", "layers": "attn.q_proj", "profile": "lite",
        "train": {"steps": 10, "batch_size": 1, "learning_rate": 0.1,
                  "sequence_length": 64, "loss_mode": "answer", "initialization": "component-v1"},
        "eval": {"sequence_length": 64, "loss_mode": "answer", "num_samples": 0, "batch_size": 1},
        "candidates": [{"id": "tiny", "pack": "tiny-pack", "mode": "fixed_rank", "rank": 4}],
    }, root=tmp_path)

    first = bakeoff.run_bakeoff(spec)
    assert phases_run == ["create", "eval", "rank-ledger", "proof"]
    assert first["artifact_provenance_verified"]
    assert first["rows"][0]["proof_status"] == "passed"
    assert first["rows"][0]["perplexity"] < first["base_metrics"]["perplexity"]
    assert bakeoff.run_bakeoff(spec) == first
    assert len(phases_run) == 4

    meta = json.loads((tmp_path / "packs/tiny-pack/meta.json").read_text())
    assert meta["training_config"]["initialization"] == "component-v1"
    original_eval = evaluate.read_text()
    evaluate.write_text('{"prompt": "changed evaluation", "answer": "yes"}\n')
    with pytest.raises(bakeoff.BakeoffError, match="Stale or unverified"):
        bakeoff.run_bakeoff(spec)
    assert len(phases_run) == 4
    evaluate.write_text(original_eval)
    (checkpoint / "model.safetensors").write_bytes(b"replaced checkpoint")
    with pytest.raises(bakeoff.BakeoffError, match="Stale or unverified"):
        bakeoff.run_bakeoff(spec)
    assert len(phases_run) == 4
