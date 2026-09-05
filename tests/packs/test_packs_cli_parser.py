"""CLI option contracts, grouped as scenarios without repeating parser setup."""

import shlex

import pytest

from mlx_plastic_rank.packs.cli import build_parser

CREATE = 'create --name demo --base /local/checkpoint --data data/train.jsonl'
EVAL = 'eval --base /local/checkpoint --data-path data/eval.jsonl'


@pytest.mark.parametrize(("command", "expected"), [
    pytest.param(
        CREATE + ' --rank-strategy stable',
        {'rank_strategy': 'stable'},
        id='create-accepts-rank-strategy',
    ),
    pytest.param(
        CREATE,
        {'rank_strategy': 'gram_energy'},
        id='create-defaults-to-gram-energy-rank-strategy',
    ),
    pytest.param(
        CREATE + ' --rank-strategy theorem',
        {'rank_strategy': 'theorem'},
        id='create-keeps-legacy-theorem-rank-strategy-alias',
    ),
    pytest.param(
        CREATE,
        {'loader': 'auto', 'layers': 'attn.q_proj,attn.k_proj,attn.v_proj'},
        id='create-keeps-lite-projection-default',
    ),
    pytest.param(
        CREATE + ' --loader mlx-vlm',
        {'loader': 'mlx-vlm'},
        id='create-accepts-explicit-vlm-loader',
    ),
    pytest.param(
        CREATE + ' --chat-template',
        {'chat_template': True},
        id='create-accepts-chat-template',
    ),
    pytest.param(
        CREATE + ' --loss-mode answer',
        {'loss_mode': 'answer'},
        id='create-accepts-answer-loss-mode',
    ),
    pytest.param(
        CREATE + ' --resume-pack phase-one',
        {'resume_pack': 'phase-one'},
        id='create-accepts-resume-pack',
    ),
    pytest.param(
        CREATE + ' --rank-map-from-pack phase-one',
        {'rank_map_from_pack': 'phase-one'},
        id='create-accepts-rank-map-from-pack',
    ),
    pytest.param(
        CREATE + ' --rank-map-json out/spectral_key_rank_map.json',
        {'rank_map_json': 'out/spectral_key_rank_map.json'},
        id='create-accepts-rank-map-json',
    ),
    pytest.param(
        EVAL + ' --chat-template',
        {'chat_template': True},
        id='eval-accepts-chat-template',
    ),
    pytest.param(
        EVAL + ' --loss-mode answer',
        {'loss_mode': 'answer'},
        id='eval-accepts-answer-loss-mode',
    ),
    pytest.param(
        'rank-ledger --name fault-codes-a --compare fault-codes-b --rank-tol 1e-4 --out'
        ' out/ledger.json --csv out/ledger.csv',
        {'name': 'fault-codes-a',
         'compare': 'fault-codes-b',
         'rank_tol': 0.0001,
         'out': 'out/ledger.json',
         'csv': 'out/ledger.csv'},
        id='rank-ledger-accepts-compare-and-outputs',
    ),
    pytest.param(
        'device-profiles --profiles 8gb,16gb --out out/device_profiles.json --markdown'
        ' out/device_profiles.md',
        {'command': 'device-profiles',
         'profiles': '8gb,16gb',
         'out': 'out/device_profiles.json'},
        id='device-profiles-accepts-outputs',
    ),
    pytest.param(
        'memory-ledger --pack hetero-source --profiles 16gb,32gb --eval-report out/eval.json'
        ' --eval-batch-report out/eval_batch.json --generation-report out/generation.json'
        ' --rank-budget-report out/rank_budget.json --base-model-gb 9.5 --extra-overhead-gb 0.5'
        ' --host-rss-peak-gb 11.0 --observed-peak-gb 10.75 --out out/memory_ledger.json'
        ' --markdown out/memory_ledger.md --csv out/memory_ledger.csv',
        {'command': 'memory-ledger',
         'pack': 'hetero-source',
         'profiles': '16gb,32gb',
         'base_model_gb': 9.5,
         'observed_peak_gb': [10.75],
         'csv': 'out/memory_ledger.csv'},
        id='memory-ledger-accepts-artifact-inputs',
    ),
    pytest.param(
        'ablation-report --pack hetero-source --unit prefix --prefix-rank 8 --top-k 12'
        ' --targets attn.q_proj,attn.k_proj --layers 0,1,2 --ablation-pack-root out/ablations'
        ' --baseline-eval out/baseline_eval.json --ablation-eval'
        ' prefix-blocks_0_attn_q_proj-keep0008=out/ablated_eval.json --out'
        ' out/ablation_report.json --markdown out/ablation_report.md --csv'
        ' out/ablation_report.csv',
        {'command': 'ablation-report',
         'pack': 'hetero-source',
         'unit': 'prefix',
         'prefix_rank': 8,
         'top_k': 12,
         'ablation_eval': ['prefix-blocks_0_attn_q_proj-keep0008=out/ablated_eval.json']},
        id='ablation-report-accepts-pack-and-eval-inputs',
    ),
    pytest.param(
        'rank-map spectral --source-pack hetero-source --q-spectral out/q.json --k-spectral'
        ' out/k.json --v-spectral out/v.json --profile heavy --policy balanced --out'
        ' out/rank-map.json',
        {'command': 'rank-map',
         'rank_map_command': 'spectral',
         'source_pack': 'hetero-source',
         'q_spectral': 'out/q.json',
         'k_spectral': 'out/k.json',
         'v_spectral': 'out/v.json',
         'profile': 'heavy',
         'out': 'out/rank-map.json'},
        id='rank-map-spectral-accepts-probe-inputs',
    ),
    pytest.param(
        'rank-map budget-report --source-pack hetero-source --fixed-rank 16 --profile heavy'
        ' --out out/r16_budget.json --markdown out/r16_budget.md --rank-map-out'
        ' out/r16_rank_map.json',
        {'rank_map_command': 'budget-report',
         'source_pack': 'hetero-source',
         'fixed_rank': 16,
         'out': 'out/r16_budget.json',
         'markdown': 'out/r16_budget.md'},
        id='rank-map-budget-report-accepts-artifact-paths',
    ),
    pytest.param(
        'rank-map normalize-budget --source-pack hetero-source --target fixed-r32-percent'
        ' --target-fixed-r32-pct 40 --out out/normalized.json --markdown out/normalized.md'
        ' --rank-map-out out/normalized_rank_map.json',
        {'rank_map_command': 'normalize-budget',
         'target': 'fixed-r32-percent',
         'target_fixed_r32_pct': 40.0,
         'rank_map_out': 'out/normalized_rank_map.json'},
        id='rank-map-normalize-budget-accepts-targets',
    ),
    pytest.param(
        'rank-map random-same-budget --source-pack hetero-source --rank-map-json'
        ' out/discovered.json --seed 17 --out out/random_control.json --markdown'
        ' out/random_control.md --rank-map-out out/random_rank_map.json',
        {'rank_map_command': 'random-same-budget',
         'seed': 17,
         'rank_map_json': 'out/discovered.json',
         'rank_map_out': 'out/random_rank_map.json'},
        id='rank-map-random-same-budget-accepts-seeded-control',
    ),
    pytest.param(
        'rank-map shuffled-discovered --source-pack hetero-source --seed 5 --out'
        ' out/shuffled_control.json --markdown out/shuffled_control.md --rank-map-out'
        ' out/shuffled_rank_map.json',
        {'rank_map_command': 'shuffled-discovered',
         'seed': 5,
         'source_pack': 'hetero-source',
         'rank_map_out': 'out/shuffled_rank_map.json'},
        id='rank-map-shuffled-discovered-accepts-seeded-control',
    ),
    pytest.param(
        'rank-map validate --source-pack hetero-source --rank-map-json out/hetero.json --out'
        ' out/validation.json --markdown out/validation.md',
        {'rank_map_command': 'validate',
         'rank_map_json': 'out/hetero.json',
         'out': 'out/validation.json'},
        id='rank-map-validate-accepts-rank-map-json',
    ),
    pytest.param(
        'proof --base mlx-community/gemma-4-12B-it-qat-mxfp8 --pack domain-pack --domain'
        ' fault-codes --train-data data/train.jsonl --eval-data data/eval.jsonl --eval-report'
        ' out/eval.json --generation-report out/generation.json --ledger-report out/ledger.json'
        ' --require-generation --require-ledger --fail-on-regression --out out/proof.json',
        {'command': 'proof',
         'pack': 'domain-pack',
         'domain': 'fault-codes',
         'require_generation': True,
         'require_ledger': True,
         'fail_on_regression': True},
        id='proof-accepts-artifact-inputs',
    ),
    pytest.param(
        'bakeoff --spec codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json --dry-run --force',
        {'command': 'bakeoff',
         'spec': 'codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json',
         'dry_run': True,
         'force': True},
        id='bakeoff-accepts-spec-dry-run-and-force',
    ),
    pytest.param(
        CREATE + ' --profile heavy',
        {'profile': 'heavy'},
        id='create-accepts-heavy-profile',
    ),
    pytest.param(
        CREATE + ' --min-rank 16',
        {'min_rank': 16},
        id='create-accepts-min-rank',
    ),
    pytest.param(
        CREATE + ' --rank 8',
        {'rank': 8},
        id='create-accepts-explicit-rank',
    ),
    pytest.param(
        CREATE + ' --rank 32 --dynamic-rank --dynamic-initial-rank 4 --dynamic-min-rank 2'
        ' --dynamic-rank-interval 25 --dynamic-rank-warmup 50 --dynamic-grow-threshold 0.4'
        ' --dynamic-prune-threshold 0.05',
        {'rank': 32,
         'dynamic_rank': True,
         'dynamic_initial_rank': 4,
         'dynamic_min_rank': 2,
         'dynamic_rank_interval': 25,
         'dynamic_rank_warmup': 50,
         'dynamic_grow_threshold': 0.4,
         'dynamic_prune_threshold': 0.05},
        id='create-accepts-dynamic-rank-controls',
    ),
    pytest.param(
        'capabilities --json --check',
        {'command': 'capabilities', 'json': True, 'check': True},
        id='capabilities-accepts-json-check',
    ),
])
def test_parser_accepts_scenario(command, expected):
    actual = vars(build_parser().parse_args(shlex.split(command)))
    for name, value in expected.items():
        if isinstance(value, bool):
            assert actual[name] is value, name
        else:
            assert actual[name] == value, name


@pytest.mark.parametrize("options", [
    pytest.param('--lora-dropout 1.0', id='create-rejects-invalid-dropout'),
    pytest.param('--rank 0', id='create-rejects-zero-explicit-rank'),
])
def test_create_rejects_invalid_values(options):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(shlex.split(CREATE + " " + options))
    assert error.value.code == 2


def test_route_parser_accepts_required_args(tmp_path):
    mapping = tmp_path / "domain map.json"
    requests = tmp_path / "requests.jsonl"
    args = build_parser().parse_args([
        "route", "--base", "/local/checkpoint", "--domain-map", str(mapping),
        "--input", str(requests), "--probe-forward",
    ])
    assert args.command == "route"
    assert args.probe_forward is True
    assert args.domain_map == mapping
    assert args.input == requests
