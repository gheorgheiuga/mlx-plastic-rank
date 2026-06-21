import pytest

from mlx_plastic_rank.packs.cli import build_parser


def test_create_parser_accepts_rank_strategy():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--rank-strategy",
            "stable",
        ]
    )
    assert args.rank_strategy == "stable"


def test_create_parser_keeps_lite_projection_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
        ]
    )
    assert args.loader == "auto"
    assert args.layers == "attn.q_proj,attn.k_proj,attn.v_proj"


def test_create_parser_accepts_explicit_vlm_loader():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--loader",
            "mlx-vlm",
        ]
    )
    assert args.loader == "mlx-vlm"


def test_create_parser_accepts_chat_template():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/industrybench_en_train.jsonl",
            "--chat-template",
        ]
    )
    assert args.chat_template is True


def test_create_parser_accepts_answer_loss_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--loss-mode",
            "answer",
        ]
    )
    assert args.loss_mode == "answer"


def test_create_parser_accepts_resume_pack():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "phase-two",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--resume-pack",
            "phase-one",
        ]
    )
    assert args.resume_pack == "phase-one"


def test_create_parser_accepts_rank_map_from_pack():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "hetero-scratch",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--rank-map-from-pack",
            "phase-one",
        ]
    )
    assert args.rank_map_from_pack == "phase-one"


def test_create_parser_accepts_rank_map_json():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "spectral-map",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--rank-map-json",
            "out/spectral_key_rank_map.json",
        ]
    )
    assert args.rank_map_json == "out/spectral_key_rank_map.json"


def test_eval_parser_accepts_chat_template():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data-path",
            "data/industrybench_en_eval.jsonl",
            "--chat-template",
        ]
    )
    assert args.chat_template is True


def test_eval_parser_accepts_answer_loss_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data-path",
            "data/fault_codes_eval.jsonl",
            "--loss-mode",
            "answer",
        ]
    )
    assert args.loss_mode == "answer"


def test_rank_ledger_parser_accepts_compare_and_outputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-ledger",
            "--name",
            "fault-codes-a",
            "--compare",
            "fault-codes-b",
            "--rank-tol",
            "1e-4",
            "--out",
            "out/ledger.json",
            "--csv",
            "out/ledger.csv",
        ]
    )
    assert args.name == "fault-codes-a"
    assert args.compare == "fault-codes-b"
    assert args.rank_tol == 1e-4
    assert args.out == "out/ledger.json"
    assert args.csv == "out/ledger.csv"


def test_device_profiles_parser_accepts_outputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "device-profiles",
            "--profiles",
            "8gb,16gb",
            "--out",
            "out/device_profiles.json",
            "--markdown",
            "out/device_profiles.md",
        ]
    )

    assert args.command == "device-profiles"
    assert args.profiles == "8gb,16gb"
    assert args.out == "out/device_profiles.json"


def test_memory_ledger_parser_accepts_artifact_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "memory-ledger",
            "--pack",
            "hetero-source",
            "--profiles",
            "16gb,32gb",
            "--eval-report",
            "out/eval.json",
            "--eval-batch-report",
            "out/eval_batch.json",
            "--generation-report",
            "out/generation.json",
            "--rank-budget-report",
            "out/rank_budget.json",
            "--base-model-gb",
            "9.5",
            "--extra-overhead-gb",
            "0.5",
            "--host-rss-peak-gb",
            "11.0",
            "--observed-peak-gb",
            "10.75",
            "--out",
            "out/memory_ledger.json",
            "--markdown",
            "out/memory_ledger.md",
            "--csv",
            "out/memory_ledger.csv",
        ]
    )

    assert args.command == "memory-ledger"
    assert args.pack == "hetero-source"
    assert args.profiles == "16gb,32gb"
    assert args.base_model_gb == 9.5
    assert args.observed_peak_gb == [10.75]
    assert args.csv == "out/memory_ledger.csv"


def test_ablation_report_parser_accepts_pack_and_eval_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "ablation-report",
            "--pack",
            "hetero-source",
            "--unit",
            "prefix",
            "--prefix-rank",
            "8",
            "--top-k",
            "12",
            "--targets",
            "attn.q_proj,attn.k_proj",
            "--layers",
            "0,1,2",
            "--ablation-pack-root",
            "out/ablations",
            "--baseline-eval",
            "out/baseline_eval.json",
            "--ablation-eval",
            "prefix-blocks_0_attn_q_proj-keep0008=out/ablated_eval.json",
            "--out",
            "out/ablation_report.json",
            "--markdown",
            "out/ablation_report.md",
            "--csv",
            "out/ablation_report.csv",
        ]
    )

    assert args.command == "ablation-report"
    assert args.pack == "hetero-source"
    assert args.unit == "prefix"
    assert args.prefix_rank == 8
    assert args.top_k == 12
    assert args.ablation_eval == [
        "prefix-blocks_0_attn_q_proj-keep0008=out/ablated_eval.json"
    ]


def test_rank_map_spectral_parser_accepts_probe_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "spectral",
            "--source-pack",
            "hetero-source",
            "--q-spectral",
            "out/q.json",
            "--k-spectral",
            "out/k.json",
            "--v-spectral",
            "out/v.json",
            "--profile",
            "heavy",
            "--policy",
            "balanced",
            "--out",
            "out/rank-map.json",
        ]
    )
    assert args.command == "rank-map"
    assert args.rank_map_command == "spectral"
    assert args.source_pack == "hetero-source"
    assert args.q_spectral == "out/q.json"
    assert args.k_spectral == "out/k.json"
    assert args.v_spectral == "out/v.json"
    assert args.profile == "heavy"
    assert args.out == "out/rank-map.json"


def test_rank_map_budget_report_parser_accepts_artifact_paths():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "budget-report",
            "--source-pack",
            "hetero-source",
            "--fixed-rank",
            "16",
            "--profile",
            "heavy",
            "--out",
            "out/r16_budget.json",
            "--markdown",
            "out/r16_budget.md",
            "--rank-map-out",
            "out/r16_rank_map.json",
        ]
    )

    assert args.rank_map_command == "budget-report"
    assert args.source_pack == "hetero-source"
    assert args.fixed_rank == 16
    assert args.out == "out/r16_budget.json"
    assert args.markdown == "out/r16_budget.md"


def test_rank_map_normalize_budget_parser_accepts_targets():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "normalize-budget",
            "--source-pack",
            "hetero-source",
            "--target",
            "fixed-r32-percent",
            "--target-fixed-r32-pct",
            "40",
            "--out",
            "out/normalized.json",
            "--markdown",
            "out/normalized.md",
            "--rank-map-out",
            "out/normalized_rank_map.json",
        ]
    )

    assert args.rank_map_command == "normalize-budget"
    assert args.target == "fixed-r32-percent"
    assert args.target_fixed_r32_pct == 40.0
    assert args.rank_map_out == "out/normalized_rank_map.json"


def test_rank_map_random_same_budget_parser_accepts_seeded_control():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "random-same-budget",
            "--source-pack",
            "hetero-source",
            "--rank-map-json",
            "out/discovered.json",
            "--seed",
            "17",
            "--out",
            "out/random_control.json",
            "--markdown",
            "out/random_control.md",
            "--rank-map-out",
            "out/random_rank_map.json",
        ]
    )

    assert args.rank_map_command == "random-same-budget"
    assert args.seed == 17
    assert args.rank_map_json == "out/discovered.json"
    assert args.rank_map_out == "out/random_rank_map.json"


def test_rank_map_shuffled_discovered_parser_accepts_seeded_control():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "shuffled-discovered",
            "--source-pack",
            "hetero-source",
            "--seed",
            "5",
            "--out",
            "out/shuffled_control.json",
            "--markdown",
            "out/shuffled_control.md",
            "--rank-map-out",
            "out/shuffled_rank_map.json",
        ]
    )

    assert args.rank_map_command == "shuffled-discovered"
    assert args.seed == 5
    assert args.source_pack == "hetero-source"
    assert args.rank_map_out == "out/shuffled_rank_map.json"


def test_rank_map_validate_parser_accepts_rank_map_json():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "validate",
            "--source-pack",
            "hetero-source",
            "--rank-map-json",
            "out/hetero.json",
            "--out",
            "out/validation.json",
            "--markdown",
            "out/validation.md",
        ]
    )

    assert args.rank_map_command == "validate"
    assert args.rank_map_json == "out/hetero.json"
    assert args.out == "out/validation.json"


def test_proof_parser_accepts_artifact_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "proof",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--pack",
            "domain-pack",
            "--domain",
            "fault-codes",
            "--train-data",
            "data/train.jsonl",
            "--eval-data",
            "data/eval.jsonl",
            "--eval-report",
            "out/eval.json",
            "--generation-report",
            "out/generation.json",
            "--ledger-report",
            "out/ledger.json",
            "--require-generation",
            "--require-ledger",
            "--fail-on-regression",
            "--out",
            "out/proof.json",
        ]
    )
    assert args.command == "proof"
    assert args.pack == "domain-pack"
    assert args.domain == "fault-codes"
    assert args.require_generation is True
    assert args.require_ledger is True
    assert args.fail_on_regression is True


def test_bakeoff_parser_accepts_spec_dry_run_and_force():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bakeoff",
            "--spec",
            "codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json",
            "--dry-run",
            "--force",
        ]
    )
    assert args.command == "bakeoff"
    assert args.spec == "codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json"
    assert args.dry_run is True
    assert args.force is True


def test_create_parser_rejects_invalid_dropout():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create",
                "--name",
                "demo",
                "--base",
                "mlx-community/gemma-4-12B-mxfp8",
                "--data",
                "data/domain_prompts.jsonl",
                "--lora-dropout",
                "1.0",
            ]
        )


def test_create_parser_accepts_heavy_profile():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--profile",
            "heavy",
        ]
    )
    assert args.profile == "heavy"


def test_create_parser_accepts_min_rank():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--min-rank",
            "16",
        ]
    )
    assert args.min_rank == 16


def test_create_parser_accepts_explicit_rank():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/industrybench_en_train.jsonl",
            "--rank",
            "8",
        ]
    )
    assert args.rank == 8


def test_create_parser_accepts_dynamic_rank_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "dynamic-demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--rank",
            "32",
            "--dynamic-rank",
            "--dynamic-initial-rank",
            "4",
            "--dynamic-min-rank",
            "2",
            "--dynamic-rank-interval",
            "25",
            "--dynamic-rank-warmup",
            "50",
            "--dynamic-grow-threshold",
            "0.4",
            "--dynamic-prune-threshold",
            "0.05",
        ]
    )
    assert args.rank == 32
    assert args.dynamic_rank is True
    assert args.dynamic_initial_rank == 4
    assert args.dynamic_min_rank == 2
    assert args.dynamic_rank_interval == 25
    assert args.dynamic_rank_warmup == 50
    assert args.dynamic_grow_threshold == 0.4
    assert args.dynamic_prune_threshold == 0.05


def test_create_parser_rejects_zero_explicit_rank():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create",
                "--name",
                "demo",
                "--base",
                "mlx-community/gemma-4-12B-mxfp8",
                "--data",
                "data/domain_prompts.jsonl",
                "--rank",
                "0",
            ]
        )


def test_route_parser_accepts_required_args(tmp_path):
    parser = build_parser()
    mapping = tmp_path / "map.json"
    reqs = tmp_path / "reqs.jsonl"
    mapping.write_text("{}", encoding="utf-8")
    reqs.write_text("", encoding="utf-8")
    args = parser.parse_args(
        [
            "route",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--domain-map",
            str(mapping),
            "--input",
            str(reqs),
            "--probe-forward",
        ]
    )
    assert args.command == "route"
    assert args.probe_forward is True


def test_capabilities_parser_accepts_json_check():
    parser = build_parser()
    args = parser.parse_args(["capabilities", "--json", "--check"])
    assert args.command == "capabilities"
    assert args.json is True
    assert args.check is True
