import json

from scripts.forgetting_machine_demo import main, render_html, run_demo


def test_demo_observes_query_removal_and_checks_policy_attestations():
    report = run_demo()

    attack = report["extraction_attack"]
    assert attack["before_forgetting"]["records_recovered"] == 3
    assert attack["after_forgetting"]["records_recovered"] == 0
    assert report["claims"] == {
        "alice_payloads_tombstoned_by_emitter": True,
        "unselected_payloads_preserved_by_emitter": True,
        "both_attestations_integrity_and_algebra_checked_at_generation": True,
        "retained_history_chain_checked": True,
        "pop_policy_identity_checked": True,
    }

    composition = report["pop_composition"]
    assert composition["shared_target_record_ids"] == ["alice-medical"]
    assert composition["ranks"]["left_sum"] == composition["ranks"]["right_sum"]

    certificate_json = json.dumps(
        [policy["certificate"] for policy in report["policies"]],
        sort_keys=True,
    )
    assert "Allergic to penicillin" not in certificate_json
    assert "emergency contact" not in certificate_json


def test_demo_cli_writes_self_contained_json_and_html(tmp_path):
    json_path = tmp_path / "forgetting-machine.json"
    html_path = tmp_path / "forgetting-machine.html"

    assert main(["--out-json", str(json_path), "--out-html", str(html_path)]) == 0

    report = json.loads(json_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")
    assert report["kind"] == "forgetting_machine_demo_report"
    assert html == render_html(report)
    assert "Two attestations checked" in html
    assert "POP POLICY IDENTITY CHECKED" not in html
    assert "https://" not in html
    assert "http://" not in html
    assert html.count("<head>") == html.count("</head>") == 1
