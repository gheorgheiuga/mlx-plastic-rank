#!/usr/bin/env python3
"""Run the stage-ready Forgetting Machine demonstration.

The demonstration uses fictional records in a deliberately controlled spectral
memory.  Its certificates do not claim deletion from model weights or from any
storage outside :class:`ForgettingVault`.
"""

from __future__ import annotations

import argparse
import html
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from research.forgetting_vault import (
    DeletionCertificate,
    ForgetPolicy,
    ForgettingVault,
    MemoryRecord,
    VerificationReport,
    verify_certificate_chain,
)

PROOF_SCOPE = (
    "The live process observes query removal inside one controlled ForgettingVault. "
    "Its HMAC integrity and Pop policy algebra are checked at generation with an "
    "ephemeral secret that is not exported; the saved artifact is therefore not "
    "publicly re-verifiable. Certificates cannot inspect historical storage. This "
    "does not prove erasure from model weights, caller-owned objects, logs, caches, "
    "backups, or process memory."
)


def _synthetic_records() -> list[MemoryRecord]:
    """Return the fictional memory corpus used by the live demonstration."""

    return [
        MemoryRecord(
            record_id="alice-medical",
            subject="alice",
            category="medical",
            payload="Allergic to penicillin; emergency contact is Mara.",
        ),
        MemoryRecord(
            record_id="alice-travel",
            subject="alice",
            category="travel",
            payload="Flight to Bucharest on 14 July; vegetarian meal requested.",
        ),
        MemoryRecord(
            record_id="alice-preference",
            subject="alice",
            category="preference",
            payload="Prefers quiet hotel rooms away from elevators.",
        ),
        MemoryRecord(
            record_id="bob-medical",
            subject="bob",
            category="medical",
            payload="Blood type O positive; annual review due in September.",
        ),
        MemoryRecord(
            record_id="carol-medical",
            subject="carol",
            category="medical",
            payload="Uses a latex-free care kit.",
        ),
        MemoryRecord(
            record_id="dylan-travel",
            subject="dylan",
            category="travel",
            payload="Train to Vienna on 18 July; bicycle reservation confirmed.",
        ),
        MemoryRecord(
            record_id="plant-maintenance",
            subject="plant",
            category="maintenance",
            payload="Pump P-204 bearing inspection scheduled for Monday.",
        ),
    ]


def _records_as_dicts(records: Sequence[MemoryRecord]) -> list[dict[str, str]]:
    return [
        {
            "record_id": record.record_id,
            "subject": record.subject,
            "category": record.category,
            "payload": record.payload,
        }
        for record in records
    ]


def _verification_as_dict(report: VerificationReport) -> dict[str, Any]:
    return {"valid": report.valid, "failures": list(report.failures)}


def _verified_certificate(
    certificate: DeletionCertificate,
    integrity_key: bytes,
) -> tuple[dict[str, Any], VerificationReport]:
    verification = certificate.verify(integrity_key)
    if not verification.valid:
        raise RuntimeError(
            "Demo generated an invalid certificate: " + ", ".join(verification.failures)
        )
    return certificate.to_dict(), verification


def run_demo(*, integrity_key: bytes | None = None) -> dict[str, Any]:
    """Execute both deletion policies and return a self-contained evidence report."""

    records = _synthetic_records()
    demo_key = integrity_key if integrity_key is not None else secrets.token_bytes(32)
    vault = ForgettingVault(
        records,
        integrity_key=demo_key,
        key_id="ephemeral-demo-v1",
    )
    initial_snapshot = vault.snapshot()

    before_alice = vault.search(subject="alice")
    before_control = vault.search(record_id="plant-maintenance")

    alice_certificate = vault.forget(
        ForgetPolicy(
            policy_id="forget-alice",
            subject="alice",
            reason="fictional user deletion request",
        )
    )
    alice_payload, alice_verification = _verified_certificate(
        alice_certificate,
        demo_key,
    )
    after_alice = vault.search(subject="alice")
    after_alice_snapshot = vault.snapshot()

    medical_certificate = vault.forget(
        ForgetPolicy(
            policy_id="forget-medical",
            category="medical",
            reason="fictional category-wide retention policy",
        )
    )
    medical_payload, medical_verification = _verified_certificate(
        medical_certificate,
        demo_key,
    )
    chain_verification = verify_certificate_chain(
        [alice_certificate, medical_certificate],
        demo_key,
        expected_head_digest=medical_certificate.digest,
    )
    final_snapshot = vault.snapshot()

    if after_alice:
        raise RuntimeError("Alice extraction unexpectedly succeeded after forgetting.")
    if vault.search(record_id="plant-maintenance") != before_control:
        raise RuntimeError("Unrelated control record changed during the demonstration.")
    if not medical_payload["pop_compositions"]:
        raise RuntimeError("The overlapping policy did not emit a Pop composition.")

    composition = medical_payload["pop_compositions"][0]
    if not composition["identity_holds"]:
        raise RuntimeError("The emitted Pop rank identity did not verify.")
    if not chain_verification.valid:
        raise RuntimeError(
            "The retained certificate history did not verify: "
            + ", ".join(chain_verification.failures)
        )

    return {
        "kind": "forgetting_machine_demo_report",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "The Forgetting Machine",
        "tagline": "A controlled AI memory that shows its deletion work.",
        "dataset": {
            "fictional": True,
            "record_count": len(records),
            "records": _records_as_dicts(records),
        },
        "extraction_attack": {
            "query": {"subject": "alice"},
            "before_forgetting": {
                "records_recovered": len(before_alice),
                "records": _records_as_dicts(before_alice),
                "succeeded": bool(before_alice),
            },
            "after_forgetting": {
                "records_recovered": len(after_alice),
                "records": _records_as_dicts(after_alice),
                "succeeded": bool(after_alice),
            },
        },
        "policies": [
            {
                "policy_id": "forget-alice",
                "selector": {"subject": "alice"},
                "certificate": alice_payload,
                "verification": _verification_as_dict(alice_verification),
            },
            {
                "policy_id": "forget-medical",
                "selector": {"category": "medical"},
                "certificate": medical_payload,
                "verification": _verification_as_dict(medical_verification),
            },
        ],
        "pop_composition": composition,
        "control_check": {
            "record_id": "plant-maintenance",
            "unchanged": True,
            "record": _records_as_dicts(before_control)[0],
        },
        "snapshots": {
            "initial": initial_snapshot,
            "after_alice": after_alice_snapshot,
            "final": final_snapshot,
        },
        "claims": {
            "alice_payloads_tombstoned_by_emitter": alice_payload["payload_erasure"][
                "emitter_check_passed"
            ],
            "unselected_payloads_preserved_by_emitter": alice_payload["preservation"][
                "emitter_check_passed"
            ],
            "both_attestations_integrity_and_algebra_checked_at_generation": (
                alice_verification.valid and medical_verification.valid
            ),
            "retained_history_chain_checked": chain_verification.valid,
            "pop_policy_identity_checked": composition["identity_holds"],
        },
        "proof_scope": PROOF_SCOPE,
    }


def _short_digest(value: str) -> str:
    return f"{value[:12]}…{value[-8:]}"


def print_narrative(report: dict[str, Any]) -> None:
    """Print a compact, stage-readable narrative."""

    attack = report["extraction_attack"]
    first = report["policies"][0]
    second = report["policies"][1]
    first_certificate = first["certificate"]
    second_certificate = second["certificate"]
    composition = report["pop_composition"]
    ranks = composition["ranks"]

    print("\nTHE FORGETTING MACHINE")
    print("A controlled AI memory that shows its deletion work.\n")
    print(
        "1  EXTRACT BEFORE   "
        f"Alice's {attack['before_forgetting']['records_recovered']} fictional records "
        "are recoverable."
    )
    for record in attack["before_forgetting"]["records"]:
        print(f"   • {record['record_id']}: {record['payload']}")
    print(
        "2  FORGET ALICE    "
        f"{len(first_certificate['newly_tombstoned_record_ids'])} payloads tombstoned; "
        f"attestation {_short_digest(first_certificate['certificate_digest'])} CHECKED."
    )
    print(
        "3  EXTRACT AFTER    "
        f"{attack['after_forgetting']['records_recovered']} Alice records recoverable."
    )
    print(
        "4  COMPOSE POLICY   Medical deletion overlaps on "
        f"{', '.join(composition['shared_target_record_ids'])}; "
        f"{len(second_certificate['newly_tombstoned_record_ids'])} new payloads tombstoned."
    )
    print(
        "5  VERIFY           "
        f"rank(f)+rank(g) = {ranks['f']}+{ranks['g']} = {ranks['left_sum']}; "
        f"rank(gcd)+rank(lcm) = {ranks['gcd']}+{ranks['lcm']} = "
        f"{ranks['right_sum']}. POP POLICY IDENTITY CHECKED."
    )
    print(
        "   Certificate 2    "
        f"{_short_digest(second_certificate['certificate_digest'])} CHECKED; "
        "unrelated control record unchanged."
    )
    print("\nPROOF SCOPE")
    print(PROOF_SCOPE)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json_for_html(value: Any) -> str:
    return html.escape(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), quote=False
    )


def render_html(report: dict[str, Any]) -> str:
    """Render a polished report with no external assets or network dependencies."""

    attack = report["extraction_attack"]
    first = report["policies"][0]["certificate"]
    second = report["policies"][1]["certificate"]
    composition = report["pop_composition"]
    ranks = composition["ranks"]
    before_cards = "".join(
        (
            '<article class="memory-card">'
            f'<span class="chip">{_escape(record["category"])}</span>'
            f'<h3>{_escape(record["record_id"])}</h3>'
            f'<p>{_escape(record["payload"])}</p>'
            "</article>"
        )
        for record in attack["before_forgetting"]["records"]
    )
    shared = ", ".join(composition["shared_target_record_ids"])
    combined = ", ".join(composition["combined_target_record_ids"])
    final_snapshot = report["snapshots"]["final"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Forgetting Machine — Controlled Demo</title>
  <style>
    :root {{ color-scheme: dark; --ink:#f7f8fb; --muted:#aeb6c5; --panel:#151923;
      --line:#2b3241; --cyan:#77e6d8; --green:#9cf0b0; --red:#ff8a91; --gold:#ffd477; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 80% 0,#213447 0,transparent 32rem),
      #090b10; color:var(--ink); font:16px/1.55 ui-sans-serif,-apple-system,BlinkMacSystemFont,
      "Segoe UI",sans-serif; }}
    main {{ width:min(1100px,calc(100% - 32px)); margin:auto; padding:64px 0 72px; }}
    header {{ padding:44px; border:1px solid var(--line); border-radius:28px;
      background:linear-gradient(135deg,rgba(119,230,216,.12),rgba(21,25,35,.94)); }}
    .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-size:.78rem;
      font-weight:800; }}
    h1 {{ margin:.2rem 0 .4rem; font-size:clamp(2.6rem,8vw,6rem); line-height:.95;
      letter-spacing:-.065em; }}
    .lead {{ max-width:720px; margin:1.2rem 0 0; color:#dfe4ec; font-size:1.2rem; }}
    .status {{ display:inline-flex; align-items:center; gap:.55rem; margin-top:1.6rem; padding:.55rem .8rem;
      border:1px solid #346748; border-radius:999px; color:var(--green); background:#102419; font-weight:750; }}
    .status::before {{ content:""; width:.65rem; height:.65rem; border-radius:50%; background:var(--green);
      box-shadow:0 0 18px var(--green); }}
    section {{ margin-top:52px; }}
    h2 {{ margin:0 0 18px; font-size:clamp(1.65rem,4vw,2.55rem); letter-spacing:-.035em; }}
    h3 {{ margin:.6rem 0 .25rem; }}
    p {{ margin:.4rem 0; }}
    .muted {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
    .memory-card,.metric,.certificate {{ border:1px solid var(--line); border-radius:18px; background:var(--panel); }}
    .memory-card {{ padding:20px; }}
    .chip {{ display:inline-block; padding:.2rem .55rem; border-radius:999px; background:#252d3c;
      color:var(--gold); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }}
    .sequence {{ display:grid; grid-template-columns:1fr auto 1fr; gap:18px; align-items:center; }}
    .metric {{ padding:28px; }}
    .metric strong {{ display:block; font-size:clamp(2.5rem,8vw,5.5rem); line-height:1; letter-spacing:-.06em; }}
    .before strong {{ color:var(--red); }} .after strong {{ color:var(--green); }}
    .arrow {{ color:var(--cyan); font-size:2rem; }}
    .certificate {{ padding:24px; position:relative; overflow:hidden; }}
    .certificate::after {{ content:"CHECKED"; position:absolute; right:-32px; top:24px; transform:rotate(35deg);
      padding:.25rem 2.7rem; color:#092113; background:var(--green); font-size:.7rem; font-weight:900; letter-spacing:.12em; }}
    .digest {{ color:var(--cyan); font:600 .82rem/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
      overflow-wrap:anywhere; padding-right:55px; }}
    .equation {{ padding:34px 24px; border:1px solid #355f5c; border-radius:20px; text-align:center;
      background:linear-gradient(130deg,#101d20,#151923); }}
    .equation .formula {{ font:700 clamp(1.15rem,4vw,2rem)/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .equation .numbers {{ margin-top:.8rem; color:var(--cyan); font-size:1.2rem; }}
    .scope {{ padding:24px; border-left:4px solid var(--gold); background:#201c13; border-radius:4px 16px 16px 4px; }}
    details {{ margin-top:14px; border:1px solid var(--line); border-radius:14px; background:#0e1118; }}
    summary {{ cursor:pointer; padding:16px 18px; color:var(--cyan); font-weight:750; }}
    pre {{ margin:0; padding:0 18px 18px; max-height:520px; overflow:auto; color:#cbd3df;
      font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }}
    footer {{ margin-top:56px; color:var(--muted); font-size:.85rem; }}
    @media(max-width:680px) {{ main {{ padding-top:20px; }} header {{ padding:28px 22px; }}
      .sequence {{ grid-template-columns:1fr; }} .arrow {{ transform:rotate(90deg); text-align:center; }} }}
    @media print {{ body {{ background:#fff; color:#111; }} main {{ width:100%; padding:0; }}
      header,.memory-card,.metric,.certificate,.equation,details {{ background:#fff; color:#111; break-inside:avoid; }}
      .muted,footer {{ color:#444; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">PopRank research prototype · controlled demonstration</div>
    <h1>The Forgetting<br>Machine</h1>
    <p class="lead">A fictional memory is extracted, a deletion request is applied, the attack is
      replayed, and its in-process integrity and policy algebra are checked.</p>
    <div class="status">Two attestations checked</div>
  </header>

  <section>
    <div class="eyebrow">01 · Extraction attack</div>
    <h2>Alice is recoverable before deletion.</h2>
    <div class="grid">{before_cards}</div>
  </section>

  <section>
    <div class="eyebrow">02 · Replay after forgetting</div>
    <h2>The same query now returns nothing.</h2>
    <div class="sequence">
      <div class="metric before"><strong>{attack['before_forgetting']['records_recovered']}</strong>
        <span>records recovered before</span></div>
      <div class="arrow">→</div>
      <div class="metric after"><strong>{attack['after_forgetting']['records_recovered']}</strong>
        <span>records recovered after</span></div>
    </div>
  </section>

  <section>
    <div class="eyebrow">03 · Evidence chain</div>
    <h2>Each transition carries an emitter attestation checked at generation.</h2>
    <div class="grid">
      <article class="certificate"><h3>Forget Alice</h3>
        <p>{len(first['newly_tombstoned_record_ids'])} payloads tombstoned · {first['preservation']['records_checked']} preserved</p>
        <p class="digest">{_escape(first['certificate_digest'])}</p></article>
      <article class="certificate"><h3>Forget medical</h3>
        <p>{len(second['newly_tombstoned_record_ids'])} new payloads tombstoned · overlaps prior policy</p>
        <p class="digest">{_escape(second['certificate_digest'])}</p></article>
    </div>
  </section>

  <section>
    <div class="eyebrow">04 · Exact composition</div>
    <h2>Two deletion policies, one checked rank balance.</h2>
    <div class="equation">
      <div class="formula">rank(f) + rank(g) = rank(gcd) + rank(lcm)</div>
      <div class="numbers">{ranks['f']} + {ranks['g']} = {ranks['gcd']} + {ranks['lcm']} = {ranks['left_sum']}</div>
      <p>Shared selection: <strong>{_escape(shared)}</strong></p>
      <p class="muted">Combined selection: {_escape(combined)}</p>
    </div>
  </section>

  <section>
    <div class="eyebrow">05 · Final state</div>
    <h2>{final_snapshot['forgotten_records']} of {final_snapshot['total_records']} controlled payloads tombstoned.</h2>
    <p class="muted">The unrelated maintenance control remained byte-for-byte represented by the same record,
      while keyed preservation commitments for both transactions remained unchanged.</p>
  </section>

  <section class="scope">
    <div class="eyebrow">Proof scope · read this</div>
    <h2>Precise evidence, deliberately narrow claim.</h2>
    <p>{_escape(report['proof_scope'])}</p>
  </section>

  <section>
    <div class="eyebrow">Audit evidence</div>
    <h2>Inspect the complete machine-readable result.</h2>
    <details><summary>Certificate 1 · Forget Alice</summary><pre>{_json_for_html(first)}</pre></details>
    <details><summary>Certificate 2 · Forget medical</summary><pre>{_json_for_html(second)}</pre></details>
    <details><summary>Complete demo report</summary><pre>{_json_for_html(report)}</pre></details>
  </section>

  <footer>Generated {_escape(report['generated_at_utc'])} · Synthetic records only · Self-contained report</footer>
</main>
</body>
</html>
"""


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_html(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the controlled Forgetting Machine stage demo."
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional path for the complete machine-readable evidence report.",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        help="Optional path for a polished, self-contained HTML report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_demo()
    print_narrative(report)
    if args.out_json is not None:
        _write_json(args.out_json, report)
        print(f"\nJSON report: {args.out_json}")
    if args.out_html is not None:
        _write_html(args.out_html, report)
        print(f"HTML report: {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
