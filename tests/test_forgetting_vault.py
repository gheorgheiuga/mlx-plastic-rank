import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, cast

import pytest

from mlx_plastic_rank import ForgetPolicy as ExportedForgetPolicy
from mlx_plastic_rank import ForgettingVault as ExportedForgettingVault
from mlx_plastic_rank import MemoryRecord as ExportedMemoryRecord
from mlx_plastic_rank import verify_certificate_chain as ExportedVerifyChain
from mlx_plastic_rank.forgetting_vault import (
    DeletionCertificate,
    ForgetPolicy,
    ForgettingVault,
    InvalidForgetPolicy,
    MemoryRecord,
    VaultIntegrityError,
    verify_certificate_chain,
)

INTEGRITY_KEY = b"test-only-forgetting-vault-integrity-key-v1"


def _resign(payload: dict) -> DeletionCertificate:
    unsigned = dict(payload)
    unsigned.pop("certificate_digest", None)
    unsigned.pop("authentication_tag", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    unsigned["certificate_digest"] = digest
    unsigned["authentication_tag"] = hmac.new(
        INTEGRITY_KEY,
        digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return DeletionCertificate(unsigned)


def _records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            record_id="alice-medical",
            subject="alice",
            category="medical",
            payload="Allergic to penicillin",
        ),
        MemoryRecord(
            record_id="alice-travel",
            subject="alice",
            category="travel",
            payload="Flight to Bucharest on 14 July",
        ),
        MemoryRecord(
            record_id="bob-medical",
            subject="bob",
            category="medical",
            payload="Blood type O positive",
        ),
        MemoryRecord(
            record_id="pump-fault",
            subject="plant",
            category="maintenance",
            payload="Pump P-204 bearing temperature high",
        ),
    ]


def test_forget_tombstones_selected_payloads_and_preserves_unmatched_records():
    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    before_bob = vault.search(record_id="bob-medical")[0]
    before_plant = vault.search(subject="plant")[0]

    certificate = vault.forget(
        ForgetPolicy(policy_id="delete-alice", subject="alice", reason="user request")
    )

    assert vault.search(subject="alice") == ()
    assert vault.search(record_id="bob-medical") == (before_bob,)
    assert vault.search(subject="plant") == (before_plant,)
    assert certificate.verify(INTEGRITY_KEY).valid is True

    report = certificate.to_dict()
    assert report["selected_record_ids"] == ["alice-medical", "alice-travel"]
    assert report["newly_tombstoned_record_ids"] == ["alice-medical", "alice-travel"]
    assert report["active_records_before"] == 4
    assert report["active_records_after"] == 2
    assert report["preservation"]["emitter_check_passed"] is True
    assert report["payload_erasure"]["emitter_check_passed"] is True
    assert "Allergic to penicillin" not in certificate.to_json()
    assert "Flight to Bucharest" not in certificate.to_json()

    snapshot = vault.snapshot()
    assert snapshot["active_record_ids"] == ["bob-medical", "pump-fault"]
    assert snapshot["forgotten_record_ids"] == ["alice-medical", "alice-travel"]


def test_policy_selectors_use_intersection_when_combined():
    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)

    certificate = vault.forget(
        ForgetPolicy(
            policy_id="delete-alice-medical",
            subject="alice",
            category="medical",
        )
    )

    assert certificate.to_dict()["selected_record_ids"] == ["alice-medical"]
    assert vault.search(subject="alice") == (
        vault.search(record_id="alice-travel")[0],
    )
    assert vault.search(record_id="bob-medical")[0].payload == "Blood type O positive"


def test_overlapping_policies_emit_exact_pop_identity_certificate():
    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    first = vault.forget(ForgetPolicy(policy_id="delete-alice", subject="alice"))
    second = vault.forget(ForgetPolicy(policy_id="delete-medical", category="medical"))

    assert first.verify(INTEGRITY_KEY).valid is True
    assert second.verify(INTEGRITY_KEY).valid is True

    [composition] = second.to_dict()["pop_compositions"]
    assert composition["other_policy_id"] == "delete-alice"
    assert composition["shared_target_record_ids"] == ["alice-medical"]
    assert composition["combined_target_record_ids"] == [
        "alice-medical",
        "alice-travel",
        "bob-medical",
    ]
    assert composition["ranks"] == {
        "f": 2,
        "g": 2,
        "gcd": 3,
        "lcm": 1,
        "left_sum": 4,
        "right_sum": 4,
    }
    assert composition["identity_holds"] is True
    assert second.to_dict()["previous_certificate_digest"] == first.digest
    assert verify_certificate_chain(
        [first, second],
        INTEGRITY_KEY,
        expected_head_digest=second.digest,
    ).valid

    fabricated_history = second.to_dict()
    fabricated_history["pop_compositions"][0]["other_certificate_digest"] = "0" * 64
    fabricated_certificate = _resign(fabricated_history)
    assert fabricated_certificate.verify(INTEGRITY_KEY).valid
    chain_report = verify_certificate_chain(
        [first, fabricated_certificate],
        INTEGRITY_KEY,
    )
    assert chain_report.valid is False
    assert "certificate_2_composition_history_invalid" in chain_report.failures


def test_certificate_verification_detects_tampering_and_wrong_key():
    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    certificate = vault.forget(ForgetPolicy(policy_id="delete-alice", subject="alice"))

    assert certificate.verify(b"wrong-key").valid is False
    assert certificate.verify(b"").failures == ("integrity_key_invalid",)

    payload = json.loads(certificate.to_json())
    payload["active_records_after"] = 99
    tampered = DeletionCertificate.from_json(json.dumps(payload))
    verification = tampered.verify(INTEGRITY_KEY)

    assert verification.valid is False
    assert "certificate_digest_mismatch" in verification.failures

    contradictory = json.loads(certificate.to_json())
    contradictory["newly_tombstoned_record_ids"] = []
    contradictory_verification = _resign(contradictory).verify(INTEGRITY_KEY)
    assert "tombstone_partition_invalid" in contradictory_verification.failures


def test_forgetting_api_is_available_from_the_package_root():
    assert ExportedForgetPolicy is ForgetPolicy
    assert ExportedForgettingVault is ForgettingVault
    assert ExportedMemoryRecord is MemoryRecord
    assert ExportedVerifyChain is verify_certificate_chain


def test_vault_rejects_invalid_records_policies_and_duplicate_requests():
    with pytest.raises(VaultIntegrityError, match="at least 32 bytes"):
        ForgettingVault(_records(), integrity_key=b"too-short")

    with pytest.raises(VaultIntegrityError, match="Duplicate record_id"):
        ForgettingVault([_records()[0], _records()[0]], integrity_key=INTEGRITY_KEY)

    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    with pytest.raises(InvalidForgetPolicy, match="at least one selector"):
        vault.forget(ForgetPolicy(policy_id="empty"))
    with pytest.raises(InvalidForgetPolicy, match="matched no records"):
        vault.forget(ForgetPolicy(policy_id="missing", subject="nobody"))
    with pytest.raises(InvalidForgetPolicy, match="subject must"):
        vault.forget(
            ForgetPolicy(policy_id="bad-subject", subject=cast(Any, 123))
        )
    with pytest.raises(InvalidForgetPolicy, match="record_ids must be a tuple"):
        vault.forget(
            ForgetPolicy(
                policy_id="bad-ids",
                record_ids=cast(Any, ["alice-medical"]),
            )
        )

    vault.forget(ForgetPolicy(policy_id="delete-alice", subject="alice"))
    with pytest.raises(InvalidForgetPolicy, match="already been used"):
        vault.forget(ForgetPolicy(policy_id="delete-alice", category="medical"))


def test_repeated_selection_is_idempotent_under_a_new_policy_id():
    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    vault.forget(ForgetPolicy(policy_id="first", subject="alice"))

    repeated = vault.forget(ForgetPolicy(policy_id="second", subject="alice"))

    report = repeated.to_dict()
    assert report["newly_tombstoned_record_ids"] == []
    assert report["already_tombstoned_record_ids"] == [
        "alice-medical",
        "alice-travel",
    ]
    assert report["active_records_before"] == report["active_records_after"] == 2
    assert repeated.verify(INTEGRITY_KEY).valid is True


def test_certificate_parsing_fails_closed_without_throwing_from_verify():
    with pytest.raises(VaultIntegrityError, match="invalid"):
        DeletionCertificate.from_json(b"\xff")
    with pytest.raises(VaultIntegrityError, match="non-finite"):
        DeletionCertificate.from_json('{"value": NaN}')
    with pytest.raises(VaultIntegrityError, match="invalid"):
        DeletionCertificate.from_json('{"value": "\ud800"}')
    with pytest.raises(VaultIntegrityError, match="invalid"):
        DeletionCertificate.from_json('{"value": ' + "9" * 5000 + "}")

    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    certificate = vault.forget(ForgetPolicy(policy_id="delete-alice", subject="alice"))
    non_ascii = certificate.to_dict()
    non_ascii["certificate_digest"] = "é" * 64
    assert DeletionCertificate(non_ascii).verify(INTEGRITY_KEY).valid is False

    invalid_fraction = json.loads(certificate.to_json())
    invalid_fraction["scrubber_projector"]["coefficients_ascending"][0][1] = 0
    report = _resign(invalid_fraction).verify(INTEGRITY_KEY)
    assert report.valid is False
    assert "certificate_structure_invalid" in report.failures

    invalid_chain = verify_certificate_chain(
        [cast(Any, object())],
        INTEGRITY_KEY,
        expected_head_digest="0" * 64,
    )
    assert invalid_chain.valid is False
    assert "certificate_1_type_invalid" in invalid_chain.failures


def test_concurrent_forget_operations_do_not_resurrect_records():
    vault = ForgettingVault(_records(), integrity_key=INTEGRITY_KEY)
    barrier = Barrier(2)

    def apply(policy: ForgetPolicy) -> DeletionCertificate:
        barrier.wait()
        return vault.forget(policy)

    with ThreadPoolExecutor(max_workers=2) as executor:
        alice_future = executor.submit(
            apply,
            ForgetPolicy(policy_id="delete-alice", subject="alice"),
        )
        plant_future = executor.submit(
            apply,
            ForgetPolicy(policy_id="delete-plant", subject="plant"),
        )
        certificates = [alice_future.result(), plant_future.result()]

    assert vault.search(subject="alice") == ()
    assert vault.search(subject="plant") == ()
    ordered = sorted(certificates, key=lambda item: item.to_dict()["event_sequence"])
    assert verify_certificate_chain(ordered, INTEGRITY_KEY).valid
