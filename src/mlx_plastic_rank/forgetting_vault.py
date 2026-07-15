"""Controlled synthetic spectral vault with authenticated deletion attestations.

The Pop identity accounts for overlap between policy targets. Payload erasure is
performed explicitly by ``ForgettingVault`` and attested by a trusted emitter; a
serialized certificate cannot independently inspect past storage state.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
ZERO_DIGEST = "0" * 64
MAX_VAULT_RECORDS = 128
MAX_CERTIFICATE_BYTES = 1_000_000
MAX_INTEGER_BITS = 4096
MIN_INTEGRITY_KEY_BYTES = 32


class VaultIntegrityError(ValueError):
    """Raised when records or serialized vault evidence violate invariants."""


class InvalidForgetPolicy(ValueError):
    """Raised when a forget policy is empty, ambiguous, or cannot be applied."""


@dataclass(frozen=True)
class MemoryRecord:
    """One payload governed by the controlled forgetting vault."""

    record_id: str
    subject: str
    category: str
    payload: str


@dataclass(frozen=True)
class ForgetPolicy:
    """Deterministic deletion selector.

    When more than one selector is present, all selectors must match. Separate
    policies should be used when union semantics are intended.
    """

    policy_id: str
    subject: str | None = None
    category: str | None = None
    record_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class VerificationReport:
    """Authentication and internal-consistency result for an attestation."""

    valid: bool
    failures: tuple[str, ...]


@dataclass
class _StoredMemory:
    record_id: str
    subject: str
    category: str
    eigenvalue: int
    payload: str | None
    gate: int = 1

    def public_record(self) -> MemoryRecord:
        if self.payload is None:
            raise VaultIntegrityError(f"Record {self.record_id!r} has been forgotten.")
        return MemoryRecord(
            record_id=self.record_id,
            subject=self.subject,
            category=self.category,
            payload=self.payload,
        )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _authentication_tag(key: bytes, digest: str) -> str:
    return hmac.new(key, digest.encode("ascii"), hashlib.sha256).hexdigest()


def _keyed_commitment(key: bytes, domain: str, payload: Any) -> str:
    message = domain.encode("ascii") + b"\x00" + _canonical_json_bytes(payload)
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _safe_compare_ascii(left: Any, right: str) -> bool:
    if not isinstance(left, str):
        return False
    try:
        return hmac.compare_digest(left.encode("ascii"), right.encode("ascii"))
    except UnicodeEncodeError:
        return False


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _strict_certificate_int(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("Certificate integer must use the JSON integer type.")
    if abs(value).bit_length() > MAX_INTEGER_BITS:
        raise ValueError("Certificate integer exceeds the supported bit length.")
    return value


def _polynomial_from_roots(roots: Sequence[int]) -> list[int]:
    """Return exact ascending coefficients for a monic polynomial."""

    coefficients = [1]
    for root in sorted(set(int(value) for value in roots)):
        next_coefficients = [0] * (len(coefficients) + 1)
        for index, coefficient in enumerate(coefficients):
            next_coefficients[index] -= root * coefficient
            next_coefficients[index + 1] += coefficient
        coefficients = next_coefficients
    return coefficients


def _evaluate_polynomial(coefficients: Sequence[int], value: int) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = result * int(value) + int(coefficient)
    return result


def _polynomial_rank(spectrum: Sequence[int], coefficients: Sequence[int]) -> int:
    return sum(
        1 for eigenvalue in spectrum if _evaluate_polynomial(coefficients, eigenvalue) != 0
    )


def _add_fraction_polynomials(
    left: Sequence[Fraction], right: Sequence[Fraction]
) -> list[Fraction]:
    size = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else Fraction(0))
        + (right[index] if index < len(right) else Fraction(0))
        for index in range(size)
    ]


def _lagrange_keep_projector(
    spectrum: Sequence[int], forgotten_roots: Sequence[int]
) -> list[Fraction]:
    """Return q with q(root)=0 for forgotten modes and q(root)=1 otherwise."""

    forgotten = set(int(value) for value in forgotten_roots)
    coefficients = [Fraction(0)]
    for eigenvalue in spectrum:
        if eigenvalue in forgotten:
            continue
        basis = [Fraction(1)]
        denominator = Fraction(1)
        for other in spectrum:
            if other == eigenvalue:
                continue
            next_basis = [Fraction(0)] * (len(basis) + 1)
            for index, coefficient in enumerate(basis):
                next_basis[index] -= other * coefficient
                next_basis[index + 1] += coefficient
            basis = next_basis
            denominator *= eigenvalue - other
        coefficients = _add_fraction_polynomials(
            coefficients, [coefficient / denominator for coefficient in basis]
        )
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return coefficients


def _evaluate_fraction_polynomial(
    coefficients: Sequence[Fraction], value: int
) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _serialize_fraction_polynomial(coefficients: Sequence[Fraction]) -> list[list[int]]:
    return [[value.numerator, value.denominator] for value in coefficients]


def _deserialize_fraction_polynomial(payload: Sequence[Sequence[int]]) -> list[Fraction]:
    if len(payload) > MAX_VAULT_RECORDS + 1:
        raise ValueError("Projector degree exceeds the supported vault size.")
    coefficients = []
    for value in payload:
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError("Fraction coefficients must be two-element arrays.")
        numerator = _strict_certificate_int(value[0])
        denominator = _strict_certificate_int(value[1])
        coefficients.append(Fraction(numerator, denominator))
    return coefficients


def _normalized_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VaultIntegrityError(f"{field} must be a non-empty string.")
    return value.strip()


def _record_digest_rows(records: Iterable[_StoredMemory]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": record.record_id,
            "subject": record.subject,
            "category": record.category,
            "eigenvalue": record.eigenvalue,
            "payload": record.payload,
            "gate": record.gate,
        }
        for record in sorted(records, key=lambda item: item.record_id)
    ]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VaultIntegrityError(f"Certificate JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise VaultIntegrityError(f"Certificate JSON contains non-finite value {value!r}.")


class DeletionCertificate:
    """Authenticated emitter attestation for one controlled vault transition."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = copy.deepcopy(dict(payload))

    @property
    def digest(self) -> str:
        value = self._payload.get("certificate_digest")
        return str(value) if value is not None else ""

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible copy of the certificate."""

        return copy.deepcopy(self._payload)

    def to_json(self) -> str:
        """Return deterministic canonical JSON."""

        return _canonical_json_bytes(self._payload).decode("utf-8")

    @classmethod
    def from_json(cls, raw: str | bytes) -> DeletionCertificate:
        """Load bounded JSON while rejecting ambiguous or non-finite values."""

        try:
            if isinstance(raw, bytes):
                if len(raw) > MAX_CERTIFICATE_BYTES:
                    raise VaultIntegrityError("Certificate JSON exceeds the size limit.")
                text = raw.decode("utf-8")
            elif isinstance(raw, str):
                if len(raw.encode("utf-8")) > MAX_CERTIFICATE_BYTES:
                    raise VaultIntegrityError("Certificate JSON exceeds the size limit.")
                text = raw
            else:
                raise VaultIntegrityError("Certificate JSON must be text or bytes.")
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            UnicodeEncodeError,
            ValueError,
            RecursionError,
        ) as exc:
            raise VaultIntegrityError(f"Certificate JSON is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise VaultIntegrityError("Certificate JSON must contain an object.")
        return cls(payload)

    def verify(self, integrity_key: bytes) -> VerificationReport:
        """Authenticate the attestation and recompute its disclosed algebra.

        This does not inspect historical vault state. Preservation and payload
        erasure remain authenticated assertions made by the trusted emitter.
        """

        failures: list[str] = []
        if (
            not isinstance(integrity_key, bytes)
            or len(integrity_key) < MIN_INTEGRITY_KEY_BYTES
        ):
            return VerificationReport(
                valid=False,
                failures=("integrity_key_invalid",),
            )
        try:
            payload = self.to_dict()
            stored_digest = payload.pop("certificate_digest", None)
            stored_tag = payload.pop("authentication_tag", None)
            encoded_payload = _canonical_json_bytes(payload)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return VerificationReport(
                valid=False,
                failures=("certificate_structure_invalid",),
            )
        if len(encoded_payload) > MAX_CERTIFICATE_BYTES:
            return VerificationReport(
                valid=False,
                failures=("certificate_size_limit_exceeded",),
            )
        computed_digest = hashlib.sha256(encoded_payload).hexdigest()
        if not _safe_compare_ascii(stored_digest, computed_digest):
            failures.append("certificate_digest_mismatch")
        expected_tag = _authentication_tag(integrity_key, computed_digest)
        if not _safe_compare_ascii(stored_tag, expected_tag):
            failures.append("authentication_tag_mismatch")
        if failures:
            return VerificationReport(valid=False, failures=tuple(failures))

        try:
            if payload.get("kind") != "spectral_forgetting_certificate":
                failures.append("certificate_kind_invalid")
            if payload.get("schema_version") != SCHEMA_VERSION:
                failures.append("schema_version_invalid")
            authentication = payload["authentication"]
            if authentication.get("algorithm") != "HMAC-SHA256":
                failures.append("authentication_algorithm_invalid")
            if not isinstance(authentication.get("key_id"), str) or not authentication[
                "key_id"
            ]:
                failures.append("authentication_key_id_invalid")
            if not isinstance(payload.get("vault_id"), str) or not payload["vault_id"]:
                failures.append("vault_id_invalid")
            if _strict_certificate_int(payload["event_sequence"]) < 1:
                failures.append("event_sequence_invalid")

            operator = payload["operator"]
            spectrum_rows = operator["spectrum"]
            if not isinstance(spectrum_rows, list) or not (
                1 <= len(spectrum_rows) <= MAX_VAULT_RECORDS
            ):
                raise ValueError("Spectrum dimension is outside supported bounds.")
            spectrum = [
                _strict_certificate_int(row["eigenvalue"]) for row in spectrum_rows
            ]
            id_by_eigenvalue = {
                _strict_certificate_int(row["eigenvalue"]): str(row["record_id"])
                for row in spectrum_rows
            }
            record_ids = [str(row["record_id"]) for row in spectrum_rows]
            if operator.get("kind") != "controlled_diagonal_integer_spectrum":
                failures.append("operator_kind_invalid")
            if _strict_certificate_int(operator["dimension"]) != len(spectrum):
                failures.append("operator_dimension_mismatch")
            if operator.get("spectrum_digest") != _sha256_payload(spectrum_rows):
                failures.append("operator_spectrum_digest_mismatch")
            if len(spectrum) != len(set(spectrum)):
                failures.append("operator_spectrum_not_unique")
            if len(record_ids) != len(set(record_ids)):
                failures.append("operator_record_ids_not_unique")

            policy_filter = payload["policy_filter"]
            roots = [
                _strict_certificate_int(value) for value in policy_filter["roots"]
            ]
            if len(roots) != len(set(roots)) or any(root not in spectrum for root in roots):
                failures.append("policy_roots_invalid")
            coefficients = _polynomial_from_roots(roots)
            supplied_coefficients = [
                _strict_certificate_int(value)
                for value in policy_filter["coefficients_ascending"]
            ]
            if coefficients != supplied_coefficients:
                failures.append("policy_polynomial_mismatch")
            if _polynomial_rank(spectrum, coefficients) != _strict_certificate_int(
                policy_filter["rank"]
            ):
                failures.append("policy_rank_mismatch")
            selected_from_roots = sorted(id_by_eigenvalue[root] for root in roots)
            if selected_from_roots != payload["selected_record_ids"]:
                failures.append("selected_records_mismatch")

            projector = _deserialize_fraction_polynomial(
                payload["scrubber_projector"]["coefficients_ascending"]
            )
            expected_projector = _lagrange_keep_projector(spectrum, roots)
            if projector != expected_projector:
                failures.append("scrubber_projector_mismatch")
            projector_values = {
                str(eigenvalue): int(_evaluate_fraction_polynomial(projector, eigenvalue))
                for eigenvalue in spectrum
            }
            if projector_values != payload["scrubber_projector"]["values_by_eigenvalue"]:
                failures.append("scrubber_projector_values_mismatch")
            if any(
                value not in (0, 1)
                or (int(eigenvalue) in roots and value != 0)
                or (int(eigenvalue) not in roots and value != 1)
                for eigenvalue, value in projector_values.items()
            ):
                failures.append("scrubber_projector_not_selective")

            selected_ids = payload["selected_record_ids"]
            newly_tombstoned = payload["newly_tombstoned_record_ids"]
            already_tombstoned = payload["already_tombstoned_record_ids"]
            if set(newly_tombstoned).intersection(already_tombstoned) or sorted(
                set(newly_tombstoned).union(already_tombstoned)
            ) != selected_ids:
                failures.append("tombstone_partition_invalid")
            active_before = _strict_certificate_int(payload["active_records_before"])
            active_after = _strict_certificate_int(payload["active_records_after"])
            newly_deleted = len(newly_tombstoned)
            if active_after != active_before - newly_deleted:
                failures.append("active_record_transition_invalid")
            if not (0 <= active_after <= active_before <= len(spectrum)):
                failures.append("active_record_counts_invalid")
            preservation = payload["preservation"]
            preservation_ids = preservation["record_ids"]
            if (
                preservation["commitment_before"]
                != preservation["commitment_after"]
                or preservation.get("emitter_check_passed") is not True
            ):
                failures.append("preservation_attestation_invalid")
            if set(preservation_ids).intersection(selected_ids) or _strict_certificate_int(
                preservation["records_checked"]
            ) != len(preservation_ids):
                failures.append("preservation_record_set_invalid")
            erasure = payload["payload_erasure"]
            if (
                erasure.get("emitter_check_passed") is not True
                or erasure["record_ids"] != selected_ids
                or _strict_certificate_int(erasure["records_checked"]) != len(selected_ids)
            ):
                failures.append("payload_erasure_attestation_invalid")
            digest_fields = [
                preservation["commitment_before"],
                preservation["commitment_after"],
                payload["state_commitment_before"],
                payload["state_commitment_after"],
                payload["previous_certificate_digest"],
                payload["policy"]["selector_commitment"],
            ]
            if not all(_is_sha256_hex(value) for value in digest_fields):
                failures.append("commitment_format_invalid")

            for composition in payload.get("pop_compositions", []):
                failures.extend(
                    _verify_composition(
                        composition,
                        spectrum=spectrum,
                        id_by_eigenvalue=id_by_eigenvalue,
                        current_policy_id=str(payload["policy"]["policy_id"]),
                        current_roots=roots,
                    )
                )
        except (KeyError, TypeError, ValueError, IndexError, ArithmeticError, OverflowError):
            failures.append("certificate_structure_invalid")

        return VerificationReport(valid=not failures, failures=tuple(dict.fromkeys(failures)))


def verify_certificate_chain(
    certificates: Sequence[DeletionCertificate],
    integrity_key: bytes,
    *,
    expected_head_digest: str | None = None,
) -> VerificationReport:
    """Verify ordering and state continuity for a retained certificate history.

    Supplying an externally retained ``expected_head_digest`` detects a truncated
    tail. Without such an anchor, a valid prefix cannot be distinguished from a
    complete history.
    """

    if not certificates:
        return VerificationReport(False, ("certificate_chain_empty",))
    failures: list[str] = []
    prior_payload: dict[str, Any] | None = None
    first_payload: dict[str, Any] | None = None
    prior_by_digest: dict[str, dict[str, Any]] = {}
    for index, certificate in enumerate(certificates, start=1):
        if not isinstance(certificate, DeletionCertificate):
            failures.append(f"certificate_{index}_type_invalid")
            continue
        report = certificate.verify(integrity_key)
        if not report.valid:
            failures.extend(
                f"certificate_{index}_{failure}" for failure in report.failures
            )
            continue
        payload = certificate.to_dict()
        if first_payload is None:
            first_payload = payload
            if payload["previous_certificate_digest"] != ZERO_DIGEST:
                failures.append("certificate_1_predecessor_invalid")
            if payload["pop_compositions"]:
                failures.append("certificate_1_compositions_invalid")
        else:
            assert prior_payload is not None
            if payload["vault_id"] != first_payload["vault_id"]:
                failures.append(f"certificate_{index}_vault_mismatch")
            if payload["operator"]["spectrum_digest"] != first_payload["operator"][
                "spectrum_digest"
            ]:
                failures.append(f"certificate_{index}_operator_mismatch")
            if payload["authentication"] != first_payload["authentication"]:
                failures.append(f"certificate_{index}_authentication_mismatch")
            if payload["previous_certificate_digest"] != prior_payload[
                "certificate_digest"
            ]:
                failures.append(f"certificate_{index}_predecessor_invalid")
            if payload["state_commitment_before"] != prior_payload[
                "state_commitment_after"
            ]:
                failures.append(f"certificate_{index}_state_continuity_invalid")
            compositions = payload["pop_compositions"]
            composition_digests = {
                composition.get("other_certificate_digest")
                for composition in compositions
            }
            if composition_digests != set(prior_by_digest):
                failures.append(f"certificate_{index}_composition_history_invalid")
            for composition in compositions:
                other = prior_by_digest.get(composition.get("other_certificate_digest"))
                if other is None:
                    continue
                if (
                    composition.get("other_policy_id")
                    != other["policy"]["policy_id"]
                    or composition.get("g_roots") != other["policy_filter"]["roots"]
                ):
                    failures.append(
                        f"certificate_{index}_composition_predecessor_mismatch"
                    )
        if payload["event_sequence"] != index:
            failures.append(f"certificate_{index}_sequence_invalid")
        prior_payload = payload
        prior_by_digest[payload["certificate_digest"]] = payload

    if expected_head_digest is not None:
        if not _is_sha256_hex(expected_head_digest):
            failures.append("expected_head_digest_invalid")
        elif not isinstance(certificates[-1], DeletionCertificate):
            failures.append("certificate_chain_head_unavailable")
        elif certificates[-1].digest != expected_head_digest:
            failures.append("certificate_chain_head_mismatch")
    return VerificationReport(
        valid=not failures,
        failures=tuple(dict.fromkeys(failures)),
    )


def _verify_composition(
    composition: Mapping[str, Any],
    *,
    spectrum: Sequence[int],
    id_by_eigenvalue: Mapping[int, str],
    current_policy_id: str,
    current_roots: Sequence[int],
) -> list[str]:
    failures: list[str] = []
    f_roots = sorted(_strict_certificate_int(value) for value in composition["f_roots"])
    g_roots = sorted(_strict_certificate_int(value) for value in composition["g_roots"])
    if (
        composition.get("current_policy_id") != current_policy_id
        or f_roots != sorted(current_roots)
    ):
        failures.append("composition_current_policy_mismatch")
    if not _is_sha256_hex(composition.get("other_certificate_digest")):
        failures.append("composition_prior_certificate_digest_invalid")
    gcd_roots = sorted(set(f_roots).intersection(g_roots))
    lcm_roots = sorted(set(f_roots).union(g_roots))
    polynomial_roots = composition["polynomial_roots"]
    if gcd_roots != polynomial_roots["gcd"] or lcm_roots != polynomial_roots["lcm"]:
        failures.append("composition_roots_mismatch")

    f_coefficients = _polynomial_from_roots(f_roots)
    g_coefficients = _polynomial_from_roots(g_roots)
    gcd_coefficients = _polynomial_from_roots(gcd_roots)
    lcm_coefficients = _polynomial_from_roots(lcm_roots)
    expected_coefficients = {
        "f": f_coefficients,
        "g": g_coefficients,
        "gcd": gcd_coefficients,
        "lcm": lcm_coefficients,
    }
    if expected_coefficients != composition["polynomial_coefficients_ascending"]:
        failures.append("composition_polynomials_mismatch")

    ranks = {
        "f": _polynomial_rank(spectrum, f_coefficients),
        "g": _polynomial_rank(spectrum, g_coefficients),
        "gcd": _polynomial_rank(spectrum, gcd_coefficients),
        "lcm": _polynomial_rank(spectrum, lcm_coefficients),
    }
    ranks["left_sum"] = ranks["f"] + ranks["g"]
    ranks["right_sum"] = ranks["gcd"] + ranks["lcm"]
    if ranks != composition["ranks"]:
        failures.append("composition_ranks_mismatch")
    if composition.get("identity_holds") is not True or ranks["left_sum"] != ranks["right_sum"]:
        failures.append("pop_identity_failed")

    shared_ids = sorted(id_by_eigenvalue[root] for root in gcd_roots)
    combined_ids = sorted(id_by_eigenvalue[root] for root in lcm_roots)
    if shared_ids != composition["shared_target_record_ids"]:
        failures.append("shared_target_records_mismatch")
    if combined_ids != composition["combined_target_record_ids"]:
        failures.append("combined_target_records_mismatch")
    return failures


class ForgettingVault:
    """Small in-memory vault with serialized, authenticated forget operations.

    The interface is intentionally small: callers can search active records,
    forget a deterministic selection, and inspect a payload-free snapshot.
    """

    def __init__(
        self,
        records: Iterable[MemoryRecord],
        *,
        integrity_key: bytes,
        key_id: str = "local-v1",
        vault_id: str | None = None,
    ):
        if (
            not isinstance(integrity_key, bytes)
            or len(integrity_key) < MIN_INTEGRITY_KEY_BYTES
        ):
            raise VaultIntegrityError(
                f"integrity_key must contain at least {MIN_INTEGRITY_KEY_BYTES} bytes."
            )
        normalized_key_id = _normalized_text(key_id, field="key_id")
        normalized_vault_id = (
            _normalized_text(vault_id, field="vault_id")
            if vault_id is not None
            else secrets.token_hex(16)
        )
        normalized: dict[str, MemoryRecord] = {}
        for raw_record in records:
            record = MemoryRecord(
                record_id=_normalized_text(raw_record.record_id, field="record_id"),
                subject=_normalized_text(raw_record.subject, field="subject"),
                category=_normalized_text(raw_record.category, field="category"),
                payload=_normalized_text(raw_record.payload, field="payload"),
            )
            if record.record_id in normalized:
                raise VaultIntegrityError(f"Duplicate record_id {record.record_id!r}.")
            normalized[record.record_id] = record
        if not normalized:
            raise VaultIntegrityError("ForgettingVault requires at least one record.")
        if len(normalized) > MAX_VAULT_RECORDS:
            raise VaultIntegrityError(
                f"ForgettingVault supports at most {MAX_VAULT_RECORDS} records."
            )

        self._integrity_key = integrity_key
        self._key_id = normalized_key_id
        self._vault_id = normalized_vault_id
        self._lock = threading.RLock()
        self._records: dict[str, _StoredMemory] = {}
        for offset, record_id in enumerate(sorted(normalized), start=2):
            record = normalized[record_id]
            self._records[record_id] = _StoredMemory(
                record_id=record.record_id,
                subject=record.subject,
                category=record.category,
                eigenvalue=offset,
                payload=record.payload,
            )
        self._history: list[DeletionCertificate] = []
        self._policy_ids: set[str] = set()

    def search(
        self,
        *,
        record_id: str | None = None,
        subject: str | None = None,
        category: str | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Return active records matching every supplied selector."""

        with self._lock:
            return self._search_locked(
                record_id=record_id,
                subject=subject,
                category=category,
            )

    def _search_locked(
        self,
        *,
        record_id: str | None,
        subject: str | None,
        category: str | None,
    ) -> tuple[MemoryRecord, ...]:
        matches = []
        for stored in sorted(self._records.values(), key=lambda item: item.record_id):
            if stored.payload is None or stored.gate == 0:
                continue
            if record_id is not None and stored.record_id != record_id:
                continue
            if subject is not None and stored.subject != subject:
                continue
            if category is not None and stored.category != category:
                continue
            matches.append(stored.public_record())
        return tuple(matches)

    def forget(self, policy: ForgetPolicy) -> DeletionCertificate:
        """Atomically tombstone a selection and return an emitter attestation."""

        with self._lock:
            return self._forget_locked(policy)

    def _forget_locked(self, policy: ForgetPolicy) -> DeletionCertificate:
        normalized_policy = self._normalize_policy(policy)
        if normalized_policy.policy_id in self._policy_ids:
            raise InvalidForgetPolicy(
                f"policy_id {normalized_policy.policy_id!r} has already been used."
            )
        selected = [
            stored
            for stored in self._records.values()
            if _policy_matches(normalized_policy, stored)
        ]
        if not selected:
            raise InvalidForgetPolicy(
                f"Forget policy {normalized_policy.policy_id!r} matched no records."
            )

        selected_ids = sorted(record.record_id for record in selected)
        selected_roots = sorted(record.eigenvalue for record in selected)
        coefficients = _polynomial_from_roots(selected_roots)
        spectrum_rows = self._spectrum_rows()
        spectrum = [int(row["eigenvalue"]) for row in spectrum_rows]
        scrubber_projector = _lagrange_keep_projector(spectrum, selected_roots)
        active_before = sum(1 for record in self._records.values() if record.payload is not None)
        newly_tombstoned = sorted(
            record.record_id for record in selected if record.payload is not None
        )
        already_tombstoned = sorted(
            record.record_id for record in selected if record.payload is None
        )
        preserved_ids = sorted(
            record.record_id
            for record in self._records.values()
            if record.payload is not None and record.record_id not in selected_ids
        )
        preservation_before = _keyed_commitment(
            self._integrity_key,
            "preserved-records-v1",
            _record_digest_rows(self._records[record_id] for record_id in preserved_ids),
        )
        state_before = self._state_digest()

        prospective = copy.deepcopy(self._records)
        for stored in prospective.values():
            keep_value = _evaluate_fraction_polynomial(
                scrubber_projector, stored.eigenvalue
            )
            if keep_value.denominator != 1 or keep_value.numerator not in (0, 1):
                raise VaultIntegrityError("Scrubber projector is not exactly selective.")
            stored.gate *= keep_value.numerator
            if stored.record_id in selected_ids:
                stored.payload = None

        preservation_after = _keyed_commitment(
            self._integrity_key,
            "preserved-records-v1",
            _record_digest_rows(prospective[record_id] for record_id in preserved_ids),
        )
        erasure_verified = all(prospective[record_id].payload is None for record_id in selected_ids)
        active_after = sum(1 for record in prospective.values() if record.payload is not None)
        state_after = _keyed_commitment(
            self._integrity_key,
            "vault-state-v1",
            _record_digest_rows(prospective.values()),
        )
        pop_compositions = [
            _build_composition(
                current_policy_id=normalized_policy.policy_id,
                current_roots=selected_roots,
                other=certificate.to_dict(),
                spectrum=spectrum,
                id_by_eigenvalue={
                    int(row["eigenvalue"]): str(row["record_id"]) for row in spectrum_rows
                },
            )
            for certificate in self._history
        ]

        payload: dict[str, Any] = {
            "kind": "spectral_forgetting_certificate",
            "schema_version": SCHEMA_VERSION,
            "proof_scope": (
                "Authenticated attestation from a trusted ForgettingVault emitter, plus "
                "internally checked policy-filter algebra. Serialized verification does not "
                "observe historical storage or prove erasure elsewhere."
            ),
            "vault_id": self._vault_id,
            "event_sequence": len(self._history) + 1,
            "authentication": {
                "algorithm": "HMAC-SHA256",
                "key_id": self._key_id,
                "trust_model": "shared secret; verifiers holding the key can also forge",
            },
            "policy": {
                "policy_id": normalized_policy.policy_id,
                "selector_semantics": "all supplied selectors must match",
                "selector_commitment": _keyed_commitment(
                    self._integrity_key,
                    "forget-policy-selector-v1",
                    {
                        "subject": normalized_policy.subject,
                        "category": normalized_policy.category,
                        "record_ids": list(normalized_policy.record_ids),
                        "reason": normalized_policy.reason,
                    },
                ),
            },
            "operator": {
                "kind": "controlled_diagonal_integer_spectrum",
                "dimension": len(spectrum),
                "spectrum": spectrum_rows,
                "spectrum_digest": _sha256_payload(spectrum_rows),
            },
            "selected_record_ids": selected_ids,
            "newly_tombstoned_record_ids": newly_tombstoned,
            "already_tombstoned_record_ids": already_tombstoned,
            "active_records_before": active_before,
            "active_records_after": active_after,
            "policy_filter": {
                "roots": selected_roots,
                "coefficients_ascending": coefficients,
                "rank": _polynomial_rank(spectrum, coefficients),
            },
            "scrubber_projector": {
                "kind": "exact_lagrange_keep_projector",
                "coefficients_ascending": _serialize_fraction_polynomial(
                    scrubber_projector
                ),
                "values_by_eigenvalue": {
                    str(eigenvalue): int(
                        _evaluate_fraction_polynomial(scrubber_projector, eigenvalue)
                    )
                    for eigenvalue in spectrum
                },
            },
            "preservation": {
                "record_ids": preserved_ids,
                "records_checked": len(preserved_ids),
                "commitment_before": preservation_before,
                "commitment_after": preservation_after,
                "emitter_check_passed": preservation_before == preservation_after,
            },
            "payload_erasure": {
                "record_ids": selected_ids,
                "records_checked": len(selected_ids),
                "emitter_check_passed": erasure_verified,
            },
            "state_commitment_before": state_before,
            "state_commitment_after": state_after,
            "previous_certificate_digest": (
                self._history[-1].digest if self._history else ZERO_DIGEST
            ),
            "pop_compositions": pop_compositions,
        }
        certificate_digest = _sha256_payload(payload)
        payload["certificate_digest"] = certificate_digest
        payload["authentication_tag"] = _authentication_tag(
            self._integrity_key, certificate_digest
        )
        certificate = DeletionCertificate(payload)
        verification = certificate.verify(self._integrity_key)
        if not verification.valid:
            raise VaultIntegrityError(
                "Generated certificate failed verification: " + ", ".join(verification.failures)
            )

        self._records = prospective
        self._history.append(certificate)
        self._policy_ids.add(normalized_policy.policy_id)
        return certificate

    def snapshot(self) -> dict[str, Any]:
        """Return a payload-free, metadata-bearing summary of controlled state."""

        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        active_ids = sorted(
            record.record_id for record in self._records.values() if record.payload is not None
        )
        forgotten_ids = sorted(
            record.record_id for record in self._records.values() if record.payload is None
        )
        return {
            "kind": "forgetting_vault_snapshot",
            "schema_version": SCHEMA_VERSION,
            "total_records": len(self._records),
            "active_records": len(active_ids),
            "forgotten_records": len(forgotten_ids),
            "active_record_ids": active_ids,
            "forgotten_record_ids": forgotten_ids,
            "state_commitment": self._state_digest(),
            "certificate_count": len(self._history),
            "latest_certificate_digest": (
                self._history[-1].digest if self._history else ZERO_DIGEST
            ),
            "proof_scope": "payload-free metadata from one controlled ForgettingVault",
        }

    def _normalize_policy(self, policy: ForgetPolicy) -> ForgetPolicy:
        if not isinstance(policy, ForgetPolicy):
            raise InvalidForgetPolicy("policy must be a ForgetPolicy instance.")
        if not isinstance(policy.policy_id, str) or not policy.policy_id.strip():
            raise InvalidForgetPolicy("policy_id must be a non-empty string.")
        policy_id = policy.policy_id.strip()
        if policy.subject is not None and (
            not isinstance(policy.subject, str) or not policy.subject.strip()
        ):
            raise InvalidForgetPolicy("subject must be a non-empty string when supplied.")
        if policy.category is not None and (
            not isinstance(policy.category, str) or not policy.category.strip()
        ):
            raise InvalidForgetPolicy("category must be a non-empty string when supplied.")
        if not isinstance(policy.record_ids, tuple):
            raise InvalidForgetPolicy("record_ids must be a tuple of non-empty strings.")
        if any(
            not isinstance(record_id, str) or not record_id.strip()
            for record_id in policy.record_ids
        ):
            raise InvalidForgetPolicy("record_ids must contain only non-empty strings.")
        if not isinstance(policy.reason, str):
            raise InvalidForgetPolicy("reason must be a string.")
        subject = policy.subject.strip() if policy.subject is not None else None
        category = policy.category.strip() if policy.category is not None else None
        record_ids = tuple(sorted({record_id.strip() for record_id in policy.record_ids}))
        if not subject and not category and not record_ids:
            raise InvalidForgetPolicy("Forget policy requires at least one selector.")
        unknown_ids = sorted(set(record_ids).difference(self._records))
        if unknown_ids:
            raise InvalidForgetPolicy(f"Forget policy references unknown record IDs: {unknown_ids}.")
        return ForgetPolicy(
            policy_id=policy_id,
            subject=subject or None,
            category=category or None,
            record_ids=record_ids,
            reason=policy.reason.strip() if isinstance(policy.reason, str) else "",
        )

    def _spectrum_rows(self) -> list[dict[str, Any]]:
        return [
            {"record_id": record.record_id, "eigenvalue": record.eigenvalue}
            for record in sorted(self._records.values(), key=lambda item: item.record_id)
        ]

    def _state_digest(self) -> str:
        return _keyed_commitment(
            self._integrity_key,
            "vault-state-v1",
            _record_digest_rows(self._records.values()),
        )


def _policy_matches(policy: ForgetPolicy, record: _StoredMemory) -> bool:
    if policy.subject is not None and record.subject != policy.subject:
        return False
    if policy.category is not None and record.category != policy.category:
        return False
    if policy.record_ids and record.record_id not in policy.record_ids:
        return False
    return True


def _build_composition(
    *,
    current_policy_id: str,
    current_roots: Sequence[int],
    other: Mapping[str, Any],
    spectrum: Sequence[int],
    id_by_eigenvalue: Mapping[int, str],
) -> dict[str, Any]:
    f_roots = sorted(set(int(value) for value in current_roots))
    g_roots = sorted(set(int(value) for value in other["policy_filter"]["roots"]))
    gcd_roots = sorted(set(f_roots).intersection(g_roots))
    lcm_roots = sorted(set(f_roots).union(g_roots))
    polynomials = {
        "f": _polynomial_from_roots(f_roots),
        "g": _polynomial_from_roots(g_roots),
        "gcd": _polynomial_from_roots(gcd_roots),
        "lcm": _polynomial_from_roots(lcm_roots),
    }
    ranks = {
        name: _polynomial_rank(spectrum, coefficients)
        for name, coefficients in polynomials.items()
    }
    ranks["left_sum"] = ranks["f"] + ranks["g"]
    ranks["right_sum"] = ranks["gcd"] + ranks["lcm"]
    return {
        "current_policy_id": current_policy_id,
        "other_policy_id": str(other["policy"]["policy_id"]),
        "other_certificate_digest": str(other["certificate_digest"]),
        "f_roots": f_roots,
        "g_roots": g_roots,
        "polynomial_roots": {"gcd": gcd_roots, "lcm": lcm_roots},
        "polynomial_coefficients_ascending": polynomials,
        "shared_target_record_ids": sorted(id_by_eigenvalue[root] for root in gcd_roots),
        "combined_target_record_ids": sorted(id_by_eigenvalue[root] for root in lcm_roots),
        "ranks": ranks,
        "identity_holds": ranks["left_sum"] == ranks["right_sum"],
    }


__all__ = [
    "DeletionCertificate",
    "ForgetPolicy",
    "ForgettingVault",
    "InvalidForgetPolicy",
    "MemoryRecord",
    "VaultIntegrityError",
    "VerificationReport",
    "verify_certificate_chain",
]
