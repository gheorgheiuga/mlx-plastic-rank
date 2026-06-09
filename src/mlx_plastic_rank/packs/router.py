"""Domain routing runtime for on-demand pack attach/detach."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Protocol


class SupportsPackLifecycle(Protocol):
    """Minimal lifecycle contract required from a pack manager."""

    def apply_pack(self, pack_dir: Path): ...

    def detach_pack(self) -> None: ...


@dataclass(frozen=True)
class RouteEvent:
    """Describe a routing action for observability and debugging."""

    requested_domain: str
    resolved_domain: str
    action: str
    active_domain: str | None
    pack: str | None
    reason: str


def _normalise_domain(domain: str) -> str:
    return domain.strip().lower()


def resolve_pack_reference(ref: str, pack_root: Path = Path("packs")) -> Path:
    """Resolve a pack reference as absolute path, relative path, or pack name."""
    candidate = Path(ref).expanduser()
    if candidate.exists():
        return candidate.resolve()
    rooted = (pack_root / ref).expanduser()
    if rooted.exists():
        return rooted.resolve()
    raise FileNotFoundError(f"Pack reference '{ref}' not found as path or under {pack_root}")


def load_domain_map(path: Path, pack_root: Path = Path("packs")) -> Dict[str, Path | None]:
    """Load domain->pack mapping JSON.

    JSON format:
    {
      "core": null,
      "taxi": "bench-r4",
      "medical": "packs/medical-v1"
    }
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Domain map must be a JSON object mapping domain -> pack reference")

    domain_map: Dict[str, Path | None] = {}
    for key, value in raw.items():
        domain = _normalise_domain(str(key))
        if value in (None, "", "core", "base"):
            domain_map[domain] = None
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"Domain '{domain}' has invalid pack reference type {type(value).__name__}; expected string/null"
            )
        domain_map[domain] = resolve_pack_reference(value, pack_root=pack_root)
    return domain_map


class DomainPackRouter:
    """On-demand domain pack router with TTL expiry and LRU tracking."""

    def __init__(
        self,
        manager: SupportsPackLifecycle,
        domain_map: Mapping[str, Path | None],
        *,
        default_domain: str = "core",
        ttl_seconds: float = 300.0,
        max_recent_domains: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_recent_domains <= 0:
            raise ValueError("max_recent_domains must be >= 1")
        self.manager = manager
        self.domain_map: Dict[str, Path | None] = {
            _normalise_domain(k): v for k, v in domain_map.items()
        }
        self.default_domain = _normalise_domain(default_domain)
        if self.default_domain not in self.domain_map:
            self.domain_map[self.default_domain] = None

        self.ttl_seconds = float(ttl_seconds)
        self.max_recent_domains = int(max_recent_domains)
        self._clock = clock

        self._active_domain: str | None = None
        self._active_pack: Path | None = None
        self._last_touch: float | None = None
        self._recent: OrderedDict[str, float] = OrderedDict()
        self._expirations = 0

    @property
    def active_domain(self) -> str | None:
        return self._active_domain

    @property
    def active_pack(self) -> Path | None:
        return self._active_pack

    @property
    def expirations(self) -> int:
        return self._expirations

    def recent_domains(self) -> list[str]:
        return list(self._recent.keys())

    def route(self, requested_domain: str | None) -> RouteEvent:
        now = self._clock()
        self.expire_if_idle(now=now)

        requested = _normalise_domain(requested_domain or self.default_domain)
        resolved = requested if requested in self.domain_map else self.default_domain
        target_pack = self.domain_map.get(resolved)
        self._touch(resolved, now)

        if target_pack is None:
            if self._active_domain is None:
                return RouteEvent(
                    requested_domain=requested,
                    resolved_domain=resolved,
                    action="noop",
                    active_domain=None,
                    pack=None,
                    reason="already_on_core",
                )
            self.manager.detach_pack()
            self._active_domain = None
            self._active_pack = None
            self._last_touch = None
            return RouteEvent(
                requested_domain=requested,
                resolved_domain=resolved,
                action="detach",
                active_domain=None,
                pack=None,
                reason="route_to_core",
            )

        if self._active_domain == resolved:
            self._last_touch = now
            return RouteEvent(
                requested_domain=requested,
                resolved_domain=resolved,
                action="noop",
                active_domain=self._active_domain,
                pack=str(self._active_pack) if self._active_pack else None,
                reason="already_active",
            )

        if self._active_domain is not None:
            self.manager.detach_pack()
        self.manager.apply_pack(target_pack)
        previous = self._active_domain
        self._active_domain = resolved
        self._active_pack = target_pack
        self._last_touch = now

        return RouteEvent(
            requested_domain=requested,
            resolved_domain=resolved,
            action="switch" if previous is not None else "attach",
            active_domain=self._active_domain,
            pack=str(target_pack),
            reason="domain_route",
        )

    def force_detach(self) -> bool:
        if self._active_domain is None:
            return False
        self.manager.detach_pack()
        self._active_domain = None
        self._active_pack = None
        self._last_touch = None
        return True

    def expire_if_idle(self, *, now: float | None = None) -> bool:
        if self._active_domain is None:
            return False
        if self.ttl_seconds <= 0:
            return False
        if self._last_touch is None:
            return False
        current = self._clock() if now is None else now
        if (current - self._last_touch) < self.ttl_seconds:
            return False
        self.manager.detach_pack()
        self._active_domain = None
        self._active_pack = None
        self._last_touch = None
        self._expirations += 1
        return True

    def _touch(self, domain: str, timestamp: float) -> None:
        if domain in self._recent:
            self._recent.pop(domain)
        self._recent[domain] = timestamp
        while len(self._recent) > self.max_recent_domains:
            self._recent.popitem(last=False)
