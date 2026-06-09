import json
from pathlib import Path

from mlx_plastic_rank.packs.router import DomainPackRouter, load_domain_map


class FakeManager:
    def __init__(self):
        self.apply_calls: list[Path] = []
        self.detach_calls = 0

    def apply_pack(self, pack_dir: Path):
        self.apply_calls.append(pack_dir)

    def detach_pack(self) -> None:
        self.detach_calls += 1


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def test_router_attach_switch_and_detach():
    manager = FakeManager()
    router = DomainPackRouter(
        manager,
        {
            "core": None,
            "taxi": Path("/tmp/taxi-pack"),
            "legal": Path("/tmp/legal-pack"),
        },
        ttl_seconds=120.0,
    )

    first = router.route("taxi")
    assert first.action == "attach"
    assert first.active_domain == "taxi"
    assert manager.apply_calls == [Path("/tmp/taxi-pack")]
    assert manager.detach_calls == 0

    second = router.route("taxi")
    assert second.action == "noop"
    assert manager.apply_calls == [Path("/tmp/taxi-pack")]
    assert manager.detach_calls == 0

    switch = router.route("legal")
    assert switch.action == "switch"
    assert switch.active_domain == "legal"
    assert manager.apply_calls == [Path("/tmp/taxi-pack"), Path("/tmp/legal-pack")]
    assert manager.detach_calls == 1

    detach = router.route("core")
    assert detach.action == "detach"
    assert router.active_domain is None
    assert manager.detach_calls == 2


def test_router_ttl_expires_idle_pack():
    manager = FakeManager()
    clock = FakeClock()
    router = DomainPackRouter(
        manager,
        {"core": None, "taxi": Path("/tmp/taxi-pack")},
        ttl_seconds=10.0,
        clock=clock,
    )

    router.route("taxi")
    assert router.active_domain == "taxi"
    clock.advance(11.0)
    expired = router.expire_if_idle()
    assert expired is True
    assert router.active_domain is None
    assert manager.detach_calls == 1
    assert router.expirations == 1


def test_router_recent_domains_lru():
    manager = FakeManager()
    router = DomainPackRouter(
        manager,
        {
            "core": None,
            "taxi": Path("/tmp/taxi-pack"),
            "legal": Path("/tmp/legal-pack"),
            "finance": Path("/tmp/finance-pack"),
        },
        max_recent_domains=2,
    )

    router.route("taxi")
    router.route("legal")
    router.route("finance")
    assert router.recent_domains() == ["legal", "finance"]


def test_router_unknown_domain_falls_back_to_default():
    manager = FakeManager()
    router = DomainPackRouter(
        manager,
        {"core": None, "taxi": Path("/tmp/taxi-pack")},
        default_domain="core",
    )

    event = router.route("unknown")
    assert event.resolved_domain == "core"
    assert event.action == "noop"
    assert manager.apply_calls == []
    assert manager.detach_calls == 0


def test_load_domain_map_resolves_pack_names_and_paths(tmp_path: Path):
    pack_root = tmp_path / "packs"
    taxi = pack_root / "taxi-pack"
    medical = tmp_path / "medical-pack"
    taxi.mkdir(parents=True)
    medical.mkdir(parents=True)

    mapping = {
        "core": None,
        "taxi": "taxi-pack",
        "medical": str(medical),
    }
    map_path = tmp_path / "domain_map.json"
    map_path.write_text(json.dumps(mapping), encoding="utf-8")

    domain_map = load_domain_map(map_path, pack_root=pack_root)
    assert domain_map["core"] is None
    assert domain_map["taxi"] == taxi.resolve()
    assert domain_map["medical"] == medical.resolve()
