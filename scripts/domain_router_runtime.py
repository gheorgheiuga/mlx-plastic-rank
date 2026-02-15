"""Run on-demand domain routing with pack attach/detach (TTL + LRU)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load as load_model

from mlx_plastic_rank.packs.manager import LoRAManager
from mlx_plastic_rank.packs.router import DomainPackRouter, load_domain_map


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model path or HF id")
    parser.add_argument("--domain-map", required=True, type=Path, help="JSON mapping domain->pack")
    parser.add_argument("--input", required=True, type=Path, help="JSONL requests with {domain, prompt}")
    parser.add_argument("--ttl-seconds", type=float, default=120.0, help="Idle TTL before detach")
    parser.add_argument("--max-recent-domains", type=int, default=8, help="LRU list length")
    parser.add_argument("--default-domain", default="core", help="Fallback domain when missing")
    parser.add_argument(
        "--probe-forward",
        action="store_true",
        help="Run a single forward pass per request to include execution timing",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum prompt tokens for --probe-forward",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between requests (useful for TTL testing)",
    )
    parser.add_argument("--out", type=Path, help="Optional JSONL output path for routing events")
    return parser


def _tokenise_prompt(tokenizer, prompt: str, max_tokens: int) -> mx.array:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [0]
    clipped = token_ids[: max(1, max_tokens)]
    return mx.array([clipped], dtype=mx.int32)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if not args.domain_map.exists():
        raise SystemExit(f"Domain map file not found: {args.domain_map}")

    domain_map = load_domain_map(args.domain_map)
    print(f"Loaded {len(domain_map)} domain entries from {args.domain_map}")

    print(f"Loading base model from {args.base}...")
    model, tokenizer = load_model(str(args.base))
    manager = LoRAManager(model, base_checkpoint=None, base_model=str(args.base))
    router = DomainPackRouter(
        manager,
        domain_map,
        default_domain=args.default_domain,
        ttl_seconds=args.ttl_seconds,
        max_recent_domains=args.max_recent_domains,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    forward_total_ms = 0.0
    with args.input.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {lineno}: {exc}") from exc
            domain = payload.get("domain")
            prompt = payload.get("prompt") or payload.get("text") or ""

            route_start = time.time()
            event = router.route(domain)
            route_ms = (time.time() - route_start) * 1000.0

            forward_ms = None
            if args.probe_forward:
                tokens = _tokenise_prompt(tokenizer, str(prompt), args.max_tokens)
                start = time.time()
                logits = model(tokens)
                mx.eval(logits)
                forward_ms = (time.time() - start) * 1000.0
                forward_total_ms += forward_ms

            row = {
                "line": lineno,
                "requested_domain": event.requested_domain,
                "resolved_domain": event.resolved_domain,
                "action": event.action,
                "reason": event.reason,
                "active_domain": event.active_domain,
                "pack": event.pack,
                "route_ms": route_ms,
                "forward_ms": forward_ms,
            }
            print(json.dumps(row))
            if args.out:
                with args.out.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row) + "\n")

            total += 1
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    detached = router.force_detach()
    summary = {
        "requests": total,
        "ttl_expirations": router.expirations,
        "recent_domains": router.recent_domains(),
        "probe_forward_total_ms": forward_total_ms,
        "detached_on_exit": detached,
    }
    print(json.dumps({"summary": summary}))


if __name__ == "__main__":
    main()
