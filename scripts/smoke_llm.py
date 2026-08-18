#!/usr/bin/env python3
"""LLM bridge smoke test.

Sends a trivial prompt through runner.llm_client and reports the result.
Use this to verify the LLM proxy is reachable and the OAuth/API-key path works
BEFORE running a full harness attempt.

Usage:
    python scripts/smoke_llm.py --llm-config proxy/claude-code-oauth-host.json
    python scripts/smoke_llm.py --llm-config proxy/claude-code-oauth-host.json --prompt "hi"

Exit codes:
    0 - success (LLM answered)
    1 - configuration/network error
    2 - HTTP or LLM protocol error (429 rate-limit, 401 auth, malformed reply)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS_ROOT / "runner"))
# legacy/harness2 modules are imported flat (`import orchestrator`), not as a package.
sys.path.insert(0, str(HARNESS_ROOT / "legacy" / "harness2"))

from harness import load_llm_config  # noqa: E402
from llm_client import LLMClient, LLMError, LLMRateLimitError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--llm-config", required=True, type=Path,
                   help="Path to JSON with model/base_url/api_key")
    p.add_argument("--prompt", default="Reply with the two characters: OK",
                   help="User prompt to send")
    p.add_argument("--retries", type=int, default=1,
                   help="Retry count on transient failures (default 1 -> 2 total attempts)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="HTTP timeout in seconds (default 60)")
    args = p.parse_args(argv)

    if not args.llm_config.exists():
        print(f"[smoke_llm] ERROR: --llm-config not found: {args.llm_config}", file=sys.stderr)
        return 1

    try:
        cfg = load_llm_config(args.llm_config)
    except Exception as exc:
        print(f"[smoke_llm] ERROR loading --llm-config: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke_llm] base_url = {cfg['base_url']}")
    print(f"[smoke_llm] model    = {cfg['model']}")
    print(f"[smoke_llm] prompt   = {args.prompt!r}")
    print(f"[smoke_llm] retries  = {args.retries} (=> {args.retries + 1} attempts max)")

    client = LLMClient(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        timeout_sec=args.timeout,
        num_retries=args.retries,
    )
    started = time.monotonic()
    try:
        resp = client.messages(
            system="You are a smoke-test bot. Reply concisely.",
            messages=[{"role": "user", "content": args.prompt}],
            tools=None,
            max_tokens=64,
        )
    except LLMRateLimitError as exc:
        print(f"\n[smoke_llm] RATE-LIMITED: {exc}", file=sys.stderr)
        print("[smoke_llm] The bridge/upstream returned HTTP 429/529.", file=sys.stderr)
        print("[smoke_llm] Anthropic OAuth throttle windows are typically 30-60s.", file=sys.stderr)
        print("[smoke_llm] Wait a few minutes and rerun, or use a different account/token.", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"\n[smoke_llm] LLM ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\n[smoke_llm] UNEXPECTED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    text = "\n".join(resp.text_blocks).strip() or "(no text blocks)"
    usage = resp.raw.get("usage", {}) if isinstance(resp.raw, dict) else {}

    print(f"\n[smoke_llm] SUCCESS in {elapsed:.2f}s")
    print(f"[smoke_llm] response    : {text}")
    print(f"[smoke_llm] stop_reason : {resp.stop_reason}")
    print(f"[smoke_llm] tool_uses   : {len(resp.tool_uses)}")
    print(f"[smoke_llm] tokens      : input={usage.get('input_tokens', '?')} "
          f"output={usage.get('output_tokens', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
