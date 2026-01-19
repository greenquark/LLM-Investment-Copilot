"""
Smoke test for UX Path A `web_search` tool.

This script runs the tool directly (no server/auth needed).

Usage (PowerShell):
  # Auto provider: uses Tavily when TAVILY_API_KEY is set; otherwise DuckDuckGo.
  uv run python .\\scripts\\web_search_smoketest.py --query "latest TQQQ news"

  # Force DuckDuckGo (no API key):
  uv run python .\\scripts\\web_search_smoketest.py --provider duckduckgo --query "latest TQQQ news"

  # Force Tavily (requires key):
  $env:TAVILY_API_KEY="tvly-..."
  uv run python .\\scripts\\web_search_smoketest.py --provider tavily --query "latest TQQQ news"

Hosted parity (Railway/Vercel) env vars:
  - WEB_SEARCH_PROVIDER=auto|tavily|duckduckgo
  - TAVILY_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _maybe_load_tavily_key_from_secrets() -> str | None:
    """
    Local-dev convenience: read Tavily key from gitignored config/secrets.yaml.
    Returns None if missing/unreadable.
    """
    try:
        secrets_path = Path(__file__).resolve().parents[1] / "config" / "secrets.yaml"
        if not secrets_path.exists():
            return None
        try:
            import yaml  # type: ignore
        except Exception:
            return None
        raw = secrets_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) if raw.strip() else None
        if not isinstance(data, dict):
            return None
        ws = data.get("web_search")
        if not isinstance(ws, dict):
            return None
        key = ws.get("tavily_api_key")
        if isinstance(key, str) and key.strip():
            return key.strip()
        return None
    except Exception:
        return None


def main() -> int:
    _ensure_repo_on_path()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "tavily", "duckduckgo"],
        help="Which provider to use (default: auto).",
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max-results", type=int, default=5, help="Max results (1-10)")
    parser.add_argument("--json", action="store_true", help="Print full JSON response")
    args = parser.parse_args()

    # Ensure env is configured BEFORE importing backend settings (which read env at import time).
    if args.provider:
        os.environ["WEB_SEARCH_PROVIDER"] = args.provider

    provider_effective = args.provider
    if provider_effective == "auto":
        provider_effective = "tavily" if os.environ.get("TAVILY_API_KEY") else "duckduckgo"

    if provider_effective == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        key = _maybe_load_tavily_key_from_secrets()
        if key:
            os.environ["TAVILY_API_KEY"] = key

    if provider_effective == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        print(
            "ERROR: provider=tavily but TAVILY_API_KEY is not set. "
            "Either set TAVILY_API_KEY or use --provider duckduckgo.",
            file=sys.stderr,
        )
        return 2

    import anyio  # type: ignore
    from ux_path_a.backend.backend_core.tools.web_search_tools import WebSearchTool

    async def _run():
        tool = WebSearchTool()
        return await tool.execute(query=args.query, max_results=args.max_results)

    payload = anyio.run(_run)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        provider = payload.get("provider")
        count = payload.get("count")
        extracted = payload.get("extracted_count")
        print(f"provider={provider} count={count} extracted_count={extracted}")
        results = payload.get("results") or []
        for i, r in enumerate(results[: min(len(results), 5)], start=1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            content = (r.get("content") or "").strip()
            print(f"\n#{i} {title}")
            print(f"- url: {url}")
            print(f"- snippet_len: {len(snippet)}")
            print(f"- content_len: {len(content)}")

    # Fail if we got no results and an error exists (helps CI usage)
    if payload.get("error"):
        print(f"\nERROR: {payload.get('error')}", file=sys.stderr)
        return 1
    if payload.get("count", 0) == 0:
        print("\nWARNING: zero results returned.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

