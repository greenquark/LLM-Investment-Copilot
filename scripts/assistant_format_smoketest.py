"""
Smoke tests for assistant post-processing (formatting/provenance hygiene).

This does NOT call OpenAI. It validates:
- tool-name leakage like "(get_symbol_data)" is stripped from the narrative
- web_search sources appendix is formatted as a compact "### Sources" list when needed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_on_path()

    from ux_path_a.backend.backend_core.orchestrator import ChatOrchestrator

    # 1) Tool marker stripping
    messy = """
Sectors & example tickers (with tool outputs and rationale)

Defense / aerospace
LMT (Lockheed Martin): current price 582.43; price change +20.44% over the period. (get_symbol_data)

Energy / oil
XOM: current price 129.89. (get_symbol_data)

Source: `get_bars` (timeframe: 1D, bars: 90)
""".strip()

    cleaned = ChatOrchestrator._strip_tool_name_markers(messy)
    if any(k in cleaned for k in ["get_symbol_data", "get_bars", "(web_search)"]):
        print("FAIL: tool markers were not stripped", file=sys.stderr)
        print(cleaned, file=sys.stderr)
        return 2

    # 1b) “Sector + tickers” readability heuristic: ticker lines become bullets.
    sector_list = """
Data window: prices and moves shown are from tool outputs for 2025-12-22 → 2026-01-16.

Defense / Aerospace
LMT (Lockheed Martin): price 582.43, +20.44%
NOC (Northrop Grumman): price 666.90, +14.07%
RTX (Raytheon): price 201.92, +8.75%
LHX (L3Harris): price 346.46, +17.40%
Why this might matter: defense budgets often rise during regional tensions.
""".strip()
    improved = ChatOrchestrator._maybe_improve_readability_for_ticker_lists(sector_list)
    if improved.count("\n- **") < 3:
        print("FAIL: ticker list was not converted to bullets as expected", file=sys.stderr)
        print(improved, file=sys.stderr)
        return 5
    if "- **Why this might matter**:" not in improved:
        print("FAIL: why-it-matters line was not normalized", file=sys.stderr)
        print(improved, file=sys.stderr)
        return 6

    # 2) Sources appendix formatting when web_search results exist but no citations in content
    web_payload = {
        "query": "example query",
        "count": 2,
        "provider": "tavily",
        "results_markdown": "\n".join(
            [
                "- [Example A](https://example.com/a) — snippet a",
                "- [Example B](https://example.com/b) — snippet b",
            ]
        ),
    }
    tool_results = [{"name": "web_search", "result": json.dumps(web_payload)}]

    base_no_citations = "Here is a summary without links."
    appended = ChatOrchestrator._maybe_append_web_sources(
        user_message="latest headlines today",
        content=base_no_citations,
        tool_results=tool_results,
    )
    if "### Sources" not in appended or "1. " not in appended:
        print("FAIL: Sources appendix not added/formatted correctly", file=sys.stderr)
        print(appended, file=sys.stderr)
        return 3

    # 3) If citations exist, do NOT append sources
    base_with_citations = "See [Example](https://example.com/a) for details."
    not_appended = ChatOrchestrator._maybe_append_web_sources(
        user_message="latest headlines today",
        content=base_with_citations,
        tool_results=tool_results,
    )
    if not_appended != base_with_citations:
        print("FAIL: Sources should not be appended when citations exist", file=sys.stderr)
        print(not_appended, file=sys.stderr)
        return 4

    print("PASS: assistant formatting smoketest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

