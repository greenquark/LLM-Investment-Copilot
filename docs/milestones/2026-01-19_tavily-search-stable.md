## Milestone: Tavily search succeeded + stable build

- **Date**: 2026-01-19
- **Tag**: `milestone-2026-01-19-tavily-search-stable`
- **Commit**: `c1c2b71`

### What’s included

- **Web search**:
  - DuckDuckGo provider upgraded to `ddgs`
  - Tavily provider implemented with a 2-step pipeline: search → extract (with HTTP+trafilatura fallback)
  - Configurable via `WEB_SEARCH_PROVIDER`, `TAVILY_API_KEY`, and extract-related env vars
- **Smoke test**:
  - `scripts/web_search_smoketest.py` supports `--provider auto|duckduckgo|tavily`
  - Tavily path verified to return results and extracted content when configured

### How to reproduce locally

```bash
uv sync --extra ux_path_a_backend --extra data
uv run python .\scripts\web_search_smoketest.py --provider duckduckgo --query "latest TQQQ news" --max-results 3
uv run python .\scripts\web_search_smoketest.py --provider tavily --query "latest TQQQ news" --max-results 3
```

