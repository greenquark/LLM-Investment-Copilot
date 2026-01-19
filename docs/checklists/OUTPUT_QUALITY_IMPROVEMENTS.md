## Output quality improvements checklist (ChatGPT parity)

Goal: make assistant responses more scannable and “ChatGPT-like” while keeping strong provenance via the existing collapsible tool UI.

### A) Structure & readability

- [ ] Use clear section headers (e.g., “Defense & Aerospace”, “Energy & Oil”) and bullet lists.
- [ ] Keep paragraphs short (1–3 lines) and avoid dense blocks.
- [ ] Use **bold** for tickers and key claims.
- [ ] Prefer tables for repeated numeric fields (Price / % change / period).

### B) Tool provenance without clutter

- [ ] Do not print tool names inline (avoid `(get_symbol_data)` / `(get_bars)` / `(web_search)`).
- [ ] If needed, include one short “Data window” line (e.g., “Prices from tools for 2025-12-22 → 2026-01-16”).
- [ ] Rely on the UI’s collapsible “Tool used / Tool results” for deep details.

### C) Sources / citations (web_search)

- [ ] When `web_search` is used, cite sources as **clickable links** near claims.
- [ ] If the model forgets, auto-append a compact “Sources” section with 2–5 links.
- [ ] Avoid duplicate sources / overly long snippets in the “Sources” block.

### D) Safety & honesty

- [ ] State clearly when web_search was not used (no live news fetched).
- [ ] Keep disclaimer short (link only), no repeated boilerplate.
- [ ] Add a short “What to watch” list for geopolitics/news-driven requests.

### E) Smoke tests (repeat until satisfied)

- [ ] `uv run python .\\scripts\\web_search_smoketest.py --provider tavily --query \"latest TQQQ news\" --max-results 3`
- [ ] `uv run python .\\scripts\\web_search_smoketest.py --provider duckduckgo --query \"Iran situation market reaction headlines\" --max-results 5`
- [ ] Run formatting smoketest script (added in this rollout) and ensure:
  - tool-name markers are stripped
  - “Sources” appendix is well-formed when needed

