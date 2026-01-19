"""
Web search tools for UX Path A.

These tools enable web search capabilities for real-time information.
Allows the LLM to search the web for current news, market information, and other data.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.tools.registry import Tool
from ux_path_a.backend.backend_core.config import settings

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """
    Search the web for information.
    
    Returns real-time search results from the web. Useful for getting
    current news, market updates, company information, and other
    information that may not be available in historical market data.
    """
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return (
            "Search the web for current information, news, or data. "
            "Use this when you need up-to-date information that may not be "
            "available in historical market data. Returns search results with "
            "titles, snippets, and URLs. This is particularly useful for "
            "recent news, company announcements, market events, or any "
            "information that requires real-time web access."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string. Be specific and include relevant keywords.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        }
    
    async def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Execute the web search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-10)
            
        Returns:
            Dictionary with search results including titles, snippets, and URLs
        """
        if not settings.WEB_SEARCH_ENABLED:
            return {
                "error": "Web search is disabled on this deployment.",
                "query": query,
            }

        # Validate max_results (prefer settings default if caller omitted/invalid)
        if not isinstance(max_results, int):
            max_results = settings.WEB_SEARCH_DEFAULT_MAX_RESULTS
        max_results = max(1, min(10, max_results))

        provider = (settings.WEB_SEARCH_PROVIDER or "auto").strip().lower()
        if provider == "auto":
            provider = "tavily" if settings.TAVILY_API_KEY else "duckduckgo"
        
        try:
            logger.info(f"Executing web search ({provider}): '{query}' (max_results={max_results})")

            if provider == "tavily":
                if not settings.TAVILY_API_KEY:
                    return {
                        "error": "WEB_SEARCH_PROVIDER is set to tavily but TAVILY_API_KEY is not configured.",
                        "query": query,
                        "provider": "tavily",
                    }
                return await self._tavily_search(query, max_results)

            if provider == "duckduckgo":
                return await self._duckduckgo_search(query, max_results)

            return {
                "error": f"Unknown WEB_SEARCH_PROVIDER: {provider}. Use auto|tavily|duckduckgo.",
                "query": query,
            }
                
        except Exception as e:
            logger.error(f"Error in web_search: {e}", exc_info=True)
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
            }

    @staticmethod
    def _results_to_markdown(results: list) -> str:
        lines = []
        for r in results:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            if not title and not url:
                continue
            if title and url:
                line = f"- [{title}]({url})"
            elif url:
                line = f"- {url}"
            else:
                line = f"- {title}"
            if snippet:
                line += f" — {snippet}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
        if not isinstance(text, str):
            return "", False
        if max_chars <= 0:
            return "", True
        if len(text) <= max_chars:
            return text, False
        return text[: max_chars - 1] + "…", True

    @staticmethod
    def _pick_urls(results: List[Dict[str, Any]], max_docs: int) -> List[str]:
        urls: List[str] = []
        seen = set()
        for r in results:
            url = (r.get("url") or "").strip()
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                continue
            if url in seen:
                continue
            urls.append(url)
            seen.add(url)
            if len(urls) >= max_docs:
                break
        return urls

    async def _tavily_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Tavily Search API (reliable for hosted deployments).
        Docs: https://docs.tavily.com/
        """
        import httpx
        from datetime import datetime, timezone

        timeout = settings.WEB_SEARCH_TIMEOUT_SECONDS
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_results = data.get("results") if isinstance(data, dict) else None
        results = []
        if isinstance(raw_results, list):
            for r in raw_results[:max_results]:
                if not isinstance(r, dict):
                    continue
                results.append(
                    {
                        "title": r.get("title", "") or "",
                        "snippet": r.get("content", "") or "",
                        "url": r.get("url", "") or "",
                    }
                )

        # Step 2 (optional): extract readable text for top URLs (kept small for latency/cost).
        extracted_count = 0
        if settings.WEB_SEARCH_EXTRACT_ENABLED and results:
            urls = self._pick_urls(results, max(0, int(settings.WEB_SEARCH_EXTRACT_MAX_DOCS)))
            if urls:
                try:
                    extracted_by_url = await self._tavily_extract(urls)
                except Exception as e:
                    logger.warning("tavily extract failed; returning snippets only", extra={"error": str(e)})
                    extracted_by_url = {}

                # If Tavily extract is empty, fall back to direct HTTP fetch+extract.
                if not extracted_by_url:
                    try:
                        extracted_by_url = await self._http_extract(urls)
                    except Exception as e:
                        logger.warning("http extract failed; returning snippets only", extra={"error": str(e)})
                        extracted_by_url = {}

                for r in results:
                    url = (r.get("url") or "").strip()
                    if not url:
                        continue
                    content = extracted_by_url.get(url)
                    if content:
                        truncated, was_truncated = self._truncate(
                            content, max(0, int(settings.WEB_SEARCH_EXTRACT_MAX_CHARS))
                        )
                        r["content"] = truncated
                        r["content_truncated"] = was_truncated
                        extracted_count += 1

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "provider": "tavily",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results_markdown": self._results_to_markdown(results),
            "extracted_count": extracted_count,
        }

    async def _tavily_extract(self, urls: List[str]) -> Dict[str, str]:
        """
        Tavily Extract API: fetch/extract readable content for a list of URLs.
        Returns: {url: extracted_text}
        """
        import httpx

        timeout = settings.WEB_SEARCH_EXTRACT_TIMEOUT_SECONDS
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "urls": urls,
            # These fields are tolerated/ignored if unsupported by the API version.
            "include_images": False,
            "extract_depth": "basic",
            "include_raw_content": False,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://api.tavily.com/extract", json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Be liberal in what we accept: Tavily SDKs have used different shapes.
        # Common: {"results":[{"url":..., "content":...}, ...]}
        raw_results = None
        if isinstance(data, dict):
            raw_results = data.get("results") or data.get("data") or data.get("documents")
        if not isinstance(raw_results, list):
            return {}

        out: Dict[str, str] = {}
        for r in raw_results:
            if not isinstance(r, dict):
                continue
            url = (r.get("url") or r.get("source_url") or r.get("link") or "").strip()
            if not url:
                continue
            content = r.get("content") or r.get("text") or r.get("raw_content") or ""
            if not isinstance(content, str):
                continue
            content = content.strip()
            if content:
                out[url] = content
        return out

    async def _http_extract(self, urls: List[str]) -> Dict[str, str]:
        """
        Fallback extractor: fetch pages directly and extract readable text.
        This is slower than Tavily extract, but more deterministic than returning only snippets.
        """
        import httpx
        import anyio

        # `trafilatura` is a robust boilerplate-removal extractor.
        try:
            from trafilatura import extract as trafi_extract  # type: ignore
        except Exception:
            return {}

        timeout = float(settings.WEB_SEARCH_EXTRACT_TIMEOUT_SECONDS)
        headers = {
            # Some sites block empty UAs; keep it generic.
            "User-Agent": "Mozilla/5.0 (compatible; SmartTradingCopilot/1.0; +https://example.com)",
        }

        async def _fetch_one(client: httpx.AsyncClient, url: str) -> Tuple[str, str]:
            try:
                resp = await client.get(url, headers=headers, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text
                text = trafi_extract(html, url=url, include_comments=False, include_tables=False) or ""
                return url, (text.strip() if isinstance(text, str) else "")
            except Exception:
                return url, ""

        async with httpx.AsyncClient(timeout=timeout) as client:
            results = await anyio.gather(*[_fetch_one(client, u) for u in urls])

        out: Dict[str, str] = {}
        for url, text in results:
            if text:
                out[url] = text
        return out

    async def _duckduckgo_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        DuckDuckGo search (no API key). This may be less reliable on hosted environments.
        """
        # Prefer `ddgs` (current package), fall back to legacy `duckduckgo_search` if present.
        # DuckDuckGo search here is scraping-based and can be intermittently rate-limited/blocked,
        # especially from shared cloud IPs.
        try:
            from ddgs import DDGS  # type: ignore
            ddgs_pkg = "ddgs"
        except Exception:
            try:
                from duckduckgo_search import DDGS  # type: ignore
                ddgs_pkg = "duckduckgo_search"
            except ImportError:
                logger.warning("DuckDuckGo search package not installed. Install `ddgs` (preferred) or `duckduckgo-search`.")
                return {
                    "error": "DuckDuckGo search is not installed. Install `ddgs` or configure Tavily (TAVILY_API_KEY).",
                    "query": query,
                    "provider": "duckduckgo",
                }

        from datetime import datetime, timezone
        import anyio

        timeout = settings.WEB_SEARCH_TIMEOUT_SECONDS

        def _create_client() -> Any:
            """
            ddgs API has changed across versions; try to pass a timeout if supported,
            but gracefully fall back when not.
            """
            try:
                return DDGS(timeout=timeout)  # type: ignore[arg-type]
            except TypeError:
                return DDGS()

        def _parse_result(r: Any) -> Optional[Dict[str, str]]:
            if not isinstance(r, dict):
                return None
            title = (r.get("title") or r.get("heading") or "").strip()
            snippet = (r.get("body") or r.get("snippet") or r.get("content") or "").strip()
            url = (r.get("href") or r.get("url") or r.get("link") or "").strip()
            if not title and not url and not snippet:
                return None
            return {"title": title, "snippet": snippet, "url": url}

        def _run_sync() -> Dict[str, Any]:
            results = []
            try:
                # DDGS doesn't always expose a clear per-request timeout; best-effort via library support.
                with _create_client() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        parsed = _parse_result(r)
                        if parsed:
                            results.append(parsed)
                return {"results": results}
            except Exception as e:
                return {"error": str(e), "results": results}

        # Retry a couple times with small backoff. This helps with transient blocks and network hiccups.
        last_error: Optional[str] = None
        results: list = []
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                with anyio.fail_after(timeout):
                    out = await anyio.to_thread.run_sync(_run_sync)
                if isinstance(out, dict):
                    results = out.get("results") if isinstance(out.get("results"), list) else []
                    last_error = out.get("error") if isinstance(out.get("error"), str) and out.get("error") else None
                else:
                    results = []
                    last_error = "DuckDuckGo search returned unexpected output type."

                # Success path: results parsed.
                if results:
                    break

                # No parsed results: might still be an upstream block. Only retry if we have attempts left.
                if attempt < attempts:
                    await anyio.sleep(0.35 * attempt)
            except TimeoutError:
                last_error = f"duckduckgo search timed out after {timeout}s"
                if attempt < attempts:
                    await anyio.sleep(0.35 * attempt)
            except Exception as e:
                last_error = str(e)
                if attempt < attempts:
                    await anyio.sleep(0.35 * attempt)

        payload = {
            "query": query,
            "results": results,
            "count": len(results),
            "provider": "duckduckgo",
            "provider_impl": ddgs_pkg,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results_markdown": self._results_to_markdown(results),
        }
        if last_error:
            payload["last_error"] = last_error

        # Observability: log count + top URLs (helps debug cases where HTTP 200 occurs but parsing yields 0)
        try:
            top_urls = [r.get("url") for r in results[:3] if isinstance(r, dict) and r.get("url")]
        except Exception:
            top_urls = []
        logger.info(
            "web_search duckduckgo completed",
            extra={
                "query": query,
                "count": payload.get("count"),
                "provider_impl": ddgs_pkg,
                "top_urls": top_urls,
                "attempts": attempts,
                "had_error": bool(last_error),
            },
        )

        if payload.get("count", 0) == 0 and "error" not in payload:
            payload["warning"] = (
                "No results were parsed from DuckDuckGo search. "
                "This can happen due to rate limits/blocks or upstream HTML changes."
            )

        return payload
    
    async def _fallback_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Fallback search method if primary method fails.
        
        Can be extended to support other search providers like:
        - Google Custom Search API
        - Bing Search API
        - SerpAPI
        """
        # For now, return an error suggesting installation
        return {
            "error": "Web search is currently unavailable. Please ensure duckduckgo-search is installed.",
            "query": query,
            "suggestion": "Install with: pip install duckduckgo-search",
        }


def register_web_search_tools(registry):
    """Register all web search tools."""
    registry.register(WebSearchTool())
    logger.info("Registered web search tools")
