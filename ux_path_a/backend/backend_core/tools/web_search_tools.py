"""
Web search tools for UX Path A.

These tools enable web search capabilities for real-time information.
Allows the LLM to search the web for current news, market information, and other data.
"""

from typing import Dict, Any
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

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "provider": "tavily",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results_markdown": self._results_to_markdown(results),
        }

    async def _duckduckgo_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        DuckDuckGo search (no API key). This may be less reliable on hosted environments.
        """
        # Prefer the renamed package if present, fall back to the legacy name.
        # duckduckgo_search emits a runtime warning: "renamed to ddgs"
        try:
            from ddgs import DDGS  # type: ignore
            ddgs_pkg = "ddgs"
        except Exception:
            try:
                from duckduckgo_search import DDGS  # type: ignore
                ddgs_pkg = "duckduckgo_search"
            except ImportError:
                logger.warning("DuckDuckGo search package not installed. Install `ddgs` or `duckduckgo-search`.")
                return {
                    "error": "DuckDuckGo search is not installed. Install `ddgs` or configure Tavily (TAVILY_API_KEY).",
                    "query": query,
                    "provider": "duckduckgo",
                }

        from datetime import datetime, timezone
        import anyio

        timeout = settings.WEB_SEARCH_TIMEOUT_SECONDS

        def _run_sync():
            results = []
            # DDGS doesn't expose a clear per-request timeout; best-effort via underlying requests.
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", "") or "",
                            "snippet": r.get("body", "") or "",
                            "url": r.get("href", "") or "",
                        }
                    )
            return results

        try:
            with anyio.fail_after(timeout):
                results = await anyio.to_thread.run_sync(_run_sync)
        except TimeoutError:
            return {
                "error": f"duckduckgo search timed out after {timeout}s",
                "query": query,
                "provider": "duckduckgo",
            }

        payload = {
            "query": query,
            "results": results,
            "count": len(results),
            "provider": "duckduckgo",
            "provider_impl": ddgs_pkg,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results_markdown": self._results_to_markdown(results),
        }

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
