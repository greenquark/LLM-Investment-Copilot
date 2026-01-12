"""
Web search tools for UX Path A.

These tools enable web search capabilities for real-time information.
Allows the LLM to search the web for current news, market information, and other data.
"""

from typing import Dict, Any, List, Optional
import logging
from ux_path_a.backend.core.tools.registry import Tool
from ux_path_a.backend.core.config import settings

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
        # Validate max_results
        max_results = max(1, min(10, max_results))
        
        try:
            # Try DuckDuckGo search (free, no API key required)
            try:
                from duckduckgo_search import DDGS
                
                logger.info(f"Executing web search: '{query}' (max_results={max_results})")
                
                with DDGS() as ddgs:
                    results = []
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                        })
                    
                    logger.info(f"Web search returned {len(results)} results for query: '{query}'")
                    
                    return {
                        "query": query,
                        "results": results,
                        "count": len(results),
                        "provider": "duckduckgo",
                    }
                    
            except ImportError:
                logger.warning("duckduckgo-search not installed. Install with: pip install duckduckgo-search")
                return {
                    "error": "Web search capability not configured. Install duckduckgo-search: pip install duckduckgo-search",
                    "query": query,
                }
            except Exception as e:
                logger.error(f"Error executing DuckDuckGo search: {e}", exc_info=True)
                # Try fallback if available
                return await self._fallback_search(query, max_results)
                
        except Exception as e:
            logger.error(f"Error in web_search: {e}", exc_info=True)
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
            }
    
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
