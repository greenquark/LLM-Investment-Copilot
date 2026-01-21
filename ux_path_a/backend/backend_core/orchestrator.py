"""
LLM Orchestrator for UX Path A.

This module handles the orchestration of LLM calls, tool selection,
and response generation. It enforces platform invariants and ensures
all analytics come from tools.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from openai import OpenAI
from openai import APIError
import json
import ast
import re

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.config import settings
from ux_path_a.backend.backend_core.tools.registry import ToolRegistry
from ux_path_a.backend.backend_core.prompts import get_system_prompt

# Try to import centralized LLM config utilities
try:
    import sys
    from pathlib import Path
    project_root = settings.PLATFORM_ROOT
    sys.path.insert(0, str(project_root))
    from core.utils.llm_config import get_llm_models, is_newer_model, get_model_capabilities
    _has_llm_config = True
except Exception:
    _has_llm_config = False

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Orchestrates chat interactions with LLM and tools.
    
    Responsibilities:
    - Manage LLM calls
    - Route tool calls
    - Enforce invariants
    - Track token usage
    - Maintain conversation context
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.tool_registry = ToolRegistry()
        self.system_prompt = get_system_prompt()
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        # Try absolute import first (for local development), fallback to relative (for deployment)
        # Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
        from ux_path_a.backend.backend_core.tools.data_tools import register_data_tools
        from ux_path_a.backend.backend_core.tools.analysis_tools import register_analysis_tools
        from ux_path_a.backend.backend_core.tools.web_search_tools import register_web_search_tools
        
        # Register all tool categories
        register_data_tools(self.tool_registry)
        register_analysis_tools(self.tool_registry)
        register_web_search_tools(self.tool_registry)
        
        logger.info(f"Registered {len(self.tool_registry.get_function_definitions())} tools")

    @staticmethod
    def _message_requests_chart(message: str) -> bool:
        m = (message or "").lower()
        return any(k in m for k in ["chart", "graph", "plot", "visualize", "visualisation", "candlestick"])

    @staticmethod
    def _parse_tool_result_payload(payload: str) -> Optional[dict]:
        """
        Tool results should be JSON strings, but older deployments may have Python repr strings.
        Try JSON first, then fall back to ast.literal_eval.
        """
        if not payload:
            return None
        try:
            obj = json.loads(payload)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        try:
            obj = ast.literal_eval(payload)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_yyyy_mm_dd(message: str) -> Optional[str]:
        if not message:
            return None
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
        return m.group(1) if m else None

    @staticmethod
    def _message_requests_daily_bar(message: str) -> bool:
        m = (message or "").lower()
        return any(
            k in m
            for k in [
                "daily",
                "daily bar",
                "daily quote",
                "daily figures",
                "quote",
                "ohlc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "last friday",
                "today",
                "yesterday",
            ]
        )

    @staticmethod
    def _message_requests_quick_quote(message: str) -> bool:
        m = (message or "").lower()
        return any(
            k in m
            for k in [
                "price",
                "quote",
                "latest",
                "current",
                "today",
                "yesterday",
                "last friday",
                "daily figures",
                "daily quote",
            ]
        )

    @staticmethod
    def _format_money(x: Any) -> str:
        try:
            f = float(x)
            return f"${f:,.2f}"
        except Exception:
            return str(x)

    @staticmethod
    def _format_int(x: Any) -> str:
        try:
            return f"{int(float(x)):,}"
        except Exception:
            return str(x)

    @staticmethod
    def _strip_source_json_blocks(text: str) -> str:
        """
        Remove raw "Source (get_bars output): { ... }" dumps from the model response.
        We keep the human-readable summary and avoid showing raw JSON to users.
        """
        if not text:
            return text

        lines = text.splitlines()
        out: List[str] = []
        skipping = False
        brace_balance = 0
        started_json = False

        for line in lines:
            s = line.strip()

            # Start skipping if we see a "Source ... get_bars" marker
            if not skipping and s.lower().startswith("source") and "get_bars" in s.lower():
                skipping = True
                brace_balance = 0
                started_json = False
                continue

            if skipping:
                # Skip optional label lines like "Source (get_bars output):"
                # Then skip a raw JSON object block.
                if "{" in line:
                    started_json = True
                    brace_balance += line.count("{")
                if "}" in line:
                    brace_balance -= line.count("}")

                # If we never saw JSON, keep skipping until a blank line, then stop.
                if not started_json and s == "":
                    skipping = False
                # If we saw JSON and the braces are balanced, stop skipping after this line.
                elif started_json and brace_balance <= 0:
                    skipping = False
                continue

            out.append(line)

        return "\n".join(out).strip()

    @staticmethod
    def _strip_tool_name_markers(text: str) -> str:
        """
        Remove tool-name leakage from the user-visible narrative.

        Examples we strip:
        - "(get_symbol_data)" / "(get_bars)" / "(web_search)"
        - "Source: `get_bars` ..." or "Data source: ... get_symbol_data ..."

        We rely on the UI’s collapsible tool sections for provenance instead.
        """
        if not text:
            return text

        # Remove inline parenthetical markers.
        cleaned = re.sub(r"\(\s*(get_symbol_data|get_bars|web_search)\s*\)", "", text, flags=re.IGNORECASE)

        # Remove whole lines that are purely provenance/tool labels.
        out_lines = []
        for line in cleaned.splitlines():
            s = line.strip()
            lower = s.lower()

            # Remove explicit "Source:" / "Data source:" lines referencing tools.
            if (lower.startswith("source:") or lower.startswith("data source:")) and any(
                k in lower for k in ["get_symbol_data", "get_bars", "web_search"]
            ):
                continue

            # Remove lines that are just the tool name (with optional backticks) or common wrappers.
            if re.fullmatch(r"`?(get_symbol_data|get_bars|web_search)`?", s, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"\(?(get_symbol_data|get_bars|web_search)\)?", s, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"source\s*\(?(get_symbol_data|get_bars|web_search)[^)]*\)?\s*:?", s, flags=re.IGNORECASE):
                continue

            out_lines.append(line)

        # Normalize whitespace where markers were removed.
        cleaned2 = "\n".join(out_lines)
        cleaned2 = re.sub(r"[ \t]{2,}", " ", cleaned2)
        cleaned2 = re.sub(r"\n{3,}", "\n\n", cleaned2)
        return cleaned2.strip()

    @classmethod
    def _maybe_reformat_sector_beneficiaries(cls, user_message: str, text: str) -> str:
        """
        Heuristic reformatter for answers that look like:
        - multiple sector headings
        - multiple ticker lines like: "LMT (Lockheed Martin): price 582.43, +20.44%"
        - repeated "Why this might matter:" paragraphs

        Goal: produce a more ChatGPT-like, scannable layout:
        - ### Summary + TL;DR
        - per-sector ### headings + a compact table
        - bullets for "Why it might matter"
        - optional "What to watch next" + "Caveats" for geopolitics/news contexts
        """
        if not text:
            return text
        if "```chart" in text:
            return text

        # Count ticker lines to decide if we should trigger.
        ticker_re = re.compile(
            r"^([A-Z]{1,6})\s*\(([^)]+)\)\s*:\s*price\s*\$?(\d+(?:\.\d+)?)\s*[;,]\s*([+-]?\d+(?:\.\d+)?%)\s*$"
        )
        lines = [ln.rstrip() for ln in text.splitlines()]
        ticker_matches = [ticker_re.match(ln.strip()) for ln in lines if ln.strip()]
        ticker_count = sum(1 for m in ticker_matches if m)
        if ticker_count < 4:
            return text

        # Helper: detect sector heading lines by "next line is ticker".
        def _is_sector_heading(idx: int) -> bool:
            s = lines[idx].strip()
            if not s:
                return False
            if s.startswith("#") or s.startswith("-") or s.startswith("|"):
                return False
            if len(s) > 70:
                return False
            # next non-empty line must be a ticker line
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                return False
            return bool(ticker_re.match(lines[j].strip()))

        # Parse data window line if present.
        data_window = None
        for ln in lines:
            if ln.strip().lower().startswith("data window"):
                data_window = ln.strip().rstrip(".")
                break

        # Parse sections.
        sections = []
        i = 0
        while i < len(lines):
            if not _is_sector_heading(i):
                i += 1
                continue
            sector = lines[i].strip().rstrip(":")
            i += 1
            # tickers
            tickers = []
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                m = ticker_re.match(s)
                if not m:
                    break
                tickers.append(
                    {
                        "ticker": m.group(1),
                        "name": m.group(2),
                        "price": m.group(3),
                        "move": m.group(4),
                    }
                )
                i += 1
            # rationale until next sector heading
            rationale_lines = []
            while i < len(lines) and not _is_sector_heading(i):
                s = lines[i].strip()
                if s:
                    # normalize the common prefix
                    if s.lower().startswith("why this might matter:"):
                        s = s.split(":", 1)[1].strip()
                    rationale_lines.append(s)
                i += 1
            rationale = " ".join(rationale_lines).strip()
            sections.append({"sector": sector, "tickers": tickers, "rationale": rationale})

        # If we parsed at least one sector, we can reformat.
        # (Some answers only include one sector block; still worth improving.)
        if len(sections) < 1:
            return text

        # Build TL;DR bullets from the first sentence of each rationale (best-effort).
        tldr = []
        for sec in sections[:6]:
            r = sec.get("rationale") or ""
            first = r.split(".")[0].strip()
            if first:
                tldr.append(f"- **{sec['sector']}**: {first}.")
            else:
                tldr.append(f"- **{sec['sector']}**")

        out = []
        out.append("### Summary")
        out.append("")
        out.append(
            "Below are sectors and example names that *sometimes* see relative strength when geopolitical risk rises. "
            "This is descriptive (not predictive) and educational only."
        )
        out.append("")
        if tldr:
            out.append("---")
            out.append("")
            out.append("### TL;DR")
            out.append("")
            out.append("\n".join(tldr))
            out.append("")
        if data_window:
            out.append("---")
            out.append("")
            out.append("### Data window")
            out.append("")
            # Keep the original "Data window ..." line verbatim (best-effort provenance).
            out.append(data_window if data_window.lower().startswith("data window") else f"Data window: {data_window}")
            out.append("")

        for sec in sections:
            out.append("---")
            out.append("")
            out.append(f"### {sec['sector']}")
            out.append("")
            out.append("| Ticker | Name | Price | Move |")
            out.append("|---|---|---:|---:|")
            for row in sec["tickers"]:
                out.append(f"| **{row['ticker']}** | {row['name']} | {row['price']} | {row['move']} |")
            out.append("")
            if sec.get("rationale"):
                out.append(f"- **Why it might matter**: {sec['rationale']}")
                out.append("")

        m = (user_message or "").lower()
        if any(k in m for k in ["iran", "middle east", "geopolit", "headline", "news", "tension", "war", "conflict"]):
            out.append("---")
            out.append("")
            out.append("### What to watch next")
            out.append("")
            out.append("- **Oil**: spot/forward moves (Brent/WTI), crack spreads, and shipping insurance premia.")
            out.append("- **Risk sentiment**: VIX, credit spreads, and USD strength (risk-off vs risk-on).")
            out.append("- **Policy headlines**: sanctions, troop movements, and defense budget guidance/contract awards.")
            out.append("- **Supply chain**: Strait of Hormuz/shipping lane disruptions and refinery outages.")
            out.append("")

        out.append("---")
        out.append("")
        out.append("### Caveats")
        out.append("")
        out.append("- **Descriptive ≠ predictive**: recent outperformance doesn’t guarantee future performance.")
        out.append("- **Single-name risk**: company-specific catalysts can dominate macro themes.")
        out.append("- Consider diversified ETFs if you prefer broad exposure to the theme.")

        return "\n".join(out).strip()

    @staticmethod
    def _maybe_normalize_section_titles_and_dividers(text: str) -> str:
        """
        Convert common section labels into ChatGPT-like markdown headers and insert horizontal rules.

        This is a conservative "cleanup" pass that only activates when we see obvious
        placeholder/meta section labels (e.g., "2–4 line summary") or bare section names.
        """
        if not text:
            return text
        if "```chart" in text:
            return text

        lines = text.splitlines()

        placeholder_re = re.compile(r"^\s*\d+\s*[–-]\s*\d+\s*line\s+summary\s*$", re.IGNORECASE)
        meta_re = re.compile(r"^\s*@.*prompts\.py.*$", re.IGNORECASE)

        # Helper: normalize a would-be section label
        def _label_key(s: str) -> str:
            s = s.strip()
            s = re.sub(r"^[\s>*#-]+", "", s)  # leading markdown markers
            s = re.sub(r"[\s:：]+$", "", s)  # trailing ":" etc
            s = re.sub(r"^\*+|\*+$", "", s)  # surrounding asterisks
            s = re.sub(r"^_+|_+$", "", s)  # surrounding underscores
            s = s.strip().lower()
            # common normalization
            s = s.replace("’", "'")
            s = re.sub(r"\s+", " ", s)
            return s
        # Helper: next non-empty line
        def _next_nonempty(i: int) -> str:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            return lines[j].strip() if j < len(lines) else ""

        # Recognize simple section header lines.
        section_map = {
            "summary": "Summary",
            "tl;dr": "TL;DR",
            "tldr": "TL;DR",
            "data window": "Data window",
            "key facts": "Key facts (tool outputs)",
            "key facts (tool outputs)": "Key facts (tool outputs)",
            "facts": "Key facts (tool outputs)",
            "quick take": "Quick take",
            "how to interpret this": "How to interpret this",
            "how to read these signals": "How to interpret this",
            "how to read these signals (educational)": "How to interpret this",
            "if you're a": "If you’re a…",
            "if youre a": "If you’re a…",
            "what would change my view": "What would change my view",
            "next steps": "Next steps (pick one)",
            "next steps (pick one)": "Next steps (pick one)",
            "practical next steps i can do for you (choose one)": "Next steps (pick one)",
            "what to watch next": "What to watch next",
            "caveats": "Caveats",
        }

        out: list[str] = []
        in_code_block = False

        for idx, ln in enumerate(lines):
            s = ln.strip()
            lower = s.lower()

            # Respect fenced code blocks; do not rewrite headings/labels inside them.
            if s.startswith("```"):
                in_code_block = not in_code_block
                out.append(ln)
                continue
            if in_code_block:
                out.append(ln)
                continue
            # Drop common prompt/meta artifacts.
            if placeholder_re.match(s):
                continue
            if meta_re.match(s):
                continue

            # Normalize existing markdown headings (### ...) to preferred titles + dividers.
            if s.startswith("#"):
                title = s.lstrip("#").strip()
                key = _label_key(title)
                if key in section_map:
                    if out:
                        out.append("")
                        out.append("---")
                        out.append("")
                    out.append(f"### {section_map[key]}")
                    out.append("")
                    continue
                out.append(ln)
                continue

            # Normalize standalone section labels into headings.
            key = _label_key(s)
            if key in section_map:
                if out:
                    out.append("")
                    out.append("---")
                    out.append("")
                out.append(f"### {section_map[key]}")
                out.append("")
                continue

            # "Data window: ..." style lines -> heading + content line.
            if lower.startswith("data window"):
                if out:
                    out.append("")
                    out.append("---")
                    out.append("")
                out.append("### Data window")
                out.append("")
                # keep content if present after ':' or otherwise keep line
                if ":" in s:
                    out.append(s.split(":", 1)[1].strip())
                else:
                    out.append(s)
                out.append("")
                continue

            # Convert likely sector headings into ### headings if followed by a table.
            # Example: "Defense / Aerospace" then a markdown table.
            if s and not s.startswith("#") and not s.startswith("-") and not s.startswith("|"):
                nxt = _next_nonempty(idx)
                if nxt.startswith("|") or nxt.lower().startswith("ticker") or nxt.lower().startswith("name"):
                    if out:
                        out.append("")
                        out.append("---")
                        out.append("")
                    out.append(f"### {s.rstrip(':')}")
                    out.append("")
                    continue

            out.append(ln)

        cleaned = "\n".join(out)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _maybe_improve_readability_for_ticker_lists(text: str) -> str:
        """
        Best-effort readability pass for “sector + tickers” answers where the model
        outputs dense lines instead of bullets/tables.

        This is intentionally conservative:
        - Only activates when we detect multiple ticker-like lines that mention price/move.
        - Converts those lines into bullet points with bold ticker.
        - Converts "Why this might matter:" lines into bullets.
        - Normalizes spacing between sections.
        """
        if not text:
            return text

        lines = text.splitlines()
        ticker_pat = re.compile(
            r"^\s*([A-Z]{1,6})\s*\(([^)]+)\)\s*:\s*(.+)$"
        )

        ticker_hits = 0
        for ln in lines:
            m = ticker_pat.match(ln)
            if not m:
                continue
            tail = m.group(3).lower()
            if "price" in tail and ("%" in tail or "change" in tail or "+" in tail or "-" in tail):
                ticker_hits += 1

        # Only rewrite if it looks like a real ticker list.
        if ticker_hits < 3:
            return text

        out: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                # Avoid stacking too many blank lines; we'll normalize later.
                out.append("")
                continue

            # Normalize "Why this might matter:" to a bullet.
            if s.lower().startswith("why this might matter:"):
                rest = s.split(":", 1)[1].strip() if ":" in s else ""
                out.append(f"- **Why this might matter**: {rest}".rstrip())
                continue

            # Convert "TICKER (Name): ..." lines into bullets with bold ticker.
            m = ticker_pat.match(ln)
            if m:
                ticker = m.group(1).strip()
                name = m.group(2).strip()
                tail = m.group(3).strip()
                # Preserve if already a bullet
                if s.startswith("- "):
                    out.append(ln)
                else:
                    out.append(f"- **{ticker}** ({name}): {tail}")
                continue

            out.append(ln)

        cleaned = "\n".join(out)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _content_has_any_citation(text: str) -> bool:
        if not text:
            return False
        # Markdown link or bare http(s)
        if re.search(r"\[[^\]]+\]\((https?://[^)]+)\)", text):
            return True
        if re.search(r"https?://\S+", text):
            return True
        return False

    @classmethod
    def _maybe_append_web_sources(
        cls,
        user_message: str,
        content: Optional[str],
        tool_results: Optional[List[Dict[str, Any]]],
    ) -> str:
        """
        If web_search returned results but the model response failed to cite them
        (or incorrectly claims "no results"), append a compact Sources section.
        """
        base = content or ""
        if not tool_results:
            return base

        web_payload = None
        for tr in tool_results:
            if not isinstance(tr, dict) or tr.get("name") != "web_search":
                continue
            web_payload = cls._parse_tool_result_payload(tr.get("result") if isinstance(tr.get("result"), str) else "")
            if web_payload:
                break

        if not isinstance(web_payload, dict) or web_payload.get("error"):
            return base

        count = web_payload.get("count")
        results_md = web_payload.get("results_markdown")
        if not results_md or not isinstance(results_md, str):
            return base
        if not isinstance(count, int) or count <= 0:
            return base

        # If the model already provided citations, don't add anything.
        if cls._content_has_any_citation(base):
            return base

        # If the model explicitly says "no results", override by appending sources.
        lowered = base.lower()
        claims_no_results = ("no results" in lowered) or ("returned no results" in lowered)

        # Format a compact, scannable sources block (ChatGPT-like).
        # Convert "- [title](url) — ..." bullets to a numbered list.
        lines = [ln.strip() for ln in results_md.splitlines() if ln.strip()]
        max_sources = 6
        numbered: List[str] = []
        n = 1
        for ln in lines:
            if ln.startswith("- "):
                ln = ln[2:].strip()
            if not ln:
                continue
            numbered.append(f"{n}. {ln}")
            n += 1
            if n > max_sources:
                break

        if not numbered:
            return base

        header = "### Sources"
        suffix = f"{header}\n" + "\n".join(numbered)

        if claims_no_results or not base.strip():
            return (base.strip() + ("\n\n" if base.strip() else "") + suffix).strip()

        # Only append sources if the user intent likely required web search
        m = (user_message or "").lower()
        if any(k in m for k in ["news", "headline", "latest", "breaking", "today", "current", "update"]):
            return (base.strip() + "\n\n" + suffix).strip()

        return base

    @classmethod
    def _build_symbol_quote_markdown(cls, payload: dict) -> Optional[str]:
        if not isinstance(payload, dict) or payload.get("error"):
            return None

        symbol = str(payload.get("symbol") or "Symbol")
        current_price = payload.get("current_price")
        ts = payload.get("timestamp")

        md = []
        # ChatGPT-like compact “card” snapshot (markdown blockquote).
        low = payload.get("low")
        high = payload.get("high")
        md.append(f"> **{symbol} — Price snapshot**")
        md.append(f"> - **Price**: {cls._format_money(current_price)}")
        if ts:
            md.append(f"> - **As of**: {ts}")
        if low is not None and high is not None:
            md.append(f"> - **Day range**: {cls._format_money(low)}–{cls._format_money(high)}")
        md.append(f"> - **Volume**: {cls._format_int(payload.get('volume'))}")
        md.append("")

        md.append(f"### {symbol} — Quick quote (tool output)")
        md.append("")
        md.append("All values below are taken directly from the market-data tool output.")
        md.append("")
        md.append(f"- **Current price**: {cls._format_money(current_price)}")
        if ts:
            md.append(f"- **Timestamp**: {ts}")
        md.append(f"- **Open**: {cls._format_money(payload.get('open'))}")
        md.append(f"- **High**: {cls._format_money(payload.get('high'))}")
        md.append(f"- **Low**: {cls._format_money(payload.get('low'))}")
        md.append(f"- **Volume**: {cls._format_int(payload.get('volume'))}")
        # Optional extra fields (if present)
        extras = []
        for k in ["price_change", "price_change_pct"]:
            if k in payload and payload.get(k) is not None:
                extras.append(f"{k}: {payload.get(k)}")
        if extras:
            md.append(f"- **Change**: {', '.join(extras)}")

        return "\n".join(md).strip()

    @classmethod
    def _maybe_inject_symbol_quote(
        cls,
        user_message: str,
        content: Optional[str],
        tool_results: Optional[List[Dict[str, Any]]],
    ) -> str:
        base = content or ""
        if not cls._message_requests_quick_quote(user_message):
            return base
        if not tool_results:
            return base
        if "Quick quote (tool output)" in base:
            return base
        # Don't override chart responses
        if "```chart" in base:
            return base

        for tr in tool_results:
            if not isinstance(tr, dict) or tr.get("name") != "get_symbol_data":
                continue
            payload = cls._parse_tool_result_payload(tr.get("result") if isinstance(tr.get("result"), str) else "")
            quote_md = cls._build_symbol_quote_markdown(payload or {})
            if not quote_md:
                continue

            # If the model is dumping raw JSON or appending boilerplate, replace with the standard quote.
            looks_messy = (
                ("Source" in base and "get_bars" in base)
                or ("{" in base and "}" in base)
                or ("disclaimer" in base.lower())
                or ("not financial advice" in base.lower())
            )
            if looks_messy or not base.strip():
                return quote_md

            return (base + ("\n\n" if base else "") + quote_md).strip()

        return base

    @classmethod
    def _build_daily_bar_quote_markdown(cls, payload: dict, requested_date: Optional[str] = None) -> Optional[str]:
        bars = payload.get("bars")
        if not isinstance(bars, list) or not bars:
            return None

        symbol = str(payload.get("symbol") or "Symbol")
        timeframe = str(payload.get("timeframe") or "")

        # Choose bar: exact date match if provided, else last bar.
        chosen = None
        if requested_date:
            for b in bars:
                if isinstance(b, dict) and isinstance(b.get("timestamp"), str) and b["timestamp"].startswith(requested_date):
                    chosen = b
                    break
        if chosen is None:
            chosen = bars[-1] if isinstance(bars[-1], dict) else None
        if not isinstance(chosen, dict):
            return None

        ts = str(chosen.get("timestamp") or "")
        date = ts.split("T")[0] if ts else (requested_date or "")

        md = []
        md.append(f"### {symbol} — Daily bar ({date or timeframe or '1D'})")
        md.append("")
        md.append("| Open | High | Low | Close | Volume | Timestamp |")
        md.append("|---:|---:|---:|---:|---:|---|")
        md.append(
            f"| {cls._format_money(chosen.get('open'))} | {cls._format_money(chosen.get('high'))} | "
            f"{cls._format_money(chosen.get('low'))} | {cls._format_money(chosen.get('close'))} | "
            f"{cls._format_int(chosen.get('volume'))} | {ts} |"
        )
        return "\n".join(md)

    @staticmethod
    def _normalize_disclaimer_and_risk(text: str) -> str:
        """
        - Remove per-message 'Risk & use' boilerplate (keep it in /disclaimer).
        - Ensure disclaimer is a clickable markdown link (single line).
        """
        if not text:
            return text
        lines = text.splitlines()
        out = []
        saw_disclaimer = False
        for line in lines:
            if line.strip().lower().startswith("risk & use"):
                continue
            if line.strip().lower().startswith("educational only"):
                continue
            if "not financial advice" in line.lower():
                continue
            # Normalize any disclaimer line to a single clickable link at the end
            if line.strip().lower().startswith("disclaimer:"):
                saw_disclaimer = True
                continue
            out.append(line)
        cleaned = "\n".join(out).strip()
        disclaimer_line = "[Disclaimer](/disclaimer)" 

        # Always include the disclaimer link once per assistant message.
        if disclaimer_line not in cleaned:
            cleaned = (cleaned + ("\n\n" if cleaned else "") + disclaimer_line).strip()
        return cleaned

    @classmethod
    def _maybe_inject_daily_quote(cls, user_message: str, content: Optional[str], tool_results: Optional[List[Dict[str, Any]]]) -> str:
        base = content or ""
        if not cls._message_requests_daily_bar(user_message):
            return base
        if not tool_results:
            return base
        # If it's already nicely formatted (table), don't add another.
        if "| Open | High | Low | Close |" in base:
            return base

        requested_date = cls._extract_yyyy_mm_dd(user_message)

        for tr in tool_results:
            if not isinstance(tr, dict) or tr.get("name") != "get_bars":
                continue
            payload = cls._parse_tool_result_payload(tr.get("result") if isinstance(tr.get("result"), str) else "")
            if not payload or payload.get("error"):
                continue
            quote_md = cls._build_daily_bar_quote_markdown(payload, requested_date=requested_date)
            if not quote_md:
                continue
            # If the model dumped raw get_bars JSON / source block, replace with the standard quote.
            looks_like_raw_dump = ("get_bars" in base) or ("Source (get_bars" in base) or ("Source:" in base and "get_bars" in base) or ("{" in base and "}" in base)
            if looks_like_raw_dump:
                return quote_md
            return (base + ("\n\n" if base else "") + quote_md).strip()

        return base

    @classmethod
    def _build_candlestick_chart_block(cls, bars_payload: dict) -> Optional[str]:
        bars = bars_payload.get("bars")
        if not isinstance(bars, list) or not bars:
            return None

        symbol = bars_payload.get("symbol") or "Symbol"
        timeframe = (bars_payload.get("timeframe") or "").upper()

        x = []
        o = []
        h = []
        l = []
        c = []
        for b in bars:
            if not isinstance(b, dict):
                continue
            ts = b.get("timestamp")
            if not isinstance(ts, str):
                continue
            # Prefer date-only for daily bars
            x.append(ts.split("T")[0] if ("1D" in timeframe or timeframe in ("D", "1D")) else ts)
            o.append(b.get("open"))
            h.append(b.get("high"))
            l.append(b.get("low"))
            c.append(b.get("close"))

        if not x:
            return None

        chart = {
            "type": "candlestick",
            "data": [
                {
                    "x": x,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "name": str(symbol),
                }
            ],
            "layout": {
                "title": f"{symbol} Price Chart ({timeframe or 'bars'})",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Price ($)"},
                "height": 450,
                "hovermode": "x unified",
            },
        }
        return "```chart\n" + json.dumps(chart, indent=2) + "\n```"

    @classmethod
    def _maybe_inject_chart(cls, user_message: str, content: Optional[str], tool_results: Optional[List[Dict[str, Any]]]) -> str:
        """
        Ensure feature parity: if user asked for a chart and we have get_bars output,
        inject a valid ```chart block even if the LLM forgets.
        """
        base = content or ""
        if "```chart" in base:
            return base
        if not cls._message_requests_chart(user_message):
            return base
        if not tool_results:
            return base

        # Find get_bars output
        for tr in tool_results:
            if not isinstance(tr, dict):
                continue
            if tr.get("name") != "get_bars":
                continue
            payload = cls._parse_tool_result_payload(tr.get("result") if isinstance(tr.get("result"), str) else "")
            if not payload or payload.get("error"):
                return base
            block = cls._build_candlestick_chart_block(payload)
            if not block:
                return base
            return (base + ("\n\n" if base else "") + block).strip()

        return base
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process a user message and generate response.
        
        Args:
            message: User message text
            session_id: Session identifier
            conversation_history: Previous messages in conversation
            
        Returns:
            Response dictionary with content, tool_calls, token_usage
        """
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get tool definitions for function calling
        tools = self.tool_registry.get_function_definitions()
        
        try:
            # Call LLM with function calling
            # Handle different parameter requirements for different models
            # Use centralized config if available, otherwise fallback to name-based detection
            if _has_llm_config:
                is_newer_model_flag = is_newer_model(settings.OPENAI_MODEL, project_root=settings.PLATFORM_ROOT)
            else:
                is_newer_model_flag = "gpt-5" in settings.OPENAI_MODEL.lower() or "o1" in settings.OPENAI_MODEL.lower()
            
            llm_params = {
                "model": settings.OPENAI_MODEL,
                "messages": messages,
            }
            
            # Temperature is not supported by all models (e.g., gpt-5-mini, o1)
            if not is_newer_model_flag:
                llm_params["temperature"] = settings.OPENAI_TEMPERATURE
            
            # Add function calling if tools available
            # Newer models (gpt-5, o1) use "tools" parameter, older models use "functions"
            if tools:
                if is_newer_model_flag:
                    # Convert to tools format for newer models
                    llm_params["tools"] = [{"type": "function", "function": tool} for tool in tools]
                    llm_params["tool_choice"] = "auto"
                else:
                    # Legacy function calling format
                    llm_params["functions"] = tools
                    llm_params["function_call"] = "auto"
            
            # Use appropriate parameter based on model
            # For newer models (gpt-5, o1): try max_completion_tokens, but SDK might not support it
            # For older models: use max_tokens
            # Handle both SDK-level errors (TypeError) and API-level errors (APIError)
            if is_newer_model_flag:
                llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
            else:
                llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
            
            try:
                response = self.client.chat.completions.create(**llm_params)
            except TypeError as e:
                # SDK-level error: parameter not recognized by SDK
                error_str = str(e).lower()
                if "max_completion_tokens" in error_str and "unexpected keyword" in error_str:
                    # SDK doesn't support max_completion_tokens parameter, omit it
                    # The API will use its default or we'll get an API error telling us what to use
                    logger.warning(f"SDK doesn't support max_completion_tokens parameter, omitting token limit...")
                    llm_params.pop("max_completion_tokens", None)
                    response = self.client.chat.completions.create(**llm_params)
                else:
                    raise
            except (APIError, Exception) as e:
                # Check if error is about unsupported parameter
                error_str = str(e).lower()
                error_message = error_str
                
                # Try multiple ways to extract error message from APIError object
                if hasattr(e, 'body') and isinstance(e.body, dict):
                    error_obj = e.body.get('error', {})
                    if isinstance(error_obj, dict):
                        error_message = str(error_obj.get('message', '')).lower()
                elif hasattr(e, 'response') and hasattr(e.response, 'json'):
                    try:
                        error_data = e.response.json()
                        if isinstance(error_data, dict) and 'error' in error_data:
                            error_obj = error_data.get('error', {})
                            if isinstance(error_obj, dict):
                                error_message = str(error_obj.get('message', '')).lower()
                    except:
                        pass
                elif hasattr(e, 'message'):
                    error_message = str(e.message).lower()
                
                # Combine all error sources for checking
                combined_error = f"{error_str} {error_message}".lower()
                
                # Check if max_tokens is not supported and needs max_completion_tokens
                if ("max_tokens" in combined_error) and \
                   ("unsupported" in combined_error) and \
                   ("max_completion_tokens" in combined_error):
                    # API says max_tokens not supported, need max_completion_tokens
                    logger.warning(f"Model {settings.OPENAI_MODEL} requires max_completion_tokens instead of max_tokens, retrying...")
                    llm_params.pop("max_tokens", None)
                    llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
                    response = self.client.chat.completions.create(**llm_params)
                # Check if max_completion_tokens is not supported and needs max_tokens
                elif ("max_completion_tokens" in combined_error) and \
                     ("unsupported" in combined_error) and \
                     ("max_tokens" in combined_error):
                    # API says max_completion_tokens not supported, need max_tokens
                    logger.warning(f"Model {settings.OPENAI_MODEL} requires max_tokens instead of max_completion_tokens, retrying...")
                    llm_params.pop("max_completion_tokens", None)
                    llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
                    response = self.client.chat.completions.create(**llm_params)
                else:
                    # Some other error, re-raise it
                    raise
            
            message_response = response.choices[0].message
            
            # Extract thinking/reasoning content if available (for models like o1)
            thinking_content = None
            if hasattr(message_response, 'reasoning_content') and message_response.reasoning_content:
                thinking_content = message_response.reasoning_content
            elif hasattr(response.choices[0], 'reasoning_content') and response.choices[0].reasoning_content:
                thinking_content = response.choices[0].reasoning_content
            
            # Handle function calls (check both function_call and tool_calls)
            function_calls = []
            if hasattr(message_response, 'function_call') and message_response.function_call:
                function_calls.append(message_response.function_call)
            elif hasattr(message_response, 'tool_calls') and message_response.tool_calls:
                function_calls = message_response.tool_calls
            
            if function_calls:
                tool_results = await self._execute_tools(
                    function_calls=function_calls,
                    session_id=session_id,
                )
                
                # Get final response with tool results
                assistant_msg = {
                    "role": "assistant",
                    "content": message_response.content or "",
                }
                if hasattr(message_response, 'function_call') and message_response.function_call:
                    assistant_msg["function_call"] = message_response.function_call
                elif hasattr(message_response, 'tool_calls') and message_response.tool_calls:
                    assistant_msg["tool_calls"] = message_response.tool_calls
                messages.append(assistant_msg)
                
                # Add tool results
                # Newer models use "tool" role, older models use "function" role
                for tool_result in tool_results:
                    if is_newer_model_flag:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result.get("tool_call_id", ""),
                            "content": tool_result["result"],
                        })
                    else:
                        messages.append({
                            "role": "function",
                            "name": tool_result["name"],
                            "content": tool_result["result"],
                        })
                
                # Get final LLM response
                # Use same model type check (is_newer_model_flag already defined above)
                final_llm_params = {
                    "model": settings.OPENAI_MODEL,
                    "messages": messages,
                }
                
                # Temperature is not supported by all models (e.g., gpt-5-mini, o1)
                if not is_newer_model_flag:
                    final_llm_params["temperature"] = settings.OPENAI_TEMPERATURE
                
                # Use appropriate parameter based on model
                # For newer models (gpt-5, o1): try max_completion_tokens, but SDK might not support it
                # For older models: use max_tokens
                # Handle both SDK-level errors (TypeError) and API-level errors (APIError)
                if is_newer_model_flag:
                    final_llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
                else:
                    final_llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
                
                try:
                    final_response = self.client.chat.completions.create(**final_llm_params)
                except TypeError as e:
                    # SDK-level error: parameter not recognized by SDK
                    error_str = str(e).lower()
                    if "max_completion_tokens" in error_str and "unexpected keyword" in error_str:
                        # SDK doesn't support max_completion_tokens parameter, omit it
                        # The API will use its default or we'll get an API error telling us what to use
                        logger.warning(f"SDK doesn't support max_completion_tokens parameter, omitting token limit...")
                        final_llm_params.pop("max_completion_tokens", None)
                        final_response = self.client.chat.completions.create(**final_llm_params)
                    else:
                        raise
                except (APIError, Exception) as e:
                    # Check if error is about unsupported parameter
                    error_str = str(e).lower()
                    error_message = error_str
                    
                    # Try multiple ways to extract error message from APIError object
                    if hasattr(e, 'body') and isinstance(e.body, dict):
                        error_obj = e.body.get('error', {})
                        if isinstance(error_obj, dict):
                            error_message = str(error_obj.get('message', '')).lower()
                    elif hasattr(e, 'response') and hasattr(e.response, 'json'):
                        try:
                            error_data = e.response.json()
                            if isinstance(error_data, dict) and 'error' in error_data:
                                error_obj = error_data.get('error', {})
                                if isinstance(error_obj, dict):
                                    error_message = str(error_obj.get('message', '')).lower()
                        except:
                            pass
                    elif hasattr(e, 'message'):
                        error_message = str(e.message).lower()
                    
                    # Combine all error sources for checking
                    combined_error = f"{error_str} {error_message}".lower()
                    
                    # Check if max_tokens is not supported and needs max_completion_tokens
                    if ("max_tokens" in combined_error) and \
                       ("unsupported" in combined_error) and \
                       ("max_completion_tokens" in combined_error):
                        # API says max_tokens not supported, need max_completion_tokens
                        logger.warning(f"Model {settings.OPENAI_MODEL} requires max_completion_tokens instead of max_tokens, retrying...")
                        final_llm_params.pop("max_tokens", None)
                        final_llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
                        final_response = self.client.chat.completions.create(**final_llm_params)
                    # Check if max_completion_tokens is not supported and needs max_tokens
                    elif ("max_completion_tokens" in combined_error) and \
                         ("unsupported" in combined_error) and \
                         ("max_tokens" in combined_error):
                        # API says max_completion_tokens not supported, need max_tokens
                        logger.warning(f"Model {settings.OPENAI_MODEL} requires max_tokens instead of max_completion_tokens, retrying...")
                        final_llm_params.pop("max_completion_tokens", None)
                        final_llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
                        final_response = self.client.chat.completions.create(**final_llm_params)
                    else:
                        # Some other error, re-raise it
                        raise
                
                final_message_response = final_response.choices[0].message
                content = final_message_response.content
                
                # Extract thinking/reasoning content from final response if available
                if not thinking_content:
                    if hasattr(final_message_response, 'reasoning_content') and final_message_response.reasoning_content:
                        thinking_content = final_message_response.reasoning_content
                    elif hasattr(final_response.choices[0], 'reasoning_content') and final_response.choices[0].reasoning_content:
                        thinking_content = final_response.choices[0].reasoning_content
                
                # Combine token usage from both calls
                total_prompt_tokens = response.usage.prompt_tokens + final_response.usage.prompt_tokens
                total_completion_tokens = response.usage.completion_tokens + final_response.usage.completion_tokens
                total_tokens = response.usage.total_tokens + final_response.usage.total_tokens
            else:
                content = message_response.content
                tool_results = None
                total_prompt_tokens = response.usage.prompt_tokens
                total_completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

            # Ensure chart rendering parity (frontend requires a ```chart JSON block)
            content = self._maybe_inject_chart(message, content, tool_results)
            # For quote intents, prefer a stable "Quick quote (tool output)" format when get_symbol_data is available.
            content = self._maybe_inject_symbol_quote(message, content, tool_results)
            # Remove raw Source(get_bars) JSON dumps if the model includes them
            content = self._strip_source_json_blocks(content or "")
            # Remove tool-name leakage like "(get_symbol_data)" from the user-visible narrative.
            content = self._strip_tool_name_markers(content or "")
            # Improve readability for multi-ticker sector lists (prefer tables + summary when the pattern matches).
            content = self._maybe_reformat_sector_beneficiaries(message, content or "")
            # Fallback readability pass for ticker lists (bullets/spacing).
            content = self._maybe_improve_readability_for_ticker_lists(content or "")
            # Normalize common section titles into ### headings and add ChatGPT-like dividers.
            content = self._maybe_normalize_section_titles_and_dividers(content or "")
            # If web_search returned results but the model didn't cite them, append sources.
            content = self._maybe_append_web_sources(message, content, tool_results)
            # Keep disclaimers as a link (and remove per-message boilerplate)
            content = self._normalize_disclaimer_and_risk(content or "")
            
            # Serialize tool_calls to ensure they're JSON serializable
            tool_calls_serialized = None
            if function_calls:
                try:
                    import json
                    def serialize_tool_call(tc):
                        """Serialize a tool call object to a dict, handling nested Function objects."""
                        # Check if it's a Function-like object (has name and arguments attributes)
                        # This check must come before checking __dict__ because Function might not expose __dict__
                        try:
                            if hasattr(tc, 'name') and hasattr(tc, 'arguments'):
                                # This looks like a Function object from OpenAI SDK
                                func_dict = {
                                    'name': getattr(tc, 'name', None),
                                    'arguments': getattr(tc, 'arguments', None),
                                }
                                # Add optional attributes
                                if hasattr(tc, 'id'):
                                    func_dict['id'] = getattr(tc, 'id', None)
                                if hasattr(tc, 'type'):
                                    func_dict['type'] = getattr(tc, 'type', None)
                                return func_dict
                        except:
                            pass
                        
                        # Handle callable objects (functions, methods)
                        if callable(tc) and not isinstance(tc, type):
                            return f"<function: {getattr(tc, '__name__', str(tc))}>"
                        
                        if isinstance(tc, dict):
                            # Recursively serialize dict values
                            return {k: serialize_tool_call(v) for k, v in tc.items()}
                        elif hasattr(tc, '__dict__'):
                            result = {}
                            for k, v in tc.__dict__.items():
                                if not k.startswith('_'):
                                    result[k] = serialize_tool_call(v)
                            # Also check for attributes that might not be in __dict__ (like properties)
                            for attr in ['name', 'arguments', 'id', 'type', 'function']:
                                if hasattr(tc, attr) and attr not in result:
                                    try:
                                        attr_value = getattr(tc, attr)
                                        result[attr] = serialize_tool_call(attr_value)
                                    except:
                                        pass
                            return result
                        elif isinstance(tc, (str, int, float, bool, type(None))):
                            return tc
                        elif isinstance(tc, (list, tuple)):
                            return [serialize_tool_call(item) for item in tc]
                        else:
                            return str(tc)
                    
                    tool_calls_serialized = [serialize_tool_call(tc) for tc in function_calls]
                    # Test serialization
                    json.dumps(tool_calls_serialized)
                except Exception as e:
                    logger.warning(f"Could not serialize tool_calls in orchestrator: {e}", exc_info=True)
                    # Fallback: convert everything to string representation
                    try:
                        tool_calls_serialized = [str(tc) for tc in function_calls] if function_calls else None
                    except:
                        tool_calls_serialized = None
            
            return {
                "content": content,
                "thinking_content": thinking_content,  # Reasoning/thinking process from models like o1
                "tool_calls": tool_calls_serialized if function_calls else None,
                "tool_results": tool_results,
                "token_usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}", exc_info=True)
            raise
    
    async def _execute_tools(
        self,
        function_calls: List[Any],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls and return results.
        
        Args:
            function_calls: List of function call objects from LLM
            session_id: Session identifier for audit logging
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for func_call in function_calls:
            # Handle both function_call and tool_call formats
            tool_call_id = None
            if hasattr(func_call, 'function'):
                # Newer tool_calls format
                tool_name = func_call.function.name
                tool_args_str = func_call.function.arguments
                if hasattr(func_call, 'id'):
                    tool_call_id = func_call.id
            elif hasattr(func_call, 'id'):
                # Tool call with id
                tool_call_id = func_call.id
                if hasattr(func_call, 'function'):
                    tool_name = func_call.function.name
                    tool_args_str = func_call.function.arguments
                else:
                    tool_name = func_call.name if hasattr(func_call, 'name') else str(func_call)
                    tool_args_str = func_call.arguments if hasattr(func_call, 'arguments') else "{}"
            else:
                # Older function_call format
                tool_name = func_call.name
                tool_args_str = func_call.arguments
            
            # Parse arguments
            import json
            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except:
                tool_args = eval(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                # Execute tool via registry
                result = await self.tool_registry.execute_tool(
                    tool_name=tool_name,
                    arguments=tool_args,
                    session_id=session_id,
                )

                # Ensure tool results are valid JSON strings (helps LLM + keeps parity across models)
                if isinstance(result, str):
                    result_str = result
                else:
                    result_str = json.dumps(result, default=str)

                results.append({
                    "name": tool_name,
                    "result": result_str,
                    "tool_call_id": tool_call_id,
                })
                
                # TODO: Audit logging (INV-AUDIT-01, INV-AUDIT-02)
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                results.append({
                    "name": tool_name,
                    "result": f"Error: {str(e)}",
                    "tool_call_id": tool_call_id,
                })
        
        return results
