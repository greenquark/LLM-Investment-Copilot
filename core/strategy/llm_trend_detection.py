from __future__ import annotations
import os
try:
    from core.utils.timestamp import normalize_bar_range  # project import
except Exception:  # pragma: no cover
    try:
        from timestamp_patched import normalize_bar_range  # fallback for standalone tests
    except Exception:
        normalize_bar_range = None  # type: ignore


import json
import logging
import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from core.strategy.base import Strategy, Context  # MysticPulse-compatible API
from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.order import Order, Side, OrderType, InstrumentType
from core.visualization.models import TradeSignal, IndicatorData, LLMTrendIndicatorData

from core.models.llm_trend import (
    LLMTrendState,
    update_state as update_trend_state,
    get_state as get_trend_state,
)

from core.utils.llm_config import get_llm_models, is_newer_model

logger = logging.getLogger(__name__)


# =========================
# Config
# =========================

@dataclass
class LLMCallBudget:
    """Simple per-run guardrail to prevent runaway LLM calls."""
    def __init__(self, max_calls: int):
        self.max_calls = int(max_calls)
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def consume(self, n: int = 1) -> None:
        n = int(n)
        if n <= 0:
            return
        if self.used + n > self.max_calls:
            raise RuntimeError(f"LLM call budget exceeded: used {self.used}, requested {n}, max {self.max_calls}")
        self.used += n


@dataclass
class LLMTrendDetectionConfig:
    """
    Configuration for LLM_Trend_Detection strategy.

    This strategy:
    - Builds a structured payload from recent OHLCV data + indicators
    - Calls GPT-5-mini (or falls back) to classify trend regimes
    - Pushes state into llm_trend_registry for DBS2.0 and other agents
    - Emits natural-language narratives for interpretability
    """

    timeframe: str = "1D"
    lookback_bars: int = 250
    slope_window: int = 60
    ma_short: int = 20
    ma_medium: int = 50
    ma_long: int = 200
    rsi_length: int = 14

    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.0
    llm_timeout: float = 180.0  # Timeout in seconds for LLM API requests (default: 3 minutes)
    use_llm: bool = True
    openai_api_key: Optional[str] = None  # OpenAI API key (can also use OPENAI_API_KEY env var)

    # If True, strategy will also place trades based purely on regime
    enable_trading: bool = False
    capital_deployment_pct: float = 1.0
    
    # Minimum bars required - calculated from indicator requirements if not specified
    min_bars_required: Optional[int] = None

    def __post_init__(self):
        tf = self.timeframe.upper()
        if tf == "D":
            tf = "1D"
        elif tf == "H":
            tf = "1H"
        elif tf.endswith("M") and not tf.endswith("H"):
            tf = tf[:-1] + "m"
        self.timeframe = tf

        if not 0.0 <= self.capital_deployment_pct <= 1.0:
            raise ValueError("capital_deployment_pct must be between 0.0 and 1.0")
    
    def get_min_bars_required(self) -> int:
        """
        Calculate the minimum number of bars required based on indicator configurations.
        
        Returns the maximum of all indicator requirements:
        - MA short: ma_short bars
        - MA medium: ma_medium bars
        - MA long: ma_long bars
        - RSI: rsi_length bars
        - Slope: slope_window bars
        
        If min_bars_required is explicitly set, use that instead.
        """
        if self.min_bars_required is not None:
            return self.min_bars_required
        
        # Calculate from indicator requirements
        return max(
            self.ma_short,
            self.ma_medium,
            self.ma_long,
            self.rsi_length,
            self.slope_window
        )


# =========================
# LLM client wrapper
# =========================

SYSTEM_PROMPT = """
You are an LLM-based quantitative technical analyst named LLM_Trend_Detection.

You will classify the market trend regime using structured OHLCV data and indicators.
You must:

1. Determine short-, medium-, and long-term trend.
2. Determine regime: TREND_UP, TREND_DOWN, or RANGE.
3. Provide reasoning.
4. Output strict JSON with this schema:

{
  "trend_short": "TREND_UP | TREND_DOWN | RANGE",
  "trend_medium": "TREND_UP | TREND_DOWN | RANGE",
  "trend_long": "TREND_UP | TREND_DOWN | RANGE",
  "scores": {
    "trend_strength": 0.0,
    "range_strength": 0.0
  },
  "key_levels": {
    "support": [float],
    "resistance": [float]
  },
  "regime_final": "TREND_UP | TREND_DOWN | RANGE",
  "explanation": "Short, human-readable explanation",
  "summary_for_user": "1–2 sentence summary suitable for UI or chat"
}

Follow rules:

- TREND_UP if MA20 > MA50, slope > threshold, RSI > 55, ADX > 25.
- TREND_DOWN if MA20 < MA50, slope < -threshold, RSI < 45, ADX > 25.
- RANGE if slope near zero OR ADX < 20 OR MAs are flat.

Always return valid JSON. Do not add extra keys.
""".strip()


class LLMTrendDetectionLLMClient:
    """
    Async wrapper around OpenAI Chat Completions for JSON trend classification.
    """

    def __init__(self, model: str = "gpt-5-mini", temperature: float = 0.0, api_key: Optional[str] = None, timeout: float = 180.0, project_root: Optional[Path] = None):
        self._model = model
        self._project_root = project_root
        self._temperature = temperature
        self._timeout = timeout

        try:
            import openai
            import os
            
            # Check if we have the new API (openai >= 1.0.0) with AsyncOpenAI
            if hasattr(openai, 'AsyncOpenAI'):
                from openai import AsyncOpenAI  # type: ignore
                
                # Get API key from parameter, environment variable, or None
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                
                # Set timeout for client using the configured timeout value
                # Try to use Timeout class if available (newer openai versions)
                try:
                    from openai import Timeout as OpenAITimeout
                    # Use the configured timeout for read operations, keep connect/write/pool at reasonable defaults
                    client_timeout = OpenAITimeout(
                        connect=30.0, 
                        read=self._timeout, 
                        write=30.0, 
                        pool=30.0
                    )
                    timeout_kwarg = {"timeout": client_timeout}
                    timeout_msg = f" (timeout: 30s connect, {self._timeout}s read)"
                except (ImportError, AttributeError):
                    # Older openai versions don't have Timeout class
                    timeout_kwarg = {}
                    timeout_msg = f" (timeout: {self._timeout}s handled by asyncio.wait_for)"
                
                if api_key:
                    self._client = AsyncOpenAI(api_key=api_key, **timeout_kwarg)
                    logger.info(f"LLM_Trend_Detection: OpenAI client initialized with API key from config{timeout_msg}")
                else:
                    # Try default (reads from OPENAI_API_KEY env var)
                    self._client = AsyncOpenAI(**timeout_kwarg)
                    if os.getenv("OPENAI_API_KEY"):
                        logger.info(f"LLM_Trend_Detection: OpenAI client initialized with API key from environment{timeout_msg}")
                    else:
                        logger.warning(
                            "LLM_Trend_Detection: No OpenAI API key found. "
                            "Set openai_api_key in config or OPENAI_API_KEY environment variable."
                        )
                
                self._use_openai = True
            else:
                # Old API (openai < 1.0.0) - AsyncOpenAI not available
                self._client = None
                self._use_openai = False
                version = getattr(openai, '__version__', 'unknown')
                logger.error(
                    f"LLM_Trend_Detection: OpenAI package version {version} is too old. "
                    "AsyncOpenAI requires openai >= 1.0.0. "
                    "Please upgrade with: pip install --upgrade openai"
                )
        except ImportError:
            self._client = None
            self._use_openai = False
            logger.warning(
                "openai package not installed. LLM_Trend_Detection will "
                "fall back to heuristic classification only. "
                "Install with: pip install openai"
            )
        except Exception as e:
            self._client = None
            self._use_openai = False
            logger.error(
                f"LLM_Trend_Detection: Failed to initialize OpenAI client: {e}. "
                "Will fall back to heuristic classification only.",
                exc_info=True
            )

    async def classify_trend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._use_openai or self._client is None:
            raise RuntimeError("OpenAI client not available")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ]

        # Get models to try from centralized config (primary + fallbacks)
        models_to_try = get_llm_models(primary=self._model, project_root=self._project_root)

        # LLM call budget guardrail (configurable via env LLM_CALL_BUDGET / config)
        budget = None
        try:
            budget = int(os.getenv('LLM_CALL_BUDGET', '0'))
        except Exception:
            budget = 0
        if not budget:
            budget = int(getattr(self, '_llm_call_budget', 0) or 0)
        if not budget:
            budget = 50
        self._llm_budget = LLMCallBudget(budget)
        last_error = None
        
        for idx, model_name in enumerate(models_to_try):
            try:
                logger.info(f"LLM_Trend_Detection: Attempting model '{model_name}' (attempt {idx + 1}/{len(models_to_try)})")
                
                # Build request parameters
                # Use centralized model capabilities to determine parameter support
                is_newer = is_newer_model(model_name, project_root=self._project_root)
                
                request_params = {
                    "model": model_name,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
                
                # Newer models (gpt-5, o1) don't support temperature - omit it entirely
                if not is_newer:
                    request_params["temperature"] = self._temperature
                    request_params["max_tokens"] = 8000  # Set initial token limit for other models
                else:
                    # For newer models, use max_completion_tokens instead of max_tokens
                    request_params["max_completion_tokens"] = 8000  # Set initial token limit
                
                # Use timeout from config
                timeout_seconds = self._timeout
                
                # Add timeout to prevent hanging
                resp = await asyncio.wait_for(
                    self._client.chat.completions.create(**request_params),
                    timeout=timeout_seconds
                )
                
                content = resp.choices[0].message.content
                finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                
                # Handle empty content or truncation - retry with higher token limit
                retry_attempted = False
                if not content or finish_reason == "length":
                    if not retry_attempted:
                        retry_attempted = True
                        initial_limit = request_params.get("max_tokens") or request_params.get("max_completion_tokens", 0)
                        # Set higher token limit (8000 if no limit was set, double if limit existed)
                        new_limit = 8000 if initial_limit == 0 else initial_limit * 2
                        
                        logger.warning(
                            f"LLM_Trend_Detection: Model '{model_name}' returned "
                            f"{'empty content' if not content else 'truncated response (finish_reason=length)'}. "
                            f"Retrying with token limit={new_limit}..."
                        )
                        
                        # Use the appropriate token parameter based on model
                        if model_name.startswith("gpt-5-mini"):
                            # gpt-5-mini uses max_completion_tokens
                            if "max_tokens" in request_params:
                                del request_params["max_tokens"]
                            request_params["max_completion_tokens"] = new_limit
                        else:
                            # Other models use max_tokens
                            if "max_completion_tokens" in request_params:
                                del request_params["max_completion_tokens"]
                            request_params["max_tokens"] = new_limit
                        
                        timeout_seconds = self._timeout
                        resp = await asyncio.wait_for(
                            self._client.chat.completions.create(**request_params),
                            timeout=timeout_seconds
                        )
                        content = resp.choices[0].message.content
                        finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                        
                        # If still empty after retry, raise error
                        if not content:
                            raise ValueError(f"LLM_Trend_Detection: Model '{model_name}' returned empty content after retry (finish_reason={finish_reason})")
                    else:
                        # Already retried once, raise error
                        raise ValueError(
                            f"LLM_Trend_Detection: Model '{model_name}' returned "
                            f"{'empty content' if not content else 'truncated response'} even after increasing token limit"
                        )
                
                # Extract token usage information
                token_info = {}
                if hasattr(resp, 'usage') and resp.usage:
                    token_info = {
                        'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0),
                        'completion_tokens': getattr(resp.usage, 'completion_tokens', 0),
                        'total_tokens': getattr(resp.usage, 'total_tokens', 0),
                    }
                
                # Log token usage
                if token_info:
                    logger.info(
                        f"LLM_Trend_Detection: Model '{model_name}' token usage - "
                        f"Prompt: {token_info['prompt_tokens']}, "
                        f"Completion: {token_info['completion_tokens']}, "
                        f"Total: {token_info['total_tokens']}"
                    )
                else:
                    logger.warning(f"LLM_Trend_Detection: Token usage not available for model '{model_name}'")
                
                # Log if we used a fallback model
                if model_name != self._model:
                    logger.info(
                        f"LLM_Trend_Detection: Primary model '{self._model}' not available, "
                        f"successfully used fallback model '{model_name}'"
                    )
                
                # Parse JSON response
                # Primary: response_format should enforce JSON output
                # Fallback: regex extraction only if JSON parsing fails (should be rare)
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback: Try to extract JSON from markdown code blocks
                    logger.warning(
                        f"LLM_Trend_Detection: Direct JSON parse failed for '{model_name}'. "
                        f"Attempting regex extraction (response_format may not be enforced)."
                    )
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group(0))
                        else:
                            raise ValueError(f"Could not parse JSON from model '{model_name}' response: {content[:200]}")
                
                # Store token info in result for logging in strategy
                result['_token_usage'] = token_info
                return result
            except asyncio.TimeoutError as e:
                last_error = e
                timeout_used = self._timeout
                logger.warning(
                    f"LLM_Trend_Detection: Model '{model_name}' request timed out after {timeout_used}s, trying next fallback..."
                )
                continue
            except Exception as e:
                error_str = str(e).lower()
                error_full = str(e)
                # Log the error for debugging
                logger.debug(f"LLM_Trend_Detection: API error for '{model_name}': {error_full[:200]}")
                # Check if it's a model_not_found error (404)
                if "model_not_found" in error_str or "404" in error_str or "does not exist" in error_str:
                    last_error = e
                    logger.warning(
                        f"LLM_Trend_Detection: Model '{model_name}' not available, trying next fallback..."
                    )
                    continue
                # Check if it's a temperature error - retry without temperature
                elif "temperature" in error_str and ("unsupported" in error_str or "unsupported_value" in error_str):
                    try:
                        # Retry without temperature parameter
                        request_params_no_temp = {
                            "model": model_name,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                        }
                        # Use appropriate token parameter
                        if model_name.startswith("gpt-5-mini"):
                            request_params_no_temp["max_completion_tokens"] = 8000
                        else:
                            request_params_no_temp["max_tokens"] = 8000
                        timeout_seconds = self._timeout
                        resp = await asyncio.wait_for(
                            self._client.chat.completions.create(**request_params_no_temp),
                            timeout=timeout_seconds
                        )
                        content = resp.choices[0].message.content
                        finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                        
                        # Handle truncation or empty content - retry with higher token limit
                        if not content or finish_reason == "length":
                            initial_limit = request_params_no_temp.get("max_tokens") or request_params_no_temp.get("max_completion_tokens", 0)
                            new_limit = 8000 if initial_limit == 0 else initial_limit * 2
                            logger.warning(
                                f"LLM_Trend_Detection: Model '{model_name}' returned "
                                f"{'empty content' if not content else 'truncated response'} after temperature retry. "
                                f"Retrying with limit={new_limit}..."
                            )
                            # Use appropriate token parameter
                            if model_name.startswith("gpt-5-mini"):
                                if "max_tokens" in request_params_no_temp:
                                    del request_params_no_temp["max_tokens"]
                                request_params_no_temp["max_completion_tokens"] = new_limit
                            else:
                                if "max_completion_tokens" in request_params_no_temp:
                                    del request_params_no_temp["max_completion_tokens"]
                                request_params_no_temp["max_tokens"] = new_limit
                            resp = await asyncio.wait_for(
                                self._client.chat.completions.create(**request_params_no_temp),
                                timeout=timeout_seconds
                            )
                            content = resp.choices[0].message.content
                            finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                            
                            if not content:
                                raise ValueError(f"LLM_Trend_Detection: Model '{model_name}' returned empty content after retry (finish_reason={finish_reason})")
                        
                        # Extract token usage information
                        token_info = {}
                        if hasattr(resp, 'usage') and resp.usage:
                            token_info = {
                                'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0),
                                'completion_tokens': getattr(resp.usage, 'completion_tokens', 0),
                                'total_tokens': getattr(resp.usage, 'total_tokens', 0),
                            }
                        
                        # Log token usage
                        if token_info:
                            logger.info(
                                f"LLM_Trend_Detection: Model '{model_name}' token usage - "
                                f"Prompt: {token_info['prompt_tokens']}, "
                                f"Completion: {token_info['completion_tokens']}, "
                                f"Total: {token_info['total_tokens']}"
                            )
                        
                        logger.info(
                            f"LLM_Trend_Detection: Model '{model_name}' doesn't support temperature, "
                            f"using default temperature"
                        )
                        result = json.loads(content)
                        result['_token_usage'] = token_info
                        return result
                    except Exception as e2:
                        # If retry also fails, continue to next model
                        last_error = e2
                        logger.warning(
                            f"LLM_Trend_Detection: Model '{model_name}' failed even without temperature, "
                            f"trying next fallback..."
                        )
                        continue
                # Check if it's a max_tokens error - retry with max_completion_tokens (but no limit)
                elif "max_tokens" in error_str and "unsupported" in error_str:
                    try:
                        # Retry with max_completion_tokens instead of max_tokens (no limit)
                        logger.warning(f"LLM_Trend_Detection: Model '{model_name}' doesn't support max_tokens, retrying with max_completion_tokens (no limit)...")
                        request_params_fixed = {
                            "model": model_name,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                            "temperature": self._temperature,
                        }
                        timeout_seconds = self._timeout
                        resp = await asyncio.wait_for(
                            self._client.chat.completions.create(**request_params_fixed),
                            timeout=timeout_seconds
                        )
                        content = resp.choices[0].message.content
                        finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                        
                        # Handle truncation or empty content - retry with higher token limit
                        if not content or finish_reason == "length":
                            initial_limit = request_params_fixed.get("max_completion_tokens", 0)
                            new_limit = 8000 if initial_limit == 0 else initial_limit * 2
                            logger.warning(
                                f"LLM_Trend_Detection: Model '{model_name}' returned "
                                f"{'empty content' if not content else 'truncated response'} after max_completion_tokens retry. "
                                f"Retrying with limit={new_limit}..."
                            )
                            request_params_fixed["max_completion_tokens"] = new_limit
                            resp = await asyncio.wait_for(
                                self._client.chat.completions.create(**request_params_fixed),
                                timeout=timeout_seconds
                            )
                            content = resp.choices[0].message.content
                            finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                            
                            if not content:
                                raise ValueError(f"LLM_Trend_Detection: Model '{model_name}' returned empty content after retry (finish_reason={finish_reason})")
                        
                        # Extract token usage information
                        token_info = {}
                        if hasattr(resp, 'usage') and resp.usage:
                            token_info = {
                                'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0),
                                'completion_tokens': getattr(resp.usage, 'completion_tokens', 0),
                                'total_tokens': getattr(resp.usage, 'total_tokens', 0),
                            }
                        
                        # Log token usage
                        if token_info:
                            logger.info(
                                f"LLM_Trend_Detection: Model '{model_name}' token usage - "
                                f"Prompt: {token_info['prompt_tokens']}, "
                                f"Completion: {token_info['completion_tokens']}, "
                                f"Total: {token_info['total_tokens']}"
                            )
                        
                        logger.info(
                            f"LLM_Trend_Detection: Model '{model_name}' requires max_completion_tokens, "
                            f"using that instead"
                        )
                        result = json.loads(content)
                        result['_token_usage'] = token_info
                        return result
                    except Exception as e2:
                        # If retry also fails, continue to next model
                        last_error = e2
                        logger.warning(
                            f"LLM_Trend_Detection: Model '{model_name}' failed even with max_completion_tokens, "
                            f"trying next fallback..."
                        )
                        continue
                # Check if it's a quota error (429) - model exists but quota exceeded
                elif "quota" in error_str or "429" in error_str or "insufficient_quota" in error_str:
                    last_error = e
                    logger.warning(
                        f"LLM_Trend_Detection: Model '{model_name}' quota exceeded (429). "
                        f"Model exists but API key needs billing/quota setup. Trying next fallback..."
                    )
                    continue
                # Check if it's a response_format error - retry without response_format
                elif "response_format" in error_str or "json_object" in error_str or ("unsupported" in error_str and "format" in error_str):
                    try:
                        # Retry without response_format (no token limit)
                        logger.warning(f"LLM_Trend_Detection: Model '{model_name}' doesn't support response_format, retrying without it...")
                        request_params_no_json = {
                            "model": model_name,
                            "messages": messages,
                            "temperature": self._temperature,
                        }
                        timeout_seconds = self._timeout
                        resp = await asyncio.wait_for(
                            self._client.chat.completions.create(**request_params_no_json),
                            timeout=timeout_seconds
                        )
                        content = resp.choices[0].message.content
                        finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                        
                        # Handle truncation or empty content - retry with higher token limit
                        if not content or finish_reason == "length":
                            initial_limit = request_params_no_json.get("max_tokens") or request_params_no_json.get("max_completion_tokens", 0)
                            new_limit = 8000 if initial_limit == 0 else initial_limit * 2
                            logger.warning(
                                f"LLM_Trend_Detection: Model '{model_name}' returned "
                                f"{'empty content' if not content else 'truncated response'} after response_format retry. "
                                f"Retrying with limit={new_limit}..."
                            )
                            # Use max_tokens (most models support it)
                            if "max_completion_tokens" in request_params_no_json:
                                del request_params_no_json["max_completion_tokens"]
                            request_params_no_json["max_tokens"] = new_limit
                            resp = await asyncio.wait_for(
                                self._client.chat.completions.create(**request_params_no_json),
                                timeout=timeout_seconds
                            )
                            content = resp.choices[0].message.content
                            finish_reason = resp.choices[0].finish_reason if hasattr(resp.choices[0], 'finish_reason') else None
                            
                            if not content:
                                raise ValueError(f"LLM_Trend_Detection: Model '{model_name}' returned empty content after retry (finish_reason={finish_reason})")
                        
                        token_info = {}
                        if hasattr(resp, 'usage') and resp.usage:
                            token_info = {
                                'prompt_tokens': getattr(resp.usage, 'prompt_tokens', 0),
                                'completion_tokens': getattr(resp.usage, 'completion_tokens', 0),
                                'total_tokens': getattr(resp.usage, 'total_tokens', 0),
                            }
                        logger.info(f"LLM_Trend_Detection: Model '{model_name}' doesn't support response_format, retrying without it succeeded")
                        # Parse JSON - will use regex extraction since no response_format
                        try:
                            result = json.loads(content)
                        except json.JSONDecodeError:
                            # Fallback: Try to extract JSON from markdown code blocks
                            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                            if json_match:
                                result = json.loads(json_match.group(1))
                            else:
                                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                                if json_match:
                                    result = json.loads(json_match.group(0))
                                else:
                                    raise ValueError(f"Could not parse JSON from model '{model_name}' response: {content[:200]}")
                        result['_token_usage'] = token_info
                        return result
                    except Exception as e2:
                        last_error = e2
                        logger.warning(f"LLM_Trend_Detection: Model '{model_name}' failed even without response_format, trying next fallback...")
                        continue
                else:
                    # For other errors (network, etc.), don't try other models
                    logger.exception(f"LLM_Trend_Detection classify_trend failed with model '{model_name}': {e}")
                    raise RuntimeError(f"LLM_Trend_Detection classify_trend failed: {e}") from e
        
        # All models failed
        if last_error:
            logger.error(
                f"LLM_Trend_Detection: All models failed. Last error: {last_error}. "
                f"Tried models: {models_to_try}"
            )
            raise RuntimeError(f"LLM_Trend_Detection: No available models. Tried: {models_to_try}. Last error: {last_error}") from last_error
        else:
            raise RuntimeError("LLM_Trend_Detection: No models to try")


# =========================
# Utility indicators
# =========================

def _sma(values: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(values) < length:
        return None
    window = values[-length:]
    return sum(window) / float(length)


def _rsi(values: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(values) <= length:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(-length, 0):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
        else:
            losses.append(-diff)
    avg_gain = sum(gains) / length if gains else 0.0
    avg_loss = sum(losses) / length if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _regression_slope(values: List[float], length: int) -> Optional[float]:
    if length <= 1 or len(values) < length:
        return None
    window = values[-length:]
    n = len(window)
    x_vals = list(range(n))
    mean_x = sum(x_vals) / n
    mean_y = sum(window) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, window))
    den = sum((x - mean_x) ** 2 for x in x_vals)
    if den == 0:
        return 0.0
    return num / den


def _calculate_bollinger_bands(
    closes: List[float], length: int, std_dev: float = 2.0
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate Bollinger Bands (upper, middle, lower) for the last value.
    
    Formula:
    - Middle Band (BB Middle) = Simple Moving Average (SMA) of closing prices
    - Upper Band = SMA + (std_dev × standard deviation)
    - Lower Band = SMA - (std_dev × standard deviation)
    
    Uses sample standard deviation (divides by n-1) to match pandas rolling().std()
    
    Args:
        closes: List of closing prices
        length: Period for SMA calculation (e.g., 20)
        std_dev: Number of standard deviations (e.g., 2.0)
        
    Returns:
        Tuple of (upper, middle, lower) for the last value, or (None, None, None) if insufficient data
    """
    if len(closes) < length:
        return None, None, None
    
    window = closes[-length:]
    sma = sum(window) / length
    
    # Calculate sample standard deviation (matches pandas rolling().std() which uses ddof=1)
    if length > 1:
        variance = sum((x - sma) ** 2 for x in window) / (length - 1)  # Sample std: divide by n-1
    else:
        variance = 0.0
    std = (variance ** 0.5) if variance > 0 else 0.0
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower


# =========================
# Strategy implementation
# =========================

class LLMTrendDetectionStrategy(Strategy):
    """
    LLM_Trend_Detection strategy.

    - MysticPulse-compatible Strategy
    - Uses GPT-5-mini (or fallback) to classify trend regimes
    - Stores structured state + narratives in llm_trend_registry
    - Emits TradeSignal / IndicatorData for visualization

    LLM calls have **daily granularity**:
    - At most one OpenAI call per symbol per calendar day.
    - Additional decisions on the same day reuse cached LLM result.
    """

    def __init__(self, symbol: str, config: LLMTrendDetectionConfig, data_engine: DataEngine, project_root: Optional[Path] = None):
        self._symbol = symbol
        self._cfg = config
        self._data = data_engine

        self._llm = LLMTrendDetectionLLMClient(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=config.openai_api_key,
            timeout=config.llm_timeout,
            project_root=project_root,
        )

        self._bars: List[Bar] = []
        self._signals: List[TradeSignal] = []
        self._indicator_history: List[IndicatorData] = []  # Keep for backward compatibility
        self._llm_indicator_history: List[LLMTrendIndicatorData] = []  # New indicator data with BB and RSI
        self._regime_history: List[Dict[str, Any]] = []  # Store regime decisions for charting
        self._previous_regime: Optional[str] = None  # Track previous regime for transition detection

        # Initialize from registry if available (cross-session cache)
        # Try to load latest state (file-based cache will be loaded on first access)
        self._last_state: Optional[LLMTrendState] = get_trend_state(self._symbol, self._cfg.timeframe)
        if self._last_state:
            self._last_llm_analysis_date = self._last_state.as_of.date()
            self._previous_regime = self._last_state.regime_final
        else:
            self._last_llm_analysis_date: Optional[date] = None  # daily granularity

    # --- Public accessors ---

    def get_signals(self) -> List[TradeSignal]:
        return self._signals

    def get_indicator_history(self) -> List[IndicatorData]:
        """Get indicator history (backward compatible)."""
        return self._indicator_history
    
    def get_llm_indicator_history(self) -> List[LLMTrendIndicatorData]:
        """Get LLM indicator history with BB and RSI."""
        return self._llm_indicator_history

    def get_bars(self) -> List[Bar]:
        return self._bars

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """Get history of regime decisions for visualization."""
        return self._regime_history

    def get_latest_state(self) -> Optional[LLMTrendState]:
        return self._last_state

    def get_narratives(self) -> List[str]:
        """
        Returns short human/LLM-readable narratives that explain the current regime.
        Future chat UIs can call this directly.
        """
        if self._last_state is None:
            return []
        texts: List[str] = []
        if self._last_state.summary_for_user:
            texts.append(self._last_state.summary_for_user)
        if self._last_state.explanation:
            texts.append(self._last_state.explanation)
        return texts

    # --- Lifecycle ---

    async def on_start(self, ctx: Context) -> None:
        ctx.log(
            f"LLM_Trend_Detection started for {self._symbol} "
            f"(timeframe={self._cfg.timeframe}, lookback_bars={self._cfg.lookback_bars}, "
            f"use_llm={self._cfg.use_llm})"
        )

    async def on_decision(self, ctx: Context, now: datetime) -> None:
        # Calculate minimum required bars from configuration
        min_bars_required = self._cfg.get_min_bars_required()
        
        # 1) Get bars (reuse ctx._bars if provided by engine AND sufficient, else fetch)
        bars: List[Bar] = None
        if hasattr(ctx, "_bars") and ctx._bars is not None:
            # Only use ctx._bars if it has enough bars
            if len(ctx._bars) >= min_bars_required:
                bars = ctx._bars
            else:
                # ctx._bars exists but is insufficient, will fetch below
                ctx.log(
                    f"[LLM_Trend_Detection] ctx._bars has insufficient bars: {len(ctx._bars)} "
                    f"(required: {min_bars_required}), fetching more..."
                )
        
        # If we don't have sufficient bars yet, fetch them
        if not bars or len(bars) < min_bars_required:
            timeframe_upper = self._cfg.timeframe.upper()
            if timeframe_upper == "1D":
                # For daily bars, calculate start date based on trading days, not calendar days
                # This ensures we only fetch the exact number of bars needed (e.g., 200 trading days, not 503)
                from core.utils.trading_days import get_trading_days_lookback_datetime
                start = get_trading_days_lookback_datetime(now, min_bars_required)
            elif timeframe_upper.endswith("H"):
                start = now - timedelta(days=365)  # 1 year for hourly
            else:
                start = now - timedelta(days=180)  # 180 days for others
            
            current_count = len(bars) if bars else 0
            ctx.log(
                f"[LLM_Trend_Detection] Fetching bars from {start.date()} to {now.date()} "
                f"(currently have {current_count}, need {min_bars_required})"
            )
            bars = await self._data.get_bars(self._symbol, start, now, self._cfg.timeframe)
            
            fetched_count = len(bars) if bars else 0
            ctx.log(
                f"[LLM_Trend_Detection] Fetched {fetched_count} bars "
                f"(required: {min_bars_required})"
            )
            
            # Final check - if still insufficient, raise error
            if not bars or len(bars) < min_bars_required:
                raise RuntimeError(
                    f"Insufficient bars for {self._symbol}: {len(bars) if bars else 0} "
                    f"(required: {min_bars_required}). "
                    f"Indicator requirements: MA_short={self._cfg.ma_short}, "
                    f"MA_medium={self._cfg.ma_medium}, MA_long={self._cfg.ma_long}, "
                    f"RSI_length={self._cfg.rsi_length}, slope_window={self._cfg.slope_window}. "
                    f"Tried fetching from {start.date()} to {now.date()}."
                )

        bars.sort(key=lambda b: b.timestamp)
        if len(bars) > self._cfg.lookback_bars:
            bars = bars[-self._cfg.lookback_bars :]

        self._bars = bars

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        timestamps = [b.timestamp for b in bars]

        last_close = closes[-1]
        last_ts = timestamps[-1]

        # 2) Compute lightweight indicators
        ma_short = _sma(closes, self._cfg.ma_short)
        ma_medium = _sma(closes, self._cfg.ma_medium)
        ma_long = _sma(closes, self._cfg.ma_long)
        rsi = _rsi(closes, self._cfg.rsi_length)
        slope = _regression_slope(closes, self._cfg.slope_window)
        
        # Calculate Bollinger Bands (20-period, 2 std dev)
        bb_upper, bb_middle, bb_lower = _calculate_bollinger_bands(closes, length=20, std_dev=2.0)

        payload: Dict[str, Any] = {
            "agent_name": "LLM_Trend_Detection",
            "symbol": self._symbol,
            "timeframe": self._cfg.timeframe,
            "as_of": last_ts.isoformat(),
            "bars": [
                {
                    "t": b.timestamp.isoformat(),
                    "o": b.open,
                    "h": b.high,
                    "l": b.low,
                    "c": b.close,
                    "v": b.volume,
                }
                for b in bars
            ],
            "features": {
                "ma": {
                    "ma_short": ma_short,
                    "ma_medium": ma_medium,
                    "ma_long": ma_long,
                },
                "rsi": rsi,
                "regression_slope": slope,
                # placeholders: extend later with ADX, ATR, Bollinger, etc.
                "adx": None,
                "atr": None,
            },
        }

        # 3) Decide whether to call LLM (daily granularity + caching)
        analysis_date = last_ts.date()
        result: Dict[str, Any]

        if self._cfg.use_llm:
            should_call_llm = (
                self._last_llm_analysis_date is None
                or self._last_llm_analysis_date != analysis_date
            )

            if should_call_llm:
                # Check registry cache first (might have state from previous run or earlier in this run)
                # Pass specific date to look up cached state for that date
                registry_state = get_trend_state(self._symbol, self._cfg.timeframe, as_of_date=analysis_date)
                if registry_state:
                    registry_date = registry_state.as_of.date()
                    if registry_date == analysis_date:
                        # Found matching state in registry cache
                        # Only use cache if it's from LLM (not fallback)
                        if registry_state.from_llm:
                            result = registry_state.raw_result
                            result_from_llm = True
                            self._last_state = registry_state
                            self._last_llm_analysis_date = analysis_date
                            # Cache HIT - silently use cached result (no log to keep output clean)
                        else:
                            # Cached state is from fallback, need to call LLM
                            ctx.log(
                                f"[LLM_Trend_Detection] Cache MISS: Cached state for {self._symbol} on {analysis_date} "
                                f"is from fallback, calling LLM"
                            )
                            try:
                                # Log partial input data before LLM call
                                ma_short_val = payload['features']['ma']['ma_short']
                                ma_medium_val = payload['features']['ma']['ma_medium']
                                ma_long_val = payload['features']['ma']['ma_long']
                                rsi_val = payload['features']['rsi']
                                slope_val = payload['features']['regression_slope']
                                
                                # Format values safely (handle None)
                                ma_short_str = f"{ma_short_val:.2f}" if ma_short_val is not None else "N/A"
                                ma_medium_str = f"{ma_medium_val:.2f}" if ma_medium_val is not None else "N/A"
                                ma_long_str = f"{ma_long_val:.2f}" if ma_long_val is not None else "N/A"
                                rsi_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
                                slope_str = f"{slope_val:.4f}" if slope_val is not None else "N/A"
                                
                                ctx.log(
                                    f"[LLM_Trend_Detection] Calling LLM for {self._symbol} on {analysis_date} - "
                                    f"Input: {len(payload.get('bars', []))} bars, "
                                    f"MA(short={ma_short_str}, medium={ma_medium_str}, long={ma_long_str}), "
                                    f"RSI={rsi_str}, Slope={slope_str}"
                                )
                                result = await self._llm.classify_trend(payload)
                                result_from_llm = True
                                self._last_llm_analysis_date = analysis_date
                                
                                # Extract and log token usage if available
                                token_usage = result.pop('_token_usage', {})
                                if token_usage:
                                    ctx.log(
                                        f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date} - "
                                        f"Tokens: {token_usage.get('total_tokens', 0)} total "
                                        f"({token_usage.get('prompt_tokens', 0)} prompt + "
                                        f"{token_usage.get('completion_tokens', 0)} completion)"
                                    )
                                else:
                                    ctx.log(
                                        f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date}"
                                    )
                            except Exception as e:
                                ctx.log(
                                    f"[LLM_Trend_Detection] LLM failed ({e}), using fallback classifier."
                                )
                                result = self._fallback_classification(
                                    ma_short, ma_medium, ma_long, rsi, slope, last_close
                                )
                                result_from_llm = False
                    else:
                        # Registry has state but for different date
                        ctx.log(
                            f"[LLM_Trend_Detection] Cache MISS: Registry has {self._symbol} state for "
                            f"{registry_date}, but need {analysis_date} - calling LLM"
                        )
                        try:
                            # Log partial input data before LLM call
                            ma_short_val = payload['features']['ma']['ma_short']
                            ma_medium_val = payload['features']['ma']['ma_medium']
                            ma_long_val = payload['features']['ma']['ma_long']
                            rsi_val = payload['features']['rsi']
                            slope_val = payload['features']['regression_slope']
                            
                            # Format values safely (handle None)
                            ma_short_str = f"{ma_short_val:.2f}" if ma_short_val is not None else "N/A"
                            ma_medium_str = f"{ma_medium_val:.2f}" if ma_medium_val is not None else "N/A"
                            ma_long_str = f"{ma_long_val:.2f}" if ma_long_val is not None else "N/A"
                            rsi_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
                            slope_str = f"{slope_val:.4f}" if slope_val is not None else "N/A"
                            
                            ctx.log(
                                f"[LLM_Trend_Detection] Calling LLM for {self._symbol} on {analysis_date} - "
                                f"Input: {len(payload.get('bars', []))} bars, "
                                f"MA(short={ma_short_str}, medium={ma_medium_str}, long={ma_long_str}), "
                                f"RSI={rsi_str}, Slope={slope_str}"
                            )
                            result = await self._llm.classify_trend(payload)
                            result_from_llm = True
                            self._last_llm_analysis_date = analysis_date
                            
                            # Extract and log token usage if available
                            token_usage = result.pop('_token_usage', {})
                            if token_usage:
                                ctx.log(
                                    f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date} - "
                                    f"Tokens: {token_usage.get('total_tokens', 0)} total "
                                    f"({token_usage.get('prompt_tokens', 0)} prompt + "
                                    f"{token_usage.get('completion_tokens', 0)} completion)"
                                )
                            else:
                                ctx.log(
                                    f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date}"
                                )
                        except Exception as e:
                            ctx.log(
                                f"[LLM_Trend_Detection] LLM failed ({e}), using fallback classifier."
                            )
                            result = self._fallback_classification(
                                ma_short, ma_medium, ma_long, rsi, slope, last_close
                            )
                            result_from_llm = False
                else:
                    # No registry state at all
                    ctx.log(
                        f"[LLM_Trend_Detection] Cache MISS: No registry state for {self._symbol} "
                        f"({self._cfg.timeframe}) - calling LLM for {analysis_date}"
                    )
                    try:
                        # Log partial input data before LLM call
                        ma_short_val = payload['features']['ma']['ma_short']
                        ma_medium_val = payload['features']['ma']['ma_medium']
                        ma_long_val = payload['features']['ma']['ma_long']
                        rsi_val = payload['features']['rsi']
                        slope_val = payload['features']['regression_slope']
                        
                        # Format values safely (handle None)
                        ma_short_str = f"{ma_short_val:.2f}" if ma_short_val is not None else "N/A"
                        ma_medium_str = f"{ma_medium_val:.2f}" if ma_medium_val is not None else "N/A"
                        ma_long_str = f"{ma_long_val:.2f}" if ma_long_val is not None else "N/A"
                        rsi_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
                        slope_str = f"{slope_val:.4f}" if slope_val is not None else "N/A"
                        
                        ctx.log(
                            f"[LLM_Trend_Detection] Calling LLM for {self._symbol} on {analysis_date} - "
                            f"Input: {len(payload.get('bars', []))} bars, "
                            f"MA(short={ma_short_str}, medium={ma_medium_str}, long={ma_long_str}), "
                            f"RSI={rsi_str}, Slope={slope_str}"
                        )
                        result = await self._llm.classify_trend(payload)
                        result_from_llm = True
                        self._last_llm_analysis_date = analysis_date
                        
                        # Extract and log token usage if available
                        token_usage = result.pop('_token_usage', {})
                        if token_usage:
                            ctx.log(
                                f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date} - "
                                f"Tokens: {token_usage.get('total_tokens', 0)} total "
                                f"({token_usage.get('prompt_tokens', 0)} prompt + "
                                f"{token_usage.get('completion_tokens', 0)} completion)"
                            )
                        else:
                            ctx.log(
                                f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date}"
                            )
                    except Exception as e:
                        ctx.log(
                            f"[LLM_Trend_Detection] LLM failed ({e}), using fallback classifier."
                        )
                        result = self._fallback_classification(
                            ma_short, ma_medium, ma_long, rsi, slope, last_close
                        )
                        result_from_llm = False
            else:
                # Same calendar day as last call → reuse cached LLM result if available
                # Check both instance cache and registry cache
                cached_state = self._last_state
                if cached_state is None:
                    # Try registry as fallback (cross-session cache)
                    # Try to get state for the specific date
                    cached_state = get_trend_state(self._symbol, self._cfg.timeframe, as_of_date=analysis_date)
                    if cached_state:
                        self._last_state = cached_state
                        self._last_llm_analysis_date = cached_state.as_of.date()
                
                if cached_state is not None:
                    # Verify cached state is for the same date
                    cached_date = cached_state.as_of.date()
                    if cached_date == analysis_date:
                        # Only use cache if it's from LLM (not fallback)
                        if cached_state.from_llm:
                            result = cached_state.raw_result
                            result_from_llm = True
                            # Cache HIT - silently use cached result (no log to keep output clean)
                        else:
                            # Cached state is from fallback, need to call LLM
                            ctx.log(
                                f"[LLM_Trend_Detection] Cached state for {self._symbol} on {analysis_date} "
                                f"is from fallback, calling LLM"
                            )
                            try:
                                # Log partial input data before LLM call
                                ma_short_val = payload['features']['ma']['ma_short']
                                ma_medium_val = payload['features']['ma']['ma_medium']
                                ma_long_val = payload['features']['ma']['ma_long']
                                rsi_val = payload['features']['rsi']
                                slope_val = payload['features']['regression_slope']
                                
                                # Format values safely (handle None)
                                ma_short_str = f"{ma_short_val:.2f}" if ma_short_val is not None else "N/A"
                                ma_medium_str = f"{ma_medium_val:.2f}" if ma_medium_val is not None else "N/A"
                                ma_long_str = f"{ma_long_val:.2f}" if ma_long_val is not None else "N/A"
                                rsi_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
                                slope_str = f"{slope_val:.4f}" if slope_val is not None else "N/A"
                                
                                ctx.log(
                                    f"[LLM_Trend_Detection] Calling LLM for {self._symbol} on {analysis_date} - "
                                    f"Input: {len(payload.get('bars', []))} bars, "
                                    f"MA(short={ma_short_str}, medium={ma_medium_str}, long={ma_long_str}), "
                                    f"RSI={rsi_str}, Slope={slope_str}"
                                )
                                result = await self._llm.classify_trend(payload)
                                result_from_llm = True
                                self._last_llm_analysis_date = analysis_date
                                
                                # Extract and log token usage if available
                                token_usage = result.pop('_token_usage', {})
                                if token_usage:
                                    ctx.log(
                                        f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date} - "
                                        f"Tokens: {token_usage.get('total_tokens', 0)} total "
                                        f"({token_usage.get('prompt_tokens', 0)} prompt + "
                                        f"{token_usage.get('completion_tokens', 0)} completion)"
                                    )
                                else:
                                    ctx.log(
                                        f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date}"
                                    )
                            except Exception as e:
                                ctx.log(
                                    f"[LLM_Trend_Detection] LLM failed ({e}), using fallback classifier."
                                )
                                result = self._fallback_classification(
                                    ma_short, ma_medium, ma_long, rsi, slope, last_close
                                )
                                result_from_llm = False
                    else:
                        # Cached state is for a different date, need to call LLM
                        ctx.log(
                            f"[LLM_Trend_Detection] Cached state is for {cached_date}, "
                            f"but current date is {analysis_date}, calling LLM"
                        )
                        try:
                            # Log partial input data before LLM call
                            ma_short_val = payload['features']['ma']['ma_short']
                            ma_medium_val = payload['features']['ma']['ma_medium']
                            ma_long_val = payload['features']['ma']['ma_long']
                            rsi_val = payload['features']['rsi']
                            slope_val = payload['features']['regression_slope']
                            
                            # Format values safely (handle None)
                            ma_short_str = f"{ma_short_val:.2f}" if ma_short_val is not None else "N/A"
                            ma_medium_str = f"{ma_medium_val:.2f}" if ma_medium_val is not None else "N/A"
                            ma_long_str = f"{ma_long_val:.2f}" if ma_long_val is not None else "N/A"
                            rsi_str = f"{rsi_val:.2f}" if rsi_val is not None else "N/A"
                            slope_str = f"{slope_val:.4f}" if slope_val is not None else "N/A"
                            
                            ctx.log(
                                f"[LLM_Trend_Detection] Calling LLM for {self._symbol} on {analysis_date} - "
                                f"Input: {len(payload.get('bars', []))} bars, "
                                f"MA(short={ma_short_str}, medium={ma_medium_str}, long={ma_long_str}), "
                                f"RSI={rsi_str}, Slope={slope_str}"
                            )
                            result = await self._llm.classify_trend(payload)
                            result_from_llm = True
                            self._last_llm_analysis_date = analysis_date
                            
                            # Extract and log token usage if available
                            token_usage = result.pop('_token_usage', {})
                            if token_usage:
                                ctx.log(
                                    f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date} - "
                                    f"Tokens: {token_usage.get('total_tokens', 0)} total "
                                    f"({token_usage.get('prompt_tokens', 0)} prompt + "
                                    f"{token_usage.get('completion_tokens', 0)} completion)"
                                )
                            else:
                                ctx.log(
                                    f"[LLM_Trend_Detection] Called LLM for {self._symbol} on {analysis_date}"
                                )
                        except Exception as e:
                            ctx.log(
                                f"[LLM_Trend_Detection] LLM failed ({e}), using fallback classifier."
                            )
                            result = self._fallback_classification(
                                ma_short, ma_medium, ma_long, rsi, slope, last_close
                            )
                            result_from_llm = False
                else:
                    # No cached state yet – fallback
                    ctx.log(
                        f"[LLM_Trend_Detection] No cached state available for "
                        f"{self._symbol} on {analysis_date}, using fallback classifier"
                    )
                    result = self._fallback_classification(
                        ma_short, ma_medium, ma_long, rsi, slope, last_close
                    )
                    result_from_llm = False
        else:
            # LLM disabled entirely → always use deterministic fallback
            result = self._fallback_classification(
                ma_short, ma_medium, ma_long, rsi, slope, last_close
            )
            result_from_llm = False

        # 4) Normalize result into LLMTrendState
        regime = result.get("regime_final") or result.get("trend_short") or "RANGE"
        scores = result.get("scores", {}) or {}
        trend_strength = float(scores.get("trend_strength", 0.0) or 0.0)
        range_strength = float(scores.get("range_strength", 0.0) or 0.0)
        key_levels = result.get("key_levels", {}) or {}
        support = key_levels.get("support") or []
        resistance = key_levels.get("resistance") or []
        explanation = result.get("explanation") or ""
        summary_for_user = result.get("summary_for_user") or explanation

        state = LLMTrendState(
            symbol=self._symbol,
            timeframe=self._cfg.timeframe,
            as_of=last_ts,
            regime_final=regime,
            trend_short=result.get("trend_short", regime),
            trend_medium=result.get("trend_medium", regime),
            trend_long=result.get("trend_long", regime),
            trend_strength=trend_strength,
            range_strength=range_strength,
            support_levels=support,
            resistance_levels=resistance,
            explanation=explanation,
            summary_for_user=summary_for_user,
            raw_result=result,
            from_llm=result_from_llm,  # Mark if result came from LLM or fallback
        )
        self._last_state = state
        # Only cache LLM results, not fallback results
        if result_from_llm:
            await update_trend_state(state)
        else:
            ctx.log(
                f"[LLM_Trend_Detection] Not caching fallback result for {self._symbol} on {analysis_date}"
            )

        # 5) Logging & visualization state
        ctx.log(
            f"[LLM_Trend_Detection] {self._symbol} @ {last_close:.2f} "
            f"regime={regime}, trend_strength={trend_strength:.2f}, "
            f"range_strength={range_strength:.2f}"
        )
        if summary_for_user:
            ctx.log(f"[LLM_Trend_Detection] summary: {summary_for_user}")

        trend_score_numeric = 0
        if regime == "TREND_UP":
            trend_score_numeric = 1
        elif regime == "TREND_DOWN":
            trend_score_numeric = -1

        ind_row = IndicatorData(
            timestamp=last_ts,
            positive_count=int(trend_strength * 100),
            negative_count=int(range_strength * 100),
            trend_score=trend_score_numeric,
            di_plus=None,
            di_minus=None,
        )
        self._indicator_history.append(ind_row)
        
        # Store LLM indicator data with BB and RSI for charting
        llm_ind_row = LLMTrendIndicatorData(
            timestamp=last_ts,
            price=last_close,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            rsi=rsi,
            ma_short=ma_short,
            ma_medium=ma_medium,
            ma_long=ma_long,
            regime=regime,
            trend_strength=trend_strength,
            range_strength=range_strength,
        )
        self._llm_indicator_history.append(llm_ind_row)

        # Store regime decision for visualization
        self._regime_history.append({
            "timestamp": last_ts,
            "regime": regime,
            "price": last_close,
            "trend_strength": trend_strength,
            "range_strength": range_strength,
            "trend_short": result.get("trend_short", regime),
            "trend_medium": result.get("trend_medium", regime),
            "trend_long": result.get("trend_long", regime),
        })
        
        # Update previous regime after storing current decision
        # (We'll use this in _maybe_generate_orders to detect transitions)

        # Create signals for all regimes to mark decisions on chart
        # TREND_UP -> BUY, TREND_DOWN -> SELL, RANGE -> HOLD (special marker)
        signal_side: Optional[str] = None
        if regime == "TREND_UP":
            signal_side = "BUY"
        elif regime == "TREND_DOWN":
            signal_side = "SELL"
        elif regime == "RANGE":
            # Use "HOLD" as a special marker for RANGE regime
            signal_side = "HOLD"

        if signal_side:
            sig = TradeSignal(
                timestamp=last_ts,
                price=last_close,
                side=signal_side,
                trend_score=trend_score_numeric,
                di_plus=None,
                di_minus=None,
            )
            self._signals.append(sig)

        # 6) Optional trading hook
        if self._cfg.enable_trading:
            await self._maybe_generate_orders(ctx, last_close, regime)
        
        # Update previous regime after trading decision
        self._previous_regime = regime

    async def on_end(self, ctx: Context) -> None:
        regime = self._last_state.regime_final if self._last_state else "N/A"
        ctx.log(
            f"LLM_Trend_Detection finished for {self._symbol}. "
            f"Last regime={regime}"
        )

    # --- Fallback classifier ---

    def _fallback_classification(
        self,
        ma_short: Optional[float],
        ma_medium: Optional[float],
        ma_long: Optional[float],
        rsi: Optional[float],
        slope: Optional[float],
        last_close: float,
    ) -> Dict[str, Any]:
        regime = "RANGE"
        trend_strength = 0.0
        range_strength = 1.0

        if ma_short is None or ma_medium is None or slope is None or rsi is None:
            return {
                "trend_short": "RANGE",
                "trend_medium": "RANGE",
                "trend_long": "RANGE",
                "scores": {
                    "trend_strength": trend_strength,
                    "range_strength": range_strength,
                },
                "key_levels": {"support": [], "resistance": []},
                "regime_final": regime,
                "explanation": "Fallback RANGE: insufficient indicator data.",
                "summary_for_user": "LLM_Trend_Detection sees a sideways / unclear market (RANGE).",
            }

        slope_threshold = abs(last_close) * 0.0005
        is_up = ma_short > ma_medium and slope > slope_threshold and rsi > 55
        is_down = ma_short < ma_medium and slope < -slope_threshold and rsi < 45

        if is_up:
            regime = "TREND_UP"
            trend_strength = 0.8
            range_strength = 0.2
        elif is_down:
            regime = "TREND_DOWN"
            trend_strength = 0.8
            range_strength = 0.2
        else:
            regime = "RANGE"
            trend_strength = 0.3
            range_strength = 0.7

        explanation = (
            f"Fallback heuristic based on MA20 vs MA50, slope, and RSI suggests {regime}."
        )
        summary = f"LLM_Trend_Detection (fallback) classifies the market as {regime}."

        return {
            "trend_short": regime,
            "trend_medium": regime,
            "trend_long": regime,
            "scores": {
                "trend_strength": trend_strength,
                "range_strength": range_strength,
            },
            "key_levels": {"support": [], "resistance": []},
            "regime_final": regime,
            "explanation": explanation,
            "summary_for_user": summary,
        }

    # --- Optional trading mapping ---

    async def _maybe_generate_orders(self, ctx: Context, last_price: float, regime: str) -> None:
        """
        Trading logic based on regime transitions:
        - Trade when obvious trend emerges: RANGE -> TREND_UP or RANGE -> TREND_DOWN
        - Trade when trend reverses: TREND_UP -> TREND_DOWN or TREND_DOWN -> TREND_UP
        - Do nothing for: RANGE, or staying in same trend
        """
        positions = ctx.portfolio.get_positions()
        current_pos = next((p for p in positions if p.symbol == self._symbol), None)
        has_position = current_pos is not None and current_pos.quantity != 0
        
        # Detect specific regime transitions
        range_to_up = (
            self._previous_regime == "RANGE" 
            and regime == "TREND_UP"
        )
        range_to_down = (
            self._previous_regime == "RANGE" 
            and regime == "TREND_DOWN"
        )
        up_to_down = (
            self._previous_regime == "TREND_UP" 
            and regime == "TREND_DOWN"
        )
        down_to_up = (
            self._previous_regime == "TREND_DOWN" 
            and regime == "TREND_UP"
        )
        
        # Trading rules:
        # 1. Trade when trend emerges from RANGE -> TREND_UP: Go long
        if range_to_up:
            if not has_position:
                # Go long: deploy 100% of capital
                cash = ctx.portfolio.state.cash
                deploy = cash * self._cfg.capital_deployment_pct
                qty = int(deploy / last_price) if last_price > 0 else 0
                if qty <= 0:
                    ctx.log("[LLM_Trend_Detection] Not enough cash to buy.")
                    return
                order = Order(
                    id=f"llmtrend_buy_{datetime.utcnow().timestamp()}",
                    symbol=self._symbol,
                    side=Side.BUY,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    instrument_type=InstrumentType.STOCK,
                )
                await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
                ctx.log(
                    f"[LLM_Trend_Detection] BUY {qty} {self._symbol} @ ~{last_price:.2f} "
                    f"(RANGE -> TREND_UP: trend emerging, deploy={self._cfg.capital_deployment_pct*100:.0f}% cash)"
                )
            # If already long, do nothing (stay long)
        
        # 2. Trade when trend emerges from RANGE -> TREND_DOWN: Close longs (go flat)
        elif range_to_down:
            if has_position and current_pos.quantity > 0:
                # Close long position
                qty = current_pos.quantity
                order = Order(
                    id=f"llmtrend_sell_{datetime.utcnow().timestamp()}",
                    symbol=self._symbol,
                    side=Side.SELL,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    instrument_type=InstrumentType.STOCK,
                )
                await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
                ctx.log(
                    f"[LLM_Trend_Detection] SELL {qty} {self._symbol} @ ~{last_price:.2f} "
                    f"(RANGE -> TREND_DOWN: bear trend emerging, closing long position)"
                )
            # If already flat, do nothing
        
        # 3. Trade when trend reverses TREND_UP -> TREND_DOWN: Close longs (go flat)
        elif up_to_down:
            if has_position and current_pos.quantity > 0:
                # Close long position
                qty = current_pos.quantity
                order = Order(
                    id=f"llmtrend_sell_{datetime.utcnow().timestamp()}",
                    symbol=self._symbol,
                    side=Side.SELL,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    instrument_type=InstrumentType.STOCK,
                )
                await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
                ctx.log(
                    f"[LLM_Trend_Detection] SELL {qty} {self._symbol} @ ~{last_price:.2f} "
                    f"(TREND_UP -> TREND_DOWN: trend reversing, closing long position)"
                )
            # If already flat, do nothing
        
        # 4. Trade when trend reverses TREND_DOWN -> TREND_UP: Go long
        elif down_to_up:
            if not has_position:
                # Go long: deploy 100% of capital
                cash = ctx.portfolio.state.cash
                deploy = cash * self._cfg.capital_deployment_pct
                qty = int(deploy / last_price) if last_price > 0 else 0
                if qty <= 0:
                    ctx.log("[LLM_Trend_Detection] Not enough cash to buy.")
                    return
                order = Order(
                    id=f"llmtrend_buy_{datetime.utcnow().timestamp()}",
                    symbol=self._symbol,
                    side=Side.BUY,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    instrument_type=InstrumentType.STOCK,
                )
                await ctx.execution.submit_order(order)  # type: ignore[attr-defined]
                ctx.log(
                    f"[LLM_Trend_Detection] BUY {qty} {self._symbol} @ ~{last_price:.2f} "
                    f"(TREND_DOWN -> TREND_UP: trend reversing, deploy={self._cfg.capital_deployment_pct*100:.0f}% cash)"
                )
            # If already long, do nothing (stay long)
        
        # 5. All other cases: Do nothing (hold current position or stay flat)
        # - RANGE -> RANGE: Do nothing
        # - TREND_UP -> TREND_UP: Stay long
        # - TREND_DOWN -> TREND_DOWN: Stay flat
        # - Any -> RANGE: Hold current position
        else:
            if has_position:
                ctx.log(
                    f"[LLM_Trend_Detection] HOLD {current_pos.quantity} {self._symbol} "
                    f"(regime={regime}, no transition trade signal)"
                )
            # If flat, stay flat
