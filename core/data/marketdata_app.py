from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import List
import httpx
import asyncio
import logging

from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.option import OptionContract

logger = logging.getLogger(__name__)

class MarketDataAppAdapter(DataEngine):
    BASE_URL = "https://api.marketdata.app"

    def __init__(
        self,
        api_token: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize MarketDataAppAdapter.
        
        Args:
            api_token: API token for MarketData.app
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        # Validate API token
        if not api_token or not api_token.strip():
            error_msg = (
                f"❌ Authentication error: API token is missing or empty.\n"
                f"   Please check your config file (env.backtest.yaml or env.live.yaml) "
                f"and ensure 'marketdata.api_token' is set correctly."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._token = api_token.strip()
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)
        
        # Log token status (without exposing the full token)
        token_preview = f"{self._token[:10]}...{self._token[-5:]}" if len(self._token) > 15 else "***"
        logger.debug(f"MarketDataAppAdapter initialized with token: {token_preview} (length: {len(self._token)})")

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """
        Get bars from MarketData.app API.
        
        This method fetches data directly from the API without caching.
        For caching, wrap this adapter with CachedDataEngine.
        """
        return await self._fetch_from_api(symbol, start, end, timeframe)
    
    async def _fetch_from_api(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """
        Fetch bars from API (internal method, no caching).
        
        Supports all MarketData.app resolutions:
        - Minutely: 1, 3, 5, 15, 30, 45, ... (as numbers) or "1m", "15m", etc.
        - Hourly: H, 1H, 2H, ... or "1H", "2H", etc.
        - Daily: D, 1D, 2D, ... or "1D", "2D", etc.
        - Weekly: W, 1W, 2W, ... or "1W", "2W", etc.
        - Monthly: M, 1M, 2M, ... or "1M", "2M", etc.
        - Yearly: Y, 1Y, 2Y, ... or "1Y", "2Y", etc.
        """
        # Normalize timeframe to API format
        timeframe_upper = timeframe.upper()
        
        # Handle minutely resolutions (e.g., "15m", "30m" or just "15", "30")
        if timeframe.endswith("m") or timeframe.endswith("M"):
            # Remove "m" or "M" suffix, use number directly
            interval = timeframe[:-1]
        elif timeframe_upper in ("H", "D", "W", "M", "Y"):
            # Single letter resolutions
            interval = timeframe_upper
        elif any(timeframe_upper.endswith(suffix) for suffix in ["H", "D", "W", "M", "Y"]):
            # Multi-period resolutions (e.g., "1H", "2D", "1W", "1M", "1Y")
            interval = timeframe_upper
        elif timeframe.isdigit():
            # Pure numeric minutely resolution (e.g., "15", "30", "60")
            interval = timeframe
        else:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported formats: minutely (1, 3, 5, 15, 30, 45, ... or '15m'), "
                f"hourly (H, 1H, 2H, ...), daily (D, 1D, 2D, ...), "
                f"weekly (W, 1W, 2W, ...), monthly (M, 1M, 2M, ...), "
                f"yearly (Y, 1Y, 2Y, ...)"
            )

        url = f"{self.BASE_URL}/v1/stocks/candles/{interval}/{symbol}/"
        params = {
            "format": "json",
        }
        
        # For daily bars, check if we're requesting a single day
        # MarketData.app API supports a 'date' parameter for single-day requests
        # Reference: https://www.marketdata.app/docs/api
        if (timeframe_upper.endswith("D") or timeframe_upper == "D") and start.date() == end.date():
            # Single day request - use 'date' parameter instead of 'from'/'to'
            # ISO 8601 date format: YYYY-MM-DD
            params["date"] = start.date().isoformat()
            logger.debug(f"Using 'date' parameter for single-day request: {params['date']}")
        elif start == end:
            # For non-daily bars with same start/end, expand end slightly to create valid range
            # Determine appropriate expansion based on timeframe
            if timeframe.endswith("m") or timeframe.endswith("M") or timeframe.isdigit():
                # Minutely: expand by 1 hour
                effective_end = end + timedelta(hours=1)
            elif timeframe_upper.endswith("H"):
                # Hourly: expand by 1 day
                effective_end = end + timedelta(days=1)
            elif timeframe_upper.endswith("W"):
                # Weekly: expand by 1 week
                effective_end = end + timedelta(weeks=1)
            elif timeframe_upper.endswith("M"):
                # Monthly: expand by 1 month (approximate)
                effective_end = end + timedelta(days=31)
            elif timeframe_upper.endswith("Y"):
                # Yearly: expand by 1 year
                effective_end = end + timedelta(days=365)
            else:
                # Default: 1 day
                effective_end = end + timedelta(days=1)
            params["from"] = start.isoformat()
            params["to"] = effective_end.isoformat()
        else:
            # Multi-day or multi-period request - use 'from' and 'to' parameters
            params["from"] = start.isoformat()
            params["to"] = end.isoformat()
        
        # Validate API token
        if not self._token or not self._token.strip():
            error_msg = (
                f"❌ Authentication error: API token is missing or empty.\n"
                f"   URL: {url}\n"
                f"   Please check your config file (env.backtest.yaml or env.live.yaml) "
                f"and ensure 'marketdata.api_token' is set correctly."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        headers = {"Authorization": f"Bearer {self._token}"}
        
        # Retry logic with exponential backoff for timeouts
        data = None
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.get(url, params=params, headers=headers)
                
                # Bail out immediately on 404 Not Found
                if resp.status_code == 404:
                    # Try to get error message from response
                    error_details = ""
                    try:
                        error_data = resp.json()
                        if isinstance(error_data, dict):
                            errmsg = error_data.get("errmsg", error_data.get("message", ""))
                            if errmsg:
                                error_details = f"Error message: {errmsg}\n   "
                    except Exception:
                        # If we can't parse JSON, try to get text
                        try:
                            error_text = resp.text[:200] if resp.text else ""
                            if error_text:
                                error_details = f"Response: {error_text}\n   "
                        except Exception:
                            pass
                    
                    token_preview = f"'{self._token[:10]}...'" if self._token and len(self._token) > 10 else "N/A"
                    error_msg = (
                        f"❌ HTTP 404 Not Found - Bailing out\n"
                        f"   URL: {url}\n"
                        f"   Params: {params}\n"
                        f"   Status: 404 Not Found\n"
                        f"   {error_details}"
                        f"Token present: {'Yes' if self._token else 'No'}, "
                        f"Token length: {len(self._token) if self._token else 0}, "
                        f"Token preview: {token_preview}\n"
                        f"   This could indicate:\n"
                        f"   - Invalid API token (check env.backtest.yaml or env.live.yaml)\n"
                        f"   - Invalid symbol or timeframe\n"
                        f"   - Date range outside available data\n"
                        f"   - API endpoint issue"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                resp.raise_for_status()
                
                # Validate response is JSON
                try:
                    data = resp.json()
                except Exception as e:
                    error_msg = (
                        f"❌ Invalid HTTP response: Failed to parse JSON response.\n"
                        f"   URL: {url}\n"
                        f"   Status: {resp.status_code}\n"
                        f"   Response text (first 500 chars): {resp.text[:500]}\n"
                        f"   Error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                
                # Validate response structure
                if not isinstance(data, dict):
                    error_msg = (
                        f"❌ Invalid HTTP response: Expected dict, got {type(data).__name__}.\n"
                        f"   URL: {url}\n"
                        f"   Status: {resp.status_code}\n"
                        f"   Response: {str(data)[:500]}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Check API status
                if data.get("s") != "ok":
                    error_msg = (
                        f"❌ MarketData API error: Status is not 'ok'.\n"
                        f"   URL: {url}\n"
                        f"   Status code: {resp.status_code}\n"
                        f"   API status: {data.get('s', 'unknown')}\n"
                        f"   Full response: {data}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Validate required fields exist
                required_fields = ["o", "h", "l", "c", "t"]
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    error_msg = (
                        f"❌ Invalid HTTP response: Missing required fields: {missing_fields}.\n"
                        f"   URL: {url}\n"
                        f"   Status: {resp.status_code}\n"
                        f"   Available fields: {list(data.keys())}\n"
                        f"   Response: {str(data)[:500]}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Validate field types and lengths
                opens = data.get("o", [])
                highs = data.get("h", [])
                lows = data.get("l", [])
                closes = data.get("c", [])
                times = data.get("t", [])
                
                # Log response summary
                num_bars = len(times) if times else 0
                if num_bars > 0:
                    # Convert first and last timestamps to readable format
                    try:
                        first_ts = datetime.fromtimestamp(times[0]).strftime('%Y-%m-%d %H:%M:%S') if times else 'N/A'
                        last_ts = datetime.fromtimestamp(times[-1]).strftime('%Y-%m-%d %H:%M:%S') if times else 'N/A'
                        # Get price range
                        if opens and closes:
                            min_price = min(min(opens), min(lows)) if lows else min(opens)
                            max_price = max(max(highs), max(closes)) if highs else max(closes)
                            price_range = f"${min_price:.2f}-${max_price:.2f}"
                        else:
                            price_range = "N/A"
                        
                        logger.info(
                            f"[MarketData] HTTP Response: {resp.status_code} OK | "
                            f"Bars: {num_bars} | "
                            f"Range: {first_ts} to {last_ts} | "
                            f"Price: {price_range} | "
                            f"Fields: {list(data.keys())}"
                        )
                    except Exception as e:
                        # If timestamp conversion fails, just log basic info
                        logger.info(
                            f"[MarketData] HTTP Response: {resp.status_code} OK | "
                            f"Bars: {num_bars} | "
                            f"Fields: {list(data.keys())} | "
                            f"Timestamp conversion error: {e}"
                        )
                else:
                    logger.info(
                        f"[MarketData] HTTP Response: {resp.status_code} OK | "
                        f"Bars: 0 (empty response) | "
                        f"Fields: {list(data.keys())}"
                    )
                
                if not all(isinstance(arr, list) for arr in [opens, highs, lows, closes, times]):
                    error_msg = (
                        f"❌ Invalid HTTP response: Expected arrays for OHLCV data.\n"
                        f"   URL: {url}\n"
                        f"   Types - o: {type(opens).__name__}, h: {type(highs).__name__}, "
                        f"l: {type(lows).__name__}, c: {type(closes).__name__}, t: {type(times).__name__}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Validate array lengths match
                lengths = [len(opens), len(highs), len(lows), len(closes), len(times)]
                if len(set(lengths)) > 1:
                    error_msg = (
                        f"❌ Invalid HTTP response: Array length mismatch.\n"
                        f"   URL: {url}\n"
                        f"   Lengths - o: {len(opens)}, h: {len(highs)}, l: {len(lows)}, "
                        f"c: {len(closes)}, t: {len(times)}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                break  # Success, exit retry loop
            except (httpx.ReadTimeout, httpx.TimeoutException, httpx.ConnectTimeout) as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s...
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, raise error instead of returning empty list
                    error_msg = (
                        f"❌ HTTP timeout: Request timed out after {self._max_retries} attempts.\n"
                        f"   URL: {url}\n"
                        f"   Timeout: {self._timeout}s\n"
                        f"   Last exception: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            except httpx.HTTPStatusError as e:
                # Bail out immediately on 404 Not Found
                if e.response.status_code == 404:
                    # Try to get error message from response
                    error_details = ""
                    try:
                        error_data = e.response.json()
                        if isinstance(error_data, dict):
                            errmsg = error_data.get("errmsg", error_data.get("message", ""))
                            if errmsg:
                                error_details = f"Error message: {errmsg}\n   "
                    except Exception:
                        # If we can't parse JSON, try to get text
                        try:
                            error_text = e.response.text[:200] if e.response.text else ""
                            if error_text:
                                error_details = f"Response: {error_text}\n   "
                        except Exception:
                            pass
                    
                    token_preview = f"'{self._token[:10]}...'" if self._token and len(self._token) > 10 else "N/A"
                    error_msg = (
                        f"❌ HTTP 404 Not Found - Bailing out\n"
                        f"   URL: {url}\n"
                        f"   Params: {params}\n"
                        f"   Status: 404 Not Found\n"
                        f"   {error_details}"
                        f"Token present: {'Yes' if self._token else 'No'}, "
                        f"Token length: {len(self._token) if self._token else 0}, "
                        f"Token preview: {token_preview}\n"
                        f"   This could indicate:\n"
                        f"   - Invalid API token (check env.backtest.yaml or env.live.yaml)\n"
                        f"   - Invalid symbol or timeframe\n"
                        f"   - Date range outside available data\n"
                        f"   - API endpoint issue"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                # For other HTTP errors, print detailed error and fail
                error_msg = (
                    f"❌ HTTP error: {e.response.status_code} {e.response.reason_phrase}.\n"
                    f"   URL: {url}\n"
                    f"   Request params: {params}\n"
                    f"   Response text (first 500 chars): {e.response.text[:500] if hasattr(e.response, 'text') else 'N/A'}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        # Safety check - if we somehow got here without data, fail with error
        if data is None:
            error_msg = (
                f"❌ Invalid HTTP response: No data received after {self._max_retries} attempts.\n"
                f"   URL: {url}\n"
                f"   Last exception: {last_exception}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        bars: List[Bar] = []
        opens = data["o"]
        highs = data["h"]
        lows = data["l"]
        closes = data["c"]
        times = data["t"]
        vols = data.get("v", [0] * len(times))

        for i, (o, h, l, c, t, v) in enumerate(zip(opens, highs, lows, closes, times, vols)):
            try:
                # Validate data types
                if not all(isinstance(x, (int, float)) for x in [o, h, l, c]):
                    error_msg = (
                        f"❌ Invalid HTTP response: Invalid data type in OHLC data at index {i}.\n"
                        f"   URL: {url}\n"
                        f"   Types - o: {type(o).__name__}, h: {type(h).__name__}, "
                        f"l: {type(l).__name__}, c: {type(c).__name__}\n"
                        f"   Values - o: {o}, h: {h}, l: {l}, c: {c}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                if not isinstance(t, (int, float)):
                    error_msg = (
                        f"❌ Invalid HTTP response: Invalid timestamp type at index {i}.\n"
                        f"   URL: {url}\n"
                        f"   Type: {type(t).__name__}, Value: {t}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # MarketData.app returns Unix timestamps (UTC-based)
                # Normalize to timezone-naive ET for consistency
                # For complete historical bars (daily, weekly, monthly), normalize to 00:00:00 ET
                from core.utils.timestamp import normalize_unix_timestamp
                ts = normalize_unix_timestamp(t, timeframe=timeframe)
                if ts < start or ts > end:
                    continue
                bars.append(
                    Bar(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v) if v is not None else 0.0,
                        timeframe=timeframe,
                    )
                )
            except (ValueError, TypeError, OSError) as e:
                error_msg = (
                    f"❌ Invalid HTTP response: Failed to parse bar data at index {i}.\n"
                    f"   URL: {url}\n"
                    f"   Values - o: {o}, h: {h}, l: {l}, c: {c}, t: {t}, v: {v}\n"
                    f"   Error: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        return bars

    async def stream_bars(self, symbol: str, timeframe: str):
        raise NotImplementedError("MarketDataAppAdapter.stream_bars is not implemented")

    async def get_option_chain(
        self,
        underlying: str,
        as_of: date,
    ) -> list[OptionContract]:
        url = f"{self.BASE_URL}/v1/options/chain/{underlying}/"
        params = {
            "date": as_of.isoformat(),
            "format": "json",
        }
        headers = {"Authorization": f"Bearer {self._token}"}
        
        # Retry logic with exponential backoff for timeouts
        data = None
        last_exception = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.get(url, params=params, headers=headers)
                
                # Bail out immediately on 404 Not Found
                if resp.status_code == 404:
                    # Try to get error message from response
                    error_details = ""
                    try:
                        error_data = resp.json()
                        if isinstance(error_data, dict):
                            errmsg = error_data.get("errmsg", error_data.get("message", ""))
                            if errmsg:
                                error_details = f"Error message: {errmsg}\n   "
                    except Exception:
                        # If we can't parse JSON, try to get text
                        try:
                            error_text = resp.text[:200] if resp.text else ""
                            if error_text:
                                error_details = f"Response: {error_text}\n   "
                        except Exception:
                            pass
                    
                    token_preview = f"'{self._token[:10]}...'" if self._token and len(self._token) > 10 else "N/A"
                    error_msg = (
                        f"❌ HTTP 404 Not Found - Bailing out (option chain)\n"
                        f"   URL: {url}\n"
                        f"   Params: {params}\n"
                        f"   Status: 404 Not Found\n"
                        f"   {error_details}"
                        f"Token present: {'Yes' if self._token else 'No'}, "
                        f"Token length: {len(self._token) if self._token else 0}, "
                        f"Token preview: {token_preview}\n"
                        f"   This could indicate:\n"
                        f"   - Invalid API token (check env.backtest.yaml or env.live.yaml)\n"
                        f"   - Invalid underlying symbol\n"
                        f"   - Date outside available data\n"
                        f"   - API endpoint issue"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                resp.raise_for_status()
                
                # Validate response is JSON
                try:
                    data = resp.json()
                except Exception as e:
                    error_msg = (
                        f"❌ Invalid HTTP response: Failed to parse JSON response for option chain.\n"
                        f"   URL: {url}\n"
                        f"   Status: {resp.status_code}\n"
                        f"   Response text (first 500 chars): {resp.text[:500]}\n"
                        f"   Error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                
                # Check status if present
                if isinstance(data, dict) and "s" in data and data.get("s") != "ok":
                    error_msg = (
                        f"❌ MarketData API error: Option chain status is not 'ok'.\n"
                        f"   URL: {url}\n"
                        f"   Status code: {resp.status_code}\n"
                        f"   API status: {data.get('s', 'unknown')}\n"
                        f"   Full response: {data}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                break  # Success, exit retry loop
            except (httpx.ReadTimeout, httpx.TimeoutException, httpx.ConnectTimeout) as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s...
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed, fail with error
                    error_msg = (
                        f"❌ HTTP timeout: Request timed out after {self._max_retries} attempts.\n"
                        f"   URL: {url}\n"
                        f"   Timeout: {self._timeout}s\n"
                        f"   Last exception: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
            except httpx.HTTPStatusError as e:
                # Bail out immediately on 404 Not Found
                if e.response.status_code == 404:
                    # Try to get error message from response
                    error_details = ""
                    try:
                        error_data = e.response.json()
                        if isinstance(error_data, dict):
                            errmsg = error_data.get("errmsg", error_data.get("message", ""))
                            if errmsg:
                                error_details = f"Error message: {errmsg}\n   "
                    except Exception:
                        # If we can't parse JSON, try to get text
                        try:
                            error_text = e.response.text[:200] if e.response.text else ""
                            if error_text:
                                error_details = f"Response: {error_text}\n   "
                        except Exception:
                            pass
                    
                    token_preview = f"'{self._token[:10]}...'" if self._token and len(self._token) > 10 else "N/A"
                    error_msg = (
                        f"❌ HTTP 404 Not Found - Bailing out (option chain)\n"
                        f"   URL: {url}\n"
                        f"   Params: {params}\n"
                        f"   Status: 404 Not Found\n"
                        f"   {error_details}"
                        f"Token present: {'Yes' if self._token else 'No'}, "
                        f"Token length: {len(self._token) if self._token else 0}, "
                        f"Token preview: {token_preview}\n"
                        f"   This could indicate:\n"
                        f"   - Invalid API token (check env.backtest.yaml or env.live.yaml)\n"
                        f"   - Invalid underlying symbol\n"
                        f"   - Date outside available data\n"
                        f"   - API endpoint issue"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # For other HTTP errors, print detailed error and fail
                error_msg = (
                    f"❌ HTTP error: {e.response.status_code} {e.response.reason_phrase}.\n"
                    f"   URL: {url}\n"
                    f"   Request params: {params}\n"
                    f"   Response text (first 500 chars): {e.response.text[:500] if hasattr(e.response, 'text') else 'N/A'}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        # Safety check - if we somehow got here without data, fail with error
        if data is None:
            error_msg = (
                f"❌ Invalid HTTP response: No data received for option chain after {self._max_retries} attempts.\n"
                f"   URL: {url}\n"
                f"   Last exception: {last_exception}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        contracts: list[OptionContract] = []
        
        # API returns columnar format: each field is an array
        # We need to transpose it into rows
        if isinstance(data, dict) and "optionSymbol" in data:
            # Columnar format - transpose arrays into rows
            option_symbols = data.get("optionSymbol", [])
            expirations = data.get("expiration", [])
            sides = data.get("side", [])  # "C" or "P"
            strikes = data.get("strike", [])
            
            # Ensure all arrays have the same length
            num_contracts = len(option_symbols)
            if not all(len(arr) == num_contracts for arr in [expirations, sides, strikes]):
                raise RuntimeError(
                    f"MarketData option chain: array length mismatch. "
                    f"optionSymbol: {len(option_symbols)}, expiration: {len(expirations)}, "
                    f"side: {len(sides)}, strike: {len(strikes)}"
                )
            
            # Parse expiration dates - could be in various formats
            for i in range(num_contracts):
                try:
                    # Try parsing expiration - could be ISO format string or timestamp
                    exp_str = expirations[i]
                    if isinstance(exp_str, (int, float)):
                        # If it's a timestamp, convert to date
                        exp_date = date.fromtimestamp(exp_str)
                    elif isinstance(exp_str, str):
                        # Try ISO format first
                        try:
                            exp_date = date.fromisoformat(exp_str.split("T")[0])  # Handle datetime strings
                        except ValueError:
                            # Try other common formats
                            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    else:
                        raise ValueError(f"Unexpected expiration format: {type(exp_str)}")
                    
                    contracts.append(
                        OptionContract(
                            symbol=option_symbols[i],
                            underlying=underlying,
                            expiry=exp_date,
                            strike=float(strikes[i]),
                            right=sides[i].upper() if isinstance(sides[i], str) else ("C" if sides[i] == 1 else "P"),
                        )
                    )
                except Exception as e:
                    # Skip invalid contracts but log the error
                    continue
        elif isinstance(data, list):
            # Row-based format (array of objects)
            for row in data:
                contracts.append(
                    OptionContract(
                        symbol=row["optionSymbol"],
                        underlying=underlying,
                        expiry=date.fromisoformat(row.get("expirationDate", row.get("expiration", ""))),
                        strike=float(row["strike"]),
                        right=row.get("type", row.get("side", ""))[0].upper(),
                    )
                )
        else:
            raise RuntimeError(
                f"MarketData option chain: unexpected response structure. "
                f"Type: {type(data)}, Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )
        
        return contracts
