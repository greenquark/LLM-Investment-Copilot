from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import List, AsyncIterator
import asyncio
import logging

from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.option import OptionContract

logger = logging.getLogger(__name__)

# Lazy import for yfinance (optional dependency)
try:
    import yfinance as yf
    import pandas as pd
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    yf = None  # type: ignore
    pd = None  # type: ignore


class YFinanceDataAdapter(DataEngine):
    """
    Yahoo Finance data adapter using yfinance library.
    
    This adapter provides free historical stock data from Yahoo Finance.
    Note: yfinance is a community-maintained library and may have rate limits.
    
    Supported timeframes:
    - Minutely: "1m", "2m", "5m", "15m", "30m", "60m", "90m"
    - Hourly: "1h"
    - Daily: "1d", "5d"
    - Weekly: "1wk"
    - Monthly: "1mo", "3mo"
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        Initialize YFinanceDataAdapter.
        
        Args:
            timeout: Request timeout in seconds (not used by yfinance, but kept for API consistency)
        """
        if not _YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is not installed. Install it with: pip install yfinance"
            )
        if pd is None:
            raise ImportError(
                "pandas is required for YFinanceDataAdapter. Install it with: pip install pandas"
            )
        self._timeout = timeout
        self._last_data_source = "YFinance"  # Track data source name for last request
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """
        Normalize timeframe to yfinance interval format.
        
        Args:
            timeframe: Timeframe string (e.g., "15m", "1H", "1D", "D")
        
        Returns:
            Normalized interval string for yfinance
        """
        timeframe_upper = timeframe.upper()
        
        # Minutely intervals
        if timeframe.endswith("m") or timeframe.endswith("M"):
            num = timeframe[:-1]
            if num.isdigit():
                interval = f"{num}m"
                # Validate yfinance supports this interval
                valid_minute_intervals = ["1", "2", "5", "15", "30", "60", "90"]
                if num not in valid_minute_intervals:
                    raise ValueError(
                        f"Unsupported minutely interval '{timeframe}'. "
                        f"Supported: {', '.join([f'{i}m' for i in valid_minute_intervals])}"
                    )
                return interval
            else:
                raise ValueError(f"Invalid minutely timeframe format: {timeframe}")
        
        # Hourly intervals
        elif timeframe_upper.endswith("H"):
            num = timeframe_upper[:-1] if timeframe_upper != "H" else "1"
            if num == "1":
                return "1h"
            else:
                raise ValueError(
                    f"Unsupported hourly interval '{timeframe}'. "
                    f"YFinance only supports '1h' or '1H'"
                )
        
        # Daily intervals
        elif timeframe_upper.endswith("D"):
            num = timeframe_upper[:-1] if timeframe_upper != "D" else "1"
            if num == "1":
                return "1d"
            elif num == "5":
                return "5d"
            else:
                raise ValueError(
                    f"Unsupported daily interval '{timeframe}'. "
                    f"YFinance supports: '1d', '5d', 'D', '1D', '5D'"
                )
        
        # Weekly intervals - check for "wk" or "WK" first
        elif timeframe_upper.endswith("WK"):
            num = timeframe_upper[:-2] if len(timeframe_upper) > 2 else "1"
            if num == "1" or num == "":
                return "1wk"
            else:
                raise ValueError(
                    f"Unsupported weekly interval '{timeframe}'. "
                    f"YFinance only supports '1wk' or '1W' or 'W'"
                )
        elif timeframe_upper.endswith("W") and not timeframe_upper.endswith("WK"):
            # Handle "1W", "W" formats (but not "1wk" which is handled above)
            num = timeframe_upper[:-1] if timeframe_upper != "W" else "1"
            if num == "1" or num == "":
                return "1wk"
            else:
                raise ValueError(
                    f"Unsupported weekly interval '{timeframe}'. "
                    f"YFinance only supports '1wk' or '1W' or 'W'"
                )
        
        # Monthly intervals - check for "mo" or "MO" first (before checking "M")
        elif timeframe_upper.endswith("MO"):
            num = timeframe_upper[:-2] if len(timeframe_upper) > 2 else "1"
            if num == "1" or num == "":
                return "1mo"
            elif num == "3":
                return "3mo"
            else:
                raise ValueError(
                    f"Unsupported monthly interval '{timeframe}'. "
                    f"YFinance supports: '1mo', '3mo', '1M', '3M', 'M'"
                )
        elif timeframe_upper.endswith("M") and not timeframe_upper.endswith("MO"):
            # Handle "1M", "3M", "M" formats (but not "1mo" which is handled above)
            num = timeframe_upper[:-1] if timeframe_upper != "M" else "1"
            if num == "1" or num == "":
                return "1mo"
            elif num == "3":
                return "3mo"
            else:
                raise ValueError(
                    f"Unsupported monthly interval '{timeframe}'. "
                    f"YFinance supports: '1mo', '3mo', '1M', '3M', 'M'"
                )
        
        # Pure numeric (assume minutes)
        elif timeframe.isdigit():
            if timeframe in ["1", "2", "5", "15", "30", "60", "90"]:
                return f"{timeframe}m"
            else:
                raise ValueError(
                    f"Unsupported numeric interval '{timeframe}'. "
                    f"Supported: 1, 2, 5, 15, 30, 60, 90 (minutes)"
                )
        
        else:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported formats: "
                f"minutely (1m, 2m, 5m, 15m, 30m, 60m, 90m), "
                f"hourly (1h, 1H), "
                f"daily (1d, 5d, 1D, 5D, D), "
                f"weekly (1wk, 1W, W), "
                f"monthly (1mo, 3mo, 1M, 3M, M)"
            )
    
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """
        Get historical bars from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string (e.g., "15m", "1H", "1D")
        
        Returns:
            List of Bar objects
        """
        # Normalize timeframe to yfinance format
        interval = self._normalize_timeframe(timeframe)
        
        # Convert datetime to date strings for yfinance
        start_date = start.date() if isinstance(start, datetime) else start
        end_date = end.date() if isinstance(end, datetime) else end
        
        # yfinance requires end_date to be after start_date (exclusive end)
        # If start and end are the same date, add 1 day to end_date
        if start_date == end_date:
            end_date = end_date + timedelta(days=1)
            logger.debug(f"Adjusted end_date from {end.date() if isinstance(end, datetime) else end} to {end_date} for yfinance (same-day request)")
        
        # Run yfinance download in executor (it's synchronous)
        # Note: For 1m data, yfinance only allows 8 days per request
        # Adjust date range if needed for 1m interval
        if interval == "1m":
            max_days = 8
            if (end_date - start_date).days > max_days:
                logger.warning(
                    f"1m data limited to {max_days} days. Adjusting start date from {start_date} to {end_date - timedelta(days=max_days)}"
                )
                start_date = end_date - timedelta(days=max_days)
        
        loop = asyncio.get_event_loop()
        try:
            # yfinance download can use start/end dates or period
            # Using start/end gives us more control over the date range
            # Note: yfinance accepts same date for start and end (e.g., start=day, end=day)
            # When there's no data, yfinance prints a warning to stderr but doesn't raise
            # an exception - it just returns an empty DataFrame, which we handle below
            df = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    tickers=symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,  # Explicitly set to avoid FutureWarning
                )
            )
        except Exception as e:
            # Check if this is a rate limit or API error that should be raised
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests', 'limit exceeded', 'quota', '403', '401', 'payment required', '402']):
                logger.error(f"YFinance API error (rate limit or authentication issue) for {symbol}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                raise RuntimeError(f"YFinance API error for {symbol}: {e}") from e
            else:
                # For other errors (network issues, etc.), log and raise
                logger.error(f"YFinance download error for {symbol}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                raise RuntimeError(f"YFinance download error for {symbol}: {e}") from e
        
        # Check if we got data
        if df is None or df.empty:
            # Check if the date is in the future (beyond today)
            from datetime import date as date_type
            today = date_type.today()
            start_date_only = start_date.date() if isinstance(start_date, datetime) else start_date
            end_date_only = end_date.date() if isinstance(end_date, datetime) else end_date
            
            if end_date_only > today:
                logger.info(
                    f"No data returned from YFinance for {symbol} from {start_date} to {end_date} (interval={interval}). "
                    f"Date is in the future (today is {today}), so data is not yet available."
                )
            else:
                logger.warning(
                    f"No data returned from YFinance for {symbol} from {start_date} to {end_date} (interval={interval}). "
                    f"This may indicate a non-trading day, holiday, or data availability issue."
                )
            return []
        
        # Debug: log DataFrame info
        logger.debug(f"YFinance returned DataFrame: shape={df.shape}, columns={list(df.columns)}, index type={type(df.index)}")
        if not df.empty:
            logger.debug(f"DataFrame index range: {df.index.min()} to {df.index.max()}")
        
        # Handle MultiIndex columns from yfinance (like the reference implementation)
        if isinstance(df.columns, pd.MultiIndex):
            levels = []
            for level_idx in range(df.columns.nlevels):
                level_values = df.columns.get_level_values(level_idx)
                levels.append(set(str(v) for v in level_values))
            
            target_level = None
            for level_idx, values in enumerate(levels):
                if {"Open", "High", "Low", "Close"}.issubset(values):
                    target_level = level_idx
                    break
            
            if target_level is not None:
                df.columns = df.columns.get_level_values(target_level)
            else:
                df.columns = ["_".join(str(x) for x in col).strip() for col in df.columns.values]
        
        # Normalize column names to title case
        df = df.rename(columns=str.title)
        
        # Check for required columns
        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns after normalization: {missing}. Available: {list(df.columns)}")
            return []
        
        if "Volume" not in df.columns:
            df["Volume"] = 0
        
        # Convert DataFrame to Bar objects
        bars: List[Bar] = []
        
        # yfinance returns DataFrame with columns: Open, High, Low, Close, Volume
        # Index is the timestamp (DatetimeIndex)
        # Normalize start/end to timezone-naive for comparison
        # Use date objects for comparison to be more lenient
        start_naive = start.replace(tzinfo=None) if start.tzinfo else start
        end_naive = end.replace(tzinfo=None) if end.tzinfo else end
        
        for idx, row in df.iterrows():
            # Handle different index types (DatetimeIndex, Timestamp, etc.)
            if hasattr(idx, 'to_pydatetime'):
                ts = idx.to_pydatetime()
            elif isinstance(idx, datetime):
                ts = idx
            else:
                # Try to convert string or other formats
                try:
                    ts = pd.to_datetime(idx).to_pydatetime()
                except:
                    logger.warning(f"Could not parse timestamp: {idx}")
                    continue
            
            # Normalize timestamp to timezone-naive for comparison
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            
            # Filter by datetime range (yfinance may return slightly different range)
            # For daily/weekly/monthly intervals, compare dates only (more lenient)
            # For intraday intervals, use full datetime comparison
            if interval in ["1d", "5d", "1wk", "1mo", "3mo"]:
                # For daily/weekly/monthly, compare dates only
                ts_date = ts_naive.date() if isinstance(ts_naive, datetime) else ts_naive
                start_date_only = start_naive.date() if isinstance(start_naive, datetime) else start_naive
                end_date_only = end_naive.date() if isinstance(end_naive, datetime) else end_naive
                if ts_date < start_date_only or ts_date > end_date_only:
                    continue
            else:
                # For intraday, use full datetime comparison but be lenient (allow 1 day buffer)
                buffer = timedelta(days=1)
                if ts_naive < (start_naive - buffer) or ts_naive > (end_naive + buffer):
                    continue
            
            # Extract OHLCV data - columns should be normalized to title case now
            # Use direct column access (pandas Series supports both [] and .get())
            try:
                open_price = row["Open"] if "Open" in row.index else (row.get("Open", 0) if hasattr(row, 'get') else 0)
                high_price = row["High"] if "High" in row.index else (row.get("High", 0) if hasattr(row, 'get') else 0)
                low_price = row["Low"] if "Low" in row.index else (row.get("Low", 0) if hasattr(row, 'get') else 0)
                close_price = row["Close"] if "Close" in row.index else (row.get("Close", 0) if hasattr(row, 'get') else 0)
                volume = row["Volume"] if "Volume" in row.index else (row.get("Volume", 0) if hasattr(row, 'get') else 0)
            except (KeyError, AttributeError, IndexError) as e:
                logger.warning(f"Error extracting OHLCV data for row {idx}: {e}, available columns: {list(df.columns)}")
                continue
            
            # Skip rows with invalid data
            if not all(isinstance(x, (int, float)) and not pd.isna(x) for x in [open_price, high_price, low_price, close_price]):
                continue
            
            # Normalize timestamp to timezone-naive UTC for consistency
            # yfinance returns timestamps in America/New_York timezone
            # For daily/weekly bars, normalize to market close (4 PM ET) for consistency
            from core.utils.timestamp import normalize_timestamp
            ts_normalized = normalize_timestamp(ts, timeframe=timeframe)
            
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=ts_normalized,
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume) if not pd.isna(volume) else 0.0,
                    timeframe=timeframe,
                )
            )
        
        # Sort by timestamp to ensure chronological order
        bars.sort(key=lambda b: b.timestamp)
        
        if len(bars) == 0:
            logger.warning(
                f"YFinance: No bars extracted for {symbol} ({timeframe}) from {start_date} to {end_date}. "
                f"DataFrame had {len(df)} rows. Check date filtering logic."
            )
        else:
            logger.debug(f"YFinance: Retrieved {len(bars)} bars for {symbol} ({timeframe}) from {start_date} to {end_date}")
        
        # Update last data source
        self._last_data_source = "YFinance"
        
        return bars
    
    async def stream_bars(self, symbol: str, timeframe: str) -> AsyncIterator[Bar]:
        """
        Stream real-time bars (not supported by yfinance).
        
        YFinance only provides historical data, not real-time streaming.
        """
        raise NotImplementedError(
            "YFinanceDataAdapter.stream_bars is not implemented. "
            "YFinance only provides historical data, not real-time streaming."
        )
    
    async def get_option_chain(
        self,
        underlying: str,
        as_of: date,
    ) -> list[OptionContract]:
        """
        Get option chain from Yahoo Finance.
        
        Args:
            underlying: Stock ticker symbol
            as_of: Date for the option chain
        
        Returns:
            List of OptionContract objects
        """
        # Run yfinance option chain fetch in executor
        loop = asyncio.get_event_loop()
        try:
            ticker = await loop.run_in_executor(
                None,
                lambda: yf.Ticker(underlying)
            )
            
            # Get option chain for the date
            # yfinance uses expiration dates, so we need to find the closest expiration
            expirations = await loop.run_in_executor(
                None,
                lambda: ticker.options
            )
            
            if not expirations:
                logger.warning(f"No option expirations found for {underlying}")
                return []
            
            # Find the closest expiration date to as_of
            as_of_datetime = datetime.combine(as_of, datetime.min.time())
            closest_exp = None
            min_diff = None
            
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                    diff = abs((exp_date - as_of_datetime).days)
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_exp = exp_str
                except:
                    continue
            
            if closest_exp is None:
                logger.warning(f"Could not find expiration date for {underlying} near {as_of}")
                return []
            
            # Get option chain for the closest expiration
            opt_chain = await loop.run_in_executor(
                None,
                lambda: ticker.option_chain(closest_exp)
            )
            
            contracts: list[OptionContract] = []
            
            # Process calls
            if hasattr(opt_chain, 'calls') and opt_chain.calls is not None and not opt_chain.calls.empty:
                for _, row in opt_chain.calls.iterrows():
                    try:
                        contracts.append(
                            OptionContract(
                                symbol=row.get("contractSymbol", ""),
                                underlying=underlying,
                                expiry=datetime.strptime(closest_exp, "%Y-%m-%d").date(),
                                strike=float(row.get("strike", 0)),
                                right="C",
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error processing call option: {e}")
                        continue
            
            # Process puts
            if hasattr(opt_chain, 'puts') and opt_chain.puts is not None and not opt_chain.puts.empty:
                for _, row in opt_chain.puts.iterrows():
                    try:
                        contracts.append(
                            OptionContract(
                                symbol=row.get("contractSymbol", ""),
                                underlying=underlying,
                                expiry=datetime.strptime(closest_exp, "%Y-%m-%d").date(),
                                strike=float(row.get("strike", 0)),
                                right="P",
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error processing put option: {e}")
                        continue
            
            logger.debug(f"YFinance: Retrieved {len(contracts)} option contracts for {underlying} expiring {closest_exp}")
            
            # Update last data source
            self._last_data_source = "YFinance"
            
            return contracts
            
        except Exception as e:
            logger.error(f"YFinance option chain error for {underlying}: {e}")
            return []
    
    @property
    def last_data_source(self) -> str:
        """Get the name of the data source for the last request."""
        return self._last_data_source

