from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Tuple, Optional
import asyncio
import logging
import time

try:
    import pandas as pd
    import pyarrow.parquet as pq
    _PANDAS_AVAILABLE = True
except (ImportError, ValueError) as e:
    # ValueError can occur with numpy/pandas version mismatches
    _PANDAS_AVAILABLE = False
    _PANDAS_ERROR = str(e)
    # Create dummy objects to prevent import errors
    pd = None  # type: ignore
    pq = None  # type: ignore

from core.models.bar import Bar

# Import trading calendar functions from core.data (centralized export)
# This ensures consistent access across the codebase
try:
    from core.data import get_trading_days, is_trading_day
    _TRADING_CALENDAR_AVAILABLE = True
except ImportError:
    # Fallback: try direct import if core.data export not available
    try:
        from core.data.trading_calendar import get_trading_days, is_trading_day
        _TRADING_CALENDAR_AVAILABLE = True
    except ImportError:
        _TRADING_CALENDAR_AVAILABLE = False
        get_trading_days = None  # type: ignore
        is_trading_day = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Result of checking cache coverage for a date range."""
    fully_covered: bool
    partial_covered: bool
    cached_start: Optional[datetime]
    cached_end: Optional[datetime]
    missing_ranges: List[Tuple[datetime, datetime]]
    overlapping_bars: List[Bar]


class DataCache:
    """Manages file-based caching of market data using Parquet format."""
    
    def __init__(self, cache_dir: str = "data_cache/bars"):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cache files (default: "data_cache/bars")
        """
        if not _PANDAS_AVAILABLE:
            error_msg = getattr(globals(), '_PANDAS_ERROR', 'Unknown error')
            raise ImportError(
                f"pandas and pyarrow are required for caching but could not be imported. "
                f"Error: {error_msg}. "
                f"Try reinstalling: pip install --upgrade --force-reinstall pandas pyarrow numpy"
            )
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_cache: dict = {}  # Cache key -> List[Bar]
        logger.info(f"DataCache initialized with cache directory: {self._cache_dir}")
    
    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        """
        Normalize timeframe to a canonical format for consistent cache file naming.
        
        Handles all variations:
        - "D" -> "1D", "1d" -> "1D"
        - "H" -> "1H", "1h" -> "1H"
        - "15m" -> "15m", "15M" -> "15m", "15" -> "15m"
        - "W" -> "1W", "1w" -> "1W"
        - "M" -> "1M", "1m" (month) -> "1M" (but "15m" stays as "15m")
        - "Y" -> "1Y", "1y" -> "1Y"
        
        Args:
            timeframe: Timeframe string in any format
            
        Returns:
            Normalized timeframe string
        """
        if not timeframe:
            raise ValueError("Timeframe cannot be empty")
        
        timeframe_upper = timeframe.upper()
        
        # Handle single letter timeframes (D, H, W, M, Y)
        if timeframe_upper == "D":
            return "1D"
        elif timeframe_upper == "H":
            return "1H"
        elif timeframe_upper == "W":
            return "1W"
        elif timeframe_upper == "M":
            # "M" alone means monthly, not minute
            return "1M"
        elif timeframe_upper == "Y":
            return "1Y"
        
        # Handle minutely vs monthly distinction
        # Monthly: "M", "1M", "2M", "3M", "6M", "12M" (single digit or specific numbers)
        # Minutely: "15m", "30m", "60m", etc. (two+ digit numbers with m/M)
        if timeframe_upper.endswith("M"):
            prefix = timeframe_upper[:-1]
            # If prefix is empty or single digit (1-9), it's monthly
            if not prefix or (len(prefix) == 1 and prefix.isdigit()):
                return (prefix if prefix else "1") + "M"
            # If prefix is multi-digit, it's minutely (e.g., "15M" -> "15m")
            elif prefix.isdigit():
                return prefix + "m"
        elif timeframe.endswith("m"):
            # Lowercase 'm' is always minutes
            prefix = timeframe[:-1]
            if prefix.isdigit():
                return prefix + "m"
        
        # Handle pure numeric minutely resolutions (e.g., "15", "30", "60")
        if timeframe.isdigit():
            return timeframe + "m"
        
        # Handle multi-period resolutions (e.g., "1H", "2D", "1W", "1M", "1Y")
        # These should already be in a consistent format, but normalize case
        if any(timeframe_upper.endswith(suffix) for suffix in ["H", "D", "W", "M", "Y"]):
            # Ensure consistent format: number + uppercase letter
            # e.g., "1h" -> "1H", "2d" -> "2D"
            for suffix in ["H", "D", "W", "M", "Y"]:
                if timeframe_upper.endswith(suffix):
                    prefix = timeframe_upper[:-len(suffix)]
                    if prefix.isdigit() or prefix == "":
                        return (prefix if prefix else "1") + suffix
        
        # If we can't normalize, return uppercase version
        # This handles edge cases and ensures at least case consistency
        return timeframe_upper
    
    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get the cache file path for a symbol and timeframe.
        
        Uses normalized timeframe to ensure consistent cache file naming
        across different timeframe format variations (e.g., "D" vs "1D", "15m" vs "15M").
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            timeframe: Timeframe in any format (e.g., "D", "1D", "15m", "15M", "1H", "H")
            
        Returns:
            Path to the cache file
        """
        # Normalize: uppercase symbol, normalized timeframe
        normalized_symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        cache_filename = f"{normalized_symbol}_{normalized_timeframe}.parquet"
        return self._cache_dir / cache_filename
    
    async def load_cached_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[List[Bar]]:
        """
        Load ALL cached bars from the cache file (not just the requested range).
        This allows proper coverage checking against the full cache.
        
        The caller should filter to the requested range after checking coverage.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            start: Start datetime (used for filtering return value, not for loading)
            end: End datetime (used for filtering return value, not for loading)
            
        Returns:
            List of cached bars within the requested range, or None if cache doesn't exist
        """
        cache_path = self.get_cache_path(symbol, timeframe)
        
        if not cache_path.exists():
            return None
        
        # Normalize timeframe for consistent cache key
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        # Check in-memory cache first (use a key that represents the full cache)
        # We'll use the cache file's modification time as part of the key to invalidate if file changes
        try:
            cache_mtime = cache_path.stat().st_mtime
            cache_key = (symbol.upper(), normalized_timeframe, cache_mtime)
        except Exception:
            cache_key = (symbol.upper(), normalized_timeframe, None)
        
        if cache_key in self._in_memory_cache:
            logger.debug(f"In-memory cache hit for {symbol} {timeframe}")
            all_cached_bars = self._in_memory_cache[cache_key]
            # Filter to requested range for return
            filtered = [b for b in all_cached_bars if start <= b.timestamp <= end]
            return filtered if filtered else None

        try:
            start_time = time.time()
            
            # Read ALL bars from Parquet file (for proper coverage checking)
            def _read_all():
                # Read the entire file - for typical cache sizes this is fast
                df = pd.read_parquet(cache_path)
                
                # Ensure timestamp is datetime and set as index if not already
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                elif df.index.name != 'timestamp':
                    df.index = pd.to_datetime(df.index)
                    df.index.name = 'timestamp'
                
                if df.empty:
                    return []
                
                # Convert to Bar objects - highly optimized using vectorized operations
                timestamps = df.index
                if hasattr(timestamps, 'to_pydatetime'):
                    # Vectorized conversion to Python datetime objects
                    ts_list = timestamps.to_pydatetime()
                else:
                    ts_list = list(timestamps)
                
                # Extract numpy arrays for faster access (no Python overhead)
                opens = df['open'].values
                highs = df['high'].values
                lows = df['low'].values
                closes = df['close'].values
                volumes = df.get('volume', pd.Series([0.0] * len(df))).values
                
                # Create bars using list comprehension with pre-extracted arrays
                # This avoids repeated DataFrame lookups and is much faster
                bars = [
                    Bar(
                        symbol=symbol,
                        timestamp=ts_list[i],
                        open=float(opens[i]),
                        high=float(highs[i]),
                        low=float(lows[i]),
                        close=float(closes[i]),
                        volume=float(volumes[i]),
                        timeframe=timeframe,
                    )
                    for i in range(len(df))
                ]
                return bars
            
            all_bars = await asyncio.to_thread(_read_all)
            
            elapsed = time.time() - start_time
            
            if not all_bars:
                return None
            
            # Store full cache in memory for future requests
            self._in_memory_cache[cache_key] = all_bars
            
            # Filter to requested range for return
            filtered = [b for b in all_bars if start <= b.timestamp <= end]
            
            # No logging when loading from cache (range covered)
            return filtered if filtered else None
            
        except Exception as e:
            logger.warning(
                f"Failed to load cache from {cache_path}: {e}. "
                "Will fetch from API instead."
            )
            # If cache is corrupted, delete it
            try:
                cache_path.unlink()
            except Exception:
                pass
            return None
    
    def get_all_cached_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[List[Bar]]:
        """
        Get all cached bars from in-memory cache if available.
        This is used for proper coverage checking.
        
        If not in memory, will load from file and populate in-memory cache.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            
        Returns:
            All cached bars from in-memory cache or file, or None if not available
        """
        # Normalize for consistent matching
        normalized_symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        # First, try to find any cache entry for this symbol/timeframe without file I/O
        # This avoids expensive stat() calls when cache is already loaded
        best_match = None
        best_mtime = 0
        for key, bars in self._in_memory_cache.items():
            # Compare normalized values
            if key[0] == normalized_symbol and key[1] == normalized_timeframe:
                key_mtime = key[2] if key[2] is not None else 0
                if key_mtime >= best_mtime:
                    best_mtime = key_mtime
                    best_match = bars
        
        # If we found a match in memory, return it immediately (no file I/O needed)
        if best_match is not None:
            return best_match
        
        # Not in memory - check file system and load if exists
        # This is the slow path - only happens on first request or after cache invalidation
        cache_path = self.get_cache_path(symbol, timeframe)
        if not cache_path.exists():
            return None
        
        # Load from file and populate in-memory cache
        try:
            cache_mtime = cache_path.stat().st_mtime
            cache_key = (normalized_symbol, normalized_timeframe, cache_mtime)
            
            # Check if already in memory with this exact key (shouldn't happen, but check anyway)
            if cache_key in self._in_memory_cache:
                return self._in_memory_cache[cache_key]
            
            # Load from file synchronously (this is the slow path anyway)
            # Note: This method is synchronous, so we can't use async operations
            # The caller (cached_engine) will handle async context
            df = pd.read_parquet(cache_path)
            
            # Ensure timestamp is datetime and set as index if not already
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif df.index.name != 'timestamp':
                df.index = pd.to_datetime(df.index)
                df.index.name = 'timestamp'
            
            if df.empty:
                return None
            
            # Convert to Bar objects
            timestamps = df.index
            if hasattr(timestamps, 'to_pydatetime'):
                ts_list = timestamps.to_pydatetime()
            else:
                ts_list = list(timestamps)
            
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df.get('volume', pd.Series([0.0] * len(df))).values
            
            all_bars = [
                Bar(
                    symbol=symbol,  # Use original symbol, not normalized
                    timestamp=ts_list[i],
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(volumes[i]),
                    timeframe=timeframe,  # Use original timeframe, not normalized
                )
                for i in range(len(df))
            ]
            
            if not all_bars:
                return None
            
            # Store in memory for future requests
            self._in_memory_cache[cache_key] = all_bars
            
            # Also remove any older cache entries for this symbol/timeframe
            keys_to_remove = [
                k for k in self._in_memory_cache.keys()
                if k[0] == normalized_symbol and k[1] == normalized_timeframe and k != cache_key
            ]
            for key in keys_to_remove:
                del self._in_memory_cache[key]
            
            return all_bars
            
        except Exception as e:
            logger.debug(f"Failed to load cache from file in get_all_cached_bars: {e}")
            return None
    
    async def save_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: List[Bar],
    ) -> None:
        """
        Save bars to cache, merging with existing data if present.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            bars: List of bars to save
        """
        if not bars:
            return
        
        cache_path = self.get_cache_path(symbol, timeframe)
        
        try:
            # Load existing cache if it exists
            # First check in-memory cache (most up-to-date)
            existing_bars: List[Bar] = []
            all_cached_bars = self.get_all_cached_bars(symbol, timeframe)
            if all_cached_bars:
                existing_bars = all_cached_bars
                logger.debug(f"Using in-memory cache: {len(existing_bars)} bars")
            elif cache_path.exists():
                # Fall back to file if not in memory
                try:
                    df_existing = await asyncio.to_thread(pd.read_parquet, cache_path)
                    if 'timestamp' in df_existing.columns:
                        df_existing = df_existing.set_index('timestamp')
                    df_existing.index = pd.to_datetime(df_existing.index)
                    
                    # Convert existing to Bar objects
                    for idx, row in df_existing.iterrows():
                        existing_bars.append(
                            Bar(
                                symbol=symbol,
                                timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                                open=float(row['open']),
                                high=float(row['high']),
                                low=float(row['low']),
                                close=float(row['close']),
                                volume=float(row.get('volume', 0.0)),
                                timeframe=timeframe,
                            )
                        )
                    logger.debug(f"Loaded from file: {len(existing_bars)} bars")
                except Exception as e:
                    logger.warning(f"Failed to load existing cache for merging: {e}")
                    existing_bars = []
            
            # Merge and deduplicate
            merged_bars = self._merge_and_deduplicate(existing_bars, bars)
            
            # Debug: Log if merge didn't add new bars (only at debug level to reduce noise)
            if len(merged_bars) == len(existing_bars) and bars:
                logger.debug(
                    f"Merge didn't add new bars: {len(bars)} new bars were duplicates. "
                    f"Existing range: {min(b.timestamp for b in existing_bars) if existing_bars else 'N/A'} to {max(b.timestamp for b in existing_bars) if existing_bars else 'N/A'}"
                )
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': b.timestamp,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume,
                    'symbol': b.symbol,
                    'timeframe': b.timeframe,
                }
                for b in merged_bars
            ])
            
            # Set timestamp as index and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Write to temporary file first (atomic write)
            temp_path = cache_path.with_suffix('.parquet.tmp')
            
            # Remove temp file if it exists from a previous failed write
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            
            await asyncio.to_thread(
                df.to_parquet,
                temp_path,
                engine='pyarrow',
                compression='snappy',
                index=True,
            )
            
            # Atomic rename with retry logic (Windows may have file locks)
            max_retries = 3
            rename_success = False
            for attempt in range(max_retries):
                try:
                    # On Windows, if target file exists and is locked, remove it first
                    if cache_path.exists():
                        # Try to remove read-only attribute if set
                        try:
                            import os
                            os.chmod(cache_path, 0o666)  # Make writable
                        except Exception:
                            pass
                        
                        # Try to delete the old file
                        try:
                            cache_path.unlink()
                        except PermissionError:
                            # File might be locked, wait and retry
                            if attempt < max_retries - 1:
                                await asyncio.sleep(0.1 * (attempt + 1))
                                continue
                            else:
                                # Last attempt: try direct write as fallback
                                logger.warning(
                                    f"Could not remove old cache file (may be locked). "
                                    f"Attempting direct write to {cache_path}"
                                )
                                await asyncio.to_thread(
                                    df.to_parquet,
                                    cache_path,
                                    engine='pyarrow',
                                    compression='snappy',
                                    index=True,
                                )
                                rename_success = True
                                break
                    
                    # Perform atomic rename
                    temp_path.replace(cache_path)
                    rename_success = True
                    break
                    
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        # Wait before retry (exponential backoff)
                        wait_time = 0.1 * (2 ** attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_retries} for cache save "
                            f"(waiting {wait_time:.2f}s): {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Last attempt failed, try direct write as fallback
                        logger.warning(
                            f"Atomic rename failed after {max_retries} attempts. "
                            f"Falling back to direct write: {e}"
                        )
                        try:
                            await asyncio.to_thread(
                                df.to_parquet,
                                cache_path,
                                engine='pyarrow',
                                compression='snappy',
                                index=True,
                            )
                            rename_success = True
                        except Exception as fallback_error:
                            logger.error(f"Direct write also failed: {fallback_error}")
                            raise
            
            # Clean up temp file if rename succeeded
            if rename_success and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            
            if rename_success:
                # Calculate actual number of new bars added
                actual_new = len(merged_bars) - len(existing_bars)
                logger.info(
                    f"Saved {len(merged_bars)} bars to cache: {cache_path} "
                    f"(added {actual_new} new, had {len(existing_bars)} existing, "
                    f"received {len(bars)} from caller)"
                )
                if actual_new == 0 and bars:
                    # Debug: why didn't new bars get added?
                    new_timestamps = [b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in bars]
                    existing_range = f"{min(b.timestamp for b in existing_bars) if existing_bars else 'N/A'} to {max(b.timestamp for b in existing_bars) if existing_bars else 'N/A'}"
                    logger.warning(
                        f"No new bars added: {len(bars)} bars were duplicates. "
                        f"New bar timestamps: {new_timestamps[:5]}. "
                        f"Existing cache range: {existing_range}"
                    )
                
                # Update in-memory cache with the merged bars
                # This ensures subsequent requests see the updated cache immediately
                try:
                    cache_mtime = cache_path.stat().st_mtime
                    normalized_symbol = symbol.upper()
                    normalized_timeframe = self._normalize_timeframe(timeframe)
                    cache_key = (normalized_symbol, normalized_timeframe, cache_mtime)
                    self._in_memory_cache[cache_key] = merged_bars
                    # Also invalidate old cache keys for this symbol/timeframe
                    # (in case file was updated but mtime didn't change significantly)
                    keys_to_remove = [
                        k for k in self._in_memory_cache.keys()
                        if k[0] == normalized_symbol and k[1] == normalized_timeframe and k != cache_key
                    ]
                    for key in keys_to_remove:
                        del self._in_memory_cache[key]
                except Exception as e:
                    logger.debug(f"Failed to update in-memory cache after save: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
            # Clean up temp file if it exists
            try:
                temp_path = cache_path.with_suffix('.parquet.tmp')
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
    
    def _merge_and_deduplicate(
        self,
        existing: List[Bar],
        new: List[Bar],
    ) -> List[Bar]:
        """
        Merge cached and new bars, handling overlaps intelligently.
        
        Strategy:
        - If same timestamp exists in both: prefer cached (assumed more reliable)
        - Sort by timestamp
        - Remove duplicates (same symbol+timestamp+timeframe)
        
        Args:
            existing: Existing cached bars
            new: New bars from API
            
        Returns:
            Merged and deduplicated list of bars
        """
        # Create a dict keyed by timestamp for fast lookup
        # Prefer existing (cached) data over new data
        merged_dict = {}
        
        # First, add all existing bars
        # Normalize timestamps to avoid timezone/microsecond issues
        for bar in existing:
            # Normalize timestamp to seconds (remove microseconds) for comparison
            normalized_ts = bar.timestamp.replace(microsecond=0)
            key = (normalized_ts, bar.symbol, bar.timeframe)
            merged_dict[key] = bar
        
        # Then add new bars, but don't overwrite existing ones
        # Normalize timestamps to avoid timezone/microsecond issues
        for bar in new:
            # Normalize timestamp to seconds (remove microseconds) for comparison
            normalized_ts = bar.timestamp.replace(microsecond=0)
            key = (normalized_ts, bar.symbol, bar.timeframe)
            if key not in merged_dict:
                merged_dict[key] = bar
        
        # Convert back to list and sort by timestamp
        merged = list(merged_dict.values())
        merged.sort(key=lambda b: b.timestamp)
        
        return merged
    
    @staticmethod
    def _get_time_unit_seconds(timeframe: Optional[str]) -> Optional[int]:
        """
        Determine the time unit in seconds for a given timeframe.
        
        This is used to normalize timestamps to the appropriate granularity
        for coverage checking. For example:
        - "15m" -> 900 seconds (15 minutes)
        - "1H" -> 3600 seconds (1 hour)
        - "1D" -> 86400 seconds (1 day)
        
        Args:
            timeframe: Timeframe string (e.g., "15m", "1H", "1D")
            
        Returns:
            Time unit in seconds, or None if timeframe is not provided or cannot be determined
        """
        if not timeframe:
            return None
        
        timeframe_upper = timeframe.upper()
        
        # Minutely timeframes (e.g., "15m", "30m", "60m")
        if timeframe.endswith("m") or timeframe.endswith("M"):
            try:
                minutes = int(timeframe[:-1])
                return minutes * 60
            except ValueError:
                return None
        
        # Hourly timeframes (e.g., "1H", "2H")
        if timeframe_upper.endswith("H"):
            try:
                hours = int(timeframe_upper[:-1]) if timeframe_upper != "H" else 1
                return hours * 3600
            except ValueError:
                return None
        
        # Daily timeframes (e.g., "1D", "D")
        if timeframe_upper.endswith("D"):
            try:
                days = int(timeframe_upper[:-1]) if timeframe_upper != "D" else 1
                return days * 86400
            except ValueError:
                return None
        
        # Weekly timeframes (e.g., "1W", "W")
        if timeframe_upper.endswith("W"):
            try:
                weeks = int(timeframe_upper[:-1]) if timeframe_upper != "W" else 1
                return weeks * 7 * 86400
            except ValueError:
                return None
        
        # Monthly timeframes (e.g., "1M", "M") - approximate as 30 days
        if timeframe_upper.endswith("M") and not timeframe_upper[:-1].isdigit() or len(timeframe_upper) == 1:
            try:
                months = int(timeframe_upper[:-1]) if timeframe_upper != "M" else 1
                return months * 30 * 86400  # Approximate
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def _normalize_timestamp_to_unit(ts: datetime, unit_seconds: int) -> datetime:
        """
        Normalize a timestamp to the nearest unit boundary.
        
        For example, if unit_seconds is 900 (15 minutes):
        - 2025-06-15 10:23:45 -> 2025-06-15 10:15:00
        - 2025-06-15 10:37:12 -> 2025-06-15 10:30:00
        
        Args:
            ts: Timestamp to normalize
            unit_seconds: Time unit in seconds
            
        Returns:
            Normalized timestamp at unit boundary
        """
        if unit_seconds is None:
            return ts
        
        # Calculate total seconds since epoch
        total_seconds = int(ts.timestamp())
        
        # Round down to nearest unit boundary
        normalized_seconds = (total_seconds // unit_seconds) * unit_seconds
        
        # Convert back to datetime
        return datetime.fromtimestamp(normalized_seconds, tz=ts.tzinfo if ts.tzinfo else None)
    
    def check_coverage(
        self,
        cached_bars: List[Bar],
        start: datetime,
        end: datetime,
        timeframe: Optional[str] = None,
    ) -> CoverageResult:
        """
        Check if cached bars cover the requested date range.
        
        This is a generic solution that works for all timeframes by:
        1. Determining the time unit from the timeframe (minutes, hours, days)
        2. Normalizing timestamps to unit boundaries
        3. Checking coverage based on normalized units
        4. Creating missing ranges based on missing units
        
        Args:
            cached_bars: List of cached bars (can be empty)
            start: Requested start datetime
            end: Requested end datetime
            timeframe: Timeframe string (e.g., "15m", "1H", "1D") - used to determine time unit
            
        Returns:
            CoverageResult with coverage information
        """
        if not cached_bars:
            return CoverageResult(
                fully_covered=False,
                partial_covered=False,
                cached_start=None,
                cached_end=None,
                missing_ranges=[(start, end)],
                overlapping_bars=[],
            )
        
        # Find cached date range
        cached_start = min(b.timestamp for b in cached_bars)
        cached_end = max(b.timestamp for b in cached_bars)
        
        # Get time unit for normalization (needed for proper overlapping check)
        unit_seconds = self._get_time_unit_seconds(timeframe)
        
        # Filter bars within requested range
        # For timeframes with units, use normalized comparison to handle timestamp differences
        if unit_seconds:
            # Normalize start/end and bar timestamps to unit boundaries for comparison
            normalized_start = self._normalize_timestamp_to_unit(start, unit_seconds)
            normalized_end = self._normalize_timestamp_to_unit(end, unit_seconds)
            overlapping = [
                b for b in cached_bars
                if normalized_start <= self._normalize_timestamp_to_unit(b.timestamp, unit_seconds) <= normalized_end
            ]
        else:
            # For unknown timeframes, use exact timestamp comparison
            overlapping = [b for b in cached_bars if start <= b.timestamp <= end]
        
        # Check if fully covered (using normalized comparison for timeframes with units)
        if unit_seconds:
            # Use normalized comparison for timeframes with units
            normalized_cached_start = self._normalize_timestamp_to_unit(cached_start, unit_seconds)
            normalized_cached_end = self._normalize_timestamp_to_unit(cached_end, unit_seconds)
            normalized_start_check = self._normalize_timestamp_to_unit(start, unit_seconds)
            normalized_end_check = self._normalize_timestamp_to_unit(end, unit_seconds)
            
            if normalized_cached_start <= normalized_start_check and normalized_cached_end >= normalized_end_check:
                if overlapping:
                    return CoverageResult(
                        fully_covered=True,
                        partial_covered=False,
                        cached_start=cached_start,
                        cached_end=cached_end,
                        missing_ranges=[],
                        overlapping_bars=overlapping,
                    )
                else:
                    # Requested range is within cached range but no bars (weekend/holiday)
                    return CoverageResult(
                        fully_covered=True,
                        partial_covered=False,
                        cached_start=cached_start,
                        cached_end=cached_end,
                        missing_ranges=[],
                        overlapping_bars=[],
                    )
        else:
            # For unknown timeframes, use exact timestamp comparison
            if cached_start <= start and cached_end >= end:
                if overlapping:
                    return CoverageResult(
                        fully_covered=True,
                        partial_covered=False,
                        cached_start=cached_start,
                        cached_end=cached_end,
                        missing_ranges=[],
                        overlapping_bars=overlapping,
                    )
                else:
                    # Requested range is within cached range but no bars (weekend/holiday)
                    return CoverageResult(
                        fully_covered=True,
                        partial_covered=False,
                        cached_start=cached_start,
                        cached_end=cached_end,
                        missing_ranges=[],
                        overlapping_bars=[],
                    )
        
        # If we can determine the time unit, use normalized coverage checking
        if unit_seconds:
            # Normalize timestamps to unit boundaries
            normalized_start = self._normalize_timestamp_to_unit(start, unit_seconds)
            normalized_end = self._normalize_timestamp_to_unit(end, unit_seconds)
            normalized_cached_start = self._normalize_timestamp_to_unit(cached_start, unit_seconds)
            normalized_cached_end = self._normalize_timestamp_to_unit(cached_end, unit_seconds)
            
            # Create set of normalized units from cached bars
            cached_units = {
                self._normalize_timestamp_to_unit(b.timestamp, unit_seconds)
                for b in cached_bars
            }
            
            # Generate all normalized units in requested range
            # For daily bars, only include trading days
            requested_units = set()
            if unit_seconds == 86400 and _TRADING_CALENDAR_AVAILABLE and get_trading_days:
                # Daily bars - only include trading days
                start_date = normalized_start.date()
                end_date = normalized_end.date()
                try:
                    trading_days = get_trading_days(start_date, end_date)
                    # Convert trading days to normalized datetime units
                    for trading_day in trading_days:
                        # Normalize trading day to Python date object first
                        # trading_day may be a pandas Timestamp
                        if hasattr(trading_day, 'date') and callable(getattr(trading_day, 'date', None)):
                            # pandas Timestamp or datetime object - convert to Python date
                            trading_day_date = trading_day.date()
                        elif isinstance(trading_day, date):
                            # Plain Python date object
                            trading_day_date = trading_day
                        else:
                            try:
                                trading_day_date = date.fromisoformat(str(trading_day))
                            except (ValueError, AttributeError):
                                continue
                        
                        # Create datetime at start of trading day
                        unit_dt = datetime.combine(trading_day_date, datetime.min.time())
                        # Normalize to unit boundary (should already be at boundary for daily)
                        normalized_unit = self._normalize_timestamp_to_unit(unit_dt, unit_seconds)
                        if normalized_start <= normalized_unit <= normalized_end:
                            requested_units.add(normalized_unit)
                except Exception as e:
                    logger.warning(f"Failed to get trading days, falling back to calendar days: {e}")
                    # Fallback to calendar days
                    current = normalized_start
                    while current <= normalized_end:
                        requested_units.add(current)
                        current += timedelta(seconds=unit_seconds)
            else:
                # For non-daily bars or if trading calendar not available, use all units
                current = normalized_start
                while current <= normalized_end:
                    requested_units.add(current)
                    current += timedelta(seconds=unit_seconds)
            
            # Check if all requested units are covered
            if requested_units.issubset(cached_units) and overlapping:
                return CoverageResult(
                    fully_covered=True,
                    partial_covered=False,
                    cached_start=cached_start,
                    cached_end=cached_end,
                    missing_ranges=[],
                    overlapping_bars=overlapping,
                )
            
            # Find missing units and create ranges
            missing_ranges = []
            missing_units = sorted(requested_units - cached_units)
            
            if missing_units:
                # For daily bars, filter out non-trading days from missing units
                # Use get_trading_days() once to get all trading days, then check membership (more efficient)
                if unit_seconds == 86400 and _TRADING_CALENDAR_AVAILABLE and get_trading_days:
                    try:
                        # Get all trading days in the range once (more efficient than calling is_trading_day per unit)
                        min_missing_date = min(unit.date() for unit in missing_units)
                        max_missing_date = max(unit.date() for unit in missing_units)
                        trading_days = get_trading_days(min_missing_date, max_missing_date)
                        
                        # Normalize trading days to Python date objects for comparison
                        # trading_days may contain pandas Timestamp objects
                        trading_days_set = set()
                        for d in trading_days:
                            # Check if it has a date() method (pandas Timestamp or datetime objects)
                            # Plain Python date objects don't have a date() method
                            if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                                trading_days_set.add(d.date())
                            elif isinstance(d, date):
                                trading_days_set.add(d)
                            else:
                                try:
                                    trading_days_set.add(date.fromisoformat(str(d)))
                                except (ValueError, AttributeError):
                                    continue
                        
                        # Filter missing units to only include trading days
                        trading_days_units = [
                            unit for unit in missing_units
                            if unit.date() in trading_days_set
                        ]
                        missing_units = trading_days_units
                        if not missing_units:
                            # No trading days missing, return empty ranges
                            return CoverageResult(
                                fully_covered=False,
                                partial_covered=len(overlapping) > 0,
                                cached_start=cached_start,
                                cached_end=cached_end,
                                missing_ranges=[],
                                overlapping_bars=overlapping,
                            )
                    except Exception as e:
                        logger.warning(f"Failed to filter trading days from missing units, using all units: {e}")
                        # Continue with all missing units if filtering fails
                
                # Group consecutive missing units into ranges
                range_start = missing_units[0]
                range_end = missing_units[0]
                
                for unit in missing_units[1:]:
                    expected_next = range_end + timedelta(seconds=unit_seconds)
                    if unit == expected_next:
                        # Consecutive unit - extend range
                        range_end = unit
                    else:
                        # Gap found - close current range and start new one
                        # Expand range_end to end of unit (inclusive of the full unit)
                        range_end_expanded = range_end + timedelta(seconds=unit_seconds - 1)
                        missing_ranges.append((range_start, range_end_expanded))
                        range_start = unit
                        range_end = unit
                
                # Close final range
                # Expand range_end to end of unit (inclusive of the full unit)
                range_end_expanded = range_end + timedelta(seconds=unit_seconds - 1)
                missing_ranges.append((range_start, range_end_expanded))
        else:
            # Fallback: Use exact timestamp comparison (for unknown timeframes)
            missing_ranges = []
            if cached_start > start:
                missing_ranges.append((start, min(cached_start, end)))
            if cached_end < end:
                missing_ranges.append((max(cached_end, start), end))
        
        return CoverageResult(
            fully_covered=False,
            partial_covered=len(overlapping) > 0,
            cached_start=cached_start,
            cached_end=cached_end,
            missing_ranges=missing_ranges,
            overlapping_bars=overlapping,
        )

