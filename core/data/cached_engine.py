"""
CachedDataEngine - Transparent caching wrapper for any DataEngine.

This module provides a caching layer that sits in front of any DataEngine
implementation, providing transparent caching without requiring changes to
the underlying data source implementations.
"""

from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import AsyncIterator, List, Optional
import logging

from core.data.base import DataEngine
from core.models.bar import Bar
from core.models.option import OptionContract
from core.utils.timestamp import normalize_timestamp_for_comparison
from core.utils.error_handling import is_api_error, format_api_error_message

try:
    from core.utils.timestamp import normalize_bar_range  # project import
except Exception:  # pragma: no cover
    try:
        from timestamp import normalize_bar_range  # fallback for standalone tests
    except Exception:
        normalize_bar_range = None  # type: ignore
logger = logging.getLogger(__name__)

# Lazy import of DataCache to avoid dependency issues if pandas/pyarrow not installed
try:
    from core.data.cache import DataCache
    _CACHE_AVAILABLE = True
except (ImportError, ValueError) as e:
    _CACHE_AVAILABLE = False
    DataCache = None  # type: ignore
    _CACHE_ERROR = str(e)

# Import trading calendar functions from core.data (centralized export)
from core.data import get_trading_days, is_trading_day


class CachedDataEngine(DataEngine):
    """
    Transparent caching wrapper for any DataEngine.
    
    This class wraps any DataEngine implementation and adds caching functionality.
    The wrapped engine doesn't need to know about caching - it's completely transparent.
    
    Example:
        >>> base_engine = MarketDataAppAdapter(api_token="...")
        >>> cached_engine = CachedDataEngine(base_engine, cache_dir="data_cache/bars")
        >>> bars = await cached_engine.get_bars("AAPL", start, end, "D")
    """
    
    def __init__(
        self,
        base_engine: DataEngine,
        cache_dir: Optional[str] = "data_cache/bars",
        cache_enabled: bool = True,
    ):
        """
        Initialize CachedDataEngine wrapper.
        
        Args:
            base_engine: The underlying DataEngine to wrap (e.g., MarketDataAppAdapter)
            cache_dir: Directory for cache files (None to disable caching)
            cache_enabled: Whether to use cache (default: True)
        """
        self._base_engine = base_engine
        self._cache_enabled = cache_enabled and cache_dir is not None and _CACHE_AVAILABLE
        
        if self._cache_enabled and DataCache is not None:
            self._cache = DataCache(cache_dir)
        else:
            self._cache = None
            if cache_enabled and not _CACHE_AVAILABLE:
                error_msg = globals().get('_CACHE_ERROR', 'Unknown error')
                logger.warning(
                    f"Caching requested but pandas/pyarrow not available. "
                    f"Error: {error_msg}. "
                    f"Try: pip install --upgrade --force-reinstall pandas pyarrow numpy"
                )
        
        # Cache statistics
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_partial_hits = 0
        self._base_engine_calls = 0
        self._total_bars_from_cache = 0
        self._total_bars_from_api = 0
        self._last_data_source = self._get_data_source_name()  # Track data source name for last request
        
        # Track "no data" ranges to avoid repeated 404s
        # Format: {(symbol, timeframe, date): True} for dates we've confirmed have no data
        self._no_data_ranges: dict = {}
    
    def _get_data_source_name(self) -> str:
        """Get the name of the underlying data source for logging."""
        engine_name = type(self._base_engine).__name__
        # Map common engine names to shorter identifiers
        name_map = {
            "MarketDataAppAdapter": "MarketData",
            "MoomooDataAdapter": "Moomoo",
        }
        return name_map.get(engine_name, engine_name)
    
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        """
        Get bars with transparent caching.
        
        First checks cache, then fetches missing data from the base engine.
        """
        # Track total requests
        self._total_requests += 1
        self._last_data_source = self._get_data_source_name()  # Reset to base source
        
        # Cache trading days lookups within this method call to avoid repeated API calls
        # Format: {(start_date, end_date): [list of trading days]}
        _trading_days_cache: dict = {}
        
        def _get_trading_days_cached(start_date: date, end_date: date) -> List[date]:
            """Get trading days with caching within this method call.
            
            Returns normalized Python date objects (not pandas Timestamps).
            """
            cache_key = (start_date, end_date)
            if cache_key not in _trading_days_cache:
                if get_trading_days:
                    try:
                        trading_days = get_trading_days(start_date, end_date)
                        # Normalize to Python date objects before caching
                        # trading_days may contain pandas Timestamp objects
                        normalized_dates = []
                        for d in trading_days:
                            # Check if it has a date() method (pandas Timestamp or datetime objects)
                            # Plain Python date objects don't have a date() method
                            if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                                normalized_dates.append(d.date())
                            elif isinstance(d, date):
                                normalized_dates.append(d)
                            else:
                                try:
                                    normalized_dates.append(date.fromisoformat(str(d)))
                                except (ValueError, AttributeError):
                                    logger.warning(f"Could not normalize trading day: {d} (type: {type(d)})")
                                    continue
                        _trading_days_cache[cache_key] = normalized_dates
                        logger.debug(f"[Cache] Retrieved {len(normalized_dates)} trading days from {start_date} to {end_date}")
                    except Exception as e:
                        logger.warning(f"[Cache] Failed to get trading days for {start_date} to {end_date}: {e}, falling back to calendar days")
                        # Fallback to calendar days
                        _trading_days_cache[cache_key] = []
                        current = start_date
                        while current <= end_date:
                            _trading_days_cache[cache_key].append(current)
                            current += timedelta(days=1)
                else:
                    # Fallback to calendar days if trading calendar not available
                    logger.debug(f"[Cache] Trading calendar not available (get_trading_days={get_trading_days is not None}), using calendar days")
                    _trading_days_cache[cache_key] = []
                    current = start_date
                    while current <= end_date:
                        _trading_days_cache[cache_key].append(current)
                        current += timedelta(days=1)
            return _trading_days_cache[cache_key]
        
        # Check cache first if enabled
        cached_bars: Optional[List[Bar]] = None
        all_cached_bars: Optional[List[Bar]] = None
        if self._cache_enabled and self._cache:
            # Get ALL cached bars first for proper coverage checking
            # Try in-memory cache first (fastest - no file I/O)
            all_cached_bars = self._cache.get_all_cached_bars(symbol, timeframe)
            
            # Only check file system if we don't have in-memory cache
            # This avoids expensive stat() calls on every request when cache is already in memory
            if not all_cached_bars:
                cache_path = self._cache.get_cache_path(symbol, timeframe)
                if cache_path.exists():
                    # Load from file (this will populate in-memory cache)
                    cached_bars = await self._cache.load_cached_bars(symbol, timeframe, start, end)
                    # Now get all cached bars from memory (should be populated now)
                    all_cached_bars = self._cache.get_all_cached_bars(symbol, timeframe)
            
            # If we have cached data, check coverage against the full cache
            if all_cached_bars:
                # Pass timeframe so check_coverage can use correct logic for daily bars
                coverage = self._cache.check_coverage(all_cached_bars, start, end, timeframe=timeframe)
                
                if coverage.fully_covered:
                    # Filter to requested range and return
                    # Use normalized timestamp comparison (same as check_coverage) for consistency
                    # This ensures we get the same bars that check_coverage identifies as overlapping
                    unit_seconds = self._cache._get_time_unit_seconds(timeframe)
                    if unit_seconds:
                        # Normalize timestamps to unit boundaries (same as check_coverage)
                        normalized_start = self._cache._normalize_timestamp_to_unit(start, unit_seconds)
                        normalized_end = self._cache._normalize_timestamp_to_unit(end, unit_seconds)
                        filtered = [
                            b for b in all_cached_bars
                            if normalized_start <= self._cache._normalize_timestamp_to_unit(b.timestamp, unit_seconds) <= normalized_end
                        ]
                    else:
                        # For unknown timeframes, use exact timestamp comparison
                        filtered = [b for b in all_cached_bars if start <= b.timestamp <= end]
                    self._cache_hits += 1
                    self._total_bars_from_cache += len(filtered)
                    self._last_data_source = "Cache"  # Mark that this request was from cache
                    
                    # Diagnostic logging for date gaps
                    if filtered and timeframe.upper() in ("D", "1D"):
                        filtered_dates = sorted(set(b.timestamp.date() for b in filtered))
                        if get_trading_days:
                            try:
                                expected_dates = get_trading_days(start.date(), end.date())
                                missing = set(expected_dates) - set(filtered_dates)
                                if missing:
                                    logger.warning(
                                        f"[Cache] Coverage check says fully covered, but {len(missing)} trading days are missing: "
                                        f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}. "
                                        f"Cache has {len(all_cached_bars)} total bars, filtered to {len(filtered)} bars. "
                                        f"Cache range: {min(b.timestamp.date() for b in all_cached_bars)} to {max(b.timestamp.date() for b in all_cached_bars)}"
                                    )
                            except Exception:
                                pass  # Don't fail if trading days check fails
                    
                    # No logging when range is fully covered (unless diagnostic warning above)
                    return filtered
                
                # Partial coverage - we'll fetch missing ranges below
                # Get filtered bars for the overlapping portion
                # Use normalized timestamp comparison (same as check_coverage) for consistency
                # This ensures we get the same bars that check_coverage identifies as overlapping
                unit_seconds = self._cache._get_time_unit_seconds(timeframe)
                if unit_seconds:
                    # Normalize timestamps to unit boundaries (same as check_coverage)
                    normalized_start = self._cache._normalize_timestamp_to_unit(start, unit_seconds)
                    normalized_end = self._cache._normalize_timestamp_to_unit(end, unit_seconds)
                    cached_bars = [
                        b for b in all_cached_bars
                        if normalized_start <= self._cache._normalize_timestamp_to_unit(b.timestamp, unit_seconds) <= normalized_end
                    ]
                else:
                    # For unknown timeframes, use exact timestamp comparison
                    cached_bars = [b for b in all_cached_bars if start <= b.timestamp <= end]
                # #region agent log
                with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"cached_engine.py:223","message":"cached_bars filtered","data":{"all_cached_bars_count":len(all_cached_bars) if all_cached_bars else 0,"cached_bars_count":len(cached_bars),"start":str(start),"end":str(end),"cached_bars_dates":[str(b.timestamp.date()) for b in cached_bars[:5]]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
        
        # Determine what needs to be fetched from base engine
        missing_ranges: List[tuple[datetime, datetime]] = []
        overlapping_bars_count = 0
        
        if all_cached_bars:
            # Use full cache for coverage check
            # Pass timeframe so check_coverage knows if it's daily
            coverage = self._cache.check_coverage(all_cached_bars, start, end, timeframe=timeframe)
            missing_ranges = coverage.missing_ranges
            # #region agent log
            with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"cached_engine.py:232","message":"coverage check result","data":{"overlapping_bars_count":len(coverage.overlapping_bars) if coverage.overlapping_bars else 0,"cached_bars_count":len(cached_bars) if 'cached_bars' in locals() else 0,"missing_ranges_count":len(missing_ranges)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            
            # Log cache state for debugging
            cached_dates = sorted(set(b.timestamp.date() for b in all_cached_bars))
            cached_dates_set = set(cached_dates)
            
            # Use trading days instead of calendar days for daily bars
            if (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                # For daily bars, only consider trading days (uses cached lookup)
                requested_dates = _get_trading_days_cached(start.date(), end.date())
            else:
                # For non-daily bars, use all calendar days for logging purposes
                requested_dates = []
                current_date = start.date()
                end_date = end.date()
                while current_date <= end_date:
                    requested_dates.append(current_date)
                    current_date += timedelta(days=1)
            
            # Find missing dates (only trading days for daily bars)
            # Exclude dates that are confirmed to have no data (404 or empty response)
            # Normalize requested_dates to Python date objects for comparison
            # requested_dates may contain pandas Timestamp objects
            # Note: pandas Timestamp is a subclass of date, so isinstance(d, date) returns True
            # We need to check if it has a date() method (Timestamp/datetime) vs plain date
            normalized_requested_dates = []
            for d in requested_dates:
                # Check if it has a date() method (pandas Timestamp or datetime objects)
                # Plain Python date objects don't have a date() method
                if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                    # pandas Timestamp or datetime object - convert to Python date
                    normalized_requested_dates.append(d.date())
                elif isinstance(d, date):
                    # Plain Python date object (no date() method)
                    normalized_requested_dates.append(d)
                else:
                    # Fallback: try to convert string or other format
                    try:
                        normalized_requested_dates.append(date.fromisoformat(str(d)))
                    except (ValueError, AttributeError):
                        logger.warning(f"Could not normalize requested date: {d} (type: {type(d)})")
                        continue
            
            missing_dates = []
            confirmed_no_data_dates = []
            missing_trading_days = []  # Track trading days missing from cache
            for d in normalized_requested_dates:
                if d not in cached_dates_set:
                    # Check if this date is confirmed to have no data
                    date_key = (symbol.upper(), timeframe, d)
                    if date_key in self._no_data_ranges:
                        confirmed_no_data_dates.append(d)
                    else:
                        missing_dates.append(d)
                        # Check if this is a trading day (for daily bars)
                        if (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                            if is_trading_day:
                                try:
                                    if is_trading_day(d):
                                        missing_trading_days.append(d)
                                except Exception:
                                    pass  # If check fails, assume it might be a trading day
            
            # Determine date type for logging and show actual range
            if (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                # Check if we actually got trading days or fell back to calendar days
                calendar_days_count = (end.date() - start.date()).days + 1
                if len(requested_dates) == calendar_days_count:
                    # We got calendar days (fallback), not trading days
                    date_type = "calendar day(s) [FALLBACK - trading calendar unavailable]"
                    requested_range_str = f"{start.date()} to {end.date()}"
                else:
                    # We got actual trading days
                    date_type = "trading day(s)"
                    if requested_dates:
                        requested_range_str = f"{requested_dates[0]} to {requested_dates[-1]}"
                    else:
                        requested_range_str = f"{start.date()} to {end.date()} (no trading days)"
            else:
                date_type = "date(s)"
                requested_range_str = f"{start.date()} to {end.date()}"
            
            logger.info(
                f"[Cache] Coverage check: Requested {len(requested_dates)} {date_type} ({requested_range_str}), "
                f"Cache has {len(cached_dates)} date(s) ({cached_dates[0] if cached_dates else 'N/A'} to {cached_dates[-1] if cached_dates else 'N/A'}), "
                f"Overlapping: {len(coverage.overlapping_bars)} bars, "
                f"Missing ranges: {len(missing_ranges)}, "
                f"Missing dates: {len(missing_dates)}"
            )
            
            # Log confirmed "no data" dates separately (these won't be fetched)
            if confirmed_no_data_dates:
                logger.debug(
                    f"[Cache] Dates confirmed as 'no data' (will not fetch): "
                    f"{[str(d) for d in confirmed_no_data_dates[:10]]}"
                    f"{'...' if len(confirmed_no_data_dates) > 10 else ''}"
                )
            
            # Log missing dates if any (only dates that will actually be fetched)
            if missing_dates:
                # Group consecutive dates for cleaner output
                if len(missing_dates) <= 10:
                    # Show all dates if 10 or fewer
                    logger.info(f"[Cache] Missing dates (will fetch): {[str(d) for d in missing_dates]}")
                else:
                    # Show first 5 and last 5 if more than 10
                    logger.info(
                        f"[Cache] Missing dates ({len(missing_dates)} total, will fetch): "
                        f"{[str(d) for d in missing_dates[:5]]} ... {[str(d) for d in missing_dates[-5:]]}"
                    )
            
            # Log error for missing trading days (these should be in cache)
            if missing_trading_days:
                if len(missing_trading_days) <= 10:
                    logger.error(
                        f"[Cache] ERROR: Missing data for {len(missing_trading_days)} trading day(s) "
                        f"(will fetch from API): {[str(d) for d in missing_trading_days]}"
                    )
                else:
                    logger.error(
                        f"[Cache] ERROR: Missing data for {len(missing_trading_days)} trading day(s) "
                        f"(will fetch from API): {[str(d) for d in missing_trading_days[:5]]} ... "
                        f"{[str(d) for d in missing_trading_days[-5:]]}"
                    )
            
            # If check_coverage returned empty missing_ranges but we detected missing dates,
            # create missing ranges from the missing dates (for daily bars)
            # This handles cases where check_coverage's normalized comparison doesn't catch all missing dates
            if not missing_ranges and missing_dates and (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                # Create individual date ranges for each missing date
                # Use normalized timestamps for consistency with check_coverage
                unit_seconds = self._cache._get_time_unit_seconds(timeframe)
                for missing_date in missing_dates:
                    # Create datetime range for the missing date (full day)
                    # For daily bars, use the full day range (00:00:00 to 23:59:59.999999)
                    # This ensures we capture the bar regardless of its timestamp
                    day_start = datetime.combine(missing_date, datetime.min.time())
                    day_end = datetime.combine(missing_date, datetime.max.time())
                    # Normalize to unit boundaries for consistency with check_coverage
                    if unit_seconds:
                        day_start = self._cache._normalize_timestamp_to_unit(day_start, unit_seconds)
                        # For daily bars, ensure day_end covers the full day
                        if unit_seconds == 86400:
                            # For daily bars, use max time to ensure we get the bar
                            day_end = datetime.combine(missing_date, datetime.max.time())
                        else:
                            day_end = self._cache._normalize_timestamp_to_unit(day_end, unit_seconds)
                    missing_ranges.append((day_start, day_end))
                    # #region agent log
                    with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                        import json
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"cached_engine.py:399","message":"created missing range from missing date","data":{"missing_date":str(missing_date),"day_start":str(day_start),"day_end":str(day_end),"request_start":str(start),"request_end":str(end)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                logger.info(f"[Cache] Created {len(missing_ranges)} missing range(s) from {len(missing_dates)} missing date(s)")
            
            # Only count as partial hit if we actually have overlapping bars
            # A partial hit means: we had SOME cached data that overlapped with the request
            # If we have cached data but NO overlapping bars (e.g., cache is for different date range),
            # this is NOT a partial hit - it's a cache miss
            if coverage.partial_covered and coverage.overlapping_bars:
                overlapping_bars_count = len(coverage.overlapping_bars)
                self._cache_partial_hits += 1
                self._total_bars_from_cache += overlapping_bars_count
                logger.info(f"[Cache] {overlapping_bars_count} bars cached, fetching {len(missing_ranges)} missing range(s)")
            elif not coverage.fully_covered:
                # We have cached data but no overlapping bars for this request
                # This is a cache miss (not a partial hit) - fetch entire range
                missing_ranges = [(start, end)]
                self._last_data_source = self._get_data_source_name()
                logger.info(f"[{self._last_data_source}] No overlapping cache, fetching entire range")
        elif cached_bars is not None:
            coverage = self._cache.check_coverage(cached_bars, start, end, timeframe=timeframe)
            missing_ranges = coverage.missing_ranges
            # Only count as partial hit if we have overlapping bars
            if coverage.partial_covered and coverage.overlapping_bars:
                overlapping_bars_count = len(coverage.overlapping_bars)
                self._cache_partial_hits += 1
                self._total_bars_from_cache += overlapping_bars_count
                logger.info(f"[Cache] {overlapping_bars_count} bars cached, fetching {len(missing_ranges)} missing range(s)")
            elif not coverage.fully_covered:
                # Cached data exists but no overlap with request
                missing_ranges = [(start, end)]
                self._last_data_source = self._get_data_source_name()
                logger.info(f"[{self._last_data_source}] No overlapping cache, fetching entire range")
        else:
            # No cache at all, fetch entire range
            # This is a cache miss (not a partial hit)
            missing_ranges = [(start, end)]
            self._last_data_source = self._get_data_source_name()  # No cache, using base engine
            logger.info(f"[{self._last_data_source}] No cache, fetching data")
        
        # Filter out ranges we've already confirmed have no data
        filtered_missing_ranges = []
        for range_start, range_end in missing_ranges:
            # For daily bars, check if we've already confirmed these dates have no data
            # Check each date in the range
            should_skip = False
            if (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                # Daily bars - check each trading day (uses cached lookup)
                start_date = range_start.date()
                end_date = range_end.date()
                trading_days_in_range = _get_trading_days_cached(start_date, end_date)
                all_dates_confirmed_no_data = True
                for trading_day in trading_days_in_range:
                    date_key = (symbol.upper(), timeframe, trading_day)
                    if date_key not in self._no_data_ranges:
                        all_dates_confirmed_no_data = False
                        break
            else:
                # Fallback to calendar days if trading calendar not available
                current_date = range_start.date()
                end_date = range_end.date()
                all_dates_confirmed_no_data = True
                while current_date <= end_date:
                    date_key = (symbol.upper(), timeframe, current_date)
                    if date_key not in self._no_data_ranges:
                        all_dates_confirmed_no_data = False
                        break
                    current_date += timedelta(days=1)
                if all_dates_confirmed_no_data and range_start.date() == range_end.date():
                    # All dates in this range are confirmed to have no data
                    logger.debug(f"[Cache] Skipping range {range_start} to {range_end} - already confirmed no data")
                    should_skip = True
            
            if not should_skip:
                filtered_missing_ranges.append((range_start, range_end))
        
        # Update missing_ranges to filtered list
        missing_ranges = filtered_missing_ranges
        
        # Log missing ranges in detail
        if missing_ranges:
            logger.info(f"[Cache] Missing ranges ({len(missing_ranges)}):")
            all_missing_dates = []
            for i, (range_start, range_end) in enumerate(missing_ranges, 1):
                if timeframe.upper().endswith("D") or timeframe.upper() == "D":
                    # For daily bars, show trading days only
                    start_date = range_start.date()
                    end_date = range_end.date()
                    missing_dates = []
                    
                    # Use cached trading days lookup
                    trading_days_in_range = _get_trading_days_cached(start_date, end_date)
                    # Filter to only trading days that are actually missing AND not confirmed as "no data"
                    cached_dates_set = set(b.timestamp.date() for b in all_cached_bars) if all_cached_bars else set()
                    # Normalize trading days to Python date objects for comparison
                    # Note: pandas Timestamp is a subclass of date, so isinstance(d, date) returns True
                    # We need to check if it has a date() method (Timestamp/datetime) vs plain date
                    missing_dates = []
                    for d in trading_days_in_range:
                        # Normalize to Python date object
                        # Check if it has a date() method (pandas Timestamp or datetime objects)
                        # Plain Python date objects don't have a date() method
                        if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                            # pandas Timestamp or datetime object - convert to Python date
                            normalized_d = d.date()
                        elif isinstance(d, date):
                            # Plain Python date object (no date() method)
                            normalized_d = d
                        else:
                            try:
                                normalized_d = date.fromisoformat(str(d))
                            except (ValueError, AttributeError):
                                continue
                        
                        # Check if missing and not confirmed as "no data"
                        if normalized_d not in cached_dates_set:
                            date_key = (symbol.upper(), timeframe, normalized_d)
                            if date_key not in self._no_data_ranges:
                                missing_dates.append(normalized_d)
                    all_missing_dates.extend(missing_dates)
                    
                    # Show all dates if 10 or fewer, otherwise show first 5 and last 5
                    if len(missing_dates) <= 10:
                        dates_str = ", ".join(str(d) for d in missing_dates)
                        logger.info(
                            f"  Range {i}: {range_start.strftime('%Y-%m-%d %H:%M:%S')} to {range_end.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"({len(missing_dates)} date(s): {dates_str})"
                        )
                    else:
                        dates_str = ", ".join(str(d) for d in missing_dates[:5]) + " ... " + ", ".join(str(d) for d in missing_dates[-5:])
                        logger.info(
                            f"  Range {i}: {range_start.strftime('%Y-%m-%d %H:%M:%S')} to {range_end.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"({len(missing_dates)} date(s): {dates_str})"
                        )
                else:
                    # For non-daily bars, just show the range
                    logger.info(
                        f"  Range {i}: {range_start.strftime('%Y-%m-%d %H:%M:%S')} to {range_end.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            
            # Log summary of all missing dates
            if all_missing_dates and timeframe.upper().endswith("D"):
                # Remove duplicates and sort
                unique_missing = sorted(set(all_missing_dates))
                if len(unique_missing) <= 20:
                    logger.info(f"[Cache] All missing dates ({len(unique_missing)}): {[str(d) for d in unique_missing]}")
                else:
                    logger.info(
                        f"[Cache] All missing dates ({len(unique_missing)} total): "
                        f"{[str(d) for d in unique_missing[:10]]} ... {[str(d) for d in unique_missing[-10:]]}"
                    )
        
        # Fetch missing data from base engine
        api_bars: List[Bar] = []
        for range_start, range_end in missing_ranges:
            self._last_data_source = self._get_data_source_name()  # Using base engine
            
            # For daily bars, try batch fetch first (much faster), then fall back to individual days if needed
            if (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                start_date = range_start.date()
                end_date = range_end.date()
                # Get only trading days in the range (uses cached lookup)
                trading_days_in_range = _get_trading_days_cached(start_date, end_date)
                if not trading_days_in_range:
                    # No trading days in this range (all holidays/weekends)
                    logger.debug(f"[{self._last_data_source}] Skipping range {start_date} to {end_date} - no trading days")
                    continue
                
                # OPTIMIZATION: Try batch fetch first (single API call for entire range)
                # This is much faster than fetching each day individually
                # Both yfinance and MarketData.app support batch fetching with date ranges
                num_trading_days = len(trading_days_in_range)
                if num_trading_days > 1:
                    # Multi-day range: try batch fetch first
                    logger.info(
                        f"[{self._last_data_source}] Batch fetching {num_trading_days} trading days: "
                        f"{start_date} to {end_date}"
                    )
                    self._base_engine_calls += 1
                    try:
                        batch_bars = await self._base_engine.get_bars(symbol, range_start, range_end, timeframe)
                    except Exception as e:
                        # API error occurred - do NOT mark as 'no data', re-raise the error
                        logger.error(format_api_error_message(
                            self._last_data_source,
                            additional_info=f"batch fetch {start_date} to {end_date}",
                            error=e
                        ))
                        raise  # Re-raise all errors
                    
                    if batch_bars:
                        # Batch fetch succeeded - check if we got all expected trading days
                        received_dates = {b.timestamp.date() for b in batch_bars}
                        # Normalize trading days to Python date objects for comparison
                        # trading_days_in_range should already be normalized by _get_trading_days_cached,
                        # but we normalize again here for safety
                        # Note: pandas Timestamp is a subclass of date, so isinstance(d, date) returns True
                        # We need to check if it has a date() method (Timestamp/datetime) vs plain date
                        expected_dates = set()
                        for d in trading_days_in_range:
                            # Check if it has a date() method (pandas Timestamp or datetime objects)
                            # Plain Python date objects don't have a date() method
                            if hasattr(d, 'date') and callable(getattr(d, 'date', None)):
                                # pandas Timestamp or datetime object - convert to Python date
                                expected_dates.add(d.date())
                            elif isinstance(d, date):
                                # Plain Python date object (no date() method)
                                expected_dates.add(d)
                            else:
                                # Fallback: try to convert string or other format
                                try:
                                    expected_dates.add(date.fromisoformat(str(d)))
                                except (ValueError, AttributeError):
                                    logger.warning(f"Could not normalize trading day: {d} (type: {type(d)})")
                                    continue
                        missing_dates = expected_dates - received_dates
                        
                        api_bars.extend(batch_bars)
                        # #region agent log
                        with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                            import json
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"cached_engine.py:583","message":"api_bars extended with batch_bars","data":{"api_bars_count":len(api_bars),"batch_bars_count":len(batch_bars),"batch_bars_dates":[str(b.timestamp.date()) for b in batch_bars[:5]]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                        # #endregion
                        self._total_bars_from_api += len(batch_bars)
                        timestamps = [b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in batch_bars]
                        logger.info(
                            f"[{self._last_data_source}] Batch fetch received {len(batch_bars)} bars "
                            f"(expected {num_trading_days} trading days): {timestamps[:3]}{'...' if len(timestamps) > 3 else ''}"
                        )
                        
                        # If some dates are missing, fetch them individually
                        if missing_dates:
                            # missing_dates are now Python date objects (normalized above)
                            missing_dates_sorted = sorted(missing_dates)
                            logger.info(
                                f"[{self._last_data_source}] Batch fetch missing {len(missing_dates)} trading day(s), "
                                f"fetching individually: {[str(d) for d in missing_dates_sorted[:5]]}{'...' if len(missing_dates) > 5 else ''}"
                            )
                            for trading_day in missing_dates_sorted:
                                self._base_engine_calls += 1
                                # trading_day is now a Python date object
                                day_start = datetime.combine(trading_day, datetime.min.time())
                                day_end = datetime.combine(trading_day, datetime.max.time())
                                logger.debug(f"[{self._last_data_source}] Fetching missing trading day: {trading_day.isoformat()}")
                                try:
                                    bars = await self._base_engine.get_bars(symbol, day_start, day_end, timeframe)
                                    # #region agent log
                                    with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                                        import json
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"cached_engine.py:654","message":"fetched bars for missing trading day","data":{"trading_day":str(trading_day),"day_start":str(day_start),"day_end":str(day_end),"bars_count":len(bars) if bars else 0,"bar_timestamps":[str(b.timestamp) for b in bars[:3]] if bars else []},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                                    # #endregion
                                    api_bars.extend(bars)
                                    if bars:
                                        self._total_bars_from_api += len(bars)
                                        logger.debug(f"[{self._last_data_source}] Retrieved {len(bars)} bar(s) for {trading_day.isoformat()}")
                                    elif not bars:
                                        # API call succeeded but returned no data
                                        # Check if it's a trading day - if so, this is an error
                                        if is_trading_day:
                                            try:
                                                is_trading = is_trading_day(trading_day)
                                                if is_trading:
                                                    # Trading day with no data - this is an error
                                                    error_msg = (
                                                        f"[{self._last_data_source}] Failed to get data for trading day {trading_day.isoformat()}. "
                                                        f"API call succeeded but returned no data. This may indicate a data availability issue."
                                                    )
                                                    logger.error(error_msg)
                                                    raise RuntimeError(error_msg)
                                                else:
                                                    # Not a trading day - mark as 'no data' (expected)
                                                    date_key = (symbol.upper(), timeframe, trading_day)
                                                    self._no_data_ranges[date_key] = True
                                                    logger.debug(
                                                        f"[{self._last_data_source}] No data available for {trading_day.isoformat()} "
                                                        f"(non-trading day). Marked as 'no data' to avoid future API calls."
                                                    )
                                            except Exception as e:
                                                # If is_trading_day check fails, assume it's a trading day and error
                                                error_msg = (
                                                    f"[{self._last_data_source}] Failed to get data for {trading_day.isoformat()}. "
                                                    f"API call succeeded but returned no data. Trading day check failed: {e}"
                                                )
                                                logger.error(error_msg)
                                                raise RuntimeError(error_msg) from e
                                        else:
                                            # Trading calendar not available - can't determine if trading day
                                            # For safety, raise error rather than marking as 'no data'
                                            error_msg = (
                                                f"[{self._last_data_source}] Failed to get data for {trading_day.isoformat()}. "
                                                f"API call succeeded but returned no data. Trading calendar not available to verify if this is a trading day."
                                            )
                                            logger.error(error_msg)
                                            raise RuntimeError(error_msg)
                                except Exception as e:
                                    # API error occurred - do NOT mark as 'no data', re-raise the error
                                    logger.error(format_api_error_message(
                                        self._last_data_source,
                                        date=trading_day.isoformat(),
                                        error=e
                                    ))
                                    raise  # Re-raise all errors
                    else:
                        # Batch fetch returned no data - try individual days
                        logger.info(
                            f"[{self._last_data_source}] Batch fetch returned no data, "
                            f"trying individual trading days ({num_trading_days} days)"
                        )
                        for trading_day in trading_days_in_range:
                            self._base_engine_calls += 1
                            day_start = datetime.combine(trading_day, datetime.min.time())
                            day_end = datetime.combine(trading_day, datetime.max.time())
                            logger.debug(f"[{self._last_data_source}] Fetching trading day: {trading_day.isoformat()}")
                            try:
                                bars = await self._base_engine.get_bars(symbol, day_start, day_end, timeframe)
                                api_bars.extend(bars)
                                if bars:
                                    self._total_bars_from_api += len(bars)
                                elif not bars:
                                    # API call succeeded but returned no data
                                    # Check if it's a trading day - if so, this is an error
                                    if is_trading_day:
                                        try:
                                            is_trading = is_trading_day(trading_day)
                                            if is_trading:
                                                # Trading day with no data - this is an error
                                                error_msg = (
                                                    f"[{self._last_data_source}] Failed to get data for trading day {trading_day.isoformat()}. "
                                                    f"API call succeeded but returned no data. This may indicate a data availability issue."
                                                )
                                                logger.error(error_msg)
                                                raise RuntimeError(error_msg)
                                            else:
                                                # Not a trading day - mark as 'no data' (expected)
                                                date_key = (symbol.upper(), timeframe, trading_day)
                                                self._no_data_ranges[date_key] = True
                                                logger.debug(f"[Cache] Marked {trading_day} as 'no data' to avoid future API calls")
                                        except Exception as e:
                                            # If is_trading_day check fails, assume it's a trading day and error
                                            error_msg = (
                                                f"[{self._last_data_source}] Failed to get data for {trading_day.isoformat()}. "
                                                f"API call succeeded but returned no data. Trading day check failed: {e}"
                                            )
                                            logger.error(error_msg)
                                            raise RuntimeError(error_msg) from e
                                    else:
                                        # Trading calendar not available - can't determine if trading day
                                        # For safety, raise error rather than marking as 'no data'
                                        error_msg = (
                                            f"[{self._last_data_source}] Failed to get data for {trading_day.isoformat()}. "
                                            f"API call succeeded but returned no data. Trading calendar not available to verify if this is a trading day."
                                        )
                                        logger.error(error_msg)
                                        raise RuntimeError(error_msg)
                            except Exception as e:
                                # API error occurred - do NOT mark as 'no data', re-raise the error
                                logger.error(format_api_error_message(
                                    self._last_data_source,
                                    date=trading_day.isoformat(),
                                    error=e
                                ))
                                raise  # Re-raise all errors
                else:
                    # Single day: fetch directly
                    trading_day = trading_days_in_range[0]
                    self._base_engine_calls += 1
                    day_start = datetime.combine(trading_day, datetime.min.time())
                    day_end = datetime.combine(trading_day, datetime.max.time())
                    logger.info(f"[{self._last_data_source}] Fetching single trading day: {trading_day.isoformat()}")
                    try:
                        bars = await self._base_engine.get_bars(symbol, day_start, day_end, timeframe)
                        api_bars.extend(bars)
                        if bars:
                            self._total_bars_from_api += len(bars)
                            timestamps = [b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in bars]
                            logger.info(f"[{self._last_data_source}] Received {len(bars)} bars: {timestamps[:3]}{'...' if len(timestamps) > 3 else ''}")
                        elif not bars:
                            # API call succeeded but returned no data
                            # Check if it's a trading day - if so, this is an error
                            if is_trading_day:
                                try:
                                    is_trading = is_trading_day(trading_day)
                                    if is_trading:
                                        # Trading day with no data - this is an error
                                        error_msg = (
                                            f"[{self._last_data_source}] Failed to get data for trading day {trading_day.isoformat()}. "
                                            f"API call succeeded but returned no data. This may indicate a data availability issue."
                                        )
                                        logger.error(error_msg)
                                        raise RuntimeError(error_msg)
                                    else:
                                        # Not a trading day - mark as 'no data' (expected)
                                        date_key = (symbol.upper(), timeframe, trading_day)
                                        self._no_data_ranges[date_key] = True
                                        logger.debug(f"[Cache] Marked {trading_day} as 'no data' to avoid future API calls")
                                except Exception as e:
                                    # If is_trading_day check fails, assume it's a trading day and error
                                    error_msg = (
                                        f"[{self._last_data_source}] Failed to get data for {trading_day.isoformat()}. "
                                        f"API call succeeded but returned no data. Trading day check failed: {e}"
                                    )
                                    logger.error(error_msg)
                                    raise RuntimeError(error_msg) from e
                            else:
                                # Trading calendar not available - can't determine if trading day
                                # For safety, raise error rather than marking as 'no data'
                                error_msg = (
                                    f"[{self._last_data_source}] Failed to get data for {trading_day.isoformat()}. "
                                    f"API call succeeded but returned no data. Trading calendar not available to verify if this is a trading day."
                                )
                                logger.error(error_msg)
                                raise RuntimeError(error_msg)
                    except Exception as e:
                        # API error occurred - do NOT mark as 'no data', re-raise the error
                        logger.error(format_api_error_message(
                            self._last_data_source,
                            date=trading_day.isoformat(),
                            error=e
                        ))
                        raise  # Re-raise all errors
            else:
                # For non-daily bars or if trading calendar not available, use the range as-is
                self._base_engine_calls += 1
                range_days = (range_end.date() - range_start.date()).days + 1 if timeframe.upper().endswith("D") else 0
                if timeframe.upper().endswith("D") and range_start.date() == range_end.date():
                    logger.info(
                        f"[{self._last_data_source}] Fetching single day: "
                        f"{range_start.strftime('%Y-%m-%d')} (using 'date' parameter)"
                    )
                else:
                    logger.info(
                        f"[{self._last_data_source}] Fetching missing range: "
                        f"{range_start.strftime('%Y-%m-%d %H:%M:%S')} to {range_end.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"({range_days} day(s) for daily bars)"
                    )
                try:
                    bars = await self._base_engine.get_bars(symbol, range_start, range_end, timeframe)
                    api_bars.extend(bars)
                    if bars:
                        self._total_bars_from_api += len(bars)
                        timestamps = [b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in bars]
                        logger.info(f"[{self._last_data_source}] Received {len(bars)} bars: {timestamps[:3]}{'...' if len(timestamps) > 3 else ''}")
                    elif not bars and range_start < range_end:
                        # API call succeeded but returned no data
                        # For daily bars, check if any trading days are missing data - if so, error
                        if timeframe.upper().endswith("D") or timeframe.upper() == "D":
                            # For daily bars, check each trading day in the range
                            start_date = range_start.date()
                            end_date = range_end.date()
                            trading_days_in_range = _get_trading_days_cached(start_date, end_date)
                            
                            if trading_days_in_range:
                                # We have trading days but got no data - this is an error
                                trading_days_str = ", ".join([str(d) for d in trading_days_in_range[:5]])
                                if len(trading_days_in_range) > 5:
                                    trading_days_str += f", ... ({len(trading_days_in_range)} total)"
                                error_msg = (
                                    f"[{self._last_data_source}] Failed to get data for trading day(s) in range {start_date} to {end_date}. "
                                    f"API call succeeded but returned no data for {len(trading_days_in_range)} trading day(s): {trading_days_str}. "
                                    f"This may indicate a data availability issue."
                                )
                                logger.error(error_msg)
                                raise RuntimeError(error_msg)
                            else:
                                # No trading days in range (all holidays/weekends) - mark as 'no data'
                                logger.debug(f"[{self._last_data_source}] No trading days in range {start_date} to {end_date} (all holidays/weekends)")
                        else:
                            # For non-daily bars, we can't easily determine if this is expected
                            # For safety, raise error rather than marking as 'no data'
                            error_msg = (
                                f"[{self._last_data_source}] Failed to get data for range {range_start} to {range_end}. "
                                f"API call succeeded but returned no data. This may indicate a data availability issue."
                            )
                            logger.error(error_msg)
                            raise RuntimeError(error_msg)
                except Exception as e:
                    # API error occurred - do NOT mark as 'no data', re-raise the error
                    logger.error(format_api_error_message(
                        self._last_data_source,
                        additional_info=f"range {range_start} to {range_end}",
                        error=e
                    ))
                    raise  # Re-raise all errors
        
        # Filter out bars that already exist in cache before merging
        # This prevents duplicate warnings and unnecessary save attempts
        if all_cached_bars and api_bars:
            from core.utils.timestamp import normalize_timestamp_for_comparison
            cached_timestamps = {(normalize_timestamp_for_comparison(b.timestamp), b.symbol, b.timeframe) for b in all_cached_bars}
            # #region agent log
            with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"cached_engine.py:841","message":"before filtering api_bars","data":{"api_bars_count":len(api_bars),"cached_timestamps_count":len(cached_timestamps),"api_bars_dates":[str(b.timestamp.date()) for b in api_bars[:5]]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            new_api_bars = [
                b for b in api_bars
                if (normalize_timestamp_for_comparison(b.timestamp), b.symbol, b.timeframe) not in cached_timestamps
            ]
            if len(new_api_bars) < len(api_bars):
                filtered_count = len(api_bars) - len(new_api_bars)
                filtered_timestamps = [
                    b.timestamp for b in api_bars
                    if (normalize_timestamp_for_comparison(b.timestamp), b.symbol, b.timeframe) in cached_timestamps
                ]
                logger.debug(
                    f"Filtered out {filtered_count} duplicate bars from {self._get_data_source_name()} response. "
                    f"Filtered timestamps: {[str(ts) for ts in filtered_timestamps[:3]]}"
                )
                # #region agent log
                with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"cached_engine.py:856","message":"after filtering api_bars","data":{"filtered_count":filtered_count,"new_api_bars_count":len(new_api_bars),"filtered_timestamps":[str(ts) for ts in filtered_timestamps]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
            api_bars = new_api_bars
        
        # Merge cached and base engine data
        if cached_bars:
            # #region agent log
            with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"cached_engine.py:859","message":"before merge","data":{"cached_bars_count":len(cached_bars),"api_bars_count":len(api_bars) if api_bars else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            all_bars = self._cache._merge_and_deduplicate(cached_bars, api_bars)
            # #region agent log
            with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
                import json
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"cached_engine.py:860","message":"after merge","data":{"all_bars_count":len(all_bars) if all_bars else 0},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
        else:
            all_bars = api_bars
        
        # Save to cache if we got new data
        # Only save the NEW bars (api_bars), not all_bars, to avoid duplicate merging
        if self._cache_enabled and self._cache and api_bars:
            # Check if any of the fetched bars are for trading days (for silent cache update)
            is_trading_day_data = False
            if (timeframe.upper().endswith("D") or timeframe.upper() == "D"):
                if is_trading_day:
                    api_dates = {b.timestamp.date() for b in api_bars}
                    is_trading_day_data = any(
                        is_trading_day(d) for d in api_dates
                    )
            
            # Save only the new bars - save_bars will merge with existing cache
            await self._cache.save_bars(symbol, timeframe, api_bars)
            
            # Log silently for trading day data (only debug level), otherwise info level
            if is_trading_day_data:
                # Silent update for trading days (debug level only)
                logger.debug(f"[Cache] Silently updated cache with {len(api_bars)} bar(s) for trading day(s)")
            else:
                # Normal logging for non-trading day data
                if api_bars:
                    timestamps = [b.timestamp.strftime('%Y-%m-%d %H:%M:%S') for b in api_bars]
                    logger.info(f"[Cache] Saved {len(api_bars)} new bars: {timestamps[:3]}{'...' if len(timestamps) > 3 else ''}")
            
            # After saving, refresh all_cached_bars from cache to include newly saved data
            # This ensures subsequent requests in the same run see the updated cache
            all_cached_bars = self._cache.get_all_cached_bars(symbol, timeframe)
        
        # Filter to exact requested range and return
        # Use normalized timestamp comparison (same as check_coverage and cached_bars filtering) for consistency
        unit_seconds = self._cache._get_time_unit_seconds(timeframe)
        if unit_seconds:
            # Normalize timestamps to unit boundaries (same as check_coverage)
            normalized_start = self._cache._normalize_timestamp_to_unit(start, unit_seconds)
            normalized_end = self._cache._normalize_timestamp_to_unit(end, unit_seconds)
            result = [
                b for b in all_bars
                if normalized_start <= self._cache._normalize_timestamp_to_unit(b.timestamp, unit_seconds) <= normalized_end
            ]
        else:
            # For unknown timeframes, use exact timestamp comparison
            result = [b for b in all_bars if start <= b.timestamp <= end]
        # #region agent log
        with open(r'c:\Users\JiantaoPan\OneDrive\Documents\Code\LLM-Investment-Copilot\.cursor\debug.log', 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"cached_engine.py:995","message":"final filter result","data":{"all_bars_count":len(all_bars) if all_bars else 0,"result_count":len(result),"start":str(start),"end":str(end),"result_dates":[str(b.timestamp.date()) for b in result[:5]]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        return result
    
    async def stream_bars(
        self,
        symbol: str,
        timeframe: str,
    ) -> AsyncIterator[Bar]:
        """
        Stream bars from base engine (no caching for streaming).
        
        Streaming data is not cached as it's real-time.
        """
        async for bar in self._base_engine.stream_bars(symbol, timeframe):
            yield bar
    
    async def get_option_chain(
        self,
        underlying: str,
        as_of: date,
    ) -> list[OptionContract]:
        """
        Get option chain from base engine (no caching for options).
        
        Option chains are not cached as they change frequently.
        """
        return await self._base_engine.get_option_chain(underlying, as_of)
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Cache Hit Classification:
        - **Full Hit**: Requested range is fully covered by cache (no API call needed)
        - **Partial Hit**: Some cached data overlaps with request, but missing ranges need API fetch
          (Only counted if overlapping_bars > 0 - having cache but no overlap is a cache miss)
        - **Cache Miss**: No cache exists or no overlapping bars (entire range fetched from API)
        
        Note: When data is fetched from API and saved, it benefits FUTURE requests.
        The CURRENT request is classified based on what was in cache BEFORE fetching.
        
        Returns:
            Dictionary with cache statistics:
            - total_requests: Total number of data requests
            - cache_hits: Number of requests fully served from cache (no API call)
            - cache_partial_hits: Number of requests with overlapping cached bars (API call needed for missing ranges)
            - api_calls: Number of calls to base engine
            - total_bars_from_cache: Total bars retrieved from cache
            - total_bars_from_api: Total bars retrieved from base engine
        """
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_partial_hits": self._cache_partial_hits,
            "api_calls": self._base_engine_calls,
            "total_bars_from_cache": self._total_bars_from_cache,
            "total_bars_from_api": self._total_bars_from_api,
        }
    
    @property
    def last_data_source(self) -> str:
        """Get the data source name for the last request (for logging context)."""
        return self._last_data_source

