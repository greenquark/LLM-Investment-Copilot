"""
CNN Fear & Greed Index Data Provider

Provides a unified interface for fetching Fear & Greed Index (FGI) data with caching:
- Cache (file-based, similar to bars cache)
- CNN API (if cache miss)
- CSV fallback (fear-greed-2011-2023.csv for historical data if API fails)

Usage:
    from core.data.fear_greed_index import get_fgi_value
    
    # Get current FGI
    value, classification = get_fgi_value()
    
    # Get historical FGI for a specific date
    value, classification = get_fgi_value(target_date=date(2024, 1, 15))
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import fear_and_greed package
try:
    import fear_and_greed
except ImportError:
    fear_and_greed = None

# Try to import httpx for CNN API
try:
    import httpx
except ImportError:
    httpx = None

# Try to import pandas for CSV loading and caching
try:
    import pandas as pd
    import pyarrow.parquet as pq
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    pq = None
    _PANDAS_AVAILABLE = False

# Cache directory for FGI data
_cache_dir: Optional[Path] = None
_cache_enabled = True

# In-memory cache: date -> (value, classification)
_in_memory_cache: dict = {}

# Fallback CSV data cache (loaded once)
_fallback_csv_cache: Optional[dict] = None


def initialize_cache(cache_dir: str = "data_cache/fgi", enabled: bool = True):
    """
    Initialize the FGI cache system.
    
    Args:
        cache_dir: Directory to store cache files (default: "data_cache/fgi")
        enabled: Whether caching is enabled (default: True)
    """
    global _cache_dir, _cache_enabled
    
    _cache_enabled = enabled and _PANDAS_AVAILABLE
    
    if _cache_enabled:
        _cache_dir = Path(cache_dir)
        _cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FGI cache initialized with directory: {_cache_dir}")
    else:
        _cache_dir = None
        if enabled and not _PANDAS_AVAILABLE:
            logger.warning("FGI caching requested but pandas/pyarrow not available")


def _get_cache_path() -> Optional[Path]:
    """Get the path to the cache file."""
    if not _cache_enabled or _cache_dir is None:
        return None
    return _cache_dir / "fear_greed_index.parquet"


def _load_from_cache(target_date: date) -> Optional[Tuple[float, str]]:
    """
    Load FGI value from cache (in-memory or file).
    
    Returns:
        Tuple of (value, classification) or None if not in cache.
    """
    # Check in-memory cache first
    if target_date in _in_memory_cache:
        return _in_memory_cache[target_date]
    
    # Check file cache
    cache_path = _get_cache_path()
    if cache_path is None or not cache_path.exists():
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        
        # Ensure date column is date type
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:
            # If date is index, convert it
            if df.index.name == 'date':
                df.index = pd.to_datetime(df.index).date
        
        # Find matching date
        if 'date' in df.columns:
            match = df[df['date'] == target_date]
        else:
            match = df[df.index == target_date]
        
        if not match.empty:
            row = match.iloc[0]
            value = float(row['value'])
            classification = str(row.get('classification', '')) if 'classification' in row else ''
            
            # Store in in-memory cache
            _in_memory_cache[target_date] = (value, classification)
            return (value, classification)
    except Exception as e:
        logger.warning(f"Failed to load FGI from cache: {e}")
    
    return None


def _save_to_cache(target_date: date, value: float, classification: str):
    """
    Save FGI value to cache (both in-memory and file).
    
    Retries up to 3 times with 1 second delay if permission errors occur.
    """
    import time
    
    # Store in in-memory cache
    _in_memory_cache[target_date] = (value, classification)
    
    # Save to file cache
    cache_path = _get_cache_path()
    if cache_path is None:
        return
    
    max_retries = 3
    retry_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            # Load existing cache or create new DataFrame
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
            else:
                df = pd.DataFrame(columns=['date', 'value', 'classification'])
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Check if date already exists
            if 'date' in df.columns:
                existing = df[df['date'] == target_date]
                if not existing.empty:
                    # Update existing row
                    df.loc[df['date'] == target_date, 'value'] = value
                    df.loc[df['date'] == target_date, 'classification'] = classification
                else:
                    # Add new row
                    new_row = pd.DataFrame({
                        'date': [target_date],
                        'value': [value],
                        'classification': [classification]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                # Date is index
                if target_date in df.index:
                    df.loc[target_date, 'value'] = value
                    df.loc[target_date, 'classification'] = classification
                else:
                    df.loc[target_date] = {'value': value, 'classification': classification}
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date')
            else:
                df = df.sort_index()
            
            # Save to parquet
            df.to_parquet(cache_path, index=False if 'date' in df.columns else True)
            logger.debug(f"Saved FGI to cache: {target_date} = {value}")
            return  # Success, exit retry loop
            
        except (PermissionError, OSError) as e:
            # Permission or file access errors - retry
            if attempt < max_retries - 1:
                logger.debug(f"Cache save failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                # Final attempt failed
                logger.warning(f"Failed to save FGI to cache after {max_retries} attempts: {e}")
        except Exception as e:
            # Other errors - don't retry
            logger.warning(f"Failed to save FGI to cache: {e}")
            return


def _load_fallback_csv() -> Optional[dict]:
    """
    Load FGI data from fallback CSV file (fear-greed-2011-2023.csv).
    
    CSV format: Date,Fear Greed (no classification column)
    
    Returns:
        Dictionary mapping date -> (value, None) or None if CSV unavailable
    """
    global _fallback_csv_cache
    
    # Return cached data if available
    if _fallback_csv_cache is not None:
        return _fallback_csv_cache
    
    if pd is None:
        logger.debug("pandas not installed, cannot load fallback CSV data")
        return None
    
    # Determine CSV file path (relative to this module)
    module_dir = Path(__file__).parent
    csv_file_path = module_dir / "fear-greed-2011-2023.csv"
    
    if not csv_file_path.exists():
        logger.debug(f"Fallback CSV file not found: {csv_file_path}")
        return None
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file_path)
        
        # Parse date column (format: M/D/YYYY)
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.date
        
        # Ensure value is numeric
        df["Fear Greed"] = pd.to_numeric(df["Fear Greed"], errors="coerce")
        
        # Remove rows with invalid values
        df = df.dropna(subset=["Fear Greed", "Date"])
        
        # Create dictionary mapping date -> (value, None)
        _fallback_csv_cache = {}
        for _, row in df.iterrows():
            date_key = row["Date"]
            value = float(row["Fear Greed"])
            # No classification in this CSV, so use None
            _fallback_csv_cache[date_key] = (value, None)
        
        logger.debug(f"Loaded {len(_fallback_csv_cache)} FGI records from fallback CSV")
        return _fallback_csv_cache
        
    except Exception as e:
        logger.warning(f"Failed to load FGI data from fallback CSV: {e}")
        return None


def _get_historical_fgi_from_fallback_csv(target_date: date) -> Tuple[Optional[float], Optional[str]]:
    """
    Get historical FGI value from fallback CSV file for a specific date.
    
    Args:
        target_date: Date to fetch FGI for.
    
    Returns:
        Tuple of (value, None) or (None, None) if not found.
    """
    fallback_data = _load_fallback_csv()
    if fallback_data is None:
        return None, None
    
    # Try exact match
    if target_date in fallback_data:
        value, _ = fallback_data[target_date]
        return value, None
    
    # Not found
    return None, None


def get_fgi_value(target_date: Optional[date] = None) -> Tuple[Optional[float], Optional[str]]:
    """
    Get Fear & Greed Index value for a specific date or current value.
    
    Priority order for historical dates:
    1. Cache (file-based, similar to bars cache)
    2. CNN API (if cache miss)
    3. Fallback CSV (fear-greed-2011-2023.csv if API fails for historical data)
    4. Current value from fear-and-greed package (final fallback)
    
    Args:
        target_date: Optional date to fetch historical FGI. If None, returns current value.
    
    Returns:
        Tuple of (value, classification) or (None, None) if unavailable.
        - value: FGI value (0-100)
        - classification: Classification string (e.g., "neutral", "fear", "greed")
    """
    # Initialize cache if not already done
    if _cache_dir is None:
        initialize_cache()
    
    if target_date is None:
        # Get current value from fear-and-greed package
        return _get_current_fgi()
    else:
        # Get historical value: try cache first, then API, then fallback CSV, then current value
        
        # 1. Try cache
        cached_value, cached_classification = _load_from_cache(target_date)
        if cached_value is not None:
            logger.debug(f"FGI cache hit for {target_date}")
            return cached_value, cached_classification
        
        logger.debug(f"FGI cache miss for {target_date}, trying API...")
        
        # 2. Try API
        api_value, api_classification = _get_historical_fgi_from_api(target_date)
        if api_value is not None:
            # Save to cache
            _save_to_cache(target_date, api_value, api_classification or "")
            logger.debug(f"FGI API success for {target_date}, saved to cache")
            return api_value, api_classification
        
        logger.debug(f"FGI API failed for {target_date}, trying fallback CSV...")
        
        # 3. Try fallback CSV (only for historical data)
        fallback_value, _ = _get_historical_fgi_from_fallback_csv(target_date)
        if fallback_value is not None:
            # Save to cache (without classification since CSV doesn't have it)
            _save_to_cache(target_date, fallback_value, "")
            logger.debug(f"FGI fallback CSV success for {target_date}, saved to cache")
            return fallback_value, None
        
        logger.debug(f"FGI fallback CSV failed for {target_date}, using current value as final fallback")
        
        # 4. Final fallback to current value
        return _get_current_fgi()


def _get_current_fgi() -> Tuple[Optional[float], Optional[str]]:
    """
    Get current FGI value from fear-and-greed package.
    
    Returns:
        Tuple of (value, classification) or (None, None) if unavailable.
    """
    if fear_and_greed is None:
        logger.debug("fear-and-greed package not installed")
        return None, None
    
    try:
        index_data = fear_and_greed.get()
        if index_data and hasattr(index_data, 'value'):
            value = float(index_data.value)
            classification = None
            if hasattr(index_data, 'description'):
                classification = str(index_data.description)
            return value, classification
    except Exception as e:
        logger.warning(f"Failed to get current FGI value: {e}")
    
    return None, None


def _get_historical_fgi_from_api(target_date: date) -> Tuple[Optional[float], Optional[str]]:
    """
    Get historical FGI value from CNN API for a specific date.
    
    API endpoint: https://production.dataviz.cnn.io/index/fearandgreed/graphdata/YYYY-MM-DD
    
    Args:
        target_date: Date to fetch FGI for.
    
    Returns:
        Tuple of (value, classification) or (None, None) if unavailable.
        Does NOT fall back - caller should handle fallback.
    """
    if httpx is None:
        logger.debug("httpx package not installed, cannot fetch historical FGI from API")
        return None, None
    
    date_str = target_date.strftime("%Y-%m-%d")
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{date_str}"
    
    # Headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.cnn.com/",
        "Origin": "https://www.cnn.com",
    }
    
    try:
        with httpx.Client(timeout=10.0, headers=headers, follow_redirects=True) as client:
            response = client.get(url)
            
            if response.status_code == 418:
                logger.warning(f"CNN API blocked request (418 - bot detection) for {date_str}")
                return None, None
            
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
            except Exception as e:
                logger.warning(f"Failed to parse CNN API response for {date_str}: {e}")
                return None, None
            
            # Parse the response structure:
            # {
            #   "fear_and_greed": { 
            #     "score": value, 
            #     "rating": classification, 
            #     "timestamp": "2025-12-29T23:59:58+00:00",
            #     "previous_close": ...,
            #     ...
            #   },
            #   "fear_and_greed_historical": { 
            #     "data": [ { "x": timestamp_ms, "y": fgi_value, "rating": classification }, ... ] 
            #   }
            # }
            value = None
            classification = None
            
            # Convert target date to date object for comparison
            target_date_only = target_date
            
            if isinstance(data, dict):
                # First, try to get from fear_and_greed_historical.data array
                if "fear_and_greed_historical" in data:
                    historical = data["fear_and_greed_historical"]
                    if isinstance(historical, dict) and "data" in historical:
                        data_array = historical["data"]
                        if isinstance(data_array, list):
                            # Convert target date to timestamp (start of day in UTC)
                            target_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                            target_timestamp_ms = int(target_datetime.timestamp() * 1000)
                            
                            # Find the data point that matches our target date
                            # Look for exact match or closest match within 24 hours
                            best_match = None
                            min_diff = float('inf')
                            
                            for point in data_array:
                                if isinstance(point, dict) and "x" in point and "y" in point:
                                    point_timestamp = float(point["x"])
                                    diff = abs(point_timestamp - target_timestamp_ms)
                                    
                                    # If exact match or within 24 hours, use it
                                    if diff < min_diff and diff < 86400000:  # 24 hours in ms
                                        min_diff = diff
                                        best_match = point
                            
                            if best_match:
                                # The 'y' field IS the FGI value (0-100)
                                value = float(best_match["y"])
                                if "rating" in best_match:
                                    classification = str(best_match["rating"])
                                logger.debug(f"Found FGI value in historical data for {date_str}")
                
                # Second, try fear_and_greed score - but ONLY if timestamp matches target date
                if value is None and "fear_and_greed" in data:
                    current = data["fear_and_greed"]
                    if isinstance(current, dict) and "score" in current:
                        # Check timestamp to verify it matches target date
                        timestamp_str = current.get("timestamp")
                        if timestamp_str:
                            try:
                                # Parse timestamp (format: "2025-12-29T23:59:58+00:00")
                                # Extract date part from ISO format timestamp
                                timestamp_date_str = timestamp_str.split("T")[0]  # Get "YYYY-MM-DD" part
                                timestamp_date = datetime.strptime(timestamp_date_str, "%Y-%m-%d").date()
                                
                                # Only use this value if timestamp date matches target date
                                if timestamp_date == target_date_only:
                                    value = float(current["score"])
                                    if "rating" in current:
                                        classification = str(current["rating"])
                                    logger.debug(f"Found FGI value in fear_and_greed for {date_str} (timestamp verified: {timestamp_date})")
                                else:
                                    logger.debug(
                                        f"fear_and_greed timestamp {timestamp_date} does not match target date {target_date_only}, "
                                        f"not using this value"
                                    )
                            except Exception as e:
                                # If timestamp parsing fails, don't use this value
                                logger.debug(f"Failed to parse timestamp '{timestamp_str}': {e}, not using fear_and_greed score")
                        else:
                            # No timestamp field - don't use this value as we can't verify it
                            logger.debug(f"fear_and_greed has no timestamp field, cannot verify date match, not using score")
            
            if value is not None:
                return value, classification
            else:
                logger.debug(f"Could not find verified FGI value for target date {date_str} in CNN API response")
                return None, None
                
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.debug(f"CNN API: Date {date_str} not found (404)")
        else:
            logger.warning(f"CNN API HTTP error {e.response.status_code} for {date_str}: {e.response.text[:200]}")
        return None, None
    except httpx.TimeoutException:
        logger.warning(f"CNN API request timeout for {date_str}")
        return None, None
    except Exception as e:
        logger.warning(f"Error fetching historical FGI from CNN API for {date_str}: {e}")
        return None, None
    
    # Should not reach here, but return None if we do
    return None, None
